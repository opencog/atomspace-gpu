/*
 * opencog/persist/gpu/GpuStorageNode.cc
 *
 * StorageNode backed by GPU SoA pools.
 * Delegates all GPU operations to a GpuBackend (CUDA or OpenCL).
 *
 * Phase 1: Store/Fetch round-trip for atoms and values.
 * Phase 2: fetchIncomingByType, fetchIncomingSet, loadType, runQuery.
 * Phase 2.5: Type-aware storage — atom Type stored in GPU pools,
 *            mixed into hash key to prevent same-name collisions.
 *
 * Copyright (C) 2025 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <sstream>
#include <functional>
#include <cstring>

#include <opencog/util/exceptions.h>
#include <opencog/util/Logger.h>
#include <opencog/atomspace/AtomSpace.h>
#include <opencog/atoms/base/Node.h>
#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/value/FloatValue.h>
#include <opencog/atoms/atom_types/atom_types.h>
#include <opencog/atoms/atom_types/NameServer.h>
#include <opencog/persist/gpu-types/atom_types.h>

#include "GpuStorageNode.h"

using namespace opencog;

// ==============================================================
// Constructors / Destructor

GpuStorageNode::GpuStorageNode(Type t, const std::string&& uri) :
	StorageNode(t, std::move(uri)),
	_connected(false),
	_num_stores(0),
	_num_fetches(0)
{
	_uri = get_name();
	parse_uri();
}

GpuStorageNode::GpuStorageNode(const std::string&& uri) :
	StorageNode(GPU_STORAGE_NODE, std::move(uri)),
	_connected(false),
	_num_stores(0),
	_num_fetches(0)
{
	_uri = get_name();
	parse_uri();
}

GpuStorageNode::~GpuStorageNode()
{
	if (_connected) close();
}

// ==============================================================
// URI parsing: "gpu://[backend:]platform:device"
//
// Extended URI format:
//   "gpu://:"                  -> auto backend, any device
//   "gpu://NVIDIA:RTX"         -> auto backend, NVIDIA RTX
//   "gpu://cuda::"             -> force CUDA, any device
//   "gpu://opencl:Intel:"      -> force OpenCL, Intel device
//   "gpu://cuda:NVIDIA:RTX"    -> CUDA, NVIDIA RTX

void GpuStorageNode::parse_uri(void)
{
	const std::string& url = _uri;
	if (0 != url.compare(0, 6, "gpu://"))
		throw RuntimeException(TRACE_INFO,
			"Unsupported URL \"%s\"\n"
			"\tExpecting 'gpu://[backend:]platform:device'",
			url.c_str());

	std::string rest = url.substr(6);

	// Check for explicit backend hint: "cuda:" or "opencl:"
	_backend_hint = "";
	if (rest.compare(0, 5, "cuda:") == 0)
	{
		_backend_hint = "cuda";
		rest = rest.substr(5);
	}
	else if (rest.compare(0, 7, "opencl:") == 0)
	{
		_backend_hint = "opencl";
		rest = rest.substr(7);
	}

	// Parse platform:device from remainder
	size_t colon = rest.find(':');
	if (std::string::npos == colon)
	{
		_splat = rest;
		_sdev = "";
	}
	else
	{
		_splat = rest.substr(0, colon);
		_sdev = rest.substr(colon + 1);
	}
}

// ==============================================================
// Connection management

void GpuStorageNode::open(void)
{
	if (_connected)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: already open! %s\n", _uri.c_str());

	// Select backend: CUDA preferred, OpenCL fallback
	bool want_cuda = (_backend_hint == "cuda" or _backend_hint.empty());
	bool want_opencl = (_backend_hint == "opencl" or _backend_hint.empty());

	GpuBackend* be = nullptr;

#ifdef HAVE_CUDA
	if (want_cuda and nullptr == be)
	{
		try
		{
			be = create_cuda_backend();
			be->init(_splat, _sdev);
		}
		catch (const RuntimeException& e)
		{
			delete be;
			be = nullptr;
			if (_backend_hint == "cuda")
				throw;  // User explicitly asked for CUDA
			logger().info("GpuStorageNode: CUDA unavailable, "
				"trying OpenCL...\n");
		}
	}
#endif

#ifdef HAVE_OPENCL
	if (want_opencl and nullptr == be)
	{
		try
		{
			be = create_opencl_backend();
			be->init(_splat, _sdev);
		}
		catch (const RuntimeException& e)
		{
			delete be;
			be = nullptr;
			if (_backend_hint == "opencl")
				throw;  // User explicitly asked for OpenCL
		}
	}
#endif

	if (nullptr == be)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: no GPU backend available for '%s'\n"
			"\tCompiled with:"
#ifdef HAVE_CUDA
			" CUDA"
#endif
#ifdef HAVE_OPENCL
			" OpenCL"
#endif
			"\n",
			_uri.c_str());

	_backend.reset(be);
	_backend->alloc_pools();
	_backend->init_pools();

	_connected = true;
	_num_stores = 0;
	_num_fetches = 0;

	logger().info("GpuStorageNode: opened %s (backend: %s, device: %s)\n",
		_uri.c_str(),
		_backend->backend_name().c_str(),
		_backend->device_info().c_str());
}

void GpuStorageNode::close(void)
{
	if (not _connected) return;

	_backend->barrier();
	_backend->shutdown();
	_backend.reset();

	// Clear CPU-side maps
	{
		std::lock_guard<std::mutex> lk(_name_mtx);
		_name_to_hash.clear();
		_hash_to_name.clear();
	}
	{
		std::lock_guard<std::mutex> lk(_atom_mtx);
		_atom_map.clear();
	}

	_connected = false;
	logger().info("GpuStorageNode: closed %s\n", _uri.c_str());
}

bool GpuStorageNode::connected(void)
{
	return _connected;
}

void GpuStorageNode::erase(void)
{
	if (not _connected) return;

	_backend->init_pools();

	{
		std::lock_guard<std::mutex> lk(_name_mtx);
		_name_to_hash.clear();
		_hash_to_name.clear();
	}
	{
		std::lock_guard<std::mutex> lk(_atom_mtx);
		_atom_map.clear();
	}
}

// ==============================================================
// Name hashing — Type is mixed into the hash to prevent
// same-name-different-type collisions.

uint64_t GpuStorageNode::name_hash(Type t, const std::string& name)
{
	std::lock_guard<std::mutex> lk(_name_mtx);

	TypedName tn{t, name};
	auto it = _name_to_hash.find(tn);
	if (it != _name_to_hash.end())
		return it->second;

	// Mix type into string hash via golden ratio constant
	std::hash<std::string> hasher;
	uint64_t h = hasher(name) ^ (uint64_t(t) * 0x9E3779B97F4A7C15ULL);

	if (h == GPU_HT_EMPTY_KEY) h = 0;

	_name_to_hash[tn] = h;
	_hash_to_name[h] = tn;
	return h;
}

// ==============================================================
// GPU pool operations (delegated to backend)

uint32_t GpuStorageNode::store_word(Type t, const std::string& name)
{
	uint64_t nhash = name_hash(t, name);
	uint32_t idx = _backend->word_find_or_create(nhash);

	if (idx == GPU_NOT_FOUND)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: word pool full, cannot store '%s'\n",
			name.c_str());

	_backend->word_write_type(idx, (uint16_t)t);
	return idx;
}

uint32_t GpuStorageNode::lookup_word(Type t, const std::string& name)
{
	uint64_t nhash = name_hash(t, name);
	return _backend->word_lookup(nhash);
}

uint32_t GpuStorageNode::store_pair(uint32_t word_a, uint32_t word_b,
                                     Type link_type)
{
	uint32_t idx = _backend->pair_find_or_create(word_a, word_b);

	if (idx == GPU_NOT_FOUND)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: pair pool full\n");

	_backend->pair_write_type(idx, (uint16_t)link_type);
	return idx;
}

uint32_t GpuStorageNode::lookup_pair(uint32_t word_a, uint32_t word_b)
{
	return _backend->pair_lookup(word_a, word_b);
}

// ==============================================================
// Value storage helpers

void GpuStorageNode::store_node_values(const Handle& h, uint32_t pool_idx)
{
	ValuePtr tv = h->getValue(truth_key());
	if (nullptr == tv) return;

	FloatValuePtr fv = FloatValueCast(tv);
	if (nullptr == fv) return;

	const std::vector<double>& vals = fv->value();
	if (vals.size() > 0)
		_backend->word_write_count(pool_idx, vals[0]);
	if (vals.size() > 1)
		_backend->word_write_marginal(pool_idx, vals[1]);
}

void GpuStorageNode::load_node_values(const Handle& h, uint32_t pool_idx)
{
	double count = 0.0, marginal = 0.0;
	_backend->word_read_values(pool_idx, count, marginal);

	if (count != 0.0 or marginal != 0.0)
	{
		std::vector<double> vals = {count, marginal};
		h->setValue(truth_key(), createFloatValue(vals));
	}
}

void GpuStorageNode::store_link_values(const Handle& h, uint32_t pool_idx)
{
	ValuePtr tv = h->getValue(truth_key());
	if (nullptr == tv) return;

	FloatValuePtr fv = FloatValueCast(tv);
	if (nullptr == fv) return;

	const std::vector<double>& vals = fv->value();
	if (vals.size() > 0)
		_backend->pair_write_count(pool_idx, vals[0]);
	if (vals.size() > 1)
		_backend->pair_write_mi(pool_idx, vals[1]);
}

void GpuStorageNode::load_link_values(const Handle& h, uint32_t pool_idx)
{
	double count = 0.0, mi = 0.0;
	_backend->pair_read_values(pool_idx, count, mi);

	if (count != 0.0 or mi != 0.0)
	{
		std::vector<double> vals = {count, mi};
		h->setValue(truth_key(), createFloatValue(vals));
	}
}

// ==============================================================
// StorageNode API: storeAtom

void GpuStorageNode::storeAtom(const Handle& h, bool synchronous)
{
	if (not _connected)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: not connected!\n");

	NodePtr np = NodeCast(h);
	if (np)
	{
		uint32_t idx = store_word(h->get_type(), np->get_name());
		store_node_values(h, idx);

		std::lock_guard<std::mutex> lk(_atom_mtx);
		_atom_map[h] = {idx, 0};
		_num_stores++;

		if (synchronous) _backend->barrier();
		return;
	}

	LinkPtr lp = LinkCast(h);
	if (lp)
	{
		for (const Handle& oh : lp->getOutgoingSet())
			storeAtom(oh, false);

		if (lp->get_arity() == 2)
		{
			const Handle& ha = lp->getOutgoingAtom(0);
			const Handle& hb = lp->getOutgoingAtom(1);

			NodePtr na = NodeCast(ha);
			NodePtr nb = NodeCast(hb);
			if (na and nb)
			{
				uint32_t ia = store_word(ha->get_type(), na->get_name());
				uint32_t ib = store_word(hb->get_type(), nb->get_name());
				uint32_t pidx = store_pair(ia, ib, h->get_type());
				store_link_values(h, pidx);

				std::lock_guard<std::mutex> lk(_atom_mtx);
				_atom_map[h] = {pidx, 1};
			}
		}

		_num_stores++;
		if (synchronous) _backend->barrier();
		return;
	}
}

// ==============================================================
// StorageNode API: getAtom

void GpuStorageNode::getAtom(const Handle& h)
{
	if (not _connected)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: not connected!\n");

	NodePtr np = NodeCast(h);
	if (np)
	{
		uint32_t idx = lookup_word(h->get_type(), np->get_name());
		if (idx == GPU_NOT_FOUND) return;

		load_node_values(h, idx);
		_num_fetches++;
		return;
	}

	LinkPtr lp = LinkCast(h);
	if (lp and lp->get_arity() == 2)
	{
		const Handle& ha = lp->getOutgoingAtom(0);
		const Handle& hb = lp->getOutgoingAtom(1);

		NodePtr na = NodeCast(ha);
		NodePtr nb = NodeCast(hb);
		if (na and nb)
		{
			uint32_t ia = lookup_word(ha->get_type(), na->get_name());
			uint32_t ib = lookup_word(hb->get_type(), nb->get_name());
			if (ia == GPU_NOT_FOUND or ib == GPU_NOT_FOUND) return;

			uint32_t pidx = lookup_pair(ia, ib);
			if (pidx == GPU_NOT_FOUND) return;

			load_link_values(h, pidx);
			_num_fetches++;
		}
	}
}

// ==============================================================
// StorageNode API: storeValue / loadValue

void GpuStorageNode::storeValue(const Handle& atom, const Handle& key)
{
	if (not _connected)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: not connected!\n");

	if (key != truth_key()) return;

	NodePtr np = NodeCast(atom);
	if (np)
	{
		uint32_t idx = store_word(atom->get_type(), np->get_name());
		store_node_values(atom, idx);
		return;
	}

	LinkPtr lp = LinkCast(atom);
	if (lp and lp->get_arity() == 2)
	{
		const Handle& ha = lp->getOutgoingAtom(0);
		const Handle& hb = lp->getOutgoingAtom(1);
		NodePtr na = NodeCast(ha);
		NodePtr nb = NodeCast(hb);
		if (na and nb)
		{
			uint32_t ia = store_word(ha->get_type(), na->get_name());
			uint32_t ib = store_word(hb->get_type(), nb->get_name());
			uint32_t pidx = store_pair(ia, ib, atom->get_type());
			store_link_values(atom, pidx);
		}
	}
}

void GpuStorageNode::loadValue(const Handle& atom, const Handle& key)
{
	if (not _connected)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: not connected!\n");

	if (key != truth_key()) return;

	NodePtr np = NodeCast(atom);
	if (np)
	{
		uint32_t idx = lookup_word(atom->get_type(), np->get_name());
		if (idx != GPU_NOT_FOUND)
			load_node_values(atom, idx);
		return;
	}

	LinkPtr lp = LinkCast(atom);
	if (lp and lp->get_arity() == 2)
	{
		const Handle& ha = lp->getOutgoingAtom(0);
		const Handle& hb = lp->getOutgoingAtom(1);
		NodePtr na = NodeCast(ha);
		NodePtr nb = NodeCast(hb);
		if (na and nb)
		{
			uint32_t ia = lookup_word(ha->get_type(), na->get_name());
			uint32_t ib = lookup_word(hb->get_type(), nb->get_name());
			if (ia != GPU_NOT_FOUND and ib != GPU_NOT_FOUND)
			{
				uint32_t pidx = lookup_pair(ia, ib);
				if (pidx != GPU_NOT_FOUND)
					load_link_values(atom, pidx);
			}
		}
	}
}

// ==============================================================
// StorageNode API: removeAtom

void GpuStorageNode::removeAtom(AtomSpace*, const Handle& h, bool recursive)
{
	if (not _connected)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: not connected!\n");

	NodePtr np = NodeCast(h);
	if (np)
	{
		uint64_t nhash = name_hash(h->get_type(), np->get_name());
		_backend->word_delete(nhash);

		std::lock_guard<std::mutex> lk(_atom_mtx);
		_atom_map.erase(h);
	}
}

// ==============================================================
// StorageNode API: bulk operations

void GpuStorageNode::fetchIncomingSet(AtomSpace* as, const Handle& h)
{
	if (not _connected) return;

	// fetchIncomingSet fetches all link types (type=0 means no filter).
	fetchIncomingByType(as, h, (Type)0);
}

void GpuStorageNode::fetchIncomingByType(AtomSpace* as,
                                          const Handle& h, Type t)
{
	if (not _connected) return;

	// We only store binary links in the pair pool.
	NodePtr np = NodeCast(h);
	if (nullptr == np) return;

	uint32_t widx = lookup_word(h->get_type(), np->get_name());
	if (widx == GPU_NOT_FOUND) return;

	// GPU parallel scan: find all pairs containing this word
	const uint32_t MAX_INCOMING = 16384;
	std::vector<uint32_t> matches(MAX_INCOMING);
	uint32_t n = _backend->incoming_scan(widx,
		matches.data(), MAX_INCOMING);
	if (0 == n) return;

	// Bulk read the pair pool to get word indices, values, and types
	uint32_t pool_count = _backend->pair_pool_count();
	if (0 == pool_count) return;

	std::vector<uint32_t> all_wa(pool_count), all_wb(pool_count);
	std::vector<double> all_counts(pool_count), all_mis(pool_count);
	std::vector<uint16_t> pair_types(pool_count);
	_backend->pair_read_bulk(pool_count,
		all_wa.data(), all_wb.data(),
		all_counts.data(), all_mis.data(),
		pair_types.data());

	// Read word pool to reconstruct names and types
	uint32_t word_count = _backend->word_pool_count();
	std::vector<uint64_t> word_hashes(word_count);
	std::vector<double> wcounts(word_count), wmarg(word_count);
	std::vector<uint16_t> word_types(word_count);
	_backend->word_read_bulk(word_count,
		word_hashes.data(), wcounts.data(), wmarg.data(),
		word_types.data());

	// Build word-index-to-(type,name) map
	struct TypeAndName { Type type; std::string name; };
	std::unordered_map<uint32_t, TypeAndName> idx_to_info;
	{
		std::lock_guard<std::mutex> lk(_name_mtx);
		for (uint32_t wi = 0; wi < word_count; wi++)
		{
			auto it = _hash_to_name.find(word_hashes[wi]);
			if (it != _hash_to_name.end())
				idx_to_info[wi] = {it->second.type, it->second.name};
		}
	}

	// Reconstruct Link atoms in the target AtomSpace
	for (uint32_t i = 0; i < n; i++)
	{
		uint32_t pidx = matches[i];
		Type link_type = (Type)pair_types[pidx];

		// Filter by type if requested (t==0 means all types)
		if (t != 0 and link_type != t) continue;

		uint32_t wa = all_wa[pidx];
		uint32_t wb = all_wb[pidx];

		auto ita = idx_to_info.find(wa);
		auto itb = idx_to_info.find(wb);
		if (ita == idx_to_info.end() or itb == idx_to_info.end())
			continue;

		Handle ha = as->add_node(ita->second.type,
			std::string(ita->second.name));
		Handle hb = as->add_node(itb->second.type,
			std::string(itb->second.name));
		Handle lnk = as->add_link(link_type, ha, hb);

		double c = all_counts[pidx];
		double m = all_mis[pidx];
		if (c != 0.0 or m != 0.0)
		{
			std::vector<double> vals = {c, m};
			lnk->setValue(truth_key(), createFloatValue(vals));
		}
	}

	_num_fetches++;
}

void GpuStorageNode::loadType(AtomSpace* as, Type t)
{
	if (not _connected) return;

	// Check if caller is asking for a node type
	if (nameserver().isA(t, NODE) or t == NODE)
	{
		// Load words from word pool, filtering by stored type
		uint32_t nwords = _backend->word_pool_count();
		if (0 == nwords) return;

		std::vector<uint64_t> hashes(nwords);
		std::vector<double> counts(nwords);
		std::vector<double> marginals(nwords);
		std::vector<uint16_t> types(nwords);
		_backend->word_read_bulk(nwords, hashes.data(),
			counts.data(), marginals.data(), types.data());

		std::lock_guard<std::mutex> lk(_name_mtx);
		for (uint32_t i = 0; i < nwords; i++)
		{
			Type stored_type = (Type)types[i];
			// Filter: t==NODE loads all node types;
			// otherwise only load matching type (including subtypes)
			if (t != NODE and not nameserver().isA(stored_type, t))
				continue;

			auto it = _hash_to_name.find(hashes[i]);
			if (it == _hash_to_name.end()) continue;

			Handle h = as->add_node(stored_type,
				std::string(it->second.name));
			if (counts[i] != 0.0 or marginals[i] != 0.0)
			{
				std::vector<double> vals = {counts[i], marginals[i]};
				h->setValue(truth_key(), createFloatValue(vals));
			}
		}
		return;
	}

	// Check if caller is asking for a link type
	if (nameserver().isA(t, LINK) or t == LINK)
	{
		// Load pairs from pair pool, filtering by stored link type
		uint32_t npairs = _backend->pair_pool_count();
		if (0 == npairs) return;

		std::vector<uint32_t> wa(npairs), wb(npairs);
		std::vector<double> counts(npairs), mis(npairs);
		std::vector<uint16_t> pair_types(npairs);
		_backend->pair_read_bulk(npairs,
			wa.data(), wb.data(), counts.data(), mis.data(),
			pair_types.data());

		// Also read word pool for name+type reconstruction
		uint32_t nwords = _backend->word_pool_count();
		std::vector<uint64_t> word_hashes(nwords);
		std::vector<double> wcounts(nwords), wmarg(nwords);
		std::vector<uint16_t> word_types(nwords);
		_backend->word_read_bulk(nwords,
			word_hashes.data(), wcounts.data(), wmarg.data(),
			word_types.data());

		// Build index-to-(type,name) map
		struct TypeAndName { Type type; std::string name; };
		std::unordered_map<uint32_t, TypeAndName> idx_to_info;
		{
			std::lock_guard<std::mutex> lk(_name_mtx);
			for (uint32_t i = 0; i < nwords; i++)
			{
				auto it = _hash_to_name.find(word_hashes[i]);
				if (it != _hash_to_name.end())
					idx_to_info[i] = {it->second.type, it->second.name};
			}
		}

		for (uint32_t i = 0; i < npairs; i++)
		{
			Type stored_link_type = (Type)pair_types[i];
			// Filter: t==LINK loads all link types;
			// otherwise only load matching type (including subtypes)
			if (t != LINK and not nameserver().isA(stored_link_type, t))
				continue;

			auto ita = idx_to_info.find(wa[i]);
			auto itb = idx_to_info.find(wb[i]);
			if (ita == idx_to_info.end() or itb == idx_to_info.end())
				continue;

			Handle ha = as->add_node(ita->second.type,
				std::string(ita->second.name));
			Handle hb = as->add_node(itb->second.type,
				std::string(itb->second.name));
			Handle lnk = as->add_link(stored_link_type, ha, hb);

			if (counts[i] != 0.0 or mis[i] != 0.0)
			{
				std::vector<double> vals = {counts[i], mis[i]};
				lnk->setValue(truth_key(), createFloatValue(vals));
			}
		}
	}
}

void GpuStorageNode::loadAtomSpace(AtomSpace* as)
{
	if (not _connected)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: not connected!\n");

	// Load all node types first (needed as outgoing atoms for links),
	// then load all link types.
	loadType(as, NODE);
	loadType(as, LINK);
}

void GpuStorageNode::storeAtomSpace(const AtomSpace* as)
{
	if (not _connected)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: not connected!\n");

	HandleSeq all_atoms;
	as->get_handles_by_type(all_atoms, ATOM, true);
	for (const Handle& h : all_atoms)
		storeAtom(h, false);

	_backend->barrier();
}

// ==============================================================
// Barrier

void GpuStorageNode::barrier(AtomSpace*)
{
	if (_connected)
		_backend->barrier();
}

// ==============================================================
// Monitor / Stats

std::string GpuStorageNode::monitor(void)
{
	std::ostringstream ss;
	ss << "GpuStorageNode: " << _uri << "\n";
	ss << "  Connected: " << (_connected ? "yes" : "no") << "\n";

	if (_connected)
	{
		ss << "  Backend:  " << _backend->backend_name() << "\n";
		ss << "  Device:   " << _backend->device_info() << "\n";
		ss << "  Words:    " << _backend->word_pool_count()
		   << " / " << GPU_WORD_CAPACITY << "\n";
		ss << "  Pairs:    " << _backend->pair_pool_count()
		   << " / " << GPU_PAIR_CAPACITY << "\n";
		ss << "  Sections: " << _backend->section_pool_count()
		   << " / " << GPU_SECTION_CAPACITY << "\n";
		ss << "  Stores:   " << _num_stores.load() << "\n";
		ss << "  Fetches:  " << _num_fetches.load() << "\n";
	}

	return ss.str();
}

// ==============================================================
// Phase 2: runQuery
//
// Delegates to the BackingStore default implementation, which uses
// our fetchIncomingByType() callback during pattern matching.
// We only add a cache check: if the result is already cached
// and fresh==false, skip re-execution.

void GpuStorageNode::runQuery(const Handle& query, const Handle& key,
                              const Handle& metadata_key, bool fresh)
{
	if (not _connected)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: not connected!\n");

	// Cache check: if result already exists and not forced fresh
	if (not fresh)
	{
		ValuePtr vp = query->getValue(key);
		if (vp) return;
	}

	// Delegate to base class — it uses our fetchIncomingByType
	// during pattern matching graph crawl.
	BackingStore::runQuery(query, key, metadata_key, fresh);
}

// ==============================================================
// Factory

DEFINE_NODE_FACTORY(GpuStorageNode, GPU_STORAGE_NODE);

// ==============================================================
