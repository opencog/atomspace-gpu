/*
 * opencog/persist/gpu/GpuStorageNode.cc
 *
 * StorageNode backed by GPU SoA pools.
 * Delegates all GPU operations to a GpuBackend (CUDA or OpenCL).
 *
 * Phase 1: Store/Fetch round-trip for atoms and values.
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
//   "gpu://:"                  → auto backend, any device
//   "gpu://NVIDIA:RTX"         → auto backend, NVIDIA RTX
//   "gpu://cuda::"             → force CUDA, any device
//   "gpu://opencl:Intel:"      → force OpenCL, Intel device
//   "gpu://cuda:NVIDIA:RTX"    → CUDA, NVIDIA RTX

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
// Name hashing

uint64_t GpuStorageNode::name_hash(const std::string& name)
{
	std::lock_guard<std::mutex> lk(_name_mtx);

	auto it = _name_to_hash.find(name);
	if (it != _name_to_hash.end())
		return it->second;

	// splitmix64 on std::hash
	std::hash<std::string> hasher;
	uint64_t h = hasher(name);

	if (h == GPU_HT_EMPTY_KEY) h = 0;

	_name_to_hash[name] = h;
	_hash_to_name[h] = name;
	return h;
}

// ==============================================================
// GPU pool operations (delegated to backend)

uint32_t GpuStorageNode::store_word(const std::string& name)
{
	uint64_t nhash = name_hash(name);
	uint32_t idx = _backend->word_find_or_create(nhash);

	if (idx == GPU_NOT_FOUND)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: word pool full, cannot store '%s'\n",
			name.c_str());

	return idx;
}

uint32_t GpuStorageNode::lookup_word(const std::string& name)
{
	uint64_t nhash = name_hash(name);
	return _backend->word_lookup(nhash);
}

uint32_t GpuStorageNode::store_pair(uint32_t word_a, uint32_t word_b)
{
	uint32_t idx = _backend->pair_find_or_create(word_a, word_b);

	if (idx == GPU_NOT_FOUND)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: pair pool full\n");

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
		uint32_t idx = store_word(np->get_name());
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
				uint32_t ia = store_word(na->get_name());
				uint32_t ib = store_word(nb->get_name());
				uint32_t pidx = store_pair(ia, ib);
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
		uint32_t idx = lookup_word(np->get_name());
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
			uint32_t ia = lookup_word(na->get_name());
			uint32_t ib = lookup_word(nb->get_name());
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
		uint32_t idx = store_word(np->get_name());
		store_node_values(atom, idx);
		return;
	}

	LinkPtr lp = LinkCast(atom);
	if (lp and lp->get_arity() == 2)
	{
		NodePtr na = NodeCast(lp->getOutgoingAtom(0));
		NodePtr nb = NodeCast(lp->getOutgoingAtom(1));
		if (na and nb)
		{
			uint32_t ia = store_word(na->get_name());
			uint32_t ib = store_word(nb->get_name());
			uint32_t pidx = store_pair(ia, ib);
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
		uint32_t idx = lookup_word(np->get_name());
		if (idx != GPU_NOT_FOUND)
			load_node_values(atom, idx);
		return;
	}

	LinkPtr lp = LinkCast(atom);
	if (lp and lp->get_arity() == 2)
	{
		NodePtr na = NodeCast(lp->getOutgoingAtom(0));
		NodePtr nb = NodeCast(lp->getOutgoingAtom(1));
		if (na and nb)
		{
			uint32_t ia = lookup_word(na->get_name());
			uint32_t ib = lookup_word(nb->get_name());
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
		uint64_t nhash = name_hash(np->get_name());
		_backend->word_delete(nhash);

		std::lock_guard<std::mutex> lk(_atom_mtx);
		_atom_map.erase(h);
	}
}

// ==============================================================
// StorageNode API: bulk operations

void GpuStorageNode::fetchIncomingSet(AtomSpace*, const Handle&)
{
}

void GpuStorageNode::fetchIncomingByType(AtomSpace*, const Handle&, Type)
{
}

void GpuStorageNode::loadType(AtomSpace*, Type)
{
}

void GpuStorageNode::loadAtomSpace(AtomSpace* as)
{
	if (not _connected)
		throw RuntimeException(TRACE_INFO,
			"GpuStorageNode: not connected!\n");

	uint32_t nwords = _backend->word_pool_count();
	if (0 == nwords) return;

	std::vector<uint64_t> hashes(nwords);
	std::vector<double> counts(nwords);
	std::vector<double> marginals(nwords);
	_backend->word_read_bulk(nwords, hashes.data(),
		counts.data(), marginals.data());

	std::lock_guard<std::mutex> lk(_name_mtx);
	for (uint32_t i = 0; i < nwords; i++)
	{
		auto it = _hash_to_name.find(hashes[i]);
		if (it == _hash_to_name.end()) continue;

		std::string name = it->second;
		Handle h = as->add_node(SCHEMA_NODE, std::move(name));
		if (counts[i] != 0.0 or marginals[i] != 0.0)
		{
			std::vector<double> vals = {counts[i], marginals[i]};
			h->setValue(truth_key(), createFloatValue(vals));
		}
	}
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
// Phase 2: runQuery (stub)

void GpuStorageNode::runQuery(const Handle& query, const Handle& key,
                              const Handle& metadata_key, bool fresh)
{
	throw RuntimeException(TRACE_INFO,
		"GpuStorageNode::runQuery() not yet implemented (Phase 2)\n");
}

// ==============================================================
// Factory

DEFINE_NODE_FACTORY(GpuStorageNode, GPU_STORAGE_NODE);

// ==============================================================
