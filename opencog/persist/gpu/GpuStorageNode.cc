/*
 * opencog/persist/gpu/GpuStorageNode.cc
 *
 * BackingStore wrapper for GpuAtomTable. Bridges the AtomSpace
 * persistence API to the GPU atom table.
 *
 * Step 3 of Linas' plan: storeAtom/getAtom via BackingStore.h.
 *
 * Copyright (C) 2026 OpenCog Foundation
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <opencog/atoms/base/Link.h>
#include <opencog/atoms/base/Node.h>
#include <opencog/util/exceptions.h>
#include <opencog/persist/gpu/GpuStorageNode.h>

using namespace opencog;

GpuStorageNode::GpuStorageNode(Type t, const std::string&& uri)
	: StorageNode(t, std::move(uri)), _is_open(false)
{
	memset(&_gpu_table, 0, sizeof(_gpu_table));
}

GpuStorageNode::GpuStorageNode(const std::string& uri)
	: StorageNode(GPU_STORAGE_NODE, uri), _is_open(false)
{
	memset(&_gpu_table, 0, sizeof(_gpu_table));
}

GpuStorageNode::~GpuStorageNode()
{
	if (_is_open) close();
}

// ================================================================
// Lifecycle

void GpuStorageNode::open(void)
{
	if (_is_open) return;

	int rc = gpu_table_alloc(&_gpu_table);
	if (rc != 0)
		throw IOException(TRACE_INFO, "GPU: failed to allocate atom table");

	_is_open = true;
}

void GpuStorageNode::close(void)
{
	if (!_is_open) return;

	gpu_table_barrier(&_gpu_table);
	gpu_table_free(&_gpu_table);

	_tlb.clear();
	_handle_to_slot.clear();
	_slot_is_node.clear();

	_is_open = false;
}

bool GpuStorageNode::connected(void)
{
	return _is_open;
}

void GpuStorageNode::destroy(void)
{
	if (!_is_open) return;
	gpu_table_clear(&_gpu_table);
	_gpu_table.atom_count = 0;
	_gpu_table.name_pool_used = 0;
	_gpu_table.out_pool_used = 0;
	_tlb.clear();
	_handle_to_slot.clear();
	_slot_is_node.clear();
}

void GpuStorageNode::erase(void)
{
	destroy();
}

// ================================================================
// Slot management
//
// Each atom gets a sequential GPU slot (0, 1, 2, ...).
// The TLB maps slot ↔ Handle (UUID = slot).
// _handle_to_slot provides fast forward lookup.

bool GpuStorageNode::has_slot(const Handle& h) const
{
	return _handle_to_slot.count(h) > 0;
}

uint32_t GpuStorageNode::assign_slot(const Handle& h)
{
	auto it = _handle_to_slot.find(h);
	if (it != _handle_to_slot.end())
		return it->second;

	uint32_t slot = _gpu_table.atom_count;
	_gpu_table.atom_count++;

	_handle_to_slot[h] = slot;
	_tlb.addAtom(h, (UUID)slot);
	_slot_is_node.push_back(h->is_node());

	return slot;
}

// ================================================================
// Store

void GpuStorageNode::do_store_atom(const Handle& h)
{
	// Already on GPU — nothing to update in Step 3 (no Values).
	if (has_slot(h)) return;

	// Links: recursively store outgoing atoms first.
	if (h->is_link())
	{
		for (const Handle& out : h->getOutgoingSet())
			do_store_atom(out);
	}

	uint32_t slot = assign_slot(h);
	int rc;

	if (h->is_node())
	{
		NodePtr np = NodeCast(h);
		const std::string& name = np->get_name();
		rc = gpu_store_node(&_gpu_table, slot,
			h->get_type(), name.c_str(), (uint16_t)name.size());
	}
	else
	{
		const HandleSeq& outgoing = h->getOutgoingSet();
		std::vector<uint32_t> out_slots;
		out_slots.reserve(outgoing.size());
		for (const Handle& out : outgoing)
			out_slots.push_back(_handle_to_slot.at(out));

		rc = gpu_store_link(&_gpu_table, slot,
			h->get_type(), out_slots.data(), (uint16_t)outgoing.size());
	}

	if (rc != 0)
		throw IOException(TRACE_INFO,
			"GPU: failed to store atom at slot %u", slot);
}

void GpuStorageNode::storeAtom(const Handle& h, bool synchronous)
{
	std::lock_guard<std::mutex> lk(_mtx);
	do_store_atom(h);
	if (synchronous)
		gpu_table_barrier(&_gpu_table);
}

// ================================================================
// Load
//
// Reconstruct atoms from GPU data. Two-pass: nodes first (no
// dependencies), then links (outgoing references resolved via
// the slot→Handle vector built during pass 1).

void GpuStorageNode::loadAtomSpace(AtomSpace* as)
{
	if (!_is_open) return;

	gpu_table_barrier(&_gpu_table);

	uint32_t n = _gpu_table.atom_count;
	std::vector<Handle> loaded(n);

	// Pass 1: nodes
	for (uint32_t slot = 0; slot < n; slot++)
	{
		if (!_slot_is_node[slot]) continue;

		uint16_t type;
		char name_buf[4096];
		uint16_t name_len = sizeof(name_buf);

		int rc = gpu_fetch_node(&_gpu_table, slot,
			&type, name_buf, &name_len);
		if (rc != 0) continue;

		std::string name(name_buf, name_len);
		loaded[slot] = as->add_node(type, std::move(name));
	}

	// Pass 2: links (outgoing slots are always < current slot
	// because do_store_atom stores outgoing first).
	for (uint32_t slot = 0; slot < n; slot++)
	{
		if (_slot_is_node[slot]) continue;

		uint16_t type;
		uint32_t out_buf[256];
		uint16_t arity = sizeof(out_buf) / sizeof(out_buf[0]);

		int rc = gpu_fetch_link(&_gpu_table, slot,
			&type, out_buf, &arity);
		if (rc != 0) continue;

		HandleSeq outgoing;
		outgoing.reserve(arity);
		for (uint16_t i = 0; i < arity; i++)
			outgoing.push_back(loaded[out_buf[i]]);

		loaded[slot] = as->add_link(type, std::move(outgoing));
	}
}

void GpuStorageNode::storeAtomSpace(const AtomSpace* as)
{
	HandleSeq all;
	as->get_handles_by_type(all, ATOM, true);

	std::lock_guard<std::mutex> lk(_mtx);
	for (const Handle& h : all)
		do_store_atom(h);
}

// ================================================================
// Fetch (Step 3: no Values stored yet)

void GpuStorageNode::getAtom(const Handle&)
{
	// Step 3: no Values on GPU. Nothing to load.
}

void GpuStorageNode::fetchIncomingSet(AtomSpace*, const Handle&) {}
void GpuStorageNode::fetchIncomingByType(AtomSpace*, const Handle&, Type) {}
void GpuStorageNode::removeAtom(AtomSpace*, const Handle&, bool) {}
void GpuStorageNode::loadType(AtomSpace*, Type) {}

// ================================================================
// Synchronization

void GpuStorageNode::barrier(AtomSpace*)
{
	if (_is_open)
		gpu_table_barrier(&_gpu_table);
}

// ================================================================
// Diagnostics

std::string GpuStorageNode::monitor(void)
{
	std::string s;
	s += "GPU Storage Monitor\n";
	s += "Connected: ";
	s += (_is_open ? "yes" : "no");
	s += "\n";
	if (_is_open)
	{
		s += "Backend: ";
		switch (_gpu_table.backend) {
			case GPU_BACKEND_CUDA:   s += "CUDA\n"; break;
			case GPU_BACKEND_OPENCL: s += "OpenCL\n"; break;
			default:                 s += "None\n"; break;
		}
		s += "Atoms: " + std::to_string(_gpu_table.atom_count) + "\n";
		s += "Name pool: " + std::to_string(_gpu_table.name_pool_used) + " bytes\n";
		s += "Out pool: " + std::to_string(_gpu_table.out_pool_used) + " slots\n";
	}
	return s;
}

// ================================================================
// Factory

Handle GpuStorageNode::factory(const Handle& base)
{
	Handle h(createGpuStorageNode(base->get_name()));
	return h;
}
