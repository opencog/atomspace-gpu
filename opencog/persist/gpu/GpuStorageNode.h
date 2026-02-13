/*
 * opencog/persist/gpu/GpuStorageNode.h
 *
 * BackingStore wrapper for GpuAtomTable. Bridges the AtomSpace
 * persistence API to the GPU atom table (Step 2).
 *
 * Step 3 of Linas' plan: storeAtom/getAtom via BackingStore.h.
 * Uses TLB for Handle ↔ GPU-slot identity mapping.
 *
 * Copyright (C) 2026 OpenCog Foundation
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef _OPENCOG_GPU_STORAGE_NODE_H
#define _OPENCOG_GPU_STORAGE_NODE_H

#include <mutex>
#include <unordered_map>
#include <vector>

#include <opencog/atomspace/AtomSpace.h>
#include <opencog/persist/api/StorageNode.h>
#include <opencog/persist/tlb/TLB.h>
#include <opencog/persist/gpu-types/atom_types.h>
#include <opencog/persist/gpu/GpuAtomTable.h>

namespace opencog
{

class GpuStorageNode : public StorageNode
{
private:
	GpuAtomTable _gpu_table;
	TLB _tlb;
	bool _is_open;
	std::mutex _mtx;

	// Handle → GPU slot (forward lookup).
	// TLB provides the reverse (slot → Handle).
	std::unordered_map<Handle, uint32_t,
	                   std::hash<opencog::Handle>,
	                   std::equal_to<opencog::Handle>> _handle_to_slot;

	// Per-slot node/link flag for two-pass loadAtomSpace.
	std::vector<bool> _slot_is_node;

	bool has_slot(const Handle&) const;
	uint32_t assign_slot(const Handle&);
	void do_store_atom(const Handle&);

public:
	GpuStorageNode(Type t, const std::string&& uri);
	GpuStorageNode(const std::string& uri);
	virtual ~GpuStorageNode();

	// StorageNode lifecycle
	void open(void);
	void close(void);
	bool connected(void);
	void create(void) {}
	void destroy(void);
	void erase(void);

	// BackingStore interface (implemented)
	void storeAtom(const Handle&, bool synchronous = false);
	void loadAtomSpace(AtomSpace*);
	void storeAtomSpace(const AtomSpace*);
	void barrier(AtomSpace* = nullptr);
	std::string monitor(void);

	// BackingStore interface (stubbed for Step 3)
	void getAtom(const Handle&);
	void fetchIncomingSet(AtomSpace*, const Handle&);
	void fetchIncomingByType(AtomSpace*, const Handle&, Type);
	void removeAtom(AtomSpace*, const Handle&, bool recursive);
	void loadType(AtomSpace*, Type);

	// Expose GPU table for verification in tests.
	const GpuAtomTable& gpu_table() const { return _gpu_table; }

	void setAtomSpace(AtomSpace* as)
	{
		if (nullptr == as) close();
		Atom::setAtomSpace(as);
	}
	static Handle factory(const Handle&);
};

NODE_PTR_DECL(GpuStorageNode)
#define createGpuStorageNode CREATE_DECL(GpuStorageNode)

} // namespace opencog

#endif // _OPENCOG_GPU_STORAGE_NODE_H
