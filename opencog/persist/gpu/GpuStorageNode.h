/*
 * opencog/persist/gpu/GpuStorageNode.h
 *
 * StorageNode implementation backed by GPU SoA pools.
 * Provides the standard StorageNode API (store/fetch/query)
 * using GPU-resident word, pair, and section pools.
 * Same pattern as RocksStorageNode (disk) and CogStorageNode
 * (network), but targeting GPU memory.
 *
 * Supports CUDA (primary, NVIDIA) and OpenCL (fallback, AMD/Intel).
 * Backend is selected at open() time based on hardware availability
 * and URI hints.
 *
 * URI format: "gpu://[backend:]platform:device"
 *   "gpu://:"              first available device (CUDA preferred)
 *   "gpu://NVIDIA:RTX"     first NVIDIA device containing "RTX"
 *   "gpu://cuda::"         force CUDA backend
 *   "gpu://opencl::"       force OpenCL backend
 *   "gpu://opencl:Intel:"  OpenCL on first Intel device
 *
 * Copyright (C) 2025 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef _OPENCOG_GPU_STORAGE_NODE_H
#define _OPENCOG_GPU_STORAGE_NODE_H

#include <memory>
#include <mutex>
#include <unordered_map>
#include <functional>

#include <opencog/persist/api/StorageNode.h>

#include "gpu-pool-defs.h"
#include "GpuBackend.h"

namespace opencog
{

/** \addtogroup grp_persist
 *  @{
 */

class GpuStorageNode : public StorageNode
{
private:
	std::string _uri;
	std::string _backend_hint;  // "cuda", "opencl", or "" (auto)
	std::string _splat;         // platform substring
	std::string _sdev;          // device substring

	// GPU backend (CUDA or OpenCL, selected at open time)
	std::unique_ptr<GpuBackend> _backend;
	bool _connected;

	// CPU-side (type,name)->hash mapping (GPU stores hashes, CPU knows strings)
	struct TypedName {
		Type type;
		std::string name;
		bool operator==(const TypedName& o) const {
			return type == o.type && name == o.name;
		}
	};
	struct TypedNameHash {
		size_t operator()(const TypedName& tn) const {
			return std::hash<std::string>()(tn.name)
			     ^ (size_t(tn.type) * 0x9E3779B97F4A7C15ULL);
		}
	};
	std::mutex _name_mtx;
	std::unordered_map<TypedName, uint64_t, TypedNameHash> _name_to_hash;
	std::unordered_map<uint64_t, TypedName> _hash_to_name;

	// Atom->GPU-index mapping for Values
	struct ValueSlot {
		uint32_t pool_index;  // index in word/pair/section pool
		int pool_type;        // 0=word, 1=pair, 2=section
	};
	std::mutex _atom_mtx;
	std::unordered_map<Handle, ValueSlot> _atom_map;

	// Internal helpers
	void parse_uri(void);

	uint64_t name_hash(Type t, const std::string& name);
	uint32_t store_word(Type t, const std::string& name);
	uint32_t store_pair(uint32_t word_a, uint32_t word_b,
	                    Type link_type);
	uint32_t lookup_word(Type t, const std::string& name);
	uint32_t lookup_pair(uint32_t word_a, uint32_t word_b);

	void store_node_values(const Handle&, uint32_t pool_idx);
	void load_node_values(const Handle&, uint32_t pool_idx);
	void store_link_values(const Handle&, uint32_t pool_idx);
	void load_link_values(const Handle&, uint32_t pool_idx);

	// Stats tracking
	std::atomic<size_t> _num_stores;
	std::atomic<size_t> _num_fetches;

public:
	GpuStorageNode(Type t, const std::string&& uri);
	GpuStorageNode(const std::string&& uri);
	GpuStorageNode(const GpuStorageNode&) = delete;
	GpuStorageNode& operator=(const GpuStorageNode&) = delete;
	virtual ~GpuStorageNode();

	// -- StorageNode interface --
	void open(void) override;
	void close(void) override;
	bool connected(void) override;

	void create(void) override { erase(); }
	void destroy(void) override { erase(); }
	void erase(void) override;

	// -- BackingStore interface --
	void getAtom(const Handle&) override;
	void storeAtom(const Handle&, bool synchronous = false) override;
	void removeAtom(AtomSpace*, const Handle&, bool recursive) override;

	void storeValue(const Handle& atom, const Handle& key) override;
	void loadValue(const Handle& atom, const Handle& key) override;

	void fetchIncomingSet(AtomSpace*, const Handle&) override;
	void fetchIncomingByType(AtomSpace*, const Handle&, Type) override;
	void loadType(AtomSpace*, Type) override;
	void loadAtomSpace(AtomSpace*) override;
	void storeAtomSpace(const AtomSpace*) override;
	void barrier(AtomSpace* = nullptr) override;

	std::string monitor(void) override;

	// -- Phase 2: Query execution on GPU --
	void runQuery(const Handle&, const Handle&,
	              const Handle& = Handle::UNDEFINED,
	              bool = false) override;

	// Atom factory
	void setAtomSpace(AtomSpace* as)
	{
		if (nullptr == as) close();
		Atom::setAtomSpace(as);
	}
	static Handle factory(const Handle&);
};

NODE_PTR_DECL(GpuStorageNode)
#define createGpuStorageNode CREATE_DECL(GpuStorageNode)

/** @}*/
} // namespace opencog

#endif // _OPENCOG_GPU_STORAGE_NODE_H
