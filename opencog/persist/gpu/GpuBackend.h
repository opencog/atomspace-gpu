/*
 * opencog/persist/gpu/GpuBackend.h
 *
 * Abstract interface for GPU compute backends (CUDA, OpenCL).
 * GpuStorageNode delegates all GPU operations to a concrete
 * GpuBackend implementation, selected at open() time based on
 * hardware availability and URI hints.
 *
 * Priority: CUDA first (more performant on NVIDIA), OpenCL
 * as fallback (works on AMD/Intel/NVIDIA).
 *
 * Copyright (C) 2025 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef _OPENCOG_GPU_BACKEND_H
#define _OPENCOG_GPU_BACKEND_H

#include <cstdint>
#include <string>

namespace opencog
{

class GpuBackend
{
public:
	virtual ~GpuBackend() = default;

	// -- Lifecycle --
	virtual void init(const std::string& platform_hint,
	                  const std::string& device_hint) = 0;
	virtual void shutdown() = 0;
	virtual std::string device_info() = 0;
	virtual std::string backend_name() = 0;  // "CUDA" or "OpenCL"

	// -- Pool allocation and initialization --
	virtual void alloc_pools() = 0;
	virtual void init_pools() = 0;

	// -- Word pool operations --
	virtual uint32_t word_find_or_create(uint64_t name_hash) = 0;
	virtual uint32_t word_lookup(uint64_t name_hash) = 0;
	virtual void word_write_count(uint32_t idx, double count) = 0;
	virtual void word_write_marginal(uint32_t idx, double marginal) = 0;
	virtual void word_read_values(uint32_t idx,
	                              double& count, double& marginal) = 0;
	virtual void word_write_type(uint32_t idx, uint16_t type) = 0;
	virtual uint16_t word_read_type(uint32_t idx) = 0;
	virtual void word_delete(uint64_t name_hash) = 0;
	virtual uint32_t word_pool_count() = 0;
	virtual void word_read_bulk(uint32_t n, uint64_t* hashes,
	                            double* counts, double* marginals,
	                            uint16_t* types) = 0;

	// -- Pair pool operations --
	virtual uint32_t pair_find_or_create(uint32_t word_a,
	                                     uint32_t word_b) = 0;
	virtual uint32_t pair_lookup(uint32_t word_a, uint32_t word_b) = 0;
	virtual void pair_write_count(uint32_t idx, double count) = 0;
	virtual void pair_write_mi(uint32_t idx, double mi) = 0;
	virtual void pair_read_values(uint32_t idx,
	                              double& count, double& mi) = 0;
	virtual void pair_write_type(uint32_t idx, uint16_t type) = 0;
	virtual uint16_t pair_read_type(uint32_t idx) = 0;
	virtual uint32_t pair_pool_count() = 0;

	// -- Section pool operations --
	virtual uint32_t section_pool_count() = 0;

	// -- Incoming-set operations (Phase 2) --

	// Parallel scan: find all pairs where word_a == target OR
	// word_b == target. Returns the number of matches found
	// (up to max_results). Matching pair pool indices are written
	// to out_pair_indices.
	virtual uint32_t incoming_scan(uint32_t target_word_idx,
	                               uint32_t* out_pair_indices,
	                               uint32_t max_results) = 0;

	// Bulk read pair pool: read word_a, word_b, count, mi, type for
	// the first `n` pair slots. Returns actual number read.
	virtual uint32_t pair_read_bulk(uint32_t n,
	                                uint32_t* word_a, uint32_t* word_b,
	                                double* counts, double* mis,
	                                uint16_t* types) = 0;

	// -- Synchronization --
	virtual void barrier() = 0;
};

// Backend factory functions (defined in respective .cc/.cu files).
// Availability depends on compile-time HAVE_CUDA / HAVE_OPENCL.
#ifdef HAVE_CUDA
GpuBackend* create_cuda_backend();
#endif

#ifdef HAVE_OPENCL
GpuBackend* create_opencl_backend();
#endif

} // namespace opencog

#endif // _OPENCOG_GPU_BACKEND_H
