/*
 * opencog/persist/gpu/OpenCLBackend.h
 *
 * OpenCL implementation of the GpuBackend interface.
 * Fallback backend for AMD/Intel GPUs, also works on NVIDIA.
 *
 * Copyright (C) 2025 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef _OPENCOG_OPENCL_BACKEND_H
#define _OPENCOG_OPENCL_BACKEND_H

#include "GpuBackend.h"
#include "gpu-pool-defs.h"
#include "opencl-headers.h"

namespace opencog
{

class OpenCLBackend : public GpuBackend
{
private:
	cl::Platform _platform;
	cl::Device _device;
	cl::Context _context;
	cl::CommandQueue _queue;
	cl::Program _program;

	// GPU SoA pool buffers -- WordPool
	cl::Buffer _word_name_hash;    // ulong[WORD_CAPACITY]
	cl::Buffer _word_count;        // double[WORD_CAPACITY]
	cl::Buffer _word_mi_marginal;  // double[WORD_CAPACITY]
	cl::Buffer _word_class_id;     // uint[WORD_CAPACITY]
	cl::Buffer _word_type;         // ushort[WORD_CAPACITY]
	cl::Buffer _word_next_free;    // uint[1]

	// GPU SoA pool buffers -- PairPool
	cl::Buffer _pair_word_a;       // uint[PAIR_CAPACITY]
	cl::Buffer _pair_word_b;       // uint[PAIR_CAPACITY]
	cl::Buffer _pair_count;        // double[PAIR_CAPACITY]
	cl::Buffer _pair_mi;           // double[PAIR_CAPACITY]
	cl::Buffer _pair_flags;        // uint[PAIR_CAPACITY]
	cl::Buffer _pair_next_free;    // uint[1]

	// GPU SoA pool buffers -- SectionPool
	cl::Buffer _sec_word;          // uint[SECTION_CAPACITY]
	cl::Buffer _sec_djh;           // ulong[SECTION_CAPACITY]
	cl::Buffer _sec_count;         // double[SECTION_CAPACITY]
	cl::Buffer _sec_next_free;     // uint[1]

	// Hash tables (keys + values for each pool)
	cl::Buffer _word_ht_keys;      // ulong[WORD_HT_CAPACITY]
	cl::Buffer _word_ht_values;    // uint[WORD_HT_CAPACITY]
	cl::Buffer _pair_ht_keys;      // ulong[PAIR_HT_CAPACITY]
	cl::Buffer _pair_ht_values;    // uint[PAIR_HT_CAPACITY]
	cl::Buffer _sec_ht_keys;       // ulong[SECTION_HT_CAPACITY]
	cl::Buffer _sec_ht_values;     // uint[SECTION_HT_CAPACITY]

	std::string _device_name;
	std::string _platform_name;

	void find_device(const std::string& platform, const std::string& device);
	void compile_kernels();

public:
	OpenCLBackend() = default;
	~OpenCLBackend() override;

	void init(const std::string& platform_hint,
	          const std::string& device_hint) override;
	void shutdown() override;
	std::string device_info() override;
	std::string backend_name() override { return "OpenCL"; }

	void alloc_pools() override;
	void init_pools() override;

	uint32_t word_find_or_create(uint64_t name_hash) override;
	uint32_t word_lookup(uint64_t name_hash) override;
	void word_write_count(uint32_t idx, double count) override;
	void word_write_marginal(uint32_t idx, double marginal) override;
	void word_read_values(uint32_t idx,
	                      double& count, double& marginal) override;
	void word_write_type(uint32_t idx, uint16_t type) override;
	uint16_t word_read_type(uint32_t idx) override;
	void word_delete(uint64_t name_hash) override;
	uint32_t word_pool_count() override;
	void word_read_bulk(uint32_t n, uint64_t* hashes,
	                    double* counts, double* marginals,
	                    uint16_t* types) override;

	uint32_t pair_find_or_create(uint32_t word_a,
	                             uint32_t word_b) override;
	uint32_t pair_lookup(uint32_t word_a, uint32_t word_b) override;
	void pair_write_count(uint32_t idx, double count) override;
	void pair_write_mi(uint32_t idx, double mi) override;
	void pair_read_values(uint32_t idx,
	                      double& count, double& mi) override;
	void pair_write_type(uint32_t idx, uint16_t type) override;
	uint16_t pair_read_type(uint32_t idx) override;
	uint32_t pair_pool_count() override;

	uint32_t section_pool_count() override;

	uint32_t incoming_scan(uint32_t target_word_idx,
	                       uint32_t* out_pair_indices,
	                       uint32_t max_results) override;
	uint32_t pair_read_bulk(uint32_t n,
	                        uint32_t* word_a, uint32_t* word_b,
	                        double* counts, double* mis,
	                        uint16_t* types) override;

	void barrier() override;
};

} // namespace opencog

#endif // _OPENCOG_OPENCL_BACKEND_H
