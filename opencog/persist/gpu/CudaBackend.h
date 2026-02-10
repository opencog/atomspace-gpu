/*
 * opencog/persist/gpu/CudaBackend.h
 *
 * CUDA implementation of the GpuBackend interface.
 * Primary backend for NVIDIA GPUs (more performant than OpenCL).
 *
 * NOTE: This header is only included by CudaBackend.cu, which is
 * compiled by nvcc. It must NOT be included by regular C++ files.
 *
 * Copyright (C) 2025 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef _OPENCOG_CUDA_BACKEND_H
#define _OPENCOG_CUDA_BACKEND_H

#include "GpuBackend.h"
#include "gpu-pool-defs.h"

#include <cuda_runtime.h>

namespace opencog
{

class CudaBackend : public GpuBackend
{
private:
	int _device_id;
	std::string _device_name;

	// Device memory -- WordPool SoA
	uint64_t* _d_word_name_hash;
	double*   _d_word_count;
	double*   _d_word_mi_marginal;
	uint32_t* _d_word_class_id;
	uint32_t* _d_word_next_free;

	// Device memory -- PairPool SoA
	uint32_t* _d_pair_word_a;
	uint32_t* _d_pair_word_b;
	double*   _d_pair_count;
	double*   _d_pair_mi;
	uint32_t* _d_pair_flags;
	uint32_t* _d_pair_next_free;

	// Device memory -- SectionPool SoA
	uint32_t* _d_sec_word;
	uint64_t* _d_sec_djh;
	double*   _d_sec_count;
	uint32_t* _d_sec_next_free;

	// Device memory -- Hash tables
	uint64_t* _d_word_ht_keys;
	uint32_t* _d_word_ht_values;
	uint64_t* _d_pair_ht_keys;
	uint32_t* _d_pair_ht_values;
	uint64_t* _d_sec_ht_keys;
	uint32_t* _d_sec_ht_values;

	// Managed staging buffers (CPU+GPU accessible, no cudaMemcpy)
	uint64_t* _staging_key;
	uint32_t* _staging_result;
	uint32_t* _staging_wa;
	uint32_t* _staging_wb;

	void check_error(cudaError_t err, const char* context);

public:
	CudaBackend();
	~CudaBackend() override;

	void init(const std::string& platform_hint,
	          const std::string& device_hint) override;
	void shutdown() override;
	std::string device_info() override;
	std::string backend_name() override { return "CUDA"; }

	void alloc_pools() override;
	void init_pools() override;

	uint32_t word_find_or_create(uint64_t name_hash) override;
	uint32_t word_lookup(uint64_t name_hash) override;
	void word_write_count(uint32_t idx, double count) override;
	void word_write_marginal(uint32_t idx, double marginal) override;
	void word_read_values(uint32_t idx,
	                      double& count, double& marginal) override;
	void word_delete(uint64_t name_hash) override;
	uint32_t word_pool_count() override;
	void word_read_bulk(uint32_t n, uint64_t* hashes,
	                    double* counts, double* marginals) override;

	uint32_t pair_find_or_create(uint32_t word_a,
	                             uint32_t word_b) override;
	uint32_t pair_lookup(uint32_t word_a, uint32_t word_b) override;
	void pair_write_count(uint32_t idx, double count) override;
	void pair_write_mi(uint32_t idx, double mi) override;
	void pair_read_values(uint32_t idx,
	                      double& count, double& mi) override;
	uint32_t pair_pool_count() override;

	uint32_t section_pool_count() override;

	void barrier() override;
};

} // namespace opencog

#endif // _OPENCOG_CUDA_BACKEND_H
