/*
 * opencog/persist/gpu/CudaBackend.cu
 *
 * CUDA implementation of GpuBackend.
 * Uses device memory with CUDA kernels for hash table operations,
 * and cudaMemcpy for direct value read/write.
 *
 * Managed staging buffers avoid per-operation cudaMalloc overhead.
 *
 * Copyright (C) 2025 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include <cstdio>
#include <cstring>

#include <opencog/util/exceptions.h>
#include <opencog/util/Logger.h>

#include "CudaBackend.h"

using namespace opencog;

// ==============================================================
// CUDA Device Kernels
// ==============================================================

// splitmix64 hash function (same as OpenCL gpu-hashtable.cl)
__device__ __forceinline__
uint64_t cuda_hash(uint64_t key)
{
	key ^= key >> 30;
	key *= 0xBF58476D1CE4E5B9ULL;
	key ^= key >> 27;
	key *= 0x94D049BB133111EBULL;
	key ^= key >> 31;
	return key;
}

#define CUDA_HT_EMPTY_KEY    0xFFFFFFFFFFFFFFFFULL
#define CUDA_HT_EMPTY_VALUE  0xFFFFFFFFU
#define CUDA_MAX_PROBES      4096

// ----------------------------------------------------------
// Generic hash table lookup kernel
__global__ void cuda_ht_lookup(
	const uint64_t* __restrict__ keys,
	const uint32_t* __restrict__ values,
	uint64_t capacity,
	const uint64_t* __restrict__ query_keys,
	uint32_t* __restrict__ results,
	uint32_t num_queries)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_queries) return;

	uint64_t key = query_keys[idx];
	uint64_t slot = cuda_hash(key) % capacity;

	for (int probe = 0; probe < CUDA_MAX_PROBES; probe++)
	{
		uint64_t existing = keys[slot];
		if (existing == key)
		{
			results[idx] = values[slot];
			return;
		}
		if (existing == CUDA_HT_EMPTY_KEY)
		{
			results[idx] = CUDA_HT_EMPTY_VALUE;
			return;
		}
		slot = (slot + 1) % capacity;
	}
	results[idx] = CUDA_HT_EMPTY_VALUE;
}

// ----------------------------------------------------------
// Generic hash table delete kernel (tombstone)
__global__ void cuda_ht_delete(
	uint64_t* __restrict__ keys,
	uint32_t* __restrict__ values,
	uint64_t capacity,
	const uint64_t* __restrict__ del_keys,
	uint32_t num_items)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_items) return;

	uint64_t key = del_keys[idx];
	uint64_t slot = cuda_hash(key) % capacity;

	for (int probe = 0; probe < CUDA_MAX_PROBES; probe++)
	{
		uint64_t existing = keys[slot];
		if (existing == key)
		{
			// Tombstone: mark as deleted (use EMPTY-1 as tombstone)
			keys[slot] = 0xFFFFFFFFFFFFFFFEULL;
			values[slot] = CUDA_HT_EMPTY_VALUE;
			return;
		}
		if (existing == CUDA_HT_EMPTY_KEY) return; // not found
		slot = (slot + 1) % capacity;
	}
}

// ----------------------------------------------------------
// Word find-or-create: hash table insert + pool allocation
__global__ void cuda_word_find_or_create(
	uint64_t* __restrict__ ht_keys,
	uint32_t* __restrict__ ht_values,
	uint64_t* __restrict__ name_hashes,
	double*   __restrict__ word_counts,
	uint32_t* __restrict__ class_ids,
	uint32_t* __restrict__ next_free,
	const uint64_t* __restrict__ in_hashes,
	uint32_t* __restrict__ out_indices,
	uint32_t num_items)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_items) return;

	uint64_t key = in_hashes[idx];
	uint64_t slot = cuda_hash(key) % GPU_WORD_HT_CAPACITY;

	for (int probe = 0; probe < CUDA_MAX_PROBES; probe++)
	{
		uint64_t old = atomicCAS(
			(unsigned long long*)&ht_keys[slot],
			(unsigned long long)CUDA_HT_EMPTY_KEY,
			(unsigned long long)key);

		if (old == key)
		{
			// Already exists
			out_indices[idx] = ht_values[slot];
			return;
		}
		if (old == CUDA_HT_EMPTY_KEY)
		{
			// We won the slot; allocate pool entry
			uint32_t pool_idx = atomicAdd(next_free, 1);
			ht_values[slot] = pool_idx;
			name_hashes[pool_idx] = key;
			word_counts[pool_idx] = 0.0;
			class_ids[pool_idx] = 0;
			out_indices[idx] = pool_idx;
			return;
		}
		slot = (slot + 1) % GPU_WORD_HT_CAPACITY;
	}
	out_indices[idx] = CUDA_HT_EMPTY_VALUE;
}

// ----------------------------------------------------------
// Pair find-or-create: hash table insert + pool allocation
__global__ void cuda_pair_find_or_create(
	uint64_t* __restrict__ ht_keys,
	uint32_t* __restrict__ ht_values,
	uint32_t* __restrict__ pair_wa,
	uint32_t* __restrict__ pair_wb,
	double*   __restrict__ pair_counts,
	double*   __restrict__ pair_mis,
	uint32_t* __restrict__ pair_flags,
	uint32_t* __restrict__ next_free,
	const uint32_t* __restrict__ in_wa,
	const uint32_t* __restrict__ in_wb,
	uint32_t* __restrict__ out_indices,
	uint32_t num_items)
{
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_items) return;

	uint32_t wa = in_wa[idx];
	uint32_t wb = in_wb[idx];
	uint32_t lo = min(wa, wb);
	uint32_t hi = max(wa, wb);
	uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;

	uint64_t slot = cuda_hash(key) % GPU_PAIR_HT_CAPACITY;

	for (int probe = 0; probe < CUDA_MAX_PROBES; probe++)
	{
		uint64_t old = atomicCAS(
			(unsigned long long*)&ht_keys[slot],
			(unsigned long long)CUDA_HT_EMPTY_KEY,
			(unsigned long long)key);

		if (old == key)
		{
			out_indices[idx] = ht_values[slot];
			return;
		}
		if (old == CUDA_HT_EMPTY_KEY)
		{
			uint32_t pool_idx = atomicAdd(next_free, 1);
			ht_values[slot] = pool_idx;
			pair_wa[pool_idx] = wa;
			pair_wb[pool_idx] = wb;
			pair_counts[pool_idx] = 0.0;
			pair_mis[pool_idx] = 0.0;
			pair_flags[pool_idx] = 0;
			out_indices[idx] = pool_idx;
			return;
		}
		slot = (slot + 1) % GPU_PAIR_HT_CAPACITY;
	}
	out_indices[idx] = CUDA_HT_EMPTY_VALUE;
}

// ==============================================================
// Host-side Implementation
// ==============================================================

CudaBackend::CudaBackend()
	: _device_id(-1),
	  _d_word_name_hash(nullptr), _d_word_count(nullptr),
	  _d_word_mi_marginal(nullptr), _d_word_class_id(nullptr),
	  _d_word_next_free(nullptr),
	  _d_pair_word_a(nullptr), _d_pair_word_b(nullptr),
	  _d_pair_count(nullptr), _d_pair_mi(nullptr),
	  _d_pair_flags(nullptr), _d_pair_next_free(nullptr),
	  _d_sec_word(nullptr), _d_sec_djh(nullptr),
	  _d_sec_count(nullptr), _d_sec_next_free(nullptr),
	  _d_word_ht_keys(nullptr), _d_word_ht_values(nullptr),
	  _d_pair_ht_keys(nullptr), _d_pair_ht_values(nullptr),
	  _d_sec_ht_keys(nullptr), _d_sec_ht_values(nullptr),
	  _staging_key(nullptr), _staging_result(nullptr),
	  _staging_wa(nullptr), _staging_wb(nullptr)
{
}

CudaBackend::~CudaBackend()
{
	shutdown();
}

void CudaBackend::check_error(cudaError_t err, const char* context)
{
	if (err != cudaSuccess)
		throw RuntimeException(TRACE_INFO,
			"CudaBackend: %s failed: %s\n",
			context, cudaGetErrorString(err));
}

// ==============================================================
// Lifecycle

void CudaBackend::init(const std::string& platform_hint,
                       const std::string& device_hint)
{
	int device_count = 0;
	check_error(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");

	if (device_count == 0)
		throw RuntimeException(TRACE_INFO,
			"CudaBackend: no CUDA devices found\n");

	// Find a matching device (by name substring)
	_device_id = -1;
	for (int i = 0; i < device_count; i++)
	{
		cudaDeviceProp prop;
		check_error(cudaGetDeviceProperties(&prop, i),
			"cudaGetDeviceProperties");

		std::string dname(prop.name);

		// Check platform hint (vendor match)
		if (0 < platform_hint.size())
		{
			// NVIDIA is always the CUDA platform, so skip platform matching
			// unless the user explicitly asked for non-NVIDIA
			if (dname.find(platform_hint) == std::string::npos and
			    platform_hint != "NVIDIA" and platform_hint != "nvidia")
				continue;
		}

		// Check device hint (model match)
		if (0 < device_hint.size() and
		    dname.find(device_hint) == std::string::npos)
			continue;

		_device_id = i;
		_device_name = dname;
		break;
	}

	if (_device_id < 0)
		throw RuntimeException(TRACE_INFO,
			"CudaBackend: no matching CUDA device for '%s:%s'\n",
			platform_hint.c_str(), device_hint.c_str());

	check_error(cudaSetDevice(_device_id), "cudaSetDevice");

	logger().info("CudaBackend: Using device %d '%s'\n",
		_device_id, _device_name.c_str());
}

void CudaBackend::shutdown()
{
	if (_device_id < 0) return;

	cudaDeviceSynchronize();

	// Free device memory
	if (_d_word_name_hash) cudaFree(_d_word_name_hash);
	if (_d_word_count)     cudaFree(_d_word_count);
	if (_d_word_mi_marginal) cudaFree(_d_word_mi_marginal);
	if (_d_word_class_id)  cudaFree(_d_word_class_id);
	if (_d_word_next_free) cudaFree(_d_word_next_free);

	if (_d_pair_word_a)    cudaFree(_d_pair_word_a);
	if (_d_pair_word_b)    cudaFree(_d_pair_word_b);
	if (_d_pair_count)     cudaFree(_d_pair_count);
	if (_d_pair_mi)        cudaFree(_d_pair_mi);
	if (_d_pair_flags)     cudaFree(_d_pair_flags);
	if (_d_pair_next_free) cudaFree(_d_pair_next_free);

	if (_d_sec_word)       cudaFree(_d_sec_word);
	if (_d_sec_djh)        cudaFree(_d_sec_djh);
	if (_d_sec_count)      cudaFree(_d_sec_count);
	if (_d_sec_next_free)  cudaFree(_d_sec_next_free);

	if (_d_word_ht_keys)   cudaFree(_d_word_ht_keys);
	if (_d_word_ht_values) cudaFree(_d_word_ht_values);
	if (_d_pair_ht_keys)   cudaFree(_d_pair_ht_keys);
	if (_d_pair_ht_values) cudaFree(_d_pair_ht_values);
	if (_d_sec_ht_keys)    cudaFree(_d_sec_ht_keys);
	if (_d_sec_ht_values)  cudaFree(_d_sec_ht_values);

	// Free managed staging buffers
	if (_staging_key)    cudaFree(_staging_key);
	if (_staging_result) cudaFree(_staging_result);
	if (_staging_wa)     cudaFree(_staging_wa);
	if (_staging_wb)     cudaFree(_staging_wb);

	// Reset all pointers
	_d_word_name_hash = nullptr;
	_d_word_count = nullptr;
	_d_word_mi_marginal = nullptr;
	_d_word_class_id = nullptr;
	_d_word_next_free = nullptr;
	_d_pair_word_a = nullptr;
	_d_pair_word_b = nullptr;
	_d_pair_count = nullptr;
	_d_pair_mi = nullptr;
	_d_pair_flags = nullptr;
	_d_pair_next_free = nullptr;
	_d_sec_word = nullptr;
	_d_sec_djh = nullptr;
	_d_sec_count = nullptr;
	_d_sec_next_free = nullptr;
	_d_word_ht_keys = nullptr;
	_d_word_ht_values = nullptr;
	_d_pair_ht_keys = nullptr;
	_d_pair_ht_values = nullptr;
	_d_sec_ht_keys = nullptr;
	_d_sec_ht_values = nullptr;
	_staging_key = nullptr;
	_staging_result = nullptr;
	_staging_wa = nullptr;
	_staging_wb = nullptr;

	_device_id = -1;
}

std::string CudaBackend::device_info()
{
	return "CUDA / " + _device_name;
}

// ==============================================================
// Pool allocation

void CudaBackend::alloc_pools()
{
	// -- WordPool --
	check_error(cudaMalloc(&_d_word_name_hash,
		sizeof(uint64_t) * GPU_WORD_CAPACITY), "alloc word_name_hash");
	check_error(cudaMalloc(&_d_word_count,
		sizeof(double) * GPU_WORD_CAPACITY), "alloc word_count");
	check_error(cudaMalloc(&_d_word_mi_marginal,
		sizeof(double) * GPU_WORD_CAPACITY), "alloc word_mi_marginal");
	check_error(cudaMalloc(&_d_word_class_id,
		sizeof(uint32_t) * GPU_WORD_CAPACITY), "alloc word_class_id");
	check_error(cudaMalloc(&_d_word_next_free,
		sizeof(uint32_t)), "alloc word_next_free");

	// -- PairPool --
	check_error(cudaMalloc(&_d_pair_word_a,
		sizeof(uint32_t) * GPU_PAIR_CAPACITY), "alloc pair_word_a");
	check_error(cudaMalloc(&_d_pair_word_b,
		sizeof(uint32_t) * GPU_PAIR_CAPACITY), "alloc pair_word_b");
	check_error(cudaMalloc(&_d_pair_count,
		sizeof(double) * GPU_PAIR_CAPACITY), "alloc pair_count");
	check_error(cudaMalloc(&_d_pair_mi,
		sizeof(double) * GPU_PAIR_CAPACITY), "alloc pair_mi");
	check_error(cudaMalloc(&_d_pair_flags,
		sizeof(uint32_t) * GPU_PAIR_CAPACITY), "alloc pair_flags");
	check_error(cudaMalloc(&_d_pair_next_free,
		sizeof(uint32_t)), "alloc pair_next_free");

	// -- SectionPool --
	check_error(cudaMalloc(&_d_sec_word,
		sizeof(uint32_t) * GPU_SECTION_CAPACITY), "alloc sec_word");
	check_error(cudaMalloc(&_d_sec_djh,
		sizeof(uint64_t) * GPU_SECTION_CAPACITY), "alloc sec_djh");
	check_error(cudaMalloc(&_d_sec_count,
		sizeof(double) * GPU_SECTION_CAPACITY), "alloc sec_count");
	check_error(cudaMalloc(&_d_sec_next_free,
		sizeof(uint32_t)), "alloc sec_next_free");

	// -- Hash tables --
	check_error(cudaMalloc(&_d_word_ht_keys,
		sizeof(uint64_t) * GPU_WORD_HT_CAPACITY), "alloc word_ht_keys");
	check_error(cudaMalloc(&_d_word_ht_values,
		sizeof(uint32_t) * GPU_WORD_HT_CAPACITY), "alloc word_ht_values");
	check_error(cudaMalloc(&_d_pair_ht_keys,
		sizeof(uint64_t) * GPU_PAIR_HT_CAPACITY), "alloc pair_ht_keys");
	check_error(cudaMalloc(&_d_pair_ht_values,
		sizeof(uint32_t) * GPU_PAIR_HT_CAPACITY), "alloc pair_ht_values");
	check_error(cudaMalloc(&_d_sec_ht_keys,
		sizeof(uint64_t) * GPU_SECTION_HT_CAPACITY), "alloc sec_ht_keys");
	check_error(cudaMalloc(&_d_sec_ht_values,
		sizeof(uint32_t) * GPU_SECTION_HT_CAPACITY), "alloc sec_ht_values");

	// -- Managed staging buffers (CPU+GPU accessible) --
	check_error(cudaMallocManaged(&_staging_key, sizeof(uint64_t)),
		"alloc staging_key");
	check_error(cudaMallocManaged(&_staging_result, sizeof(uint32_t)),
		"alloc staging_result");
	check_error(cudaMallocManaged(&_staging_wa, sizeof(uint32_t)),
		"alloc staging_wa");
	check_error(cudaMallocManaged(&_staging_wb, sizeof(uint32_t)),
		"alloc staging_wb");
}

void CudaBackend::init_pools()
{
	// Zero bump allocators
	uint32_t zero = 0;
	cudaMemcpy(_d_word_next_free, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(_d_pair_next_free, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(_d_sec_next_free, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice);

	// Initialize hash table keys to EMPTY (0xFF bytes)
	cudaMemset(_d_word_ht_keys,   0xFF, sizeof(uint64_t) * GPU_WORD_HT_CAPACITY);
	cudaMemset(_d_word_ht_values, 0xFF, sizeof(uint32_t) * GPU_WORD_HT_CAPACITY);
	cudaMemset(_d_pair_ht_keys,   0xFF, sizeof(uint64_t) * GPU_PAIR_HT_CAPACITY);
	cudaMemset(_d_pair_ht_values, 0xFF, sizeof(uint32_t) * GPU_PAIR_HT_CAPACITY);
	cudaMemset(_d_sec_ht_keys,    0xFF, sizeof(uint64_t) * GPU_SECTION_HT_CAPACITY);
	cudaMemset(_d_sec_ht_values,  0xFF, sizeof(uint32_t) * GPU_SECTION_HT_CAPACITY);

	cudaDeviceSynchronize();

	logger().info("CudaBackend: GPU pools initialized\n");
}

// ==============================================================
// Word pool operations

uint32_t CudaBackend::word_find_or_create(uint64_t name_hash)
{
	*_staging_key = name_hash;
	*_staging_result = GPU_NOT_FOUND;

	cuda_word_find_or_create<<<1, 1>>>(
		_d_word_ht_keys, _d_word_ht_values,
		_d_word_name_hash, _d_word_count, _d_word_class_id,
		_d_word_next_free,
		_staging_key, _staging_result, 1);
	cudaDeviceSynchronize();

	return *_staging_result;
}

uint32_t CudaBackend::word_lookup(uint64_t name_hash)
{
	*_staging_key = name_hash;
	*_staging_result = GPU_NOT_FOUND;

	cuda_ht_lookup<<<1, 1>>>(
		_d_word_ht_keys, _d_word_ht_values,
		(uint64_t)GPU_WORD_HT_CAPACITY,
		_staging_key, _staging_result, 1);
	cudaDeviceSynchronize();

	return *_staging_result;
}

void CudaBackend::word_write_count(uint32_t idx, double count)
{
	cudaMemcpy(_d_word_count + idx, &count,
		sizeof(double), cudaMemcpyHostToDevice);
}

void CudaBackend::word_write_marginal(uint32_t idx, double marginal)
{
	cudaMemcpy(_d_word_mi_marginal + idx, &marginal,
		sizeof(double), cudaMemcpyHostToDevice);
}

void CudaBackend::word_read_values(uint32_t idx,
                                   double& count, double& marginal)
{
	cudaMemcpy(&count, _d_word_count + idx,
		sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&marginal, _d_word_mi_marginal + idx,
		sizeof(double), cudaMemcpyDeviceToHost);
}

void CudaBackend::word_delete(uint64_t name_hash)
{
	*_staging_key = name_hash;

	cuda_ht_delete<<<1, 1>>>(
		_d_word_ht_keys, _d_word_ht_values,
		(uint64_t)GPU_WORD_HT_CAPACITY,
		_staging_key, 1);
	cudaDeviceSynchronize();
}

uint32_t CudaBackend::word_pool_count()
{
	uint32_t cnt = 0;
	cudaMemcpy(&cnt, _d_word_next_free,
		sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return cnt;
}

void CudaBackend::word_read_bulk(uint32_t n, uint64_t* hashes,
                                 double* counts, double* marginals)
{
	cudaMemcpy(hashes, _d_word_name_hash,
		sizeof(uint64_t) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(counts, _d_word_count,
		sizeof(double) * n, cudaMemcpyDeviceToHost);
	cudaMemcpy(marginals, _d_word_mi_marginal,
		sizeof(double) * n, cudaMemcpyDeviceToHost);
}

// ==============================================================
// Pair pool operations

uint32_t CudaBackend::pair_find_or_create(uint32_t word_a, uint32_t word_b)
{
	*_staging_wa = word_a;
	*_staging_wb = word_b;
	*_staging_result = GPU_NOT_FOUND;

	cuda_pair_find_or_create<<<1, 1>>>(
		_d_pair_ht_keys, _d_pair_ht_values,
		_d_pair_word_a, _d_pair_word_b,
		_d_pair_count, _d_pair_mi, _d_pair_flags,
		_d_pair_next_free,
		_staging_wa, _staging_wb, _staging_result, 1);
	cudaDeviceSynchronize();

	return *_staging_result;
}

uint32_t CudaBackend::pair_lookup(uint32_t word_a, uint32_t word_b)
{
	uint32_t lo = std::min(word_a, word_b);
	uint32_t hi = std::max(word_a, word_b);
	uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;

	*_staging_key = key;
	*_staging_result = GPU_NOT_FOUND;

	cuda_ht_lookup<<<1, 1>>>(
		_d_pair_ht_keys, _d_pair_ht_values,
		(uint64_t)GPU_PAIR_HT_CAPACITY,
		_staging_key, _staging_result, 1);
	cudaDeviceSynchronize();

	return *_staging_result;
}

void CudaBackend::pair_write_count(uint32_t idx, double count)
{
	cudaMemcpy(_d_pair_count + idx, &count,
		sizeof(double), cudaMemcpyHostToDevice);
}

void CudaBackend::pair_write_mi(uint32_t idx, double mi)
{
	cudaMemcpy(_d_pair_mi + idx, &mi,
		sizeof(double), cudaMemcpyHostToDevice);
}

void CudaBackend::pair_read_values(uint32_t idx,
                                   double& count, double& mi)
{
	cudaMemcpy(&count, _d_pair_count + idx,
		sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(&mi, _d_pair_mi + idx,
		sizeof(double), cudaMemcpyDeviceToHost);
}

uint32_t CudaBackend::pair_pool_count()
{
	uint32_t cnt = 0;
	cudaMemcpy(&cnt, _d_pair_next_free,
		sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return cnt;
}

// ==============================================================
// Section pool operations

uint32_t CudaBackend::section_pool_count()
{
	uint32_t cnt = 0;
	cudaMemcpy(&cnt, _d_sec_next_free,
		sizeof(uint32_t), cudaMemcpyDeviceToHost);
	return cnt;
}

// ==============================================================

void CudaBackend::barrier()
{
	cudaDeviceSynchronize();
}

// ==============================================================
// Factory function

namespace opencog {

GpuBackend* create_cuda_backend()
{
	return new CudaBackend();
}

} // namespace opencog

// ==============================================================
