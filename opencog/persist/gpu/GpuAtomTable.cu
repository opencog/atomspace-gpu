/*
 * opencog/persist/gpu/GpuAtomTable.cu
 *
 * CUDA implementation of the GpuAtomTable API.
 *
 * AoS for per-atom fixed fields: one GpuAtom struct per slot,
 * one cudaMemcpy to store/fetch all fields of one atom.
 * Variable-length data (names, outgoing sets) in side pools.
 *
 * Copyright (C) 2026 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include "GpuAtomTable.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

/* ---------------------------------------------------------------- */
#define CUDA_CHECK(call, ctx) do {                                 \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error in %s: %s\n",                 \
                (ctx), cudaGetErrorString(_e));                    \
        return -1;                                                 \
    }                                                              \
} while(0)

#define CUDA_CHECK_VOID(call, ctx) do {                            \
    cudaError_t _e = (call);                                       \
    if (_e != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error in %s: %s\n",                 \
                (ctx), cudaGetErrorString(_e));                    \
    }                                                              \
} while(0)

/* ================================================================
 * gpu_table_alloc
 * ================================================================ */
extern "C"
int cuda_table_alloc(GpuAtomTable* t)
{
    memset(t, 0, sizeof(GpuAtomTable));

    CUDA_CHECK(cudaMalloc(&t->atoms,     GPU_ATOM_CAPACITY * sizeof(GpuAtom)),
               "alloc atoms");
    CUDA_CHECK(cudaMalloc(&t->name_pool, GPU_NAME_POOL_BYTES),
               "alloc name_pool");
    CUDA_CHECK(cudaMalloc(&t->out_pool,  GPU_OUT_POOL_SLOTS * sizeof(uint32_t)),
               "alloc out_pool");

    CUDA_CHECK(cudaMemset(t->atoms,     0, GPU_ATOM_CAPACITY * sizeof(GpuAtom)),
               "zero atoms");
    CUDA_CHECK(cudaMemset(t->name_pool, 0, GPU_NAME_POOL_BYTES),
               "zero name_pool");
    CUDA_CHECK(cudaMemset(t->out_pool,  0, GPU_OUT_POOL_SLOTS * sizeof(uint32_t)),
               "zero out_pool");

    t->atom_count     = 0;
    t->name_pool_used = 0;
    t->out_pool_used  = 0;

    return 0;
}

/* ================================================================
 * gpu_table_free
 * ================================================================ */
extern "C"
void cuda_table_free(GpuAtomTable* t)
{
    if (t->atoms)     cudaFree(t->atoms);
    if (t->name_pool) cudaFree(t->name_pool);
    if (t->out_pool)  cudaFree(t->out_pool);

    memset(t, 0, sizeof(GpuAtomTable));
}

/* ================================================================
 * gpu_table_clear
 * ================================================================ */
extern "C"
void cuda_table_clear(GpuAtomTable* t)
{
    CUDA_CHECK_VOID(cudaMemset(t->atoms,     0, GPU_ATOM_CAPACITY * sizeof(GpuAtom)),
                    "clear atoms");
    CUDA_CHECK_VOID(cudaMemset(t->name_pool, 0, GPU_NAME_POOL_BYTES),
                    "clear name_pool");
    CUDA_CHECK_VOID(cudaMemset(t->out_pool,  0, GPU_OUT_POOL_SLOTS * sizeof(uint32_t)),
                    "clear out_pool");

    t->atom_count     = 0;
    t->name_pool_used = 0;
    t->out_pool_used  = 0;
}

/* ================================================================
 * gpu_store_node -- store type + name at slot.
 * ================================================================ */
extern "C"
int cuda_store_node(GpuAtomTable* t, uint32_t slot,
                   uint16_t type, const char* name, uint16_t name_len)
{
    if (slot >= GPU_ATOM_CAPACITY) return -1;
    if (t->name_pool_used + name_len > GPU_NAME_POOL_BYTES) return -1;

    uint32_t off = t->name_pool_used;

    /* Build the atom struct on host, copy in one shot */
    GpuAtom a;
    memset(&a, 0, sizeof(a));
    a.type        = type;
    a.is_node     = 1;
    a.data_offset = off;
    a.data_len    = name_len;

    CUDA_CHECK(cudaMemcpy(t->atoms + slot, &a, sizeof(GpuAtom),
                           cudaMemcpyHostToDevice), "store node atom");

    /* Append name to pool */
    if (name_len > 0)
    {
        CUDA_CHECK(cudaMemcpy(t->name_pool + off, name, name_len,
                               cudaMemcpyHostToDevice), "store node name");
    }

    t->name_pool_used += name_len;
    if (slot >= t->atom_count) t->atom_count = slot + 1;

    return 0;
}

/* ================================================================
 * gpu_store_link -- store type + outgoing set at slot.
 * ================================================================ */
extern "C"
int cuda_store_link(GpuAtomTable* t, uint32_t slot,
                   uint16_t type,
                   const uint32_t* outgoing, uint16_t arity)
{
    if (slot >= GPU_ATOM_CAPACITY) return -1;
    if (t->out_pool_used + arity > GPU_OUT_POOL_SLOTS) return -1;

    uint32_t off = t->out_pool_used;

    GpuAtom a;
    memset(&a, 0, sizeof(a));
    a.type        = type;
    a.is_node     = 0;
    a.data_offset = off;
    a.data_len    = arity;

    CUDA_CHECK(cudaMemcpy(t->atoms + slot, &a, sizeof(GpuAtom),
                           cudaMemcpyHostToDevice), "store link atom");

    /* Append outgoing set to pool */
    if (arity > 0)
    {
        CUDA_CHECK(cudaMemcpy(t->out_pool + off, outgoing,
                               arity * sizeof(uint32_t),
                               cudaMemcpyHostToDevice), "store link outgoing");
    }

    t->out_pool_used += arity;
    if (slot >= t->atom_count) t->atom_count = slot + 1;

    return 0;
}

/* ================================================================
 * gpu_fetch_node -- read node back to host.
 * ================================================================ */
extern "C"
int cuda_fetch_node(const GpuAtomTable* t, uint32_t slot,
                   uint16_t* type, char* name_buf, uint16_t* name_len)
{
    if (slot >= GPU_ATOM_CAPACITY) return -1;

    GpuAtom a;
    CUDA_CHECK(cudaMemcpy(&a, t->atoms + slot, sizeof(GpuAtom),
                           cudaMemcpyDeviceToHost), "fetch node atom");

    *type     = a.type;
    *name_len = a.data_len;

    if (a.data_len > 0 && name_buf)
    {
        CUDA_CHECK(cudaMemcpy(name_buf, t->name_pool + a.data_offset,
                               a.data_len,
                               cudaMemcpyDeviceToHost), "fetch node name");
    }

    return 0;
}

/* ================================================================
 * gpu_fetch_link -- read link back to host.
 * ================================================================ */
extern "C"
int cuda_fetch_link(const GpuAtomTable* t, uint32_t slot,
                   uint16_t* type, uint32_t* out_buf, uint16_t* arity)
{
    if (slot >= GPU_ATOM_CAPACITY) return -1;

    GpuAtom a;
    CUDA_CHECK(cudaMemcpy(&a, t->atoms + slot, sizeof(GpuAtom),
                           cudaMemcpyDeviceToHost), "fetch link atom");

    *type  = a.type;
    *arity = a.data_len;

    if (a.data_len > 0 && out_buf)
    {
        CUDA_CHECK(cudaMemcpy(out_buf, t->out_pool + a.data_offset,
                               a.data_len * sizeof(uint32_t),
                               cudaMemcpyDeviceToHost), "fetch link outgoing");
    }

    return 0;
}

/* ================================================================
 * gpu_table_barrier
 * ================================================================ */
extern "C"
void cuda_table_barrier(GpuAtomTable* t)
{
    (void)t;
    cudaDeviceSynchronize();
}
