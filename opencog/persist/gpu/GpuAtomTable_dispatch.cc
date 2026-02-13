/*
 * opencog/persist/gpu/GpuAtomTable_dispatch.cc
 *
 * Runtime dispatch for GpuAtomTable: tries CUDA first, falls
 * back to OpenCL. All public gpu_* functions route through here.
 *
 * Copyright (C) 2026 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include "GpuAtomTable.h"

#include <cstdio>
#include <cstring>

/* ================================================================
 * gpu_table_alloc -- try CUDA, fall back to OpenCL.
 * ================================================================ */
extern "C"
int gpu_table_alloc(GpuAtomTable* t)
{
    memset(t, 0, sizeof(GpuAtomTable));
    t->backend = GPU_BACKEND_NONE;

#ifdef HAVE_CUDA
    if (cuda_table_alloc(t) == 0)
    {
        t->backend = GPU_BACKEND_CUDA;
        fprintf(stderr, "GpuAtomTable: using CUDA backend\n");
        return 0;
    }
    fprintf(stderr, "GpuAtomTable: CUDA failed, trying OpenCL...\n");
#endif

#ifdef HAVE_OPENCL
    if (ocl_table_alloc(t) == 0)
    {
        t->backend = GPU_BACKEND_OPENCL;
        fprintf(stderr, "GpuAtomTable: using OpenCL backend\n");
        return 0;
    }
    fprintf(stderr, "GpuAtomTable: OpenCL failed\n");
#endif

    fprintf(stderr, "GpuAtomTable: no GPU backend available\n");
    return -1;
}

/* ================================================================
 * gpu_table_free
 * ================================================================ */
extern "C"
void gpu_table_free(GpuAtomTable* t)
{
    switch (t->backend)
    {
#ifdef HAVE_CUDA
        case GPU_BACKEND_CUDA:   cuda_table_free(t); break;
#endif
#ifdef HAVE_OPENCL
        case GPU_BACKEND_OPENCL: ocl_table_free(t);  break;
#endif
        default: break;
    }
    t->backend = GPU_BACKEND_NONE;
}

/* ================================================================
 * gpu_table_clear
 * ================================================================ */
extern "C"
void gpu_table_clear(GpuAtomTable* t)
{
    switch (t->backend)
    {
#ifdef HAVE_CUDA
        case GPU_BACKEND_CUDA:   cuda_table_clear(t); break;
#endif
#ifdef HAVE_OPENCL
        case GPU_BACKEND_OPENCL: ocl_table_clear(t);  break;
#endif
        default: break;
    }
}

/* ================================================================
 * gpu_store_node
 * ================================================================ */
extern "C"
int gpu_store_node(GpuAtomTable* t, uint32_t slot,
                   uint16_t type, const char* name, uint16_t name_len)
{
    switch (t->backend)
    {
#ifdef HAVE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_store_node(t, slot, type, name, name_len);
#endif
#ifdef HAVE_OPENCL
        case GPU_BACKEND_OPENCL:
            return ocl_store_node(t, slot, type, name, name_len);
#endif
        default: return -1;
    }
}

/* ================================================================
 * gpu_store_link
 * ================================================================ */
extern "C"
int gpu_store_link(GpuAtomTable* t, uint32_t slot,
                   uint16_t type,
                   const uint32_t* outgoing, uint16_t arity)
{
    switch (t->backend)
    {
#ifdef HAVE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_store_link(t, slot, type, outgoing, arity);
#endif
#ifdef HAVE_OPENCL
        case GPU_BACKEND_OPENCL:
            return ocl_store_link(t, slot, type, outgoing, arity);
#endif
        default: return -1;
    }
}

/* ================================================================
 * gpu_fetch_node
 * ================================================================ */
extern "C"
int gpu_fetch_node(const GpuAtomTable* t, uint32_t slot,
                   uint16_t* type, char* name_buf, uint16_t* name_len)
{
    switch (t->backend)
    {
#ifdef HAVE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_fetch_node(t, slot, type, name_buf, name_len);
#endif
#ifdef HAVE_OPENCL
        case GPU_BACKEND_OPENCL:
            return ocl_fetch_node(t, slot, type, name_buf, name_len);
#endif
        default: return -1;
    }
}

/* ================================================================
 * gpu_fetch_link
 * ================================================================ */
extern "C"
int gpu_fetch_link(const GpuAtomTable* t, uint32_t slot,
                   uint16_t* type, uint32_t* out_buf, uint16_t* arity)
{
    switch (t->backend)
    {
#ifdef HAVE_CUDA
        case GPU_BACKEND_CUDA:
            return cuda_fetch_link(t, slot, type, out_buf, arity);
#endif
#ifdef HAVE_OPENCL
        case GPU_BACKEND_OPENCL:
            return ocl_fetch_link(t, slot, type, out_buf, arity);
#endif
        default: return -1;
    }
}

/* ================================================================
 * gpu_table_barrier
 * ================================================================ */
extern "C"
void gpu_table_barrier(GpuAtomTable* t)
{
    switch (t->backend)
    {
#ifdef HAVE_CUDA
        case GPU_BACKEND_CUDA:   cuda_table_barrier(t); break;
#endif
#ifdef HAVE_OPENCL
        case GPU_BACKEND_OPENCL: ocl_table_barrier(t);  break;
#endif
        default: break;
    }
}
