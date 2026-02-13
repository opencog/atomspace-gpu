/*
 * opencog/persist/gpu/GpuAtomTable.h
 *
 * GPU-resident atom table: holds nodes and links in GPU RAM.
 *
 * AoS layout for per-atom fixed fields (one struct per atom),
 * with separate pools for variable-length data (names, outgoing
 * sets). The graph is accessed per-atom (store one, fetch one),
 * so AoS is the natural layout. SoA slicing is reserved for
 * Values (Step 4, slice-table) where GPU kernels process one
 * field across many atoms in parallel.
 *
 * Step 2 of Linas' plan: hold nodes and links in GPU RAM.
 * No ContentHash, no flags, no Values, no incoming sets --
 * just type + name (nodes) and type + outgoing (links).
 *
 * CUDA is preferred; OpenCL is fallback. Both can be compiled
 * in; gpu_table_alloc() tries CUDA first, then OpenCL.
 *
 * Copyright (C) 2026 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef _OPENCOG_GPU_ATOM_TABLE_H
#define _OPENCOG_GPU_ATOM_TABLE_H

#include <stdint.h>

/* -----------------------------------------------------------
 * Capacity defaults (overridable at compile time with -D).
 *
 * GPU memory at defaults:
 *   atoms      1M * 12B =  12 MB
 *   name_pool           =  64 MB
 *   out_pool   4M *  4B =  16 MB
 *                 Total ~  92 MB
 * ----------------------------------------------------------- */

#ifndef GPU_ATOM_CAPACITY
#define GPU_ATOM_CAPACITY       1048576   /* 1M atoms */
#endif

#ifndef GPU_NAME_POOL_BYTES
#define GPU_NAME_POOL_BYTES     67108864  /* 64 MB for packed names */
#endif

#ifndef GPU_OUT_POOL_SLOTS
#define GPU_OUT_POOL_SLOTS      4194304   /* 4M outgoing-set entries */
#endif

/* -----------------------------------------------------------
 * GpuBackendType -- which GPU backend is active for a table.
 * ----------------------------------------------------------- */
typedef enum GpuBackendType
{
	GPU_BACKEND_NONE   = 0,
	GPU_BACKEND_CUDA   = 1,
	GPU_BACKEND_OPENCL = 2
} GpuBackendType;

/* -----------------------------------------------------------
 * GpuAtom -- fixed-size per-atom record (12 bytes).
 * One cudaMemcpy to store/fetch all fields of one atom.
 * ----------------------------------------------------------- */
typedef struct GpuAtom
{
	uint16_t type;           /* atom type id */
	uint8_t  is_node;        /* 1 = node, 0 = link */
	uint8_t  _pad;           /* alignment padding */
	uint32_t data_offset;    /* byte offset into name_pool (node)
	                            or slot index into out_pool (link) */
	uint16_t data_len;       /* name length in bytes (node)
	                            or arity (link) */
	uint16_t _pad2;          /* pad to 12 bytes */
} GpuAtom;

/* -----------------------------------------------------------
 * GpuAtomTable -- device pointers to atom array + pools.
 * The struct itself lives on the host.
 * ----------------------------------------------------------- */
typedef struct GpuAtomTable
{
	GpuBackendType backend;  /* which GPU backend owns this table */

	/* AoS: one GpuAtom per slot */
	GpuAtom*  atoms;         /* [CAPACITY] device array */

	/* Variable-length pools (shared across all atoms) */
	char*     name_pool;     /* [NAME_POOL_BYTES] packed node names */
	uint32_t* out_pool;      /* [OUT_POOL_SLOTS] packed outgoing sets */

	/* Host-side bookkeeping (NOT on device) */
	uint32_t  atom_count;     /* next free slot index */
	uint32_t  name_pool_used; /* bytes used in name_pool */
	uint32_t  out_pool_used;  /* slots used in out_pool */
} GpuAtomTable;

/* -----------------------------------------------------------
 * Public C-linkage API -- dispatches to CUDA or OpenCL backend.
 * ----------------------------------------------------------- */
#ifdef __cplusplus
extern "C" {
#endif

/* Lifecycle */
int  gpu_table_alloc(GpuAtomTable* t);
void gpu_table_free(GpuAtomTable* t);
void gpu_table_clear(GpuAtomTable* t);

/* Store (host -> device). Slot assignment is the caller's job. */
int gpu_store_node(GpuAtomTable* t, uint32_t slot,
                   uint16_t type, const char* name, uint16_t name_len);

int gpu_store_link(GpuAtomTable* t, uint32_t slot,
                   uint16_t type,
                   const uint32_t* outgoing, uint16_t arity);

/* Fetch (device -> host). Caller provides buffers. */
int gpu_fetch_node(const GpuAtomTable* t, uint32_t slot,
                   uint16_t* type, char* name_buf, uint16_t* name_len);

int gpu_fetch_link(const GpuAtomTable* t, uint32_t slot,
                   uint16_t* type, uint32_t* out_buf, uint16_t* arity);

/* Synchronization */
void gpu_table_barrier(GpuAtomTable* t);

/* -----------------------------------------------------------
 * Backend-specific functions (internal, called by dispatch).
 * ----------------------------------------------------------- */

#ifdef HAVE_CUDA
int  cuda_table_alloc(GpuAtomTable* t);
void cuda_table_free(GpuAtomTable* t);
void cuda_table_clear(GpuAtomTable* t);
int  cuda_store_node(GpuAtomTable* t, uint32_t slot,
                     uint16_t type, const char* name, uint16_t name_len);
int  cuda_store_link(GpuAtomTable* t, uint32_t slot,
                     uint16_t type,
                     const uint32_t* outgoing, uint16_t arity);
int  cuda_fetch_node(const GpuAtomTable* t, uint32_t slot,
                     uint16_t* type, char* name_buf, uint16_t* name_len);
int  cuda_fetch_link(const GpuAtomTable* t, uint32_t slot,
                     uint16_t* type, uint32_t* out_buf, uint16_t* arity);
void cuda_table_barrier(GpuAtomTable* t);
#endif

#ifdef HAVE_OPENCL
int  ocl_table_alloc(GpuAtomTable* t);
void ocl_table_free(GpuAtomTable* t);
void ocl_table_clear(GpuAtomTable* t);
int  ocl_store_node(GpuAtomTable* t, uint32_t slot,
                    uint16_t type, const char* name, uint16_t name_len);
int  ocl_store_link(GpuAtomTable* t, uint32_t slot,
                    uint16_t type,
                    const uint32_t* outgoing, uint16_t arity);
int  ocl_fetch_node(const GpuAtomTable* t, uint32_t slot,
                    uint16_t* type, char* name_buf, uint16_t* name_len);
int  ocl_fetch_link(const GpuAtomTable* t, uint32_t slot,
                    uint16_t* type, uint32_t* out_buf, uint16_t* arity);
void ocl_table_barrier(GpuAtomTable* t);
#endif

#ifdef __cplusplus
}
#endif

#endif /* _OPENCOG_GPU_ATOM_TABLE_H */
