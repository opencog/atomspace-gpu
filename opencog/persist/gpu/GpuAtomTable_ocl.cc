/*
 * opencog/persist/gpu/GpuAtomTable_ocl.cc
 *
 * OpenCL implementation of the GpuAtomTable API.
 *
 * AoS for per-atom fixed fields: one GpuAtom struct per slot,
 * one enqueueWriteBuffer/enqueueReadBuffer per atom.
 * Variable-length data (names, outgoing sets) in side pools.
 *
 * Copyright (C) 2026 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifdef HAVE_OPENCL

#include "GpuAtomTable.h"
#include "opencl-headers.h"

#include <cstdio>
#include <cstring>
#include <vector>

/* ----------------------------------------------------------------
 * Internal state: OpenCL handles stored alongside the table.
 * ---------------------------------------------------------------- */
struct OclAtomTableState
{
    cl::Context       context;
    cl::CommandQueue  queue;

    cl::Buffer  b_atoms;       /* GpuAtom[CAPACITY] */
    cl::Buffer  b_name_pool;   /* char[NAME_POOL_BYTES] */
    cl::Buffer  b_out_pool;    /* uint32_t[OUT_POOL_SLOTS] */
};

static GpuAtomTable* s_active_table = nullptr;
static OclAtomTableState* s_state = nullptr;

static OclAtomTableState* get_state(const GpuAtomTable* t)
{
    if (s_active_table == t) return s_state;
    return nullptr;
}

static bool pick_device(cl::Platform& plat, cl::Device& dev)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) return false;

    for (auto& p : platforms)
    {
        std::vector<cl::Device> devs;
        p.getDevices(CL_DEVICE_TYPE_GPU, &devs);
        if (!devs.empty()) { plat = p; dev = devs[0]; return true; }
    }
    for (auto& p : platforms)
    {
        std::vector<cl::Device> devs;
        p.getDevices(CL_DEVICE_TYPE_ALL, &devs);
        if (!devs.empty()) { plat = p; dev = devs[0]; return true; }
    }
    return false;
}

/* ================================================================
 * gpu_table_alloc
 * ================================================================ */
extern "C"
int ocl_table_alloc(GpuAtomTable* t)
{
    memset(t, 0, sizeof(GpuAtomTable));

    cl::Platform plat;
    cl::Device dev;
    if (!pick_device(plat, dev))
    {
        fprintf(stderr, "OpenCL: no device found\n");
        return -1;
    }

    auto* st = new OclAtomTableState();

    try {
        st->context = cl::Context(dev);
        st->queue   = cl::CommandQueue(st->context, dev);

        cl_int err;

        #define MAKE_BUF(field, bytes) \
            st->field = cl::Buffer(st->context, CL_MEM_READ_WRITE, (bytes), nullptr, &err); \
            if (err != CL_SUCCESS) { fprintf(stderr, "OpenCL alloc " #field " failed\n"); delete st; return -1; }

        MAKE_BUF(b_atoms,     (size_t)GPU_ATOM_CAPACITY * sizeof(GpuAtom))
        MAKE_BUF(b_name_pool, GPU_NAME_POOL_BYTES)
        MAKE_BUF(b_out_pool,  (size_t)GPU_OUT_POOL_SLOTS * sizeof(uint32_t))

        #undef MAKE_BUF

        /* Zero all buffers */
        uint8_t zero = 0;
        st->queue.enqueueFillBuffer(st->b_atoms,     zero, 0,
                                     (size_t)GPU_ATOM_CAPACITY * sizeof(GpuAtom));
        st->queue.enqueueFillBuffer(st->b_name_pool,  zero, 0, GPU_NAME_POOL_BYTES);
        st->queue.enqueueFillBuffer(st->b_out_pool,   (uint32_t)0, 0,
                                     (size_t)GPU_OUT_POOL_SLOTS * sizeof(uint32_t));
        st->queue.finish();

    } catch (cl::Error& e) {
        fprintf(stderr, "OpenCL exception in alloc: %s (%d)\n",
                e.what(), e.err());
        delete st;
        return -1;
    }

    /* Opaque non-null markers */
    t->atoms     = (GpuAtom*)(uintptr_t)1;
    t->name_pool = (char*)(uintptr_t)1;
    t->out_pool  = (uint32_t*)(uintptr_t)1;

    t->atom_count     = 0;
    t->name_pool_used = 0;
    t->out_pool_used  = 0;

    s_active_table = t;
    s_state = st;

    return 0;
}

/* ================================================================
 * gpu_table_free
 * ================================================================ */
extern "C"
void ocl_table_free(GpuAtomTable* t)
{
    if (s_active_table == t && s_state)
    {
        delete s_state;
        s_state = nullptr;
        s_active_table = nullptr;
    }
    memset(t, 0, sizeof(GpuAtomTable));
}

/* ================================================================
 * gpu_table_clear
 * ================================================================ */
extern "C"
void ocl_table_clear(GpuAtomTable* t)
{
    auto* st = get_state(t);
    if (!st) return;

    try {
        uint8_t zero = 0;
        st->queue.enqueueFillBuffer(st->b_atoms,     zero, 0,
                                     (size_t)GPU_ATOM_CAPACITY * sizeof(GpuAtom));
        st->queue.enqueueFillBuffer(st->b_name_pool,  zero, 0, GPU_NAME_POOL_BYTES);
        st->queue.enqueueFillBuffer(st->b_out_pool,   (uint32_t)0, 0,
                                     (size_t)GPU_OUT_POOL_SLOTS * sizeof(uint32_t));
        st->queue.finish();
    } catch (cl::Error& e) {
        fprintf(stderr, "OpenCL exception in clear: %s (%d)\n",
                e.what(), e.err());
    }

    t->atom_count     = 0;
    t->name_pool_used = 0;
    t->out_pool_used  = 0;
}

/* ================================================================
 * gpu_store_node
 * ================================================================ */
extern "C"
int ocl_store_node(GpuAtomTable* t, uint32_t slot,
                   uint16_t type, const char* name, uint16_t name_len)
{
    if (slot >= GPU_ATOM_CAPACITY) return -1;
    if (t->name_pool_used + name_len > GPU_NAME_POOL_BYTES) return -1;

    auto* st = get_state(t);
    if (!st) return -1;

    uint32_t off = t->name_pool_used;

    GpuAtom a;
    memset(&a, 0, sizeof(a));
    a.type        = type;
    a.is_node     = 1;
    a.data_offset = off;
    a.data_len    = name_len;

    try {
        st->queue.enqueueWriteBuffer(st->b_atoms, CL_FALSE,
            slot * sizeof(GpuAtom), sizeof(GpuAtom), &a);

        if (name_len > 0)
            st->queue.enqueueWriteBuffer(st->b_name_pool, CL_FALSE,
                off, name_len, name);

        st->queue.finish();
    } catch (cl::Error& e) {
        fprintf(stderr, "OpenCL exception in store_node: %s (%d)\n",
                e.what(), e.err());
        return -1;
    }

    t->name_pool_used += name_len;
    if (slot >= t->atom_count) t->atom_count = slot + 1;
    return 0;
}

/* ================================================================
 * gpu_store_link
 * ================================================================ */
extern "C"
int ocl_store_link(GpuAtomTable* t, uint32_t slot,
                   uint16_t type,
                   const uint32_t* outgoing, uint16_t arity)
{
    if (slot >= GPU_ATOM_CAPACITY) return -1;
    if (t->out_pool_used + arity > GPU_OUT_POOL_SLOTS) return -1;

    auto* st = get_state(t);
    if (!st) return -1;

    uint32_t off = t->out_pool_used;

    GpuAtom a;
    memset(&a, 0, sizeof(a));
    a.type        = type;
    a.is_node     = 0;
    a.data_offset = off;
    a.data_len    = arity;

    try {
        st->queue.enqueueWriteBuffer(st->b_atoms, CL_FALSE,
            slot * sizeof(GpuAtom), sizeof(GpuAtom), &a);

        if (arity > 0)
            st->queue.enqueueWriteBuffer(st->b_out_pool, CL_FALSE,
                off * sizeof(uint32_t), arity * sizeof(uint32_t), outgoing);

        st->queue.finish();
    } catch (cl::Error& e) {
        fprintf(stderr, "OpenCL exception in store_link: %s (%d)\n",
                e.what(), e.err());
        return -1;
    }

    t->out_pool_used += arity;
    if (slot >= t->atom_count) t->atom_count = slot + 1;
    return 0;
}

/* ================================================================
 * gpu_fetch_node
 * ================================================================ */
extern "C"
int ocl_fetch_node(const GpuAtomTable* t, uint32_t slot,
                   uint16_t* type, char* name_buf, uint16_t* name_len)
{
    if (slot >= GPU_ATOM_CAPACITY) return -1;

    auto* st = get_state(t);
    if (!st) return -1;

    try {
        GpuAtom a;
        st->queue.enqueueReadBuffer(st->b_atoms, CL_TRUE,
            slot * sizeof(GpuAtom), sizeof(GpuAtom), &a);

        *type     = a.type;
        *name_len = a.data_len;

        if (a.data_len > 0 && name_buf)
        {
            st->queue.enqueueReadBuffer(st->b_name_pool, CL_TRUE,
                a.data_offset, a.data_len, name_buf);
        }
    } catch (cl::Error& e) {
        fprintf(stderr, "OpenCL exception in fetch_node: %s (%d)\n",
                e.what(), e.err());
        return -1;
    }
    return 0;
}

/* ================================================================
 * gpu_fetch_link
 * ================================================================ */
extern "C"
int ocl_fetch_link(const GpuAtomTable* t, uint32_t slot,
                   uint16_t* type, uint32_t* out_buf, uint16_t* arity)
{
    if (slot >= GPU_ATOM_CAPACITY) return -1;

    auto* st = get_state(t);
    if (!st) return -1;

    try {
        GpuAtom a;
        st->queue.enqueueReadBuffer(st->b_atoms, CL_TRUE,
            slot * sizeof(GpuAtom), sizeof(GpuAtom), &a);

        *type  = a.type;
        *arity = a.data_len;

        if (a.data_len > 0 && out_buf)
        {
            st->queue.enqueueReadBuffer(st->b_out_pool, CL_TRUE,
                a.data_offset * sizeof(uint32_t),
                a.data_len * sizeof(uint32_t), out_buf);
        }
    } catch (cl::Error& e) {
        fprintf(stderr, "OpenCL exception in fetch_link: %s (%d)\n",
                e.what(), e.err());
        return -1;
    }
    return 0;
}

/* ================================================================
 * gpu_table_barrier
 * ================================================================ */
extern "C"
void ocl_table_barrier(GpuAtomTable* t)
{
    auto* st = get_state(t);
    if (st)
    {
        try { st->queue.finish(); }
        catch (cl::Error& e) {
            fprintf(stderr, "OpenCL exception in barrier: %s (%d)\n",
                    e.what(), e.err());
        }
    }
}

#endif /* HAVE_OPENCL */
