/*
 * opencog/persist/gpu/gpu-pool-defs.h
 *
 * Shared pool capacity constants for GPU SoA pools.
 * Used by both CUDA and OpenCL backends.
 *
 * Copyright (C) 2025 OpenCog Foundation
 *
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#ifndef _OPENCOG_GPU_POOL_DEFS_H
#define _OPENCOG_GPU_POOL_DEFS_H

// Pool capacity defaults (can be overridden at compile time).
//
// GPU memory usage at default capacities:
//   WordPool     128K slots, 256K hash table  ~  4 MB
//   PairPool       4M slots,   8M hash table  ~160 MB
//   SectionPool    1M slots,   2M hash table  ~ 40 MB
//   Total                                     ~204 MB
#ifndef GPU_WORD_CAPACITY
#define GPU_WORD_CAPACITY       131072    // 128K words
#endif
#ifndef GPU_PAIR_CAPACITY
#define GPU_PAIR_CAPACITY      4194304    // 4M pairs
#endif
#ifndef GPU_SECTION_CAPACITY
#define GPU_SECTION_CAPACITY   1048576    // 1M sections
#endif

// Hash tables: 2x the pool capacity (50% load factor)
#ifndef GPU_WORD_HT_CAPACITY
#define GPU_WORD_HT_CAPACITY    262144
#endif
#ifndef GPU_PAIR_HT_CAPACITY
#define GPU_PAIR_HT_CAPACITY   8388608
#endif
#ifndef GPU_SECTION_HT_CAPACITY
#define GPU_SECTION_HT_CAPACITY 2097152
#endif

// Sentinel values (must match gpu-hashtable.cl and CUDA kernels)
#define GPU_HT_EMPTY_KEY    0xFFFFFFFFFFFFFFFFULL
#define GPU_HT_EMPTY_VALUE  0xFFFFFFFFU
#define GPU_NOT_FOUND       0xFFFFFFFFU

#endif // _OPENCOG_GPU_POOL_DEFS_H
