/* gpu-connector-rewrite.cu — Grammar cost computation and class MI cache
 *
 * Replaces CPU grammar export computations with parallel GPU kernels:
 *   1. grammar_compute_costs: One thread per section → cost = -0.5 * log2(count/max) + 0.1
 *   2. grammar_build_class_mi: Aggregate word-pair MI into class-pair MI
 *   3. grammar_class_mi_normalize: Normalize class MI by pair count
 *
 * Data layout matches existing OpenCL SoA pools (gpu-atomspace.cl).
 *
 * Build: nvcc -O2 -arch=sm_75 -rdc=true -c gpu-connector-rewrite.cu
 */

#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cmath>

/* ─── Constants ─── */

#ifndef CLASS_MI_HT_CAPACITY
#define CLASS_MI_HT_CAPACITY  65536   /* power of 2, hash table for class pairs */
#endif

/* Hash table sentinels (matching OpenCL conventions) */
#define GR_HT_EMPTY_KEY    0xFFFFFFFFFFFFFFFFULL
#define GR_HT_EMPTY_VALUE  0xFFFFFFFFU

/* ─── Hash function (same splitmix64 as gpu-hashtable.cl) ─── */

__device__ __forceinline__
uint64_t gr_hash(uint64_t key) {
    key ^= key >> 30;
    key *= 0xBF58476D1CE4E5B9ULL;
    key ^= key >> 27;
    key *= 0x94D049BB133111EBULL;
    key ^= key >> 31;
    return key;
}

/* ─── Class pair key encoding ─── */

__device__ __forceinline__
uint64_t class_pair_key(uint32_t class_a, uint32_t class_b) {
    uint32_t lo = (class_a <= class_b) ? class_a : class_b;
    uint32_t hi = (class_a <= class_b) ? class_b : class_a;
    return ((uint64_t)lo << 32) | (uint64_t)hi;
}

/* ─── Atomic double add (CAS loop, same pattern as OpenCL version) ─── */

__device__ __forceinline__
void atomic_add_double(double* addr, double val) {
    unsigned long long int* addr_as_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
            __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
}

/* ═══════════════════════════════════════════════════════════════
 *  KERNEL 1: GRAMMAR COST COMPUTATION
 *
 *  One thread per section. Computes grammar dictionary cost from
 *  section counts:
 *    cost = -0.5 * log2(count / max_count) + 0.1
 *
 *  Higher count → lower cost (more common = preferred).
 *  The +0.1 base cost prevents zero-cost entries.
 *  Sections with count < 1 get cost = 99.0 (effectively pruned).
 * ═══════════════════════════════════════════════════════════════ */

__global__ void grammar_compute_costs(
    const double*   __restrict__ sec_count,   /* section counts */
    double*         sec_cost,                  /* output: grammar costs */
    uint32_t        num_sections,
    double          max_count                  /* max section count across all sections */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_sections) return;

    double count = sec_count[tid];
    if (count < 1.0) {
        sec_cost[tid] = 99.0;  /* pruned */
        return;
    }

    /* cost = -0.5 * log2(count / max_count) + 0.1 */
    double ratio = count / fmax(max_count, 1.0);
    double cost = -0.5 * log2(fmax(ratio, 1e-20)) + 0.1;

    /* Clamp to [0.1, 10.0] — reasonable grammar cost range */
    cost = fmax(0.1, fmin(cost, 10.0));
    sec_cost[tid] = cost;
}

/* ═══════════════════════════════════════════════════════════════
 *  FIND MAX SECTION COUNT (reduction)
 *
 *  Parallel reduction to find max section count for cost normalization.
 *  Uses warp shuffle for final reduction.
 * ═══════════════════════════════════════════════════════════════ */

__global__ void grammar_find_max_count(
    const double*   __restrict__ sec_count,
    uint32_t        num_sections,
    double*         max_count_out   /* single double, initialized to 0 */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    double my_max = 0.0;
    if (tid < num_sections) {
        my_max = sec_count[tid];
    }

    /* Warp-level max reduction */
    for (int offset = 16; offset > 0; offset >>= 1) {
        double other = __shfl_down_sync(0xFFFFFFFF, my_max, offset);
        my_max = fmax(my_max, other);
    }

    /* Lane 0 of each warp does atomicMax (via CAS since no native double atomicMax) */
    if ((threadIdx.x & 31) == 0 && my_max > 0.0) {
        unsigned long long int* addr = (unsigned long long int*)max_count_out;
        unsigned long long int old = *addr;
        unsigned long long int assumed;
        do {
            assumed = old;
            double old_val = __longlong_as_double(assumed);
            if (my_max <= old_val) break;
            old = atomicCAS(addr, assumed, __double_as_longlong(my_max));
        } while (assumed != old);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  KERNEL 2: BUILD CLASS MI CACHE
 *
 *  One thread per word pair. If both words are classified, accumulate
 *  MI into a class-pair hash table using atomic CAS.
 *
 *  class_mi_sum[class_pair] += mi
 *  class_mi_count[class_pair] += 1
 *
 *  Hash table layout (SoA):
 *    ht_keys[cap]    — class pair key ((lo << 32) | hi)
 *    ht_mi_sum[cap]  — accumulated MI sum
 *    ht_count[cap]   — number of word pairs contributing
 * ═══════════════════════════════════════════════════════════════ */

__global__ void grammar_build_class_mi(
    /* Word pair data */
    const uint32_t* __restrict__ pair_word_a,
    const uint32_t* __restrict__ pair_word_b,
    const double*   __restrict__ pair_mi,
    const double*   __restrict__ pair_count,
    uint32_t        num_pairs,
    /* Word → class mapping */
    const uint32_t* __restrict__ word_class_id,
    /* Class MI hash table (output) */
    volatile uint64_t*  ht_keys,
    double*             ht_mi_sum,
    uint32_t*           ht_count,
    uint32_t            ht_capacity
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_pairs) return;

    /* Skip pairs with no count */
    if (pair_count[tid] < 1.0) return;

    uint32_t wa = pair_word_a[tid];
    uint32_t wb = pair_word_b[tid];

    /* Both words must be classified */
    uint32_t ca = word_class_id[wa];
    uint32_t cb = word_class_id[wb];
    if (ca == 0xFFFFFFFFU || cb == 0xFFFFFFFFU) return;

    /* Skip self-class pairs (same class) — not useful for grammar */
    if (ca == cb) return;

    double mi = pair_mi[tid];
    if (mi <= 0.0) return;  /* only positive MI */

    uint64_t key = class_pair_key(ca, cb);
    uint64_t mask = (uint64_t)(ht_capacity - 1);
    uint64_t slot = gr_hash(key) & mask;

    /* Linear probe to find or create slot */
    for (int probe = 0; probe < 4096; probe++) {
        uint64_t existing = atomicCAS(
            (unsigned long long int*)&ht_keys[slot],
            GR_HT_EMPTY_KEY,
            key);

        if (existing == GR_HT_EMPTY_KEY || existing == key) {
            /* Slot is ours — accumulate */
            atomic_add_double(&ht_mi_sum[slot], mi);
            atomicAdd(&ht_count[slot], 1U);
            return;
        }

        slot = (slot + 1) & mask;
    }
    /* Hash table full — drop this pair (shouldn't happen with proper sizing) */
}

/* ═══════════════════════════════════════════════════════════════
 *  KERNEL 3: NORMALIZE CLASS MI
 *
 *  One thread per hash table slot. Divides accumulated MI sum
 *  by count to get average MI per class pair.
 * ═══════════════════════════════════════════════════════════════ */

__global__ void grammar_class_mi_normalize(
    const volatile uint64_t*  ht_keys,
    double*                   ht_mi_sum,    /* in/out: sum → average */
    const uint32_t*           ht_count,
    uint32_t                  ht_capacity
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ht_capacity) return;

    if (ht_keys[tid] == GR_HT_EMPTY_KEY) return;

    uint32_t cnt = ht_count[tid];
    if (cnt > 0) {
        ht_mi_sum[tid] /= (double)cnt;
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  CLASS MI LOOKUP (device function for other kernels)
 *
 *  Look up class-pair MI from the hash table. Returns 0.0 if not found.
 * ═══════════════════════════════════════════════════════════════ */

__device__
double class_mi_lookup(
    uint32_t class_a,
    uint32_t class_b,
    const volatile uint64_t*  ht_keys,
    const double*             ht_mi_sum,
    uint32_t                  ht_capacity
) {
    if (class_a == 0xFFFFFFFFU || class_b == 0xFFFFFFFFU) return 0.0;
    if (class_a == class_b) return 0.0;

    uint64_t key = class_pair_key(class_a, class_b);
    uint64_t mask = (uint64_t)(ht_capacity - 1);
    uint64_t slot = gr_hash(key) & mask;

    for (int probe = 0; probe < 4096; probe++) {
        uint64_t existing = ht_keys[slot];
        if (existing == key) return ht_mi_sum[slot];
        if (existing == GR_HT_EMPTY_KEY) return 0.0;
        slot = (slot + 1) & mask;
    }
    return 0.0;
}

/* ═══════════════════════════════════════════════════════════════
 *  HOST API
 * ═══════════════════════════════════════════════════════════════ */

/* Initialize class MI hash table (zero keys, sums, counts) */
extern "C"
void grammar_init_class_mi_ht(
    uint64_t* d_ht_keys,
    double*   d_ht_mi_sum,
    uint32_t* d_ht_count,
    uint32_t  ht_capacity
) {
    cudaMemset(d_ht_keys, 0xFF, ht_capacity * sizeof(uint64_t));  /* all GR_HT_EMPTY_KEY */
    cudaMemset(d_ht_mi_sum, 0, ht_capacity * sizeof(double));
    cudaMemset(d_ht_count, 0, ht_capacity * sizeof(uint32_t));
}

/* Run full grammar pipeline: find max → compute costs → build class MI → normalize */
extern "C"
void grammar_pipeline_run(
    /* Section data */
    const double*   d_sec_count,
    double*         d_sec_cost,
    uint32_t        num_sections,
    /* Word pair data */
    const uint32_t* d_pair_word_a,
    const uint32_t* d_pair_word_b,
    const double*   d_pair_mi,
    const double*   d_pair_count,
    uint32_t        num_pairs,
    /* Word → class */
    const uint32_t* d_word_class_id,
    /* Class MI hash table */
    uint64_t*       d_ht_keys,
    double*         d_ht_mi_sum,
    uint32_t*       d_ht_count,
    uint32_t        ht_capacity,
    /* Work buffer */
    double*         d_max_count  /* single double */
) {
    int threads = 256;

    /* Step 1: Find max section count */
    cudaMemset(d_max_count, 0, sizeof(double));
    int blocks_sec = (num_sections + threads - 1) / threads;
    if (blocks_sec > 0) {
        grammar_find_max_count<<<blocks_sec, threads>>>(
            d_sec_count, num_sections, d_max_count);
    }
    cudaDeviceSynchronize();

    double h_max_count;
    cudaMemcpy(&h_max_count, d_max_count, sizeof(double), cudaMemcpyDeviceToHost);

    /* Step 2: Compute costs */
    if (blocks_sec > 0) {
        grammar_compute_costs<<<blocks_sec, threads>>>(
            d_sec_count, d_sec_cost, num_sections, h_max_count);
    }
    cudaDeviceSynchronize();

    /* Step 3: Build class MI cache */
    grammar_init_class_mi_ht(d_ht_keys, d_ht_mi_sum, d_ht_count, ht_capacity);
    cudaDeviceSynchronize();

    int blocks_pairs = (num_pairs + threads - 1) / threads;
    if (blocks_pairs > 0) {
        grammar_build_class_mi<<<blocks_pairs, threads>>>(
            d_pair_word_a, d_pair_word_b, d_pair_mi, d_pair_count,
            num_pairs, d_word_class_id,
            (volatile uint64_t*)d_ht_keys, d_ht_mi_sum, d_ht_count,
            ht_capacity);
    }
    cudaDeviceSynchronize();

    /* Step 4: Normalize */
    int blocks_ht = (ht_capacity + threads - 1) / threads;
    grammar_class_mi_normalize<<<blocks_ht, threads>>>(
        (const volatile uint64_t*)d_ht_keys, d_ht_mi_sum, d_ht_count, ht_capacity);
    cudaDeviceSynchronize();
}

/* Read class MI for a specific class pair (host convenience) */
extern "C"
double grammar_read_class_mi(
    uint32_t  class_a,
    uint32_t  class_b,
    uint64_t* d_ht_keys,
    double*   d_ht_mi_sum,
    uint32_t  ht_capacity
) {
    uint64_t key = ((uint64_t)(class_a <= class_b ? class_a : class_b) << 32)
                 | (uint64_t)(class_a <= class_b ? class_b : class_a);

    uint64_t mask = (uint64_t)(ht_capacity - 1);

    /* Read hash table on host — copy relevant slots */
    uint64_t slot = 0;
    {
        /* Compute hash on host (same splitmix64) */
        uint64_t h = key;
        h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 27; h *= 0x94D049BB133111EBULL;
        h ^= h >> 31;
        slot = h & mask;
    }

    /* Linear probe on device memory */
    for (int probe = 0; probe < 4096; probe++) {
        uint64_t existing_key;
        cudaMemcpy(&existing_key, &d_ht_keys[slot], sizeof(uint64_t), cudaMemcpyDeviceToHost);

        if (existing_key == key) {
            double mi;
            cudaMemcpy(&mi, &d_ht_mi_sum[slot], sizeof(double), cudaMemcpyDeviceToHost);
            return mi;
        }
        if (existing_key == GR_HT_EMPTY_KEY) return 0.0;
        slot = (slot + 1) & mask;
    }
    return 0.0;
}

/* Count occupied slots in class MI hash table */
extern "C"
uint32_t grammar_class_mi_count(
    uint64_t* d_ht_keys,
    uint32_t  ht_capacity
) {
    /* Copy keys to host and count non-empty */
    uint64_t* h_keys = (uint64_t*)malloc(ht_capacity * sizeof(uint64_t));
    cudaMemcpy(h_keys, d_ht_keys, ht_capacity * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    uint32_t count = 0;
    for (uint32_t i = 0; i < ht_capacity; i++) {
        if (h_keys[i] != GR_HT_EMPTY_KEY) count++;
    }
    free(h_keys);
    return count;
}
