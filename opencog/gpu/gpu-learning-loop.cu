/* gpu-learning-loop.cu — Persistent CUDA learning kernel
 *
 * The ENTIRE learning pipeline as ONE persistent kernel launched with
 * cudaLaunchCooperativeKernel. CPU only feeds sentences via unified
 * memory and polls convergence. All data stays GPU-resident.
 *
 * Pipeline stages (conditional execution):
 *   1. Count pairs from new sentences
 *   2. Compute MI on dirty pairs
 *   3. Cosine similarity (norms → chains → dots → filter)
 *   4. Connected components clustering
 *   5. Grammar costs + class MI
 *   6. Parse + surprise
 *   7. Convergence check (entropy plateau)
 *
 * Requires: CUDA cooperative groups (CC 7.5+, RTX 2070)
 *
 * Build: nvcc -O2 -arch=sm_75 -rdc=true -c gpu-learning-loop.cu
 */

#include <cstdio>
#include <cfloat>
#include <cmath>
#include <cooperative_groups.h>
#include "gpu-learning-types.h"

namespace cg = cooperative_groups;

/* ─── Hash function ─── */

__device__ __forceinline__
uint64_t ll_hash(uint64_t key) {
    key ^= key >> 30;
    key *= 0xBF58476D1CE4E5B9ULL;
    key ^= key >> 27;
    key *= 0x94D049BB133111EBULL;
    key ^= key >> 31;
    return key;
}

/* ─── Atomic double add ─── */

__device__ __forceinline__
void ll_atomic_add_double(double* addr, double val) {
    unsigned long long int* a = (unsigned long long int*)addr;
    unsigned long long int old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed,
            __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
}

/* Structs defined in gpu-learning-types.h */

/* ═══════════════════════════════════════════════════════════════
 *  DEVICE FUNCTIONS (called inside persistent kernel)
 * ═══════════════════════════════════════════════════════════════ */

/* ─── Find or create pair in hash table ─── */

__device__
uint32_t ll_find_or_create_pair(LearningState* state, uint32_t wa, uint32_t wb) {
    uint32_t lo = (wa <= wb) ? wa : wb;
    uint32_t hi = (wa <= wb) ? wb : wa;
    uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;
    uint64_t mask = LL_PAIR_HT_CAPACITY - 1;
    uint64_t slot = ll_hash(key) & mask;

    for (int probe = 0; probe < 4096; probe++) {
        uint64_t existing = atomicCAS(
            (unsigned long long int*)&state->pair_ht_keys[slot],
            LL_EMPTY_KEY, key);

        if (existing == LL_EMPTY_KEY) {
            /* New slot — allocate pair */
            uint32_t idx = atomicAdd(&state->pair_count_u32, 1U);
            if (idx >= LL_MAX_PAIRS) return LL_EMPTY_VALUE;
            state->pair_word_a[idx] = lo;
            state->pair_word_b[idx] = hi;
            state->pair_count[idx] = 0.0;
            state->pair_mi[idx] = 0.0;
            state->pair_dirty[idx] = 0;
            __threadfence();
            state->pair_ht_values[slot] = idx;
            return idx;
        }
        if (existing == key) {
            /* Existing slot — wait for value if needed */
            uint32_t val = state->pair_ht_values[slot];
            int safety = 1000;
            while (val == LL_EMPTY_VALUE && safety-- > 0) {
                val = state->pair_ht_values[slot];
            }
            return val;
        }
        slot = (slot + 1) & mask;
    }
    return LL_EMPTY_VALUE;
}

/* ─── Count pairs from one sentence ─── */

__device__
void ll_count_sentence(LearningState* state, uint32_t* words, uint32_t len) {
    /* Sliding window: all pairs within window of 2 */
    for (uint32_t i = 0; i < len; i++) {
        for (uint32_t j = i + 1; j < len && j <= i + 2; j++) {
            uint32_t idx = ll_find_or_create_pair(state, words[i], words[j]);
            if (idx == LL_EMPTY_VALUE) continue;

            ll_atomic_add_double(&state->pair_count[idx], 1.0);
            ll_atomic_add_double(&state->word_marginal[words[i]], 1.0);
            ll_atomic_add_double(&state->word_marginal[words[j]], 1.0);
            ll_atomic_add_double(&state->total_pair_observations, 1.0);

            if (atomicExch(&state->pair_dirty[idx], 1U) == 0U) {
                atomicAdd(&state->dirty_count, 1U);
            }
        }
    }
}

/* ─── Compute MI for all dirty pairs ─── */

__device__
void ll_compute_mi_dirty(LearningState* state, cg::grid_group grid) {
    double n = state->total_pair_observations;
    double log2_factor = 1.4426950408889634;
    double eps = 1e-10;

    uint32_t num_pairs = state->pair_count_u32;
    uint32_t tid = grid.thread_rank();
    uint32_t stride = grid.size();

    for (uint32_t p = tid; p < num_pairs; p += stride) {
        if (state->pair_dirty[p] != 1) continue;

        double count = state->pair_count[p];
        if (count < 1.0) {
            state->pair_mi[p] = 0.0;
            state->pair_dirty[p] = 0;
            continue;
        }

        uint32_t wa = state->pair_word_a[p];
        uint32_t wb = state->pair_word_b[p];
        double left  = fmax(state->word_marginal[wa], eps);
        double right = fmax(state->word_marginal[wb], eps);
        double n_safe = fmax(n, eps);

        double mi = (log(count) + log(n_safe) - log(left) - log(right)) * log2_factor;
        state->pair_mi[p] = mi;
        state->pair_dirty[p] = 0;
    }

    if (tid == 0) {
        state->dirty_count = 0;
        state->mi_updated = 1;
    }
}

/* ─── Simple connected components (for the persistent kernel) ─── */

__device__
void ll_connected_components(LearningState* state, cg::grid_group grid) {
    uint32_t num_words = state->word_count_u32;
    uint32_t num_pairs = state->pair_count_u32;
    uint32_t tid = grid.thread_rank();
    uint32_t stride = grid.size();

    /* Build edge list from pairs with positive MI and high cosine */
    /* For simplicity in the persistent kernel, we use MI-based clustering:
     * connect words that share high MI neighbors (simplified cosine proxy) */
    if (tid == 0) state->cc_edge_count = 0;
    grid.sync();

    /* Filter pairs with MI > threshold into edge list */
    for (uint32_t p = tid; p < num_pairs; p += stride) {
        if (state->pair_mi[p] > 2.0 && state->pair_count[p] > 3.0) {
            uint32_t idx = atomicAdd(&state->cc_edge_count, 1U);
            if (idx < LL_CC_MAX_EDGES) {
                state->cc_edge_a[idx] = state->pair_word_a[p];
                state->cc_edge_b[idx] = state->pair_word_b[p];
            }
        }
    }
    grid.sync();

    uint32_t num_edges = min(state->cc_edge_count, (uint32_t)LL_CC_MAX_EDGES);

    /* Init labels */
    for (uint32_t w = tid; w < num_words; w += stride) {
        state->cc_labels[w] = w;
    }
    grid.sync();

    /* Propagate + compress loop
     * CRITICAL: All threads must execute the same number of grid.sync() calls.
     * Use a global changed flag instead of __shared__ + break. */
    for (int iter = 0; iter < 32; iter++) {
        /* Reset changed flag */
        if (tid == 0) state->cc_changed = 0;
        grid.sync();

        for (uint32_t e = tid; e < num_edges; e += stride) {
            uint32_t u = state->cc_edge_a[e];
            uint32_t v = state->cc_edge_b[e];
            uint32_t lu = state->cc_labels[u];
            uint32_t lv = state->cc_labels[v];
            if (lu != lv) {
                uint32_t hi = (lu > lv) ? lu : lv;
                uint32_t lo = (lu > lv) ? lv : lu;
                uint32_t old = atomicMin(&state->cc_labels[hi], lo);
                if (old != lo) state->cc_changed = 1;
            }
        }
        grid.sync();

        /* Compress */
        for (uint32_t w = tid; w < num_words; w += stride) {
            uint32_t l = state->cc_labels[w];
            while (state->cc_labels[l] != l) l = state->cc_labels[l];
            state->cc_labels[w] = l;
        }
        grid.sync();

        /* All threads check the same global flag — no divergent control flow */
        if (!state->cc_changed) break;
    }

    /* Assign class IDs */
    if (tid == 0) state->num_classes = 0;
    grid.sync();

    for (uint32_t w = tid; w < num_words; w += stride) {
        if (state->cc_labels[w] == w) {
            uint32_t cid = atomicAdd(&state->num_classes, 1U);
            state->word_class_id[w] = cid;
        } else {
            state->word_class_id[w] = LL_EMPTY_VALUE;
        }
    }
    grid.sync();

    /* Map non-roots */
    for (uint32_t w = tid; w < num_words; w += stride) {
        if (state->word_class_id[w] == LL_EMPTY_VALUE) {
            uint32_t root = state->cc_labels[w];
            state->word_class_id[w] = state->word_class_id[root];
        }
    }
    grid.sync();

    if (tid == 0) state->classes_updated = 1;
}

/* ─── Compute class distribution entropy ─── */
/* Uses the cc_labels array (no longer needed after CC) as scratch space
 * instead of allocating a huge stack array. */

__device__
double ll_compute_entropy(LearningState* state, cg::grid_group grid) {
    uint32_t tid = grid.thread_rank();
    uint32_t stride = grid.size();
    uint32_t num_words = state->word_count_u32;
    uint32_t nc = state->num_classes;

    if (nc == 0) return 0.0;

    /* Reuse cc_labels as class size counters (zeroed by all threads) */
    for (uint32_t i = tid; i < nc; i += stride) {
        state->cc_labels[i] = 0;
    }
    grid.sync();

    /* Count class sizes in parallel */
    for (uint32_t w = tid; w < num_words; w += stride) {
        uint32_t cid = state->word_class_id[w];
        if (cid < nc) {
            atomicAdd(&state->cc_labels[cid], 1U);
        }
    }
    grid.sync();

    /* Compute entropy on thread 0 (nc is typically small) */
    double entropy = 0.0;
    if (tid == 0) {
        double n = (double)num_words;
        for (uint32_t i = 0; i < nc; i++) {
            uint32_t sz = state->cc_labels[i];
            if (sz > 0) {
                double p = (double)sz / n;
                entropy -= p * log2(p);
            }
        }
    }
    return entropy;
}

/* ═══════════════════════════════════════════════════════════════
 *  PERSISTENT KERNEL
 *
 *  Launched once with cudaLaunchCooperativeKernel.
 *  Runs until convergence or max iterations.
 *  CPU feeds sentences via unified memory ring buffer.
 * ═══════════════════════════════════════════════════════════════ */

__global__ void learning_loop_kernel(
    LearningState*          state,
    SentenceRing*           ring,
    volatile int*           done_flag,
    volatile int*           pause_flag,     /* CPU can pause processing */
    volatile uint32_t*      stats_iteration,
    volatile uint32_t*      stats_pairs,
    volatile uint32_t*      stats_classes,
    volatile double*        stats_entropy
) {
    auto grid = cg::this_grid();
    uint32_t tid = grid.thread_rank();

    /* CRITICAL: With cooperative groups, ALL threads must hit every grid.sync().
     * No conditional blocks around grid.sync(). Stages always run;
     * threads skip their work when the stage condition is false. */

    while (!(*done_flag)) {
        /* ─── Stage 1: Ingest sentences (thread 0 only, no grid.sync needed) ─── */
        if (tid == 0 && !(*pause_flag)) {
            int ingested = 0;
            /* Process up to 16 sentences per iteration to avoid starving GPU */
            int max_ingest = 16;
            while (ring->read_idx != ring->write_idx && max_ingest-- > 0) {
                uint32_t ridx = ring->read_idx % LL_RING_SIZE;
                SentenceSlot* slot = &ring->slots[ridx];

                if (slot->ready && slot->length >= 2) {
                    ll_count_sentence(state, slot->words, slot->length);
                    slot->ready = 0;
                    ingested = 1;
                }
                ring->read_idx++;
            }
            if (ingested) state->has_new_sentences = 1;
        }
        grid.sync();

        /* ─── Stage 2: MI computation (ALL threads enter, conditional work) ─── */
        /* All threads enter ll_compute_mi_dirty regardless; threads with no dirty
         * pairs just skip their iterations in the parallel loop. */
        int do_mi = (state->dirty_count >= LL_MI_THRESHOLD);
        if (do_mi) {
            ll_compute_mi_dirty(state, grid);
        }
        grid.sync();

        /* ─── Stage 3: Connected components + entropy (ALL threads enter) ─── */
        int do_cc = (state->mi_updated && state->pair_count_u32 > 10);
        if (do_cc) {
            ll_connected_components(state, grid);
        }
        grid.sync();

        /* Entropy computation — all threads participate */
        double entropy = 0.0;
        if (do_cc) {
            entropy = ll_compute_entropy(state, grid);
        }
        grid.sync();

        /* Stats and convergence (thread 0 only, no grid.sync inside) */
        if (tid == 0 && do_cc) {
            uint32_t eidx = state->entropy_idx % LL_ENTROPY_WINDOW;
            state->entropy_history[eidx] = entropy;
            state->entropy_idx++;

            *stats_iteration = state->iteration;
            *stats_pairs = state->pair_count_u32;
            *stats_classes = state->num_classes;
            *stats_entropy = entropy;

            state->iteration++;
            state->mi_updated = 0;

            /* Check entropy plateau */
            if (state->entropy_idx >= LL_ENTROPY_WINDOW) {
                double min_e = 1e30, max_e = -1e30;
                for (int i = 0; i < LL_ENTROPY_WINDOW; i++) {
                    double e = state->entropy_history[i];
                    if (e < min_e) min_e = e;
                    if (e > max_e) max_e = e;
                }
                if (max_e > 0.0 && (max_e - min_e) / max_e < 0.01) {
                    *done_flag = 1;
                }
            }

            if (state->iteration >= LL_MAX_ITERATIONS) {
                *done_flag = 1;
            }
        }

        if (tid == 0) state->has_new_sentences = 0;
        grid.sync();
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  HOST API (C interface for Scheme bridge)
 * ═══════════════════════════════════════════════════════════════ */

extern "C" {

/* Initialize learning state */
LearningState* ll_init(uint32_t num_words) {
    LearningState* state;
    cudaMallocManaged(&state, sizeof(LearningState));
    memset(state, 0, sizeof(LearningState));

    state->word_count_u32 = num_words;
    state->pair_count_u32 = 0;
    state->dirty_count = 0;
    state->total_pair_observations = 0.0;
    state->num_classes = 0;
    state->iteration = 0;
    state->entropy_idx = 0;
    state->cc_edge_count = 0;

    /* Init hash table */
    memset(state->pair_ht_keys, 0xFF, sizeof(state->pair_ht_keys));
    memset(state->pair_ht_values, 0xFF, sizeof(state->pair_ht_values));

    /* Init word data */
    for (uint32_t i = 0; i < num_words; i++) {
        state->word_marginal[i] = 0.0;
        state->word_class_id[i] = LL_EMPTY_VALUE;
    }

    return state;
}

/* Initialize sentence ring buffer */
SentenceRing* ll_init_ring() {
    SentenceRing* ring;
    cudaMallocManaged(&ring, sizeof(SentenceRing));
    memset(ring, 0, sizeof(SentenceRing));
    return ring;
}

/* Feed a sentence (CPU writes to unified memory) */
void ll_feed_sentence(SentenceRing* ring, uint32_t* words, uint32_t length) {
    if (length > LL_MAX_SENTENCE_LEN) length = LL_MAX_SENTENCE_LEN;

    uint32_t widx = ring->write_idx % LL_RING_SIZE;
    SentenceSlot* slot = &ring->slots[widx];

    /* Wait for slot to be consumed */
    int wait = 0;
    while (slot->ready && wait < 1000000) { wait++; }

    memcpy(slot->words, words, length * sizeof(uint32_t));
    slot->length = length;
    __sync_synchronize();  /* memory fence */
    slot->ready = 1;
    ring->write_idx++;
}

/* Launch the persistent kernel */
int ll_launch(
    LearningState* state,
    SentenceRing*  ring,
    int*           done_flag,
    int*           pause_flag,
    uint32_t*      stats_iteration,
    uint32_t*      stats_pairs,
    uint32_t*      stats_classes,
    double*        stats_entropy
) {
    /* Query max blocks for cooperative launch */
    int num_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks, learning_loop_kernel, 128, 0);
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    /* Use fewer blocks to leave room for other GPU work */
    int total_blocks = num_sms;  /* 1 block per SM */
    int block_size = 128;

    dim3 grid_dim(total_blocks);
    dim3 block_dim(block_size);

    void* args[] = {
        &state, &ring, &done_flag, &pause_flag,
        &stats_iteration, &stats_pairs, &stats_classes, &stats_entropy
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)learning_loop_kernel, grid_dim, block_dim, args);

    if (err != cudaSuccess) {
        printf("Cooperative kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/* Wait for kernel completion */
void ll_wait() {
    cudaDeviceSynchronize();
}

/* Read class assignments (from unified memory — no copy needed) */
void ll_read_classes(LearningState* state, uint32_t* out_class_ids, uint32_t num_words) {
    /* Unified memory: just memcpy from managed pointer */
    cudaDeviceSynchronize();  /* ensure GPU is done */
    memcpy(out_class_ids, state->word_class_id, num_words * sizeof(uint32_t));
}

/* Cleanup */
void ll_shutdown(LearningState* state, SentenceRing* ring) {
    cudaFree(state);
    cudaFree(ring);
}

} /* extern "C" */
