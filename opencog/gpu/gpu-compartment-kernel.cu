/* gpu-compartment-kernel.cu — Hardware-native compartmentalized learning
 *
 * Maps 1 SM = 1 compartment (AtomSpace). Each SM accumulates pair counts
 * independently with ZERO cross-SM atomics. After accumulation, a grid-wide
 * merge phase combines compartment stats into the shared LearningState,
 * then MI/CC/entropy run on merged data.
 *
 * Architecture:
 *   Phase A: Snapshot ring buffer (thread 0)
 *   Phase B: Per-SM sentence counting (each block processes its stride)
 *   Phase C: Merge compartments → shared LearningState (atomics)
 *   Phase D: MI computation on merged data (all threads)
 *   Phase E: Connected components clustering (all threads)
 *   Phase F: Entropy + convergence check (all threads + thread 0)
 *   Phase G: Reset compartments for next round (all threads)
 *
 * 7 grid.sync() per iteration — all unconditional.
 *
 * Requires: CUDA cooperative groups (CC 7.5+, RTX 2070)
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true -c gpu-compartment-kernel.cu
 */

#include <cstdio>
#include <cstring>
#include <cfloat>
#include <cmath>
#include <cooperative_groups.h>
#include "gpu-learning-types.h"

namespace cg = cooperative_groups;

/* ─── Hash function (same splitmix64 as gpu-learning-loop.cu) ─── */

__device__ __forceinline__
uint64_t ck_hash(uint64_t key) {
    key ^= key >> 30;
    key *= 0xBF58476D1CE4E5B9ULL;
    key ^= key >> 27;
    key *= 0x94D049BB133111EBULL;
    key ^= key >> 31;
    return key;
}

/* ─── Atomic double add ─── */

__device__ __forceinline__
void ck_atomic_add_double(double* addr, double val) {
    unsigned long long int* a = (unsigned long long int*)addr;
    unsigned long long int old = *a, assumed;
    do {
        assumed = old;
        old = atomicCAS(a, assumed,
            __double_as_longlong(__longlong_as_double(assumed) + val));
    } while (assumed != old);
}

/* ═══════════════════════════════════════════════════════════════
 *  COMPARTMENT DEVICE FUNCTIONS
 *
 *  These operate on a SINGLE CompartmentState — no atomics needed
 *  because only threads within one block write to their compartment.
 * ═══════════════════════════════════════════════════════════════ */

/* Find or create pair in compartment's local hash table.
 * SINGLE-WRITER: only thread 0 of each block calls this,
 * so no atomicCAS needed — plain reads/writes. */

__device__
uint32_t comp_find_or_create_pair(CompartmentState* comp,
                                  uint32_t wa, uint32_t wb) {
    uint32_t lo = (wa <= wb) ? wa : wb;
    uint32_t hi = (wa <= wb) ? wb : wa;
    uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;
    uint64_t mask = LL_COMP_HT_CAPACITY - 1;
    uint64_t slot = ck_hash(key) & mask;

    for (int probe = 0; probe < 4096; probe++) {
        uint64_t existing = comp->pair_ht_keys[slot];

        if (existing == LL_EMPTY_KEY) {
            /* New slot */
            uint32_t idx = comp->pair_count;
            if (idx >= LL_COMP_MAX_PAIRS) return LL_EMPTY_VALUE;
            comp->pair_count = idx + 1;
            comp->pair_ht_keys[slot] = key;
            comp->pair_ht_values[slot] = idx;
            comp->pair_word_a[idx] = lo;
            comp->pair_word_b[idx] = hi;
            comp->pair_obs[idx] = 0.0;
            return idx;
        }
        if (existing == key) {
            return comp->pair_ht_values[slot];
        }
        slot = (slot + 1) & mask;
    }
    return LL_EMPTY_VALUE;
}

/* Count pairs from one sentence into compartment (single-writer) */

__device__
void comp_count_sentence(CompartmentState* comp,
                         uint32_t* words, uint32_t len) {
    for (uint32_t i = 0; i < len; i++) {
        for (uint32_t j = i + 1; j < len && j <= i + 2; j++) {
            uint32_t idx = comp_find_or_create_pair(comp, words[i], words[j]);
            if (idx == LL_EMPTY_VALUE) continue;

            comp->pair_obs[idx] += 1.0;
            comp->word_marginal[words[i]] += 1.0;
            comp->word_marginal[words[j]] += 1.0;
            comp->total_observations += 1.0;
        }
    }
    comp->sentences_processed++;
}

/* Merge one compartment's accumulated stats into shared LearningState.
 * ALL threads in the block participate — each thread handles a stride
 * of the compartment's pairs. Uses atomics on the shared state. */

__device__
void comp_merge_into_shared(CompartmentState* comp,
                            LearningState* shared,
                            uint32_t local_tid,
                            uint32_t block_size) {
    uint32_t num_local_pairs = comp->pair_count;

    /* Each thread in the block merges a stride of pairs */
    for (uint32_t p = local_tid; p < num_local_pairs; p += block_size) {
        uint32_t wa = comp->pair_word_a[p];
        uint32_t wb = comp->pair_word_b[p];
        double obs = comp->pair_obs[p];
        if (obs < 1.0) continue;

        /* Find or create in shared hash table (uses atomicCAS) */
        uint64_t key = ((uint64_t)wa << 32) | (uint64_t)wb;
        uint64_t mask = LL_PAIR_HT_CAPACITY - 1;
        uint64_t slot = ck_hash(key) & mask;

        uint32_t shared_idx = LL_EMPTY_VALUE;
        for (int probe = 0; probe < 4096; probe++) {
            uint64_t existing = atomicCAS(
                (unsigned long long int*)&shared->pair_ht_keys[slot],
                LL_EMPTY_KEY, key);

            if (existing == LL_EMPTY_KEY) {
                /* New pair in shared state */
                uint32_t idx = atomicAdd(&shared->pair_count_u32, 1U);
                if (idx < LL_MAX_PAIRS) {
                    shared->pair_word_a[idx] = wa;
                    shared->pair_word_b[idx] = wb;
                    shared->pair_count[idx] = 0.0;
                    shared->pair_mi[idx] = 0.0;
                    shared->pair_dirty[idx] = 0;
                    __threadfence();
                    shared->pair_ht_values[slot] = idx;
                    shared_idx = idx;
                }
                break;
            }
            if (existing == key) {
                /* Existing pair — wait for value */
                uint32_t val = shared->pair_ht_values[slot];
                int safety = 1000;
                while (val == LL_EMPTY_VALUE && safety-- > 0) {
                    val = shared->pair_ht_values[slot];
                }
                shared_idx = val;
                break;
            }
            slot = (slot + 1) & mask;
        }

        if (shared_idx != LL_EMPTY_VALUE && shared_idx < LL_MAX_PAIRS) {
            ck_atomic_add_double(&shared->pair_count[shared_idx], obs);

            /* Mark dirty for MI recomputation */
            if (atomicExch(&shared->pair_dirty[shared_idx], 1U) == 0U) {
                atomicAdd(&shared->dirty_count, 1U);
            }
        }
    }

    /* Merge marginals — each thread handles a stride of words */
    uint32_t num_words = shared->word_count_u32;
    for (uint32_t w = local_tid; w < num_words; w += block_size) {
        double m = comp->word_marginal[w];
        if (m > 0.0) {
            ck_atomic_add_double(&shared->word_marginal[w], m);
        }
    }

    /* Thread 0 of each block merges total observations */
    if (local_tid == 0) {
        ck_atomic_add_double(&shared->total_pair_observations,
                             comp->total_observations);
        atomicAdd(&shared->total_sentences, comp->sentences_processed);
    }
}

/* Clear a compartment for next round (all threads in block participate) */

__device__
void comp_clear(CompartmentState* comp,
                uint32_t num_words,
                uint32_t local_tid,
                uint32_t block_size) {
    /* Clear HT keys */
    for (uint32_t i = local_tid; i < LL_COMP_HT_CAPACITY; i += block_size) {
        comp->pair_ht_keys[i] = LL_EMPTY_KEY;
        comp->pair_ht_values[i] = LL_EMPTY_VALUE;
    }
    /* Clear marginals */
    for (uint32_t w = local_tid; w < num_words; w += block_size) {
        comp->word_marginal[w] = 0.0;
    }
    /* Thread 0 resets scalars */
    if (local_tid == 0) {
        comp->pair_count = 0;
        comp->total_observations = 0.0;
        comp->sentences_processed = 0;
        comp->surprise_score = 0.0;
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  SHARED DEVICE FUNCTIONS (reused from gpu-learning-loop.cu pattern)
 * ═══════════════════════════════════════════════════════════════ */

/* MI computation — identical to ll_compute_mi_dirty */

__device__
void ck_compute_mi(LearningState* state, cg::grid_group grid) {
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

/* Connected components — identical to ll_connected_components */

__device__
void ck_connected_components(LearningState* state, cg::grid_group grid) {
    uint32_t num_words = state->word_count_u32;
    uint32_t num_pairs = state->pair_count_u32;
    uint32_t tid = grid.thread_rank();
    uint32_t stride = grid.size();

    if (tid == 0) state->cc_edge_count = 0;
    grid.sync();

    /* Filter high-MI pairs into edge list */
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

    /* Propagate + compress (fixed iteration count for grid.sync safety) */
    for (int iter = 0; iter < 32; iter++) {
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

    for (uint32_t w = tid; w < num_words; w += stride) {
        if (state->word_class_id[w] == LL_EMPTY_VALUE) {
            uint32_t root = state->cc_labels[w];
            state->word_class_id[w] = state->word_class_id[root];
        }
    }
    grid.sync();

    if (tid == 0) {
        state->classes_updated = 1;
        state->cc_runs++;
    }
}

/* Entropy computation — serial on thread 0.
 * cc_labels is shared with CC (root labels). Parallel atomicAdd on
 * cc_labels has a stale-write race across iterations. Thread-0 serial
 * is correct and fast enough (O(num_words) ≈ 8K, <0.1 ms). */

__device__
double ck_compute_entropy(LearningState* state, cg::grid_group grid) {
    uint32_t tid = grid.thread_rank();
    uint32_t num_words = state->word_count_u32;
    uint32_t nc = state->num_classes;

    if (nc == 0) return 0.0;
    double entropy = 0.0;
    if (tid == 0) {
        double n = (double)num_words;

        /* Clear histogram bins */
        for (uint32_t i = 0; i < nc; i++)
            state->cc_labels[i] = 0;

        /* Count class sizes */
        for (uint32_t w = 0; w < num_words; w++) {
            uint32_t cid = state->word_class_id[w];
            if (cid < nc)
                state->cc_labels[cid]++;
        }

        /* Compute Shannon entropy */
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
 *  PERSISTENT COMPARTMENT KERNEL
 *
 *  Launched once with cudaLaunchCooperativeKernel.
 *  gridDim.x = number of SMs = number of compartments.
 *  Each block IS a compartment.
 *
 *  7 grid.sync() per iteration (all unconditional).
 * ═══════════════════════════════════════════════════════════════ */

__global__ void compartment_kernel(
    LearningState*          shared,
    CompartmentState*       compartments,  /* array[gridDim.x] */
    SentenceRing*           ring,
    volatile int*           done_flag,
    volatile int*           pause_flag,
    volatile uint32_t*      stats_iteration,
    volatile uint32_t*      stats_pairs,
    volatile uint32_t*      stats_classes,
    volatile double*        stats_entropy
) {
    auto grid = cg::this_grid();
    uint32_t tid = grid.thread_rank();
    uint32_t bid = blockIdx.x;
    uint32_t local_tid = threadIdx.x;
    uint32_t block_size = blockDim.x;
    uint32_t num_blocks = gridDim.x;

    CompartmentState* my_comp = &compartments[bid];

    while (!(*done_flag)) {

        /* ═══ Phase A: Snapshot ring buffer (thread 0 only) ═══
         * Thread 0 captures how many sentences are available.
         * Cap batch to num_blocks * 16 per iteration so the pipeline
         * runs multiple rounds and the entropy window fills up.
         * This prevents the "consume-all-then-spin" problem. */
        if (tid == 0 && !(*pause_flag)) {
            uint32_t w = ring->write_idx;
            uint32_t r = ring->read_idx;
            uint32_t avail = (w >= r) ? (w - r) : 0;
            /* Cap: each SM gets ~16 sentences per round */
            uint32_t max_batch = num_blocks * 16;
            if (avail > max_batch) avail = max_batch;
            shared->batch_start = r;
            shared->batch_size = avail;
            if (avail > 0) {
                ring->read_idx = r + avail;  /* consume only the batch */
                shared->has_new_sentences = 1;
            }
        }
        grid.sync(); /* ── sync 1 ── */

        /* ═══ Phase B: Per-SM sentence counting ═══
         * Block `bid` processes sentences [bid, bid+num_blocks, bid+2*num_blocks, ...]
         * from the snapshot. Only thread 0 of each block writes to its compartment
         * (single-writer, no atomics needed). */
        if (shared->has_new_sentences) {
            uint32_t batch_start = shared->batch_start;
            uint32_t batch_size  = shared->batch_size;

            /* Thread 0 of each block counts sentences for its compartment */
            if (local_tid == 0) {
                for (uint32_t s = bid; s < batch_size; s += num_blocks) {
                    uint32_t ridx = (batch_start + s) % LL_RING_SIZE;
                    SentenceSlot* slot = &ring->slots[ridx];

                    if (slot->ready && slot->length >= 2) {
                        comp_count_sentence(my_comp, slot->words, slot->length);
                        slot->ready = 0;
                    }
                }
            }
        }
        grid.sync(); /* ── sync 2 ── */

        /* ═══ Phase C: Merge compartments → shared state ═══
         * All threads in each block participate in merging their
         * compartment's accumulated stats into the shared LearningState.
         * This phase uses atomics on shared state. */
        if (shared->has_new_sentences && my_comp->pair_count > 0) {
            comp_merge_into_shared(my_comp, shared, local_tid, block_size);
        }
        grid.sync(); /* ── sync 3 ── */

        /* ═══ Phase D: MI computation on merged data ═══
         * All threads participate, operating on shared LearningState.
         * Flush remaining dirty pairs even below threshold when idle. */
        int do_mi = (shared->dirty_count >= LL_MI_THRESHOLD) ||
                    (shared->dirty_count > 0 && !shared->has_new_sentences);
        if (do_mi) {
            ck_compute_mi(shared, grid);
        }
        grid.sync(); /* ── sync 4 ── */

        /* ═══ Phase E: Connected components clustering ═══
         * Run CC when MI freshly updated, OR on idle rounds to fill
         * the entropy window (up to ENTROPY_WINDOW+2 extra rounds).
         * CC is idempotent — same data gives same classes. */
        int had_new_data = shared->has_new_sentences;
        int need_fill = (!had_new_data &&
                         shared->total_sentences > 0 &&
                         shared->pair_count_u32 > 10 &&
                         shared->entropy_idx < (uint32_t)(LL_ENTROPY_WINDOW + 2));
        int do_cc = (shared->mi_updated && shared->pair_count_u32 > 10)
                    || need_fill;
        if (do_cc) {
            ck_connected_components(shared, grid);
        }
        grid.sync(); /* ── sync 5 ── */

        /* ═══ Phase F: Entropy + convergence check ═══ */
        double entropy = 0.0;
        if (do_cc) {
            entropy = ck_compute_entropy(shared, grid);
        }
        grid.sync(); /* ── sync 6 ── */

        /* Thread 0: update stats, check phase-gated convergence */
        if (tid == 0) {
            if (do_cc) {
                uint32_t eidx = shared->entropy_idx % LL_ENTROPY_WINDOW;
                shared->entropy_history[eidx] = entropy;
                shared->entropy_idx++;

                *stats_pairs = shared->pair_count_u32;
                *stats_classes = shared->num_classes;
                *stats_entropy = entropy;

                /* Only count data rounds as iterations */
                if (had_new_data) {
                    shared->iteration++;
                    *stats_iteration = shared->iteration;
                }
                shared->mi_updated = 0;
            }

            /* Track idle rounds (no new data) */
            if (!had_new_data) {
                shared->idle_rounds++;
            } else {
                shared->idle_rounds = 0;
            }

            /* Convergence check runs EVERY iteration.
             * Two convergence paths:
             *   1. Phase-gated: all gates pass + entropy plateau
             *   2. Data-exhausted: no new data for 50+ rounds + entropy stable
             * Path 2 prevents infinite spinning when data runs out. */
            if (shared->entropy_idx >= LL_ENTROPY_WINDOW) {
                double min_e = 1e30, max_e = -1e30;
                for (int i = 0; i < LL_ENTROPY_WINDOW; i++) {
                    double e = shared->entropy_history[i];
                    if (e < min_e) min_e = e;
                    if (e > max_e) max_e = e;
                }
                int entropy_stable = (max_e > 0.0 &&
                                      (max_e - min_e) / max_e < 0.01);

                /* Path 1: Full convergence (meaningful clustering) */
                int gate_sentences  = (shared->total_sentences >= LL_MIN_SENTENCES);
                int gate_cc_runs    = (shared->cc_runs >= LL_MIN_CC_RUNS);
                int gate_classes    = (shared->num_classes >= LL_MIN_CLASSES);
                int gate_nontrivial = (shared->num_classes <
                                       shared->word_count_u32 * 95 / 100);

                if (entropy_stable && gate_sentences && gate_cc_runs &&
                    gate_classes && gate_nontrivial) {
                    *done_flag = 1;
                }

                /* Path 2: Data exhausted (no more info to extract) */
                if (entropy_stable && shared->idle_rounds > 50 &&
                    shared->total_sentences > 0) {
                    *done_flag = 1;
                }
            }

            if (shared->iteration >= LL_MAX_ITERATIONS) {
                *done_flag = 1;
            }

            shared->has_new_sentences = 0;
        }

        /* ═══ Phase G: Reset compartments for next round ═══ */
        comp_clear(my_comp, shared->word_count_u32, local_tid, block_size);
        grid.sync(); /* ── sync 7 ── */
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  HOST API
 * ═══════════════════════════════════════════════════════════════ */

extern "C" {

/* Initialize compartment-aware learning state */
LearningState* ck_init(uint32_t num_words) {
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
    state->total_sentences = 0;
    state->cc_runs = 0;
    state->batch_size = 0;
    state->batch_start = 0;
    state->idle_rounds = 0;

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

/* Initialize compartment array — one per SM */
CompartmentState* ck_init_compartments(uint32_t num_compartments,
                                       uint32_t num_words) {
    CompartmentState* comps;
    cudaMallocManaged(&comps, num_compartments * sizeof(CompartmentState));

    for (uint32_t c = 0; c < num_compartments; c++) {
        comps[c].pair_count = 0;
        comps[c].total_observations = 0.0;
        comps[c].sentences_processed = 0;
        comps[c].surprise_score = 0.0;

        memset(comps[c].pair_ht_keys, 0xFF, sizeof(comps[c].pair_ht_keys));
        memset(comps[c].pair_ht_values, 0xFF, sizeof(comps[c].pair_ht_values));

        for (uint32_t w = 0; w < num_words; w++) {
            comps[c].word_marginal[w] = 0.0;
        }
    }

    return comps;
}

/* Initialize sentence ring buffer (larger for compartment kernel) */
SentenceRing* ck_init_ring() {
    SentenceRing* ring;
    cudaMallocManaged(&ring, sizeof(SentenceRing));
    memset(ring, 0, sizeof(SentenceRing));
    return ring;
}

/* Feed a sentence (CPU writes to unified memory) */
void ck_feed_sentence(SentenceRing* ring, uint32_t* words, uint32_t length) {
    if (length > LL_MAX_SENTENCE_LEN) length = LL_MAX_SENTENCE_LEN;

    uint32_t widx = ring->write_idx % LL_RING_SIZE;
    SentenceSlot* slot = &ring->slots[widx];

    /* Wait for slot to be consumed (with timeout) */
    int wait = 0;
    while (slot->ready && wait < 1000000) { wait++; }

    memcpy(slot->words, words, length * sizeof(uint32_t));
    slot->length = length;
    __sync_synchronize();  /* memory fence */
    slot->ready = 1;
    ring->write_idx++;
}

/* Launch the compartment kernel */
int ck_launch(
    LearningState*     state,
    CompartmentState*  compartments,
    SentenceRing*      ring,
    int*               done_flag,
    int*               pause_flag,
    uint32_t*          stats_iteration,
    uint32_t*          stats_pairs,
    uint32_t*          stats_classes,
    double*            stats_entropy
) {
    /* Query SM count */
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);

    /* 1 block per SM, 128 threads per block */
    int total_blocks = num_sms;
    int block_size = 128;

    state->num_compartments = (uint32_t)total_blocks;

    dim3 grid_dim(total_blocks);
    dim3 block_dim(block_size);

    void* args[] = {
        &state, &compartments, &ring, &done_flag, &pause_flag,
        &stats_iteration, &stats_pairs, &stats_classes, &stats_entropy
    };

    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)compartment_kernel, grid_dim, block_dim, args);

    if (err != cudaSuccess) {
        printf("Compartment kernel launch failed: %s\n",
               cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/* Wait for kernel completion */
void ck_wait() {
    cudaDeviceSynchronize();
}

/* Read class assignments */
void ck_read_classes(LearningState* state, uint32_t* out, uint32_t n) {
    cudaDeviceSynchronize();
    memcpy(out, state->word_class_id, n * sizeof(uint32_t));
}

/* Cleanup */
void ck_shutdown(LearningState* state, CompartmentState* comps,
                 SentenceRing* ring) {
    cudaFree(state);
    cudaFree(comps);
    cudaFree(ring);
}

/* Query number of SMs (for allocating compartments) */
int ck_get_num_sms() {
    int num_sms = 0;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    return num_sms;
}

/* Memory report */
void ck_memory_report(int num_compartments) {
    size_t state_sz = sizeof(LearningState);
    size_t comp_sz = sizeof(CompartmentState);
    size_t ring_sz = sizeof(SentenceRing);
    size_t total = state_sz + comp_sz * num_compartments + ring_sz;

    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);

    printf("Memory report:\n");
    printf("  LearningState:    %8.2f MB\n", state_sz / (1024.0 * 1024.0));
    printf("  CompartmentState: %8.2f MB (x%d = %.2f MB)\n",
           comp_sz / (1024.0 * 1024.0), num_compartments,
           comp_sz * num_compartments / (1024.0 * 1024.0));
    printf("  SentenceRing:     %8.2f MB (%d slots)\n",
           ring_sz / (1024.0 * 1024.0), LL_RING_SIZE);
    printf("  Total allocated:  %8.2f MB\n", total / (1024.0 * 1024.0));
    printf("  GPU free:         %8.2f MB / %.2f MB\n",
           free_mem / (1024.0 * 1024.0), total_mem / (1024.0 * 1024.0));
}

} /* extern "C" */
