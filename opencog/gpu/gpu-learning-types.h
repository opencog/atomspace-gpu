/* gpu-learning-types.h — Shared type definitions for the learning loop
 *
 * Included by gpu-learning-loop.cu, gpu-learning-bridge.cu, and test files.
 */

#ifndef GPU_LEARNING_TYPES_H
#define GPU_LEARNING_TYPES_H

#include <cstdint>

/* ─── Constants ─── */

#define LL_MAX_WORDS        8192
#define LL_MAX_PAIRS        262144
#define LL_PAIR_HT_CAPACITY 524288
#define LL_MAX_SENTENCE_LEN 64
#define LL_RING_SIZE        2048
#define LL_MI_THRESHOLD     100
#define LL_CC_MAX_EDGES     65536
#define LL_CC_THRESHOLD     0.15f
#define LL_MAX_ITERATIONS   500
#define LL_ENTROPY_WINDOW   5
#define LL_EMPTY_KEY        0xFFFFFFFFFFFFFFFFULL
#define LL_EMPTY_VALUE      0xFFFFFFFFU

/* ─── Compartment constants ─── */

#define LL_MAX_COMPARTMENTS   64      /* max SMs we support */
#define LL_COMP_MAX_PAIRS     32768   /* per-compartment pair slots */
#define LL_COMP_HT_CAPACITY   65536   /* per-compartment HT (2x pairs) */
#ifndef LL_MIN_SENTENCES
#define LL_MIN_SENTENCES      500     /* min sentences before convergence check */
#endif
#ifndef LL_MIN_CC_RUNS
#define LL_MIN_CC_RUNS        5       /* min CC rounds before convergence check */
#endif
#ifndef LL_MIN_CLASSES
#define LL_MIN_CLASSES        10      /* min classes before convergence check */
#endif

/* ═══════════════════════════════════════════════════════════════
 *  GPU-RESIDENT STATE (shared/merged)
 * ═══════════════════════════════════════════════════════════════ */

struct LearningState {
    /* Word pool */
    uint32_t  word_count_u32;
    double    word_marginal[LL_MAX_WORDS];
    uint32_t  word_class_id[LL_MAX_WORDS];

    /* Pair pool */
    uint32_t  pair_count_u32;
    uint32_t  pair_word_a[LL_MAX_PAIRS];
    uint32_t  pair_word_b[LL_MAX_PAIRS];
    double    pair_count[LL_MAX_PAIRS];
    double    pair_mi[LL_MAX_PAIRS];
    uint32_t  pair_dirty[LL_MAX_PAIRS];

    /* Pair hash table */
    uint64_t  pair_ht_keys[LL_PAIR_HT_CAPACITY];
    uint32_t  pair_ht_values[LL_PAIR_HT_CAPACITY];

    /* Stats */
    uint32_t  dirty_count;
    double    total_pair_observations;
    uint32_t  num_classes;
    uint32_t  iteration;

    /* Entropy */
    double    entropy_history[LL_ENTROPY_WINDOW];
    uint32_t  entropy_idx;

    /* CC work buffers */
    uint32_t  cc_edge_a[LL_CC_MAX_EDGES];
    uint32_t  cc_edge_b[LL_CC_MAX_EDGES];
    uint32_t  cc_labels[LL_MAX_WORDS];
    uint32_t  cc_edge_count;

    /* Pipeline flags */
    int       has_new_sentences;
    int       mi_updated;
    int       cosine_ready;
    int       classes_updated;
    int       cc_changed;  /* used by CC convergence check */

    /* Phase-gated convergence (compartment kernel) */
    uint32_t  total_sentences;     /* cumulative across all rounds */
    uint32_t  cc_runs;             /* how many times CC has run */
    uint32_t  batch_size;          /* snapshot of available sentences */
    uint32_t  batch_start;         /* ring read position at snapshot */
    uint32_t  num_compartments;    /* actual number of SMs in use */
    uint32_t  idle_rounds;         /* rounds with no new data */
};

/* ═══════════════════════════════════════════════════════════════
 *  PER-COMPARTMENT STATE (one per SM)
 *
 *  Each SM accumulates pair counts independently — NO cross-SM
 *  atomics during counting. Only the merge phase writes to shared
 *  LearningState using atomics.
 *
 *  ~1.5 MB per compartment × 36 SMs = ~54 MB total.
 * ═══════════════════════════════════════════════════════════════ */

struct CompartmentState {
    /* Per-compartment pair accumulation */
    uint32_t  pair_count;
    uint32_t  pair_word_a[LL_COMP_MAX_PAIRS];
    uint32_t  pair_word_b[LL_COMP_MAX_PAIRS];
    double    pair_obs[LL_COMP_MAX_PAIRS];

    /* Per-compartment hash table (pair key → local pair index) */
    uint64_t  pair_ht_keys[LL_COMP_HT_CAPACITY];
    uint32_t  pair_ht_values[LL_COMP_HT_CAPACITY];

    /* Per-compartment marginals */
    double    word_marginal[LL_MAX_WORDS];
    double    total_observations;

    /* Stats */
    uint32_t  sentences_processed;
    double    surprise_score;       /* future: for surprise-weighted allocation */
};

/* Sentence ring buffer */
struct SentenceSlot {
    uint32_t  words[LL_MAX_SENTENCE_LEN];
    uint32_t  length;
    uint32_t  ready;
};

struct SentenceRing {
    SentenceSlot slots[LL_RING_SIZE];
    volatile uint32_t write_idx;
    volatile uint32_t read_idx;
};

#endif /* GPU_LEARNING_TYPES_H */
