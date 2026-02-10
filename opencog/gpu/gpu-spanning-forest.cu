/* gpu-spanning-forest.cu — Batch PMFG parsing + surprise computation
 *
 * Grammar-based parsing and surprise measurement on GPU:
 *   1. parse_with_grammar: Batch PMFG with two-level MI lookup
 *      (class-MI first, fallback to word-pair MI)
 *   2. compute_grammar_surprise: Compare word-parse vs grammar-parse
 *      edges, score sentences by how unexpected they are
 *
 * One thread per sentence — embarrassingly parallel since sentences
 * are independent. Small n per sentence (≤MAX_SENTENCE_LEN) makes
 * O(n²) greedy PMFG tractable per thread.
 *
 * Build: nvcc -O2 -arch=sm_75 -rdc=true -c gpu-spanning-forest.cu
 */

#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cmath>

/* ─── Constants ─── */

#ifndef MAX_SENTENCE_LEN
#define MAX_SENTENCE_LEN  64   /* max words per sentence */
#endif

#ifndef MAX_SENTENCES
#define MAX_SENTENCES     4096  /* max sentences in a batch */
#endif

#ifndef MAX_TREE_EDGES
#define MAX_TREE_EDGES    (MAX_SENTENCE_LEN - 1)  /* MST has n-1 edges */
#endif

/* Hash table sentinels */
#define SF_HT_EMPTY_KEY    0xFFFFFFFFFFFFFFFFULL
#define SF_HT_EMPTY_VALUE  0xFFFFFFFFU

/* ─── Hash function (same splitmix64) ─── */

__device__ __forceinline__
uint64_t sf_hash(uint64_t key) {
    key ^= key >> 30;
    key *= 0xBF58476D1CE4E5B9ULL;
    key ^= key >> 27;
    key *= 0x94D049BB133111EBULL;
    key ^= key >> 31;
    return key;
}

/* ─── Pair key encoding (matches gpu-atomspace.cl) ─── */

__device__ __forceinline__
uint64_t pair_key(uint32_t a, uint32_t b) {
    uint32_t lo = (a <= b) ? a : b;
    uint32_t hi = (a <= b) ? b : a;
    return ((uint64_t)lo << 32) | (uint64_t)hi;
}

/* ─── Word-pair MI lookup from pair hash table ─── */

__device__
double word_pair_mi_lookup(
    uint32_t word_a, uint32_t word_b,
    const uint64_t* __restrict__ pair_ht_keys,
    const uint32_t* __restrict__ pair_ht_values,
    const double*   __restrict__ pair_mi,
    uint32_t        pair_ht_capacity
) {
    uint64_t key = pair_key(word_a, word_b);
    uint64_t mask = (uint64_t)(pair_ht_capacity - 1);
    uint64_t slot = sf_hash(key) & mask;

    for (int probe = 0; probe < 4096; probe++) {
        uint64_t existing = pair_ht_keys[slot];
        if (existing == key) {
            uint32_t idx = pair_ht_values[slot];
            if (idx == SF_HT_EMPTY_VALUE) return 0.0;
            return pair_mi[idx];
        }
        if (existing == SF_HT_EMPTY_KEY) return 0.0;
        slot = (slot + 1) & mask;
    }
    return 0.0;
}

/* ─── Class-pair MI lookup from class MI hash table ─── */

__device__
double class_pair_mi_lookup(
    uint32_t class_a, uint32_t class_b,
    const uint64_t* __restrict__ class_ht_keys,
    const double*   __restrict__ class_ht_mi,
    uint32_t        class_ht_capacity
) {
    if (class_a == 0xFFFFFFFFU || class_b == 0xFFFFFFFFU) return 0.0;
    if (class_a == class_b) return 0.0;

    uint32_t lo = (class_a <= class_b) ? class_a : class_b;
    uint32_t hi = (class_a <= class_b) ? class_b : class_a;
    uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;
    uint64_t mask = (uint64_t)(class_ht_capacity - 1);
    uint64_t slot = sf_hash(key) & mask;

    for (int probe = 0; probe < 4096; probe++) {
        uint64_t existing = class_ht_keys[slot];
        if (existing == key) return class_ht_mi[slot];
        if (existing == SF_HT_EMPTY_KEY) return 0.0;
        slot = (slot + 1) & mask;
    }
    return 0.0;
}

/* ─── Two-level MI lookup: class-MI first, fallback to word-pair MI ─── */

__device__
double two_level_mi_lookup(
    uint32_t word_a, uint32_t word_b,
    /* Word → class mapping */
    const uint32_t* __restrict__ word_class_id,
    /* Class MI hash table */
    const uint64_t* __restrict__ class_ht_keys,
    const double*   __restrict__ class_ht_mi,
    uint32_t        class_ht_capacity,
    /* Word-pair MI hash table */
    const uint64_t* __restrict__ pair_ht_keys,
    const uint32_t* __restrict__ pair_ht_values,
    const double*   __restrict__ pair_mi,
    uint32_t        pair_ht_capacity
) {
    /* Try class MI first */
    uint32_t ca = word_class_id[word_a];
    uint32_t cb = word_class_id[word_b];

    if (ca != 0xFFFFFFFFU && cb != 0xFFFFFFFFU && ca != cb) {
        double cmi = class_pair_mi_lookup(ca, cb,
            class_ht_keys, class_ht_mi, class_ht_capacity);
        if (cmi > 0.0) return cmi;
    }

    /* Fallback to word-pair MI */
    return word_pair_mi_lookup(word_a, word_b,
        pair_ht_keys, pair_ht_values, pair_mi, pair_ht_capacity);
}

/* ═══════════════════════════════════════════════════════════════
 *  KERNEL 1: GREEDY PMFG (Planar Maximally Filtered Graph)
 *
 *  One thread per sentence. Greedy maximum spanning tree:
 *  repeatedly add the highest-MI edge that doesn't create a cycle.
 *
 *  Uses Union-Find (path compression) for cycle detection.
 *  Result: n-1 edges per n-word sentence.
 *
 *  Sentence format:
 *    sentence_words[sent_offset + i] = word pool index
 *    sentence_lengths[sent_idx] = number of words
 *    sentence_offsets[sent_idx] = offset into sentence_words
 *
 *  Output:
 *    parse_edges: packed edge lists per sentence
 *    parse_edge_mi: MI value for each edge
 *    parse_edge_counts: number of edges per sentence
 * ═══════════════════════════════════════════════════════════════ */

__global__ void parse_with_grammar(
    /* Sentence data */
    const uint32_t* __restrict__ sentence_words,
    const uint32_t* __restrict__ sentence_lengths,
    const uint32_t* __restrict__ sentence_offsets,
    uint32_t        num_sentences,
    /* Word → class mapping */
    const uint32_t* __restrict__ word_class_id,
    /* Class MI hash table (grammar) */
    const uint64_t* __restrict__ class_ht_keys,
    const double*   __restrict__ class_ht_mi,
    uint32_t        class_ht_capacity,
    /* Word-pair MI hash table (fallback) */
    const uint64_t* __restrict__ pair_ht_keys,
    const uint32_t* __restrict__ pair_ht_values,
    const double*   __restrict__ pair_mi,
    uint32_t        pair_ht_capacity,
    /* Output: parse edges (packed, MAX_TREE_EDGES per sentence) */
    uint32_t*       parse_edge_a,      /* [sent_idx * MAX_TREE_EDGES + edge_idx] */
    uint32_t*       parse_edge_b,
    double*         parse_edge_mi,
    uint32_t*       parse_edge_count   /* [sent_idx] = number of edges found */
) {
    uint32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_sentences) return;

    uint32_t len = sentence_lengths[sid];
    uint32_t off = sentence_offsets[sid];

    if (len < 2 || len > MAX_SENTENCE_LEN) {
        parse_edge_count[sid] = 0;
        return;
    }

    /* Local word indices for this sentence */
    uint32_t words[MAX_SENTENCE_LEN];
    for (uint32_t i = 0; i < len; i++) {
        words[i] = sentence_words[off + i];
    }

    /* Union-Find for cycle detection */
    uint32_t parent[MAX_SENTENCE_LEN];
    for (uint32_t i = 0; i < len; i++) parent[i] = i;

    /* Find root with path compression */
    #define UF_FIND(x) ({ \
        uint32_t _r = (x); \
        while (parent[_r] != _r) { \
            parent[_r] = parent[parent[_r]]; \
            _r = parent[_r]; \
        } \
        _r; \
    })

    /* Greedy PMFG: pick highest-MI edge that doesn't create cycle */
    uint32_t num_edges = 0;
    uint32_t max_edges = len - 1;
    uint32_t out_base = sid * MAX_TREE_EDGES;

    /* We need to consider all O(n²) edges. Sort by MI descending.
     * For small n (≤64), an O(n²) selection sort approach works:
     * repeatedly find the highest unused non-cycle edge. */

    /* Greedy approach: iterate max_edges times,
     * each time scanning all pairs for the best non-cycle edge. */

    for (uint32_t round = 0; round < max_edges && num_edges < max_edges; round++) {
        double best_mi = -1e30;
        uint32_t best_i = 0, best_j = 0;

        for (uint32_t i = 0; i < len; i++) {
            for (uint32_t j = i + 1; j < len; j++) {
                /* Check if adding edge (i,j) would create a cycle */
                uint32_t ri = UF_FIND(i);
                uint32_t rj = UF_FIND(j);
                if (ri == rj) continue;  /* same component → cycle */

                double mi = two_level_mi_lookup(
                    words[i], words[j],
                    word_class_id,
                    class_ht_keys, class_ht_mi, class_ht_capacity,
                    pair_ht_keys, pair_ht_values, pair_mi, pair_ht_capacity);

                if (mi > best_mi) {
                    best_mi = mi;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_mi <= -1e29) break;  /* no more edges available */

        /* Union */
        uint32_t ri = UF_FIND(best_i);
        uint32_t rj = UF_FIND(best_j);
        parent[ri] = rj;

        /* Store edge */
        parse_edge_a[out_base + num_edges] = words[best_i];
        parse_edge_b[out_base + num_edges] = words[best_j];
        parse_edge_mi[out_base + num_edges] = best_mi;
        num_edges++;
    }

    parse_edge_count[sid] = num_edges;

    #undef UF_FIND
}

/* ═══════════════════════════════════════════════════════════════
 *  KERNEL 2: WORD-LEVEL PMFG (for comparison)
 *
 *  Same as above but ONLY uses word-pair MI (no class-MI).
 *  This gives the "true" parse that the grammar should approximate.
 * ═══════════════════════════════════════════════════════════════ */

__global__ void parse_word_only(
    const uint32_t* __restrict__ sentence_words,
    const uint32_t* __restrict__ sentence_lengths,
    const uint32_t* __restrict__ sentence_offsets,
    uint32_t        num_sentences,
    const uint64_t* __restrict__ pair_ht_keys,
    const uint32_t* __restrict__ pair_ht_values,
    const double*   __restrict__ pair_mi,
    uint32_t        pair_ht_capacity,
    uint32_t*       parse_edge_a,
    uint32_t*       parse_edge_b,
    double*         parse_edge_mi,
    uint32_t*       parse_edge_count
) {
    uint32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_sentences) return;

    uint32_t len = sentence_lengths[sid];
    uint32_t off = sentence_offsets[sid];

    if (len < 2 || len > MAX_SENTENCE_LEN) {
        parse_edge_count[sid] = 0;
        return;
    }

    uint32_t words[MAX_SENTENCE_LEN];
    for (uint32_t i = 0; i < len; i++) words[i] = sentence_words[off + i];

    uint32_t parent[MAX_SENTENCE_LEN];
    for (uint32_t i = 0; i < len; i++) parent[i] = i;

    #define UF_FIND(x) ({ \
        uint32_t _r = (x); \
        while (parent[_r] != _r) { \
            parent[_r] = parent[parent[_r]]; \
            _r = parent[_r]; \
        } \
        _r; \
    })

    uint32_t num_edges = 0;
    uint32_t max_edges = len - 1;
    uint32_t out_base = sid * MAX_TREE_EDGES;

    for (uint32_t round = 0; round < max_edges && num_edges < max_edges; round++) {
        double best_mi = -1e30;
        uint32_t best_i = 0, best_j = 0;

        for (uint32_t i = 0; i < len; i++) {
            for (uint32_t j = i + 1; j < len; j++) {
                uint32_t ri = UF_FIND(i);
                uint32_t rj = UF_FIND(j);
                if (ri == rj) continue;

                double mi = word_pair_mi_lookup(
                    words[i], words[j],
                    pair_ht_keys, pair_ht_values, pair_mi,
                    pair_ht_capacity);

                if (mi > best_mi) {
                    best_mi = mi;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_mi <= -1e29) break;

        uint32_t ri = UF_FIND(best_i);
        uint32_t rj = UF_FIND(best_j);
        parent[ri] = rj;

        parse_edge_a[out_base + num_edges] = words[best_i];
        parse_edge_b[out_base + num_edges] = words[best_j];
        parse_edge_mi[out_base + num_edges] = best_mi;
        num_edges++;
    }

    parse_edge_count[sid] = num_edges;
    #undef UF_FIND
}

/* ═══════════════════════════════════════════════════════════════
 *  KERNEL 3: GRAMMAR SURPRISE
 *
 *  One thread per sentence. Compares word-parse edges vs
 *  grammar-parse edges. Surprise = edges in word-parse NOT in
 *  grammar-parse, weighted by MI, normalized by sentence length.
 *
 *  Higher surprise → grammar doesn't explain this sentence well.
 * ═══════════════════════════════════════════════════════════════ */

__global__ void compute_grammar_surprise(
    /* Word-parse results */
    const uint32_t* __restrict__ word_edge_a,
    const uint32_t* __restrict__ word_edge_b,
    const double*   __restrict__ word_edge_mi,
    const uint32_t* __restrict__ word_edge_count,
    /* Grammar-parse results */
    const uint32_t* __restrict__ gram_edge_a,
    const uint32_t* __restrict__ gram_edge_b,
    const uint32_t* __restrict__ gram_edge_count,
    /* Sentence data (for normalization) */
    const uint32_t* __restrict__ sentence_lengths,
    uint32_t        num_sentences,
    /* Output */
    double*         surprise_scores  /* [num_sentences] */
) {
    uint32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= num_sentences) return;

    uint32_t w_count = word_edge_count[sid];
    uint32_t g_count = gram_edge_count[sid];
    uint32_t len = sentence_lengths[sid];

    if (w_count == 0 || len < 2) {
        surprise_scores[sid] = 0.0;
        return;
    }

    uint32_t w_base = sid * MAX_TREE_EDGES;
    uint32_t g_base = sid * MAX_TREE_EDGES;

    /* For each edge in word-parse, check if it exists in grammar-parse */
    double surprise = 0.0;
    for (uint32_t wi = 0; wi < w_count; wi++) {
        uint32_t wa = word_edge_a[w_base + wi];
        uint32_t wb = word_edge_b[w_base + wi];
        double   wmi = word_edge_mi[w_base + wi];

        /* Normalize edge for comparison (canonical order) */
        uint32_t wlo = (wa <= wb) ? wa : wb;
        uint32_t whi = (wa <= wb) ? wb : wa;

        /* Search grammar parse for this edge */
        int found = 0;
        for (uint32_t gi = 0; gi < g_count; gi++) {
            uint32_t ga = gram_edge_a[g_base + gi];
            uint32_t gb = gram_edge_b[g_base + gi];
            uint32_t glo = (ga <= gb) ? ga : gb;
            uint32_t ghi = (ga <= gb) ? gb : ga;

            if (wlo == glo && whi == ghi) {
                found = 1;
                break;
            }
        }

        if (!found) {
            /* This word-parse edge is NOT in grammar-parse → surprising */
            surprise += fabs(wmi);
        }
    }

    /* Normalize by sentence length */
    surprise_scores[sid] = surprise / (double)len;
}

/* ═══════════════════════════════════════════════════════════════
 *  HOST API
 * ═══════════════════════════════════════════════════════════════ */

/* Run full parse + surprise pipeline */
extern "C"
void spanning_forest_run(
    /* Sentence data */
    const uint32_t* d_sentence_words,
    const uint32_t* d_sentence_lengths,
    const uint32_t* d_sentence_offsets,
    uint32_t        num_sentences,
    /* Word → class */
    const uint32_t* d_word_class_id,
    /* Class MI (grammar) */
    const uint64_t* d_class_ht_keys,
    const double*   d_class_ht_mi,
    uint32_t        class_ht_capacity,
    /* Word-pair MI */
    const uint64_t* d_pair_ht_keys,
    const uint32_t* d_pair_ht_values,
    const double*   d_pair_mi,
    uint32_t        pair_ht_capacity,
    /* Pre-allocated output buffers */
    uint32_t*       d_word_parse_a,    /* [num_sentences * MAX_TREE_EDGES] */
    uint32_t*       d_word_parse_b,
    double*         d_word_parse_mi,
    uint32_t*       d_word_parse_count, /* [num_sentences] */
    uint32_t*       d_gram_parse_a,
    uint32_t*       d_gram_parse_b,
    double*         d_gram_parse_mi,
    uint32_t*       d_gram_parse_count,
    double*         d_surprise          /* [num_sentences] */
) {
    int threads = 64;  /* fewer threads — each does O(n³) work */

    int blocks = (num_sentences + threads - 1) / threads;
    if (blocks < 1) blocks = 1;

    /* Step 1: Word-only parse */
    parse_word_only<<<blocks, threads>>>(
        d_sentence_words, d_sentence_lengths, d_sentence_offsets,
        num_sentences,
        d_pair_ht_keys, d_pair_ht_values, d_pair_mi, pair_ht_capacity,
        d_word_parse_a, d_word_parse_b, d_word_parse_mi, d_word_parse_count);
    cudaDeviceSynchronize();

    /* Step 2: Grammar-based parse */
    parse_with_grammar<<<blocks, threads>>>(
        d_sentence_words, d_sentence_lengths, d_sentence_offsets,
        num_sentences,
        d_word_class_id,
        d_class_ht_keys, d_class_ht_mi, class_ht_capacity,
        d_pair_ht_keys, d_pair_ht_values, d_pair_mi, pair_ht_capacity,
        d_gram_parse_a, d_gram_parse_b, d_gram_parse_mi, d_gram_parse_count);
    cudaDeviceSynchronize();

    /* Step 3: Surprise */
    compute_grammar_surprise<<<blocks, threads>>>(
        d_word_parse_a, d_word_parse_b, d_word_parse_mi, d_word_parse_count,
        d_gram_parse_a, d_gram_parse_b, d_gram_parse_count,
        d_sentence_lengths, num_sentences,
        d_surprise);
    cudaDeviceSynchronize();
}
