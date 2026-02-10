/* gpu-divergence.cu — Polysemy detection across compartments
 *
 * MI-neighborhood divergence: for each word present in two compartments,
 * build MI-partner vectors and compute cosine similarity between them.
 * Low cosine → word has different associations → polysemous.
 *
 * Validated by exp9c: MI-neighborhood divergence finds plausible polysemy
 * at 2.1% rate (5/235 words) — "just right" selectivity.
 *
 * Build: nvcc -O2 -arch=sm_75 -rdc=true -c gpu-divergence.cu
 */

#include <cstdint>
#include <cstdio>
#include <cfloat>
#include <cmath>

/* ─── Constants ─── */

#ifndef DIV_TOP_K
#define DIV_TOP_K  32  /* number of top MI partners to compare */
#endif

#ifndef DIV_MAX_WORDS
#define DIV_MAX_WORDS  131072
#endif

/* Hash table sentinels */
#define DIV_HT_EMPTY_KEY    0xFFFFFFFFFFFFFFFFULL
#define DIV_HT_EMPTY_VALUE  0xFFFFFFFFU

/* ─── Hash function ─── */

__device__ __forceinline__
uint64_t div_hash(uint64_t key) {
    key ^= key >> 30;
    key *= 0xBF58476D1CE4E5B9ULL;
    key ^= key >> 27;
    key *= 0x94D049BB133111EBULL;
    key ^= key >> 31;
    return key;
}

/* ─── Pair key ─── */

__device__ __forceinline__
uint64_t div_pair_key(uint32_t a, uint32_t b) {
    uint32_t lo = (a <= b) ? a : b;
    uint32_t hi = (a <= b) ? b : a;
    return ((uint64_t)lo << 32) | (uint64_t)hi;
}

/* ═══════════════════════════════════════════════════════════════
 *  KERNEL 1: BUILD MI NEIGHBORHOOD VECTORS
 *
 *  One thread per word. For each word, scan all pairs involving
 *  that word and collect the top-K MI partners into a dense vector.
 *
 *  The "neighborhood vector" is indexed by partner word index.
 *  For cosine comparison, we store MI values at partner positions.
 *
 *  Simple approach: scan all pairs, keep top-K by MI value using
 *  an insertion sort (K is small, 32).
 * ═══════════════════════════════════════════════════════════════ */

__global__ void build_mi_neighborhood(
    /* Pair data (one compartment) */
    const uint32_t* __restrict__ pair_word_a,
    const uint32_t* __restrict__ pair_word_b,
    const double*   __restrict__ pair_mi,
    const double*   __restrict__ pair_count,
    uint32_t        num_pairs,
    uint32_t        num_words,
    /* Output: top-K neighbors per word */
    uint32_t*       neighbor_ids,    /* [word_idx * DIV_TOP_K + k] = partner word */
    double*         neighbor_mi,     /* [word_idx * DIV_TOP_K + k] = MI value */
    uint32_t*       neighbor_count   /* [word_idx] = actual number of neighbors found */
) {
    uint32_t wid = blockIdx.x * blockDim.x + threadIdx.x;
    if (wid >= num_words) return;

    /* Local top-K tracking */
    uint32_t top_ids[DIV_TOP_K];
    double   top_mi[DIV_TOP_K];
    int      count = 0;

    for (int k = 0; k < DIV_TOP_K; k++) {
        top_ids[k] = 0xFFFFFFFFU;
        top_mi[k] = -1e30;
    }

    /* Scan all pairs for this word */
    for (uint32_t p = 0; p < num_pairs; p++) {
        if (pair_count[p] < 1.0) continue;

        uint32_t wa = pair_word_a[p];
        uint32_t wb = pair_word_b[p];

        uint32_t partner;
        if (wa == wid) partner = wb;
        else if (wb == wid) partner = wa;
        else continue;

        double mi = pair_mi[p];
        if (mi <= 0.0) continue;

        /* Insert into top-K (insertion sort, K=32 is small) */
        if (count < DIV_TOP_K) {
            /* Find insertion point */
            int pos = count;
            while (pos > 0 && top_mi[pos - 1] < mi) {
                if (pos < DIV_TOP_K) {
                    top_ids[pos] = top_ids[pos - 1];
                    top_mi[pos] = top_mi[pos - 1];
                }
                pos--;
            }
            top_ids[pos] = partner;
            top_mi[pos] = mi;
            count++;
        } else if (mi > top_mi[DIV_TOP_K - 1]) {
            /* Replace smallest */
            int pos = DIV_TOP_K - 1;
            while (pos > 0 && top_mi[pos - 1] < mi) {
                top_ids[pos] = top_ids[pos - 1];
                top_mi[pos] = top_mi[pos - 1];
                pos--;
            }
            top_ids[pos] = partner;
            top_mi[pos] = mi;
        }
    }

    /* Write to output */
    uint32_t base = wid * DIV_TOP_K;
    int actual = (count < DIV_TOP_K) ? count : DIV_TOP_K;
    for (int k = 0; k < actual; k++) {
        neighbor_ids[base + k] = top_ids[k];
        neighbor_mi[base + k] = top_mi[k];
    }
    for (int k = actual; k < DIV_TOP_K; k++) {
        neighbor_ids[base + k] = 0xFFFFFFFFU;
        neighbor_mi[base + k] = 0.0;
    }
    neighbor_count[wid] = actual;
}

/* ═══════════════════════════════════════════════════════════════
 *  KERNEL 2: COMPUTE DIVERGENCE (COSINE BETWEEN NEIGHBORHOODS)
 *
 *  One thread per word. For each word, compare its MI-neighborhood
 *  in compartment A vs compartment B using cosine similarity on
 *  the intersection of partner sets.
 *
 *  Low cosine = high divergence = polysemy candidate.
 * ═══════════════════════════════════════════════════════════════ */

__global__ void compute_divergence(
    /* Compartment A neighborhoods */
    const uint32_t* __restrict__ nbr_a_ids,
    const double*   __restrict__ nbr_a_mi,
    const uint32_t* __restrict__ nbr_a_count,
    /* Compartment B neighborhoods */
    const uint32_t* __restrict__ nbr_b_ids,
    const double*   __restrict__ nbr_b_mi,
    const uint32_t* __restrict__ nbr_b_count,
    uint32_t        num_words,
    /* Output */
    double*         divergence_score,  /* [word_idx] = 1.0 - cosine */
    uint32_t*       polysemy_flag,     /* [word_idx] = 1 if polysemous */
    double          polysemy_threshold /* divergence above this → polysemous */
) {
    uint32_t wid = blockIdx.x * blockDim.x + threadIdx.x;
    if (wid >= num_words) return;

    uint32_t ca = nbr_a_count[wid];
    uint32_t cb = nbr_b_count[wid];

    /* Skip words with insufficient data in either compartment */
    if (ca < 3 || cb < 3) {
        divergence_score[wid] = 0.0;
        polysemy_flag[wid] = 0;
        return;
    }

    uint32_t base_a = wid * DIV_TOP_K;
    uint32_t base_b = wid * DIV_TOP_K;

    /* Compute cosine between neighborhood vectors.
     * Build dense vectors over the union of partner sets,
     * then compute dot product and norms.
     *
     * Simple O(K²) approach since K=32 is small. */
    double dot = 0.0;
    double norm_a_sq = 0.0;
    double norm_b_sq = 0.0;

    /* For each partner in A, find matching partner in B */
    for (uint32_t i = 0; i < ca; i++) {
        uint32_t pid_a = nbr_a_ids[base_a + i];
        double   mi_a  = nbr_a_mi[base_a + i];
        norm_a_sq += mi_a * mi_a;

        for (uint32_t j = 0; j < cb; j++) {
            if (nbr_b_ids[base_b + j] == pid_a) {
                dot += mi_a * nbr_b_mi[base_b + j];
                break;
            }
        }
    }

    for (uint32_t j = 0; j < cb; j++) {
        double mi_b = nbr_b_mi[base_b + j];
        norm_b_sq += mi_b * mi_b;
    }

    /* Cosine similarity */
    double denom = sqrt(norm_a_sq) * sqrt(norm_b_sq);
    double cosine = (denom > 1e-10) ? (dot / denom) : 1.0;
    cosine = fmax(-1.0, fmin(1.0, cosine));  /* clamp */

    double div = 1.0 - cosine;
    divergence_score[wid] = div;
    polysemy_flag[wid] = (div > polysemy_threshold) ? 1 : 0;
}

/* ═══════════════════════════════════════════════════════════════
 *  HOST API
 * ═══════════════════════════════════════════════════════════════ */

extern "C"
void divergence_run(
    /* Compartment A pair data */
    const uint32_t* d_pair_a_word_a,
    const uint32_t* d_pair_a_word_b,
    const double*   d_pair_a_mi,
    const double*   d_pair_a_count,
    uint32_t        num_pairs_a,
    /* Compartment B pair data */
    const uint32_t* d_pair_b_word_a,
    const uint32_t* d_pair_b_word_b,
    const double*   d_pair_b_mi,
    const double*   d_pair_b_count,
    uint32_t        num_pairs_b,
    /* Common */
    uint32_t        num_words,
    double          polysemy_threshold,
    /* Pre-allocated work buffers */
    uint32_t*       d_nbr_a_ids,     /* [num_words * DIV_TOP_K] */
    double*         d_nbr_a_mi,
    uint32_t*       d_nbr_a_count,   /* [num_words] */
    uint32_t*       d_nbr_b_ids,
    double*         d_nbr_b_mi,
    uint32_t*       d_nbr_b_count,
    /* Output */
    double*         d_divergence,     /* [num_words] */
    uint32_t*       d_polysemy_flag   /* [num_words] */
) {
    int threads = 256;
    int blocks = (num_words + threads - 1) / threads;
    if (blocks < 1) blocks = 1;

    /* Build neighborhoods for compartment A */
    build_mi_neighborhood<<<blocks, threads>>>(
        d_pair_a_word_a, d_pair_a_word_b, d_pair_a_mi, d_pair_a_count,
        num_pairs_a, num_words,
        d_nbr_a_ids, d_nbr_a_mi, d_nbr_a_count);
    cudaDeviceSynchronize();

    /* Build neighborhoods for compartment B */
    build_mi_neighborhood<<<blocks, threads>>>(
        d_pair_b_word_a, d_pair_b_word_b, d_pair_b_mi, d_pair_b_count,
        num_pairs_b, num_words,
        d_nbr_b_ids, d_nbr_b_mi, d_nbr_b_count);
    cudaDeviceSynchronize();

    /* Compute divergence */
    compute_divergence<<<blocks, threads>>>(
        d_nbr_a_ids, d_nbr_a_mi, d_nbr_a_count,
        d_nbr_b_ids, d_nbr_b_mi, d_nbr_b_count,
        num_words,
        d_divergence, d_polysemy_flag, polysemy_threshold);
    cudaDeviceSynchronize();
}
