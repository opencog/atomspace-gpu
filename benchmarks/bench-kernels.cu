/* bench-kernels.cu — Micro-benchmarks for individual CUDA kernels
 *
 * Measures each kernel in isolation to identify bottlenecks.
 * 100 iterations per benchmark, reports min/mean/max/stddev.
 *
 * Benchmarks:
 *   B1a: Connected Components (Shiloach-Vishkin)
 *   B1b: Grammar cost computation
 *   B1c: Class MI aggregation
 *   B1d: PMFG parse (word-only)
 *   B1e: Surprise computation
 *   B1f: MI-neighborhood divergence
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true -o bench-kernels \
 *     bench-kernels.cu gpu-connected-components.cu gpu-connector-rewrite.cu \
 *     gpu-spanning-forest.cu gpu-divergence.cu -lcudadevrt -lm
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <cuda_runtime.h>

/* ─── Forward declarations from other .cu files ─── */

extern "C" {
    int cc_run(
        const uint32_t* d_cand_word_a, const uint32_t* d_cand_word_b,
        const double* d_cand_cosine, uint32_t num_candidates,
        uint32_t num_words, float threshold,
        uint32_t* d_edge_a, uint32_t* d_edge_b, float* d_edge_weight,
        uint32_t* d_edge_count, uint32_t edge_capacity,
        uint32_t* d_label, int* d_changed,
        uint32_t* d_component_flags, uint32_t* d_component_count,
        uint32_t* d_class_id, uint32_t* d_next_class_id,
        uint32_t* d_class_sizes,
        uint32_t* h_num_components, uint32_t* h_num_edges);

    void grammar_pipeline_run(
        const double* d_sec_count, double* d_sec_cost, uint32_t num_sections,
        const uint32_t* d_pair_word_a, const uint32_t* d_pair_word_b,
        const double* d_pair_mi, const double* d_pair_count,
        uint32_t num_pairs, const uint32_t* d_word_class_id,
        uint64_t* d_ht_keys, double* d_ht_mi_sum, uint32_t* d_ht_count,
        uint32_t ht_capacity, double* d_max_count);

    void spanning_forest_run(
        const uint32_t* d_sentence_words, const uint32_t* d_sentence_lengths,
        const uint32_t* d_sentence_offsets, uint32_t num_sentences,
        const uint32_t* d_word_class_id,
        const uint64_t* d_class_ht_keys, const double* d_class_ht_mi,
        uint32_t class_ht_capacity,
        const uint64_t* d_pair_ht_keys, const uint32_t* d_pair_ht_values,
        const double* d_pair_mi, uint32_t pair_ht_capacity,
        uint32_t* d_word_parse_a, uint32_t* d_word_parse_b,
        double* d_word_parse_mi, uint32_t* d_word_parse_count,
        uint32_t* d_gram_parse_a, uint32_t* d_gram_parse_b,
        double* d_gram_parse_mi, uint32_t* d_gram_parse_count,
        double* d_surprise);

    void divergence_run(
        const uint32_t* d_pair_a_word_a, const uint32_t* d_pair_a_word_b,
        const double* d_pair_a_mi, const double* d_pair_a_count,
        uint32_t num_pairs_a,
        const uint32_t* d_pair_b_word_a, const uint32_t* d_pair_b_word_b,
        const double* d_pair_b_mi, const double* d_pair_b_count,
        uint32_t num_pairs_b,
        uint32_t num_words, double polysemy_threshold,
        uint32_t* d_nbr_a_ids, double* d_nbr_a_mi, uint32_t* d_nbr_a_count,
        uint32_t* d_nbr_b_ids, double* d_nbr_b_mi, uint32_t* d_nbr_b_count,
        double* d_divergence, uint32_t* d_polysemy_flag);
}

/* ─── Timing helpers ─── */

#define ITERS 100

struct BenchStats {
    double min_ms, max_ms, mean_ms, stddev_ms;
};

static BenchStats compute_stats(double* times, int n) {
    BenchStats s;
    s.min_ms = 1e30; s.max_ms = -1e30;
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        if (times[i] < s.min_ms) s.min_ms = times[i];
        if (times[i] > s.max_ms) s.max_ms = times[i];
        sum += times[i];
        sum_sq += times[i] * times[i];
    }
    s.mean_ms = sum / n;
    s.stddev_ms = sqrt(sum_sq / n - s.mean_ms * s.mean_ms);
    return s;
}

/* ─── Data generation helpers ─── */

static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Generate random cosine candidate pairs */
static void gen_cosine_candidates(
    uint32_t* h_word_a, uint32_t* h_word_b, double* h_cosine,
    uint32_t num_cand, uint32_t num_words, uint32_t seed
) {
    uint32_t rng = seed;
    for (uint32_t i = 0; i < num_cand; i++) {
        uint32_t wa = xorshift32(&rng) % num_words;
        uint32_t wb = xorshift32(&rng) % num_words;
        if (wb == wa) wb = (wa + 1) % num_words;
        h_word_a[i] = (wa <= wb) ? wa : wb;
        h_word_b[i] = (wa <= wb) ? wb : wa;
        /* Cosine values: mostly low, some high (realistic distribution) */
        double cos_val = (double)(xorshift32(&rng) % 10000) / 10000.0;
        cos_val = cos_val * cos_val; /* quadratic: bias towards low values */
        h_cosine[i] = cos_val;
    }
}

/* Generate word pairs with MI/count data */
static void gen_word_pairs(
    uint32_t* h_wa, uint32_t* h_wb, double* h_mi, double* h_count,
    uint32_t num_pairs, uint32_t num_words, uint32_t seed
) {
    uint32_t rng = seed;
    for (uint32_t i = 0; i < num_pairs; i++) {
        uint32_t wa = xorshift32(&rng) % num_words;
        uint32_t wb = xorshift32(&rng) % num_words;
        if (wb == wa) wb = (wa + 1) % num_words;
        h_wa[i] = (wa <= wb) ? wa : wb;
        h_wb[i] = (wa <= wb) ? wb : wa;
        h_mi[i] = 0.5 + 3.0 * (double)(xorshift32(&rng) % 10000) / 10000.0;
        h_count[i] = 1.0 + (double)(xorshift32(&rng) % 100);
    }
}

/* Build a pair hash table on GPU */
static void build_pair_ht(
    uint32_t* h_wa, uint32_t* h_wb, uint32_t num_pairs,
    uint64_t* h_ht_keys, uint32_t* h_ht_values, uint32_t ht_cap
) {
    memset(h_ht_keys, 0xFF, ht_cap * sizeof(uint64_t));
    memset(h_ht_values, 0xFF, ht_cap * sizeof(uint32_t));

    for (uint32_t i = 0; i < num_pairs; i++) {
        uint32_t lo = (h_wa[i] <= h_wb[i]) ? h_wa[i] : h_wb[i];
        uint32_t hi = (h_wa[i] <= h_wb[i]) ? h_wb[i] : h_wa[i];
        uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;

        /* splitmix64 */
        uint64_t h = key;
        h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;
        h ^= h >> 27; h *= 0x94D049BB133111EBULL;
        h ^= h >> 31;

        uint64_t slot = h & (uint64_t)(ht_cap - 1);
        for (int probe = 0; probe < 4096; probe++) {
            if (h_ht_keys[slot] == 0xFFFFFFFFFFFFFFFFULL) {
                h_ht_keys[slot] = key;
                h_ht_values[slot] = i;
                break;
            }
            slot = (slot + 1) & (uint64_t)(ht_cap - 1);
        }
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  B1a: CONNECTED COMPONENTS
 * ═══════════════════════════════════════════════════════════════ */

static void bench_cc() {
    printf("\n=== Benchmark B1a: Connected Components ===\n");
    printf("%-8s | %-8s | %-6s | %-10s | %-10s | %-10s | %-10s\n",
           "Words", "Cands", "Comps", "Min(ms)", "Mean(ms)", "Max(ms)", "StdDev");
    printf("---------|----------|--------|------------|------------|------------|------------\n");

    uint32_t word_scales[] = {100, 500, 1000, 5000, 8000};
    int num_scales = 5;

    for (int s = 0; s < num_scales; s++) {
        uint32_t nw = word_scales[s];
        uint32_t nc = nw * 2;  /* 2 candidates per word */

        /* Host data */
        uint32_t* h_wa = (uint32_t*)malloc(nc * sizeof(uint32_t));
        uint32_t* h_wb = (uint32_t*)malloc(nc * sizeof(uint32_t));
        double*   h_cos = (double*)malloc(nc * sizeof(double));
        gen_cosine_candidates(h_wa, h_wb, h_cos, nc, nw, 42 + s);

        /* Device buffers */
        uint32_t *d_cand_wa, *d_cand_wb;
        double *d_cand_cos;
        cudaMalloc(&d_cand_wa, nc * sizeof(uint32_t));
        cudaMalloc(&d_cand_wb, nc * sizeof(uint32_t));
        cudaMalloc(&d_cand_cos, nc * sizeof(double));
        cudaMemcpy(d_cand_wa, h_wa, nc * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cand_wb, h_wb, nc * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cand_cos, h_cos, nc * sizeof(double), cudaMemcpyHostToDevice);

        /* CC work buffers */
        uint32_t edge_cap = nc;
        uint32_t *d_edge_a, *d_edge_b, *d_edge_count;
        float *d_edge_weight;
        uint32_t *d_label, *d_comp_flags, *d_comp_count;
        uint32_t *d_class_id, *d_next_id, *d_class_sizes;
        int *d_changed;

        cudaMalloc(&d_edge_a, edge_cap * sizeof(uint32_t));
        cudaMalloc(&d_edge_b, edge_cap * sizeof(uint32_t));
        cudaMalloc(&d_edge_weight, edge_cap * sizeof(float));
        cudaMalloc(&d_edge_count, sizeof(uint32_t));
        cudaMalloc(&d_label, nw * sizeof(uint32_t));
        cudaMalloc(&d_changed, sizeof(int));
        cudaMalloc(&d_comp_flags, nw * sizeof(uint32_t));
        cudaMalloc(&d_comp_count, sizeof(uint32_t));
        cudaMalloc(&d_class_id, nw * sizeof(uint32_t));
        cudaMalloc(&d_next_id, sizeof(uint32_t));
        cudaMalloc(&d_class_sizes, nw * sizeof(uint32_t));

        /* Warmup */
        uint32_t h_nc_out, h_ne_out;
        cc_run(d_cand_wa, d_cand_wb, d_cand_cos, nc, nw, 0.15f,
               d_edge_a, d_edge_b, d_edge_weight, d_edge_count, edge_cap,
               d_label, d_changed, d_comp_flags, d_comp_count,
               d_class_id, d_next_id, d_class_sizes,
               &h_nc_out, &h_ne_out);

        /* Benchmark */
        double times[ITERS];
        uint32_t last_comps = 0;

        for (int i = 0; i < ITERS; i++) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            cc_run(d_cand_wa, d_cand_wb, d_cand_cos, nc, nw, 0.15f,
                   d_edge_a, d_edge_b, d_edge_weight, d_edge_count, edge_cap,
                   d_label, d_changed, d_comp_flags, d_comp_count,
                   d_class_id, d_next_id, d_class_sizes,
                   &h_nc_out, &h_ne_out);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times[i] = (double)ms;
            last_comps = h_nc_out;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        BenchStats bs = compute_stats(times, ITERS);
        printf("%-8u | %-8u | %-6u | %10.3f | %10.3f | %10.3f | %10.3f\n",
               nw, nc, last_comps, bs.min_ms, bs.mean_ms, bs.max_ms, bs.stddev_ms);

        /* Cleanup */
        free(h_wa); free(h_wb); free(h_cos);
        cudaFree(d_cand_wa); cudaFree(d_cand_wb); cudaFree(d_cand_cos);
        cudaFree(d_edge_a); cudaFree(d_edge_b); cudaFree(d_edge_weight);
        cudaFree(d_edge_count); cudaFree(d_label); cudaFree(d_changed);
        cudaFree(d_comp_flags); cudaFree(d_comp_count);
        cudaFree(d_class_id); cudaFree(d_next_id); cudaFree(d_class_sizes);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  B1b + B1c: GRAMMAR COSTS + CLASS MI
 * ═══════════════════════════════════════════════════════════════ */

static void bench_grammar() {
    printf("\n=== Benchmark B1b: Grammar Costs ===\n");
    printf("%-10s | %-10s | %-10s | %-10s | %-10s\n",
           "Sections", "Min(ms)", "Mean(ms)", "Max(ms)", "StdDev");
    printf("-----------|------------|------------|------------|------------\n");

    uint32_t sec_scales[] = {100, 1000, 10000, 64000};
    int num_scales = 4;

    for (int s = 0; s < num_scales; s++) {
        uint32_t ns = sec_scales[s];

        /* Generate section counts */
        double* h_sec_count = (double*)malloc(ns * sizeof(double));
        uint32_t rng = 123 + s;
        for (uint32_t i = 0; i < ns; i++) {
            h_sec_count[i] = 1.0 + (double)(xorshift32(&rng) % 1000);
        }

        /* Pair data for class MI */
        uint32_t np = (ns < 10000) ? ns : 10000;
        uint32_t nw = 500;
        uint32_t* h_wa = (uint32_t*)malloc(np * sizeof(uint32_t));
        uint32_t* h_wb = (uint32_t*)malloc(np * sizeof(uint32_t));
        double* h_mi = (double*)malloc(np * sizeof(double));
        double* h_count = (double*)malloc(np * sizeof(double));
        gen_word_pairs(h_wa, h_wb, h_mi, h_count, np, nw, 456 + s);

        /* Class assignments (10 classes) */
        uint32_t* h_class = (uint32_t*)malloc(nw * sizeof(uint32_t));
        for (uint32_t i = 0; i < nw; i++) h_class[i] = i % 10;

        /* Allocate device memory */
        double *d_sec_count, *d_sec_cost, *d_max_count;
        uint32_t *d_wa, *d_wb, *d_class_id;
        double *d_mi, *d_pcount;
        uint64_t *d_ht_keys;
        double *d_ht_mi_sum;
        uint32_t *d_ht_count;
        uint32_t ht_cap = 65536;

        cudaMalloc(&d_sec_count, ns * sizeof(double));
        cudaMalloc(&d_sec_cost, ns * sizeof(double));
        cudaMalloc(&d_max_count, sizeof(double));
        cudaMalloc(&d_wa, np * sizeof(uint32_t));
        cudaMalloc(&d_wb, np * sizeof(uint32_t));
        cudaMalloc(&d_mi, np * sizeof(double));
        cudaMalloc(&d_pcount, np * sizeof(double));
        cudaMalloc(&d_class_id, nw * sizeof(uint32_t));
        cudaMalloc(&d_ht_keys, ht_cap * sizeof(uint64_t));
        cudaMalloc(&d_ht_mi_sum, ht_cap * sizeof(double));
        cudaMalloc(&d_ht_count, ht_cap * sizeof(uint32_t));

        cudaMemcpy(d_sec_count, h_sec_count, ns * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wa, h_wa, np * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wb, h_wb, np * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mi, h_mi, np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pcount, h_count, np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_class_id, h_class, nw * sizeof(uint32_t), cudaMemcpyHostToDevice);

        /* Warmup */
        grammar_pipeline_run(d_sec_count, d_sec_cost, ns,
                             d_wa, d_wb, d_mi, d_pcount, np, d_class_id,
                             d_ht_keys, d_ht_mi_sum, d_ht_count, ht_cap, d_max_count);

        /* Benchmark */
        double times[ITERS];
        for (int i = 0; i < ITERS; i++) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            grammar_pipeline_run(d_sec_count, d_sec_cost, ns,
                                 d_wa, d_wb, d_mi, d_pcount, np, d_class_id,
                                 d_ht_keys, d_ht_mi_sum, d_ht_count, ht_cap, d_max_count);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times[i] = (double)ms;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        BenchStats bs = compute_stats(times, ITERS);
        printf("%-10u | %10.3f | %10.3f | %10.3f | %10.3f\n",
               ns, bs.min_ms, bs.mean_ms, bs.max_ms, bs.stddev_ms);

        /* Cleanup */
        free(h_sec_count); free(h_wa); free(h_wb); free(h_mi); free(h_count); free(h_class);
        cudaFree(d_sec_count); cudaFree(d_sec_cost); cudaFree(d_max_count);
        cudaFree(d_wa); cudaFree(d_wb); cudaFree(d_mi); cudaFree(d_pcount);
        cudaFree(d_class_id);
        cudaFree(d_ht_keys); cudaFree(d_ht_mi_sum); cudaFree(d_ht_count);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  B1d + B1e: PMFG PARSE + SURPRISE
 * ═══════════════════════════════════════════════════════════════ */

#define MAX_TREE_EDGES 63

static void bench_parse() {
    printf("\n=== Benchmark B1d: PMFG Parse + Surprise ===\n");
    printf("%-6s | %-6s | %-10s | %-10s | %-10s | %-10s | %-12s\n",
           "Sents", "AvgLen", "Min(ms)", "Mean(ms)", "Max(ms)", "StdDev", "Sents/sec");
    printf("-------|--------|------------|------------|------------|------------|-------------\n");

    struct ParseScale {
        uint32_t num_sentences;
        uint32_t avg_len;
    };
    ParseScale scales[] = {
        {1, 10}, {1, 32}, {1, 64},
        {10, 10}, {10, 32},
        {64, 10}, {64, 32}
    };
    int num_scales = 7;

    for (int s = 0; s < num_scales; s++) {
        uint32_t ns = scales[s].num_sentences;
        uint32_t avg_len = scales[s].avg_len;
        uint32_t nw = 500; /* word pool */
        uint32_t np = 2000; /* pairs */

        /* Generate pair data + hash table */
        uint32_t* h_wa = (uint32_t*)malloc(np * sizeof(uint32_t));
        uint32_t* h_wb = (uint32_t*)malloc(np * sizeof(uint32_t));
        double* h_mi = (double*)malloc(np * sizeof(double));
        double* h_count = (double*)malloc(np * sizeof(double));
        gen_word_pairs(h_wa, h_wb, h_mi, h_count, np, nw, 789 + s);

        uint32_t ht_cap = 8192;
        uint64_t* h_ht_keys = (uint64_t*)malloc(ht_cap * sizeof(uint64_t));
        uint32_t* h_ht_vals = (uint32_t*)malloc(ht_cap * sizeof(uint32_t));
        build_pair_ht(h_wa, h_wb, np, h_ht_keys, h_ht_vals, ht_cap);

        /* Class data (10 classes) */
        uint32_t* h_class = (uint32_t*)malloc(nw * sizeof(uint32_t));
        for (uint32_t i = 0; i < nw; i++) h_class[i] = i % 10;

        /* Generate sentences */
        uint32_t total_words = 0;
        uint32_t* h_sent_lengths = (uint32_t*)malloc(ns * sizeof(uint32_t));
        uint32_t* h_sent_offsets = (uint32_t*)malloc(ns * sizeof(uint32_t));
        uint32_t rng = 321 + s;
        for (uint32_t i = 0; i < ns; i++) {
            h_sent_lengths[i] = avg_len;
            h_sent_offsets[i] = total_words;
            total_words += avg_len;
        }
        uint32_t* h_sent_words = (uint32_t*)malloc(total_words * sizeof(uint32_t));
        for (uint32_t i = 0; i < total_words; i++) {
            h_sent_words[i] = xorshift32(&rng) % nw;
        }

        /* Allocate device buffers */
        uint32_t *d_wa, *d_wb;
        double *d_mi;
        uint64_t *d_ht_keys;
        uint32_t *d_ht_vals, *d_class_id;
        uint32_t *d_sent_words, *d_sent_lengths, *d_sent_offsets;
        uint32_t *d_word_pa, *d_word_pb, *d_word_pc;
        double *d_word_pmi;
        uint32_t *d_gram_pa, *d_gram_pb, *d_gram_pc;
        double *d_gram_pmi;
        double *d_surprise;

        /* Class MI hash table (empty — grammar parse will fallback to word-pair) */
        uint64_t *d_class_ht_keys;
        double *d_class_ht_mi;
        uint32_t class_ht_cap = 1024;

        cudaMalloc(&d_wa, np * sizeof(uint32_t));
        cudaMalloc(&d_wb, np * sizeof(uint32_t));
        cudaMalloc(&d_mi, np * sizeof(double));
        cudaMalloc(&d_ht_keys, ht_cap * sizeof(uint64_t));
        cudaMalloc(&d_ht_vals, ht_cap * sizeof(uint32_t));
        cudaMalloc(&d_class_id, nw * sizeof(uint32_t));
        cudaMalloc(&d_class_ht_keys, class_ht_cap * sizeof(uint64_t));
        cudaMalloc(&d_class_ht_mi, class_ht_cap * sizeof(double));
        cudaMalloc(&d_sent_words, total_words * sizeof(uint32_t));
        cudaMalloc(&d_sent_lengths, ns * sizeof(uint32_t));
        cudaMalloc(&d_sent_offsets, ns * sizeof(uint32_t));
        cudaMalloc(&d_word_pa, ns * MAX_TREE_EDGES * sizeof(uint32_t));
        cudaMalloc(&d_word_pb, ns * MAX_TREE_EDGES * sizeof(uint32_t));
        cudaMalloc(&d_word_pmi, ns * MAX_TREE_EDGES * sizeof(double));
        cudaMalloc(&d_word_pc, ns * sizeof(uint32_t));
        cudaMalloc(&d_gram_pa, ns * MAX_TREE_EDGES * sizeof(uint32_t));
        cudaMalloc(&d_gram_pb, ns * MAX_TREE_EDGES * sizeof(uint32_t));
        cudaMalloc(&d_gram_pmi, ns * MAX_TREE_EDGES * sizeof(double));
        cudaMalloc(&d_gram_pc, ns * sizeof(uint32_t));
        cudaMalloc(&d_surprise, ns * sizeof(double));

        cudaMemcpy(d_wa, h_wa, np * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wb, h_wb, np * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mi, h_mi, np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ht_keys, h_ht_keys, ht_cap * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ht_vals, h_ht_vals, ht_cap * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_class_id, h_class, nw * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemset(d_class_ht_keys, 0xFF, class_ht_cap * sizeof(uint64_t));
        cudaMemset(d_class_ht_mi, 0, class_ht_cap * sizeof(double));
        cudaMemcpy(d_sent_words, h_sent_words, total_words * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sent_lengths, h_sent_lengths, ns * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sent_offsets, h_sent_offsets, ns * sizeof(uint32_t), cudaMemcpyHostToDevice);

        /* Warmup */
        spanning_forest_run(
            d_sent_words, d_sent_lengths, d_sent_offsets, ns,
            d_class_id, d_class_ht_keys, d_class_ht_mi, class_ht_cap,
            d_ht_keys, d_ht_vals, d_mi, ht_cap,
            d_word_pa, d_word_pb, d_word_pmi, d_word_pc,
            d_gram_pa, d_gram_pb, d_gram_pmi, d_gram_pc,
            d_surprise);

        /* Benchmark */
        double times[ITERS];
        for (int i = 0; i < ITERS; i++) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            spanning_forest_run(
                d_sent_words, d_sent_lengths, d_sent_offsets, ns,
                d_class_id, d_class_ht_keys, d_class_ht_mi, class_ht_cap,
                d_ht_keys, d_ht_vals, d_mi, ht_cap,
                d_word_pa, d_word_pb, d_word_pmi, d_word_pc,
                d_gram_pa, d_gram_pb, d_gram_pmi, d_gram_pc,
                d_surprise);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times[i] = (double)ms;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        BenchStats bs = compute_stats(times, ITERS);
        double sents_per_sec = (bs.mean_ms > 0.001) ? (ns * 1000.0 / bs.mean_ms) : 0.0;
        printf("%-6u | %-6u | %10.3f | %10.3f | %10.3f | %10.3f | %12.1f\n",
               ns, avg_len, bs.min_ms, bs.mean_ms, bs.max_ms, bs.stddev_ms, sents_per_sec);

        /* Cleanup */
        free(h_wa); free(h_wb); free(h_mi); free(h_count); free(h_class);
        free(h_ht_keys); free(h_ht_vals);
        free(h_sent_words); free(h_sent_lengths); free(h_sent_offsets);
        cudaFree(d_wa); cudaFree(d_wb); cudaFree(d_mi);
        cudaFree(d_ht_keys); cudaFree(d_ht_vals); cudaFree(d_class_id);
        cudaFree(d_class_ht_keys); cudaFree(d_class_ht_mi);
        cudaFree(d_sent_words); cudaFree(d_sent_lengths); cudaFree(d_sent_offsets);
        cudaFree(d_word_pa); cudaFree(d_word_pb); cudaFree(d_word_pmi); cudaFree(d_word_pc);
        cudaFree(d_gram_pa); cudaFree(d_gram_pb); cudaFree(d_gram_pmi); cudaFree(d_gram_pc);
        cudaFree(d_surprise);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  B1f: MI-NEIGHBORHOOD DIVERGENCE
 * ═══════════════════════════════════════════════════════════════ */

static void bench_divergence() {
    printf("\n=== Benchmark B1f: MI-Neighborhood Divergence ===\n");
    printf("%-8s | %-8s | %-10s | %-10s | %-10s | %-10s | %-8s\n",
           "Words", "Pairs", "Min(ms)", "Mean(ms)", "Max(ms)", "StdDev", "Polysem");
    printf("---------|----------|------------|------------|------------|------------|--------\n");

    uint32_t word_scales[] = {100, 500, 1000};
    int num_scales = 3;

    for (int s = 0; s < num_scales; s++) {
        uint32_t nw = word_scales[s];
        uint32_t np = nw * 4; /* pairs per compartment */

        /* Generate pair data for two compartments */
        uint32_t* h_wa_a = (uint32_t*)malloc(np * sizeof(uint32_t));
        uint32_t* h_wb_a = (uint32_t*)malloc(np * sizeof(uint32_t));
        double* h_mi_a = (double*)malloc(np * sizeof(double));
        double* h_cnt_a = (double*)malloc(np * sizeof(double));
        gen_word_pairs(h_wa_a, h_wb_a, h_mi_a, h_cnt_a, np, nw, 111 + s);

        uint32_t* h_wa_b = (uint32_t*)malloc(np * sizeof(uint32_t));
        uint32_t* h_wb_b = (uint32_t*)malloc(np * sizeof(uint32_t));
        double* h_mi_b = (double*)malloc(np * sizeof(double));
        double* h_cnt_b = (double*)malloc(np * sizeof(double));
        gen_word_pairs(h_wa_b, h_wb_b, h_mi_b, h_cnt_b, np, nw, 222 + s);

        /* Device buffers */
        uint32_t *d_wa_a, *d_wb_a, *d_wa_b, *d_wb_b;
        double *d_mi_a, *d_cnt_a, *d_mi_b, *d_cnt_b;
        uint32_t *d_nbr_a_ids, *d_nbr_b_ids, *d_nbr_a_cnt, *d_nbr_b_cnt;
        double *d_nbr_a_mi, *d_nbr_b_mi;
        double *d_divergence;
        uint32_t *d_polysemy;

        cudaMalloc(&d_wa_a, np * sizeof(uint32_t));
        cudaMalloc(&d_wb_a, np * sizeof(uint32_t));
        cudaMalloc(&d_mi_a, np * sizeof(double));
        cudaMalloc(&d_cnt_a, np * sizeof(double));
        cudaMalloc(&d_wa_b, np * sizeof(uint32_t));
        cudaMalloc(&d_wb_b, np * sizeof(uint32_t));
        cudaMalloc(&d_mi_b, np * sizeof(double));
        cudaMalloc(&d_cnt_b, np * sizeof(double));
        cudaMalloc(&d_nbr_a_ids, nw * 32 * sizeof(uint32_t));
        cudaMalloc(&d_nbr_a_mi, nw * 32 * sizeof(double));
        cudaMalloc(&d_nbr_a_cnt, nw * sizeof(uint32_t));
        cudaMalloc(&d_nbr_b_ids, nw * 32 * sizeof(uint32_t));
        cudaMalloc(&d_nbr_b_mi, nw * 32 * sizeof(double));
        cudaMalloc(&d_nbr_b_cnt, nw * sizeof(uint32_t));
        cudaMalloc(&d_divergence, nw * sizeof(double));
        cudaMalloc(&d_polysemy, nw * sizeof(uint32_t));

        cudaMemcpy(d_wa_a, h_wa_a, np * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wb_a, h_wb_a, np * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mi_a, h_mi_a, np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cnt_a, h_cnt_a, np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wa_b, h_wa_b, np * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_wb_b, h_wb_b, np * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mi_b, h_mi_b, np * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cnt_b, h_cnt_b, np * sizeof(double), cudaMemcpyHostToDevice);

        /* Warmup */
        divergence_run(
            d_wa_a, d_wb_a, d_mi_a, d_cnt_a, np,
            d_wa_b, d_wb_b, d_mi_b, d_cnt_b, np,
            nw, 0.5,
            d_nbr_a_ids, d_nbr_a_mi, d_nbr_a_cnt,
            d_nbr_b_ids, d_nbr_b_mi, d_nbr_b_cnt,
            d_divergence, d_polysemy);

        /* Benchmark */
        double times[ITERS];
        for (int i = 0; i < ITERS; i++) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);

            divergence_run(
                d_wa_a, d_wb_a, d_mi_a, d_cnt_a, np,
                d_wa_b, d_wb_b, d_mi_b, d_cnt_b, np,
                nw, 0.5,
                d_nbr_a_ids, d_nbr_a_mi, d_nbr_a_cnt,
                d_nbr_b_ids, d_nbr_b_mi, d_nbr_b_cnt,
                d_divergence, d_polysemy);

            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            times[i] = (double)ms;
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }

        /* Count polysemy flags */
        uint32_t* h_poly = (uint32_t*)malloc(nw * sizeof(uint32_t));
        cudaMemcpy(h_poly, d_polysemy, nw * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t poly_count = 0;
        for (uint32_t i = 0; i < nw; i++) if (h_poly[i]) poly_count++;

        BenchStats bs = compute_stats(times, ITERS);
        printf("%-8u | %-8u | %10.3f | %10.3f | %10.3f | %10.3f | %u\n",
               nw, np, bs.min_ms, bs.mean_ms, bs.max_ms, bs.stddev_ms, poly_count);

        /* Cleanup */
        free(h_wa_a); free(h_wb_a); free(h_mi_a); free(h_cnt_a);
        free(h_wa_b); free(h_wb_b); free(h_mi_b); free(h_cnt_b);
        free(h_poly);
        cudaFree(d_wa_a); cudaFree(d_wb_a); cudaFree(d_mi_a); cudaFree(d_cnt_a);
        cudaFree(d_wa_b); cudaFree(d_wb_b); cudaFree(d_mi_b); cudaFree(d_cnt_b);
        cudaFree(d_nbr_a_ids); cudaFree(d_nbr_a_mi); cudaFree(d_nbr_a_cnt);
        cudaFree(d_nbr_b_ids); cudaFree(d_nbr_b_mi); cudaFree(d_nbr_b_cnt);
        cudaFree(d_divergence); cudaFree(d_polysemy);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════ */

int main() {
    printf("GPU Learning Pipeline — Micro-Benchmarks\n");
    printf("=========================================\n");

    /* Print GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs, %d MB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           (int)(prop.totalGlobalMem / (1024*1024)));
    printf("Iterations per benchmark: %d\n", ITERS);

    bench_cc();
    bench_grammar();
    bench_parse();
    bench_divergence();

    printf("\n=== All micro-benchmarks complete ===\n");

    /* Print LearningState size estimate */
    printf("\nMemory estimates:\n");
    size_t ls_size = sizeof(uint32_t) /* word_count */
        + 8192 * sizeof(double) /* word_marginal */
        + 8192 * sizeof(uint32_t) /* word_class_id */
        + sizeof(uint32_t) /* pair_count */
        + 262144 * sizeof(uint32_t) * 3 /* pair_word_a/b + pair_dirty */
        + 262144 * sizeof(double) * 2 /* pair_count + pair_mi */
        + 524288 * sizeof(uint64_t) /* pair_ht_keys */
        + 524288 * sizeof(uint32_t) /* pair_ht_values */
        + sizeof(uint32_t) * 4 + sizeof(double) /* stats */
        + 5 * sizeof(double) + sizeof(uint32_t) /* entropy */
        + 65536 * sizeof(uint32_t) * 2 /* cc_edge_a/b */
        + 8192 * sizeof(uint32_t) /* cc_labels */
        + sizeof(uint32_t) /* cc_edge_count */
        + sizeof(int) * 5; /* pipeline flags */
    printf("  LearningState struct: %.2f MB\n", (double)ls_size / (1024.0 * 1024.0));
    printf("  SentenceRing struct: %.2f KB\n",
           (double)(64 * (64 * sizeof(uint32_t) + 2 * sizeof(uint32_t)) + 2 * sizeof(uint32_t)) / 1024.0);

    return 0;
}
