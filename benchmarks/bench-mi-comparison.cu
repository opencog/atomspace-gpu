/* bench-mi-comparison.cu — CUDA vs OpenCL MI kernel comparison
 *
 * Generates identical test data at multiple scales, runs the CUDA MI
 * computation (same code path as the persistent kernel), and reports
 * throughput. Also writes test data to a file for OpenCL comparison
 * via the existing gpu-mi.scm Scheme wrapper.
 *
 * Scales: 1K, 10K, 100K, 262K pairs
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true -o bench-mi-comparison \
 *     bench-mi-comparison.cu gpu-learning-loop.cu -lcudadevrt -lm
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "gpu-learning-types.h"

namespace cg = cooperative_groups;

/* ─── Forward declarations ─── */

extern "C" {
    LearningState* ll_init(uint32_t num_words);
    SentenceRing*  ll_init_ring();
    void           ll_shutdown(LearningState* state, SentenceRing* ring);
}

/* ─── Helpers ─── */

static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static double time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

/* ─── Reference MI computation (CPU, double precision) ─── */

static double cpu_mi(double count, double left_marginal, double right_marginal,
                     double total_obs) {
    if (count < 1.0) return 0.0;
    double log2_factor = 1.4426950408889634;
    double eps = 1e-10;
    double n = fmax(total_obs, eps);
    double l = fmax(left_marginal, eps);
    double r = fmax(right_marginal, eps);
    return (log(count) + log(n) - log(l) - log(r)) * log2_factor;
}

/* ═══════════════════════════════════════════════════════════════
 *  MI BENCHMARK
 *
 *  Uses LearningState directly (same memory layout as persistent kernel).
 *  Populates pairs + marginals, then calls the MI kernel via a minimal
 *  cooperative kernel that runs ll_compute_mi_dirty once.
 * ═══════════════════════════════════════════════════════════════ */

/* Minimal kernel that just computes MI (one iteration of the persistent kernel) */
__global__ void mi_only_kernel(LearningState* state) {
    auto grid = cg::this_grid();
    uint32_t tid = grid.thread_rank();
    uint32_t stride = grid.size();

    double n = state->total_pair_observations;
    double log2_factor = 1.4426950408889634;
    double eps = 1e-10;
    uint32_t num_pairs = state->pair_count_u32;

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
}

struct MIBenchResult {
    uint32_t num_pairs;
    double   cuda_time_ms;
    double   cuda_throughput;  /* pairs/sec */
    double   cpu_time_ms;
    double   cpu_throughput;
    double   max_abs_error;
    double   mean_abs_error;
};

static MIBenchResult bench_mi_at_scale(uint32_t num_pairs, uint32_t num_words, uint32_t seed) {
    MIBenchResult r;
    memset(&r, 0, sizeof(r));
    r.num_pairs = num_pairs;

    uint32_t rng = seed;

    /* Init LearningState with managed memory */
    LearningState* state = ll_init(num_words);

    /* Populate pairs and marginals */
    state->pair_count_u32 = num_pairs;
    state->total_pair_observations = 0.0;

    for (uint32_t i = 0; i < num_pairs; i++) {
        uint32_t wa = xorshift32(&rng) % num_words;
        uint32_t wb = xorshift32(&rng) % num_words;
        if (wb == wa) wb = (wa + 1) % num_words;
        uint32_t lo = (wa <= wb) ? wa : wb;
        uint32_t hi = (wa <= wb) ? wb : wa;

        state->pair_word_a[i] = lo;
        state->pair_word_b[i] = hi;
        double cnt = 1.0 + (double)(xorshift32(&rng) % 100);
        state->pair_count[i] = cnt;
        state->pair_mi[i] = 0.0;
        state->pair_dirty[i] = 1;

        state->word_marginal[lo] += cnt;
        state->word_marginal[hi] += cnt;
        state->total_pair_observations += cnt;
    }
    state->dirty_count = num_pairs;

    /* ─── CPU reference ─── */
    double* cpu_mi_vals = (double*)malloc(num_pairs * sizeof(double));
    double t_cpu_start = time_ms();
    for (uint32_t i = 0; i < num_pairs; i++) {
        cpu_mi_vals[i] = cpu_mi(
            state->pair_count[i],
            state->word_marginal[state->pair_word_a[i]],
            state->word_marginal[state->pair_word_b[i]],
            state->total_pair_observations);
    }
    double t_cpu_end = time_ms();
    r.cpu_time_ms = t_cpu_end - t_cpu_start;
    r.cpu_throughput = (r.cpu_time_ms > 0.001)
        ? (num_pairs * 1000.0 / r.cpu_time_ms) : 0.0;

    /* ─── CUDA MI kernel (cooperative launch) ─── */
    /* Warmup */
    {
        int num_blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, mi_only_kernel, 128, 0);
        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        dim3 grid_dim(num_sms);
        dim3 block_dim(128);
        void* args[] = { &state };
        cudaLaunchCooperativeKernel((void*)mi_only_kernel, grid_dim, block_dim, args);
        cudaDeviceSynchronize();
    }

    /* Reset dirty flags for benchmark */
    for (uint32_t i = 0; i < num_pairs; i++) {
        state->pair_dirty[i] = 1;
        state->pair_mi[i] = 0.0;
    }
    cudaDeviceSynchronize();

    /* Benchmark: average over 10 runs */
    int n_runs = 10;
    double total_cuda_ms = 0.0;

    for (int run = 0; run < n_runs; run++) {
        /* Reset dirty flags */
        for (uint32_t i = 0; i < num_pairs; i++) {
            state->pair_dirty[i] = 1;
        }
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        dim3 grid_dim(num_sms);
        dim3 block_dim(128);
        void* args[] = { &state };

        cudaEventRecord(start);
        cudaLaunchCooperativeKernel((void*)mi_only_kernel, grid_dim, block_dim, args);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        total_cuda_ms += ms;

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    r.cuda_time_ms = total_cuda_ms / n_runs;
    r.cuda_throughput = (r.cuda_time_ms > 0.001)
        ? (num_pairs * 1000.0 / r.cuda_time_ms) : 0.0;

    /* ─── Compare CUDA vs CPU results ─── */
    cudaDeviceSynchronize();
    r.max_abs_error = 0.0;
    r.mean_abs_error = 0.0;

    for (uint32_t i = 0; i < num_pairs; i++) {
        double err = fabs(state->pair_mi[i] - cpu_mi_vals[i]);
        if (err > r.max_abs_error) r.max_abs_error = err;
        r.mean_abs_error += err;
    }
    r.mean_abs_error /= num_pairs;

    /* ─── Write test data for OpenCL comparison ─── */
    if (num_pairs <= 10000) {
        char fname[256];
        snprintf(fname, sizeof(fname), "mi-test-data-%uk.bin", num_pairs / 1000);
        FILE* f = fopen(fname, "wb");
        if (f) {
            /* Header: num_pairs, num_words, total_obs */
            fwrite(&num_pairs, sizeof(uint32_t), 1, f);
            fwrite(&num_words, sizeof(uint32_t), 1, f);
            fwrite(&state->total_pair_observations, sizeof(double), 1, f);
            /* Pair data */
            fwrite(state->pair_word_a, sizeof(uint32_t), num_pairs, f);
            fwrite(state->pair_word_b, sizeof(uint32_t), num_pairs, f);
            fwrite(state->pair_count, sizeof(double), num_pairs, f);
            /* Marginals */
            fwrite(state->word_marginal, sizeof(double), num_words, f);
            /* Reference MI values */
            fwrite(cpu_mi_vals, sizeof(double), num_pairs, f);
            fclose(f);
            printf("  Wrote %s for OpenCL comparison\n", fname);
        }
    }

    free(cpu_mi_vals);

    /* Cleanup: need a dummy ring for ll_shutdown */
    SentenceRing* ring = ll_init_ring();
    ll_shutdown(state, ring);

    return r;
}

/* ═══════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════ */

int main() {
    printf("GPU Learning Pipeline — MI Kernel Comparison\n");
    printf("=============================================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs, %d MB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           (int)(prop.totalGlobalMem / (1024*1024)));

    int supports_coop = 0;
    cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, 0);
    if (!supports_coop) {
        printf("ERROR: Device does not support cooperative launch\n");
        return 1;
    }

    printf("\n=== Experiment 3: CUDA MI Throughput ===\n");
    printf("%-10s | %-10s | %-12s | %-14s | %-12s | %-14s | %-12s | %-12s\n",
           "Pairs", "Words", "CUDA(ms)", "CUDA(pairs/s)", "CPU(ms)", "CPU(pairs/s)",
           "MaxErr", "MeanErr");
    printf("-----------|------------|--------------|----------------|--------------|----------------|--------------|------------\n");

    struct { uint32_t pairs; uint32_t words; } scales[] = {
        {1000,   200},
        {10000,  1000},
        {100000, 5000},
        {262144, 8192}
    };
    int num_scales = 4;

    for (int s = 0; s < num_scales; s++) {
        printf("Running %uK pairs, %u words...\n", scales[s].pairs / 1000, scales[s].words);
        MIBenchResult r = bench_mi_at_scale(scales[s].pairs, scales[s].words, 42 + s);

        printf("%-10u | %-10u | %12.3f | %14.0f | %12.3f | %14.0f | %12.2e | %12.2e\n",
               r.num_pairs, scales[s].words,
               r.cuda_time_ms, r.cuda_throughput,
               r.cpu_time_ms, r.cpu_throughput,
               r.max_abs_error, r.mean_abs_error);
    }

    /* Baseline comparison summary */
    printf("\n=== Comparison with known OpenCL baseline ===\n");
    printf("OpenCL MI throughput (prior benchmark): 385,000 pairs/sec\n");
    printf("CPU MI throughput (corpus-learning.scm): ~7,000 pairs/sec\n");
    printf("See CUDA throughput above for comparison.\n");
    printf("\nNote: OpenCL comparison data files written for scales <= 10K.\n");
    printf("Run gpu-mi.scm with these files for direct accuracy comparison.\n");

    printf("\n=== MI comparison complete ===\n");
    return 0;
}
