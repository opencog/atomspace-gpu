/* bench-pipeline.cu — End-to-end persistent kernel benchmark
 *
 * Tests the full learning pipeline at realistic scales:
 *   P1: Small  (500 words, 200 sentences)
 *   P2: Medium (2K words, 1K sentences)
 *   P3: Full   (5K words, 5K sentences)
 *   P4: Max    (8K words, 10K sentences)
 *
 * Also measures (Experiments 5 & 6):
 *   - Sentence feeding rate (CPU→GPU bottleneck)
 *   - GPU memory profile (cudaMemGetInfo)
 *   - Ring buffer utilization
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true -o bench-pipeline \
 *     bench-pipeline.cu gpu-learning-loop.cu -lcudadevrt -lm
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <unistd.h>
#include <cuda_runtime.h>
#include "gpu-learning-types.h"

/* ─── Forward declarations ─── */

extern "C" {
    LearningState* ll_init(uint32_t num_words);
    SentenceRing*  ll_init_ring();
    void           ll_feed_sentence(SentenceRing* ring, uint32_t* words, uint32_t length);
    int            ll_launch(LearningState* state, SentenceRing* ring,
                             int* done_flag, int* pause_flag,
                             uint32_t* stats_iteration, uint32_t* stats_pairs,
                             uint32_t* stats_classes, double* stats_entropy);
    void           ll_wait();
    void           ll_read_classes(LearningState* state, uint32_t* out, uint32_t n);
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

/* Generate synthetic sentences with Zipf-like word distribution */
static void gen_sentences(
    uint32_t** out_sentences,  /* array of sentence word arrays */
    uint32_t*  out_lengths,    /* sentence lengths */
    uint32_t   num_sentences,
    uint32_t   num_words,
    uint32_t   seed
) {
    uint32_t rng = seed;

    /* Zipf distribution: word i has frequency proportional to 1/(i+1) */
    for (uint32_t s = 0; s < num_sentences; s++) {
        /* Sentence length: 5 to 40 words (realistic) */
        uint32_t len = 5 + (xorshift32(&rng) % 36);
        if (len > LL_MAX_SENTENCE_LEN) len = LL_MAX_SENTENCE_LEN;

        out_sentences[s] = (uint32_t*)malloc(len * sizeof(uint32_t));
        out_lengths[s] = len;

        for (uint32_t w = 0; w < len; w++) {
            /* Zipf: higher probability for lower-indexed words */
            double u = (double)(xorshift32(&rng) % 10000) / 10000.0;
            uint32_t idx = (uint32_t)(num_words * pow(u, 2.0));
            if (idx >= num_words) idx = num_words - 1;
            out_sentences[s][w] = idx;
        }
    }
}

static void free_sentences(uint32_t** sents, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) free(sents[i]);
    free(sents);
}

/* ─── GPU memory reporting ─── */

static void report_gpu_memory(const char* label) {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  GPU memory [%s]: %.1f MB used / %.1f MB total (%.1f MB free)\n",
           label, (total_mem - free_mem) / (1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0),
           free_mem / (1024.0 * 1024.0));
}

/* ═══════════════════════════════════════════════════════════════
 *  PIPELINE BENCHMARK
 * ═══════════════════════════════════════════════════════════════ */

struct PipelineResult {
    uint32_t num_words;
    uint32_t num_sentences;
    double   total_time_ms;
    uint32_t iterations;
    uint32_t pairs;
    uint32_t classes;
    double   entropy;
    double   sents_per_sec;
    int      converged;
    double   feeding_time_ms;
    double   gpu_used_mb;
};

static PipelineResult run_pipeline(
    uint32_t num_words,
    uint32_t num_sentences,
    uint32_t seed
) {
    PipelineResult result;
    memset(&result, 0, sizeof(result));
    result.num_words = num_words;
    result.num_sentences = num_sentences;

    /* Generate sentences */
    uint32_t** sentences = (uint32_t**)malloc(num_sentences * sizeof(uint32_t*));
    uint32_t* lengths = (uint32_t*)malloc(num_sentences * sizeof(uint32_t));
    gen_sentences(sentences, lengths, num_sentences, num_words, seed);

    /* Measure memory before init */
    size_t mem_before_free, mem_before_total;
    cudaMemGetInfo(&mem_before_free, &mem_before_total);

    /* Init */
    LearningState* state = ll_init(num_words);
    SentenceRing* ring = ll_init_ring();

    int* done_flag;
    int* pause_flag;
    uint32_t* stats_iteration;
    uint32_t* stats_pairs;
    uint32_t* stats_classes;
    double* stats_entropy;

    cudaMallocManaged(&done_flag, sizeof(int));
    cudaMallocManaged(&pause_flag, sizeof(int));
    cudaMallocManaged(&stats_iteration, sizeof(uint32_t));
    cudaMallocManaged(&stats_pairs, sizeof(uint32_t));
    cudaMallocManaged(&stats_classes, sizeof(uint32_t));
    cudaMallocManaged(&stats_entropy, sizeof(double));

    *done_flag = 0;
    *pause_flag = 0;
    *stats_iteration = 0;
    *stats_pairs = 0;
    *stats_classes = 0;
    *stats_entropy = 0.0;

    /* Measure memory after init */
    size_t mem_after_free, mem_after_total;
    cudaMemGetInfo(&mem_after_free, &mem_after_total);
    result.gpu_used_mb = (double)(mem_before_free - mem_after_free) / (1024.0 * 1024.0);

    /* Launch persistent kernel */
    double t_start = time_ms();

    int err = ll_launch(state, ring, done_flag, pause_flag,
                        stats_iteration, stats_pairs, stats_classes, stats_entropy);
    if (err != 0) {
        printf("  FAILED to launch kernel!\n");
        ll_shutdown(state, ring);
        free_sentences(sentences, num_sentences);
        free(lengths);
        cudaFree(done_flag); cudaFree(pause_flag);
        cudaFree(stats_iteration); cudaFree(stats_pairs);
        cudaFree(stats_classes); cudaFree(stats_entropy);
        return result;
    }

    /* Feed sentences */
    double t_feed_start = time_ms();
    for (uint32_t s = 0; s < num_sentences; s++) {
        ll_feed_sentence(ring, sentences[s], lengths[s]);
    }
    double t_feed_end = time_ms();
    result.feeding_time_ms = t_feed_end - t_feed_start;

    /* Wait for convergence (or timeout after 60 seconds) */
    double timeout = 60000.0; /* 60 seconds */
    uint32_t last_iter = 0;
    double last_check = time_ms();

    while (!(*done_flag)) {
        usleep(10000); /* 10ms poll interval */

        if (time_ms() - t_start > timeout) {
            printf("  TIMEOUT after %.1f seconds\n", timeout / 1000.0);
            *done_flag = 1;
            break;
        }

        /* Progress reporting (every 2 seconds) */
        if (time_ms() - last_check > 2000.0 && *stats_iteration > last_iter) {
            printf("  ... iter=%u pairs=%u classes=%u entropy=%.3f\n",
                   *stats_iteration, *stats_pairs, *stats_classes, *stats_entropy);
            last_iter = *stats_iteration;
            last_check = time_ms();
        }
    }

    ll_wait();
    double t_end = time_ms();

    result.total_time_ms = t_end - t_start;
    result.iterations = *stats_iteration;
    result.pairs = *stats_pairs;
    result.classes = *stats_classes;
    result.entropy = *stats_entropy;
    result.sents_per_sec = (result.total_time_ms > 0.001)
        ? (num_sentences * 1000.0 / result.total_time_ms) : 0.0;
    result.converged = (*stats_iteration < LL_MAX_ITERATIONS);

    /* Cleanup */
    ll_shutdown(state, ring);
    free_sentences(sentences, num_sentences);
    free(lengths);
    cudaFree(done_flag); cudaFree(pause_flag);
    cudaFree(stats_iteration); cudaFree(stats_pairs);
    cudaFree(stats_classes); cudaFree(stats_entropy);

    return result;
}

/* ═══════════════════════════════════════════════════════════════
 *  EXPERIMENT 5: SENTENCE FEEDING RATE
 * ═══════════════════════════════════════════════════════════════ */

static void bench_feeding_rate() {
    printf("\n=== Experiment 5: Sentence Feeding Rate ===\n");
    printf("%-10s | %-10s | %-12s | %-14s\n",
           "Sentences", "AvgLen", "Feed(ms)", "Feed(sents/s)");
    printf("-----------|------------|--------------|----------------\n");

    uint32_t counts[] = {100, 1000, 5000, 10000};
    int n_counts = 4;

    for (int c = 0; c < n_counts; c++) {
        uint32_t ns = counts[c];
        uint32_t nw = 5000;

        /* Generate sentences */
        uint32_t** sentences = (uint32_t**)malloc(ns * sizeof(uint32_t*));
        uint32_t* lengths = (uint32_t*)malloc(ns * sizeof(uint32_t));
        gen_sentences(sentences, lengths, ns, nw, 555 + c);

        /* Compute average length */
        double avg_len = 0;
        for (uint32_t i = 0; i < ns; i++) avg_len += lengths[i];
        avg_len /= ns;

        /* Init ring only (no kernel — measure pure feeding overhead) */
        SentenceRing* ring = ll_init_ring();

        /* Mark all slots as consumed (ring starts empty) */
        for (int i = 0; i < LL_RING_SIZE; i++) {
            ring->slots[i].ready = 0;
        }

        /* We can't feed without a consumer, so just measure the memcpy cost */
        double t_start = time_ms();
        for (uint32_t s = 0; s < ns; s++) {
            /* Simulate feed: copy to slot, mark ready, advance */
            uint32_t widx = s % LL_RING_SIZE;
            SentenceSlot* slot = &ring->slots[widx];
            memcpy(slot->words, sentences[s], lengths[s] * sizeof(uint32_t));
            slot->length = lengths[s];
            __sync_synchronize();
            slot->ready = 1;
            /* Immediately mark consumed so next iteration works */
            slot->ready = 0;
        }
        double t_end = time_ms();

        double feed_ms = t_end - t_start;
        double sps = (feed_ms > 0.001) ? (ns * 1000.0 / feed_ms) : 0.0;

        printf("%-10u | %-10.1f | %12.3f | %14.0f\n",
               ns, avg_len, feed_ms, sps);

        free_sentences(sentences, ns);
        free(lengths);
        cudaFree(ring);
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  EXPERIMENT 6: MEMORY PROFILE
 * ═══════════════════════════════════════════════════════════════ */

static void bench_memory() {
    printf("\n=== Experiment 6: GPU Memory Profile ===\n");

    report_gpu_memory("baseline");

    /* Allocate LearningState */
    LearningState* state = ll_init(8192);
    report_gpu_memory("after LearningState init");

    SentenceRing* ring = ll_init_ring();
    report_gpu_memory("after SentenceRing init");

    /* Control flags */
    int* done_flag;
    int* pause_flag;
    uint32_t* si; uint32_t* sp; uint32_t* sc;
    double* se;
    cudaMallocManaged(&done_flag, sizeof(int));
    cudaMallocManaged(&pause_flag, sizeof(int));
    cudaMallocManaged(&si, sizeof(uint32_t));
    cudaMallocManaged(&sp, sizeof(uint32_t));
    cudaMallocManaged(&sc, sizeof(uint32_t));
    cudaMallocManaged(&se, sizeof(double));
    report_gpu_memory("after control flags");

    /* Computed struct sizes */
    printf("\n  Computed struct sizes:\n");
    printf("    sizeof(LearningState) = %zu bytes (%.2f MB)\n",
           sizeof(LearningState), sizeof(LearningState) / (1024.0 * 1024.0));
    printf("    sizeof(SentenceRing)  = %zu bytes (%.2f KB)\n",
           sizeof(SentenceRing), sizeof(SentenceRing) / 1024.0);
    printf("    sizeof(SentenceSlot)  = %zu bytes\n", sizeof(SentenceSlot));

    /* Max vocabulary extrapolation */
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("\n  Available GPU memory: %.1f MB\n", free_mem / (1024.0 * 1024.0));
    printf("  LearningState overhead: %.2f MB\n", sizeof(LearningState) / (1024.0 * 1024.0));

    /* Cleanup */
    ll_shutdown(state, ring);
    cudaFree(done_flag); cudaFree(pause_flag);
    cudaFree(si); cudaFree(sp); cudaFree(sc); cudaFree(se);
}

/* ═══════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════ */

int main() {
    printf("GPU Learning Pipeline — Integration Benchmarks\n");
    printf("===============================================\n");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs, %d MB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           (int)(prop.totalGlobalMem / (1024*1024)));

    int supports_coop = 0;
    cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, 0);
    if (!supports_coop) {
        printf("ERROR: Device does not support cooperative launch (required for persistent kernel)\n");
        return 1;
    }

    /* Run memory profile first (before any pipeline allocation) */
    bench_memory();

    /* Pipeline benchmarks */
    struct {
        const char* name;
        uint32_t words;
        uint32_t sentences;
    } scales[] = {
        {"P1: Small",   500,  200},
        {"P2: Medium", 2000, 1000},
        {"P3: Full",   5000, 5000},
        {"P4: Max",    8000, 10000}
    };
    int num_scales = 4;

    printf("\n=== Experiment 2: Pipeline Integration Benchmarks ===\n");
    printf("%-12s | %-6s | %-8s | %-8s | %-8s | %-6s | %-8s | %-10s | %-8s | %-8s\n",
           "Test", "Words", "Sents", "Time(s)", "Iters", "Pairs", "Classes", "Entropy",
           "S/sec", "GPU(MB)");
    printf("-------------|--------|----------|----------|----------|--------|----------|------------|----------|--------\n");

    for (int s = 0; s < num_scales; s++) {
        printf("\nRunning %s (%u words, %u sentences)...\n",
               scales[s].name, scales[s].words, scales[s].sentences);

        PipelineResult r = run_pipeline(scales[s].words, scales[s].sentences, 1000 + s);

        printf("%-12s | %-6u | %-8u | %8.2f | %-8u | %-6u | %-8u | %10.4f | %8.1f | %8.1f\n",
               scales[s].name, r.num_words, r.num_sentences,
               r.total_time_ms / 1000.0, r.iterations, r.pairs, r.classes,
               r.entropy, r.sents_per_sec, r.gpu_used_mb);
        printf("  Feed time: %.1f ms (%.0f sents/sec feeding rate)\n",
               r.feeding_time_ms,
               r.feeding_time_ms > 0.001 ? r.num_sentences * 1000.0 / r.feeding_time_ms : 0.0);
        printf("  Converged: %s, Iterations: %u\n",
               r.converged ? "YES" : "NO (hit max)", r.iterations);
    }

    /* Feeding rate benchmark */
    bench_feeding_rate();

    printf("\n=== All pipeline benchmarks complete ===\n");
    return 0;
}
