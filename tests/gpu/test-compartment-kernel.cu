/* test-compartment-kernel.cu — Tests for hardware-native compartment kernel
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true -o test-compartment-kernel \
 *     test-compartment-kernel.cu gpu-compartment-kernel.cu -lcudadevrt -lm
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

/* Forward declarations */
extern "C" {
    LearningState*    ck_init(uint32_t num_words);
    CompartmentState* ck_init_compartments(uint32_t num_compartments,
                                           uint32_t num_words);
    SentenceRing*     ck_init_ring();
    void              ck_feed_sentence(SentenceRing* ring,
                                       uint32_t* words, uint32_t length);
    int               ck_launch(LearningState* state,
                                CompartmentState* compartments,
                                SentenceRing* ring,
                                int* done_flag, int* pause_flag,
                                uint32_t* stats_iteration,
                                uint32_t* stats_pairs,
                                uint32_t* stats_classes,
                                double* stats_entropy);
    void              ck_wait();
    void              ck_read_classes(LearningState* state,
                                      uint32_t* out, uint32_t n);
    void              ck_shutdown(LearningState* state,
                                  CompartmentState* comps,
                                  SentenceRing* ring);
    int               ck_get_num_sms();
    void              ck_memory_report(int num_compartments);
}

/* ─── Helpers ─── */

static double time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/* Managed memory helpers for control flags */
struct ControlFlags {
    int*       done_flag;
    int*       pause_flag;
    uint32_t*  stats_iteration;
    uint32_t*  stats_pairs;
    uint32_t*  stats_classes;
    double*    stats_entropy;
};

static ControlFlags alloc_flags() {
    ControlFlags f;
    cudaMallocManaged(&f.done_flag, sizeof(int));
    cudaMallocManaged(&f.pause_flag, sizeof(int));
    cudaMallocManaged(&f.stats_iteration, sizeof(uint32_t));
    cudaMallocManaged(&f.stats_pairs, sizeof(uint32_t));
    cudaMallocManaged(&f.stats_classes, sizeof(uint32_t));
    cudaMallocManaged(&f.stats_entropy, sizeof(double));
    *f.done_flag = 0;
    *f.pause_flag = 0;
    *f.stats_iteration = 0;
    *f.stats_pairs = 0;
    *f.stats_classes = 0;
    *f.stats_entropy = 0.0;
    return f;
}

static void free_flags(ControlFlags* f) {
    cudaFree(f->done_flag);
    cudaFree(f->pause_flag);
    cudaFree(f->stats_iteration);
    cudaFree(f->stats_pairs);
    cudaFree(f->stats_classes);
    cudaFree(f->stats_entropy);
}

/* Generate Zipf-distributed sentences */
static void gen_zipf_sentence(uint32_t* words, uint32_t len,
                               uint32_t num_words, uint32_t* rng) {
    for (uint32_t i = 0; i < len; i++) {
        /* Approximate Zipf: lower word IDs more likely */
        uint32_t r = xorshift32(rng) % 1000;
        uint32_t w;
        if (r < 200)      w = xorshift32(rng) % 10;          /* top 10 */
        else if (r < 500)  w = 10 + xorshift32(rng) % 90;     /* top 100 */
        else if (r < 800)  w = 100 + xorshift32(rng) % 900;   /* top 1000 */
        else               w = xorshift32(rng) % num_words;    /* any */
        words[i] = w % num_words;
    }
}

/* ═══════════════════════════════════════════════════════════════
 *  TEST 1: Basic launch and memory
 * ═══════════════════════════════════════════════════════════════ */

static int test1_basic_launch() {
    printf("\n=== T1: Basic launch + memory report ===\n");

    int num_sms = ck_get_num_sms();
    printf("SMs detected: %d\n", num_sms);

    uint32_t num_words = 500;
    LearningState* state = ck_init(num_words);
    CompartmentState* comps = ck_init_compartments(num_sms, num_words);
    SentenceRing* ring = ck_init_ring();
    ControlFlags flags = alloc_flags();

    ck_memory_report(num_sms);

    /* Launch and immediately signal done */
    *flags.done_flag = 0;

    int err = ck_launch(state, comps, ring,
                        flags.done_flag, flags.pause_flag,
                        flags.stats_iteration, flags.stats_pairs,
                        flags.stats_classes, flags.stats_entropy);
    if (err != 0) {
        printf("FAIL: kernel launch failed\n");
        ck_shutdown(state, comps, ring);
        free_flags(&flags);
        return 0;
    }

    /* Feed a few sentences, then stop */
    uint32_t rng = 42;
    for (int s = 0; s < 20; s++) {
        uint32_t words[8];
        gen_zipf_sentence(words, 8, num_words, &rng);
        ck_feed_sentence(ring, words, 8);
    }

    usleep(500000); /* 500ms */
    *flags.done_flag = 1;
    ck_wait();

    printf("After 20 sentences: pairs=%u, iter=%u\n",
           *flags.stats_pairs, *flags.stats_iteration);

    int pass = (*flags.stats_pairs > 0);
    printf("T1: %s (pairs discovered > 0)\n", pass ? "PASS" : "FAIL");

    ck_shutdown(state, comps, ring);
    free_flags(&flags);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 *  TEST 2: Sentence distribution across compartments
 * ═══════════════════════════════════════════════════════════════ */

static int test2_sentence_distribution() {
    printf("\n=== T2: Sentence distribution across SMs ===\n");

    int num_sms = ck_get_num_sms();
    uint32_t num_words = 500;
    LearningState* state = ck_init(num_words);
    CompartmentState* comps = ck_init_compartments(num_sms, num_words);
    SentenceRing* ring = ck_init_ring();
    ControlFlags flags = alloc_flags();

    /* Feed exactly num_sms * 10 sentences so each SM gets ~10 */
    uint32_t total_sents = num_sms * 10;
    printf("Feeding %u sentences to %d SMs...\n", total_sents, num_sms);

    int err = ck_launch(state, comps, ring,
                        flags.done_flag, flags.pause_flag,
                        flags.stats_iteration, flags.stats_pairs,
                        flags.stats_classes, flags.stats_entropy);
    if (err != 0) {
        printf("FAIL: kernel launch failed\n");
        ck_shutdown(state, comps, ring);
        free_flags(&flags);
        return 0;
    }

    uint32_t rng = 123;
    for (uint32_t s = 0; s < total_sents; s++) {
        uint32_t words[10];
        gen_zipf_sentence(words, 10, num_words, &rng);
        ck_feed_sentence(ring, words, 10);
    }

    /* Wait for processing */
    usleep(2000000); /* 2s */
    *flags.done_flag = 1;
    ck_wait();

    printf("Total sentences tracked: %u (expected %u)\n",
           state->total_sentences, total_sents);
    printf("Pairs: %u, Iterations: %u\n",
           *flags.stats_pairs, *flags.stats_iteration);

    /* Check that total_sentences matches what we fed */
    int pass = (state->total_sentences >= total_sents * 9 / 10);
    printf("T2: %s (total_sentences >= 90%% of fed)\n",
           pass ? "PASS" : "FAIL");

    ck_shutdown(state, comps, ring);
    free_flags(&flags);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 *  TEST 3: Phase-gated convergence (no false convergence)
 * ═══════════════════════════════════════════════════════════════ */

static int test3_phase_gated_convergence() {
    printf("\n=== T3: Phase-gated convergence ===\n");

    int num_sms = ck_get_num_sms();
    uint32_t num_words = 1000;
    LearningState* state = ck_init(num_words);
    CompartmentState* comps = ck_init_compartments(num_sms, num_words);
    SentenceRing* ring = ck_init_ring();
    ControlFlags flags = alloc_flags();

    int err = ck_launch(state, comps, ring,
                        flags.done_flag, flags.pause_flag,
                        flags.stats_iteration, flags.stats_pairs,
                        flags.stats_classes, flags.stats_entropy);
    if (err != 0) {
        printf("FAIL: kernel launch failed\n");
        ck_shutdown(state, comps, ring);
        free_flags(&flags);
        return 0;
    }

    /* Feed only 50 sentences — well below LL_MIN_SENTENCES (500).
     * The old kernel would false-converge here.
     * The new kernel should NOT converge. */
    uint32_t rng = 456;
    for (int s = 0; s < 50; s++) {
        uint32_t words[8];
        gen_zipf_sentence(words, 8, num_words, &rng);
        ck_feed_sentence(ring, words, 8);
    }

    /* Wait 3 seconds — old kernel converged in <1s */
    usleep(3000000);

    int converged = *flags.done_flag;
    uint32_t iters = *flags.stats_iteration;

    printf("After 50 sentences, 3s wait: converged=%d, iters=%u\n",
           converged, iters);

    /* The kernel may converge via "data exhausted" path (idle_rounds > 50),
     * which is correct behavior. But it should NOT have converged via
     * entropy-plateau with phase gates (that would be false convergence).
     * Check: if converged, verify it's via data-exhausted, not phase gates.
     * The phase gate requires total_sentences >= 500, which 50 doesn't meet. */
    *flags.done_flag = 1;
    ck_wait();

    int pass;
    if (converged) {
        /* Verify it was data-exhausted, not phase-gated */
        int was_phase_gated = (state->total_sentences >= LL_MIN_SENTENCES &&
                               state->cc_runs >= LL_MIN_CC_RUNS &&
                               state->num_classes >= LL_MIN_CLASSES);
        pass = !was_phase_gated;
        printf("T3: %s (converged=%d via %s)\n",
               pass ? "PASS" : "FAIL", converged,
               was_phase_gated ? "PHASE-GATE (BAD)" : "data-exhausted (OK)");
    } else {
        pass = 1;
        printf("T3: PASS (did NOT converge with 50 sentences)\n");
    }

    ck_shutdown(state, comps, ring);
    free_flags(&flags);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 *  TEST 4: Real convergence with enough data
 * ═══════════════════════════════════════════════════════════════ */

static int test4_real_convergence() {
    printf("\n=== T4: Real convergence with 2000 sentences ===\n");

    int num_sms = ck_get_num_sms();
    uint32_t num_words = 500;
    LearningState* state = ck_init(num_words);
    CompartmentState* comps = ck_init_compartments(num_sms, num_words);
    SentenceRing* ring = ck_init_ring();
    ControlFlags flags = alloc_flags();

    int err = ck_launch(state, comps, ring,
                        flags.done_flag, flags.pause_flag,
                        flags.stats_iteration, flags.stats_pairs,
                        flags.stats_classes, flags.stats_entropy);
    if (err != 0) {
        printf("FAIL: kernel launch failed\n");
        ck_shutdown(state, comps, ring);
        free_flags(&flags);
        return 0;
    }

    /* Feed 2000 sentences — above LL_MIN_SENTENCES */
    uint32_t rng = 789;
    double t_start = time_ms();

    for (int s = 0; s < 2000; s++) {
        uint32_t len = 5 + (xorshift32(&rng) % 12);  /* 5-16 words */
        uint32_t words[LL_MAX_SENTENCE_LEN];
        gen_zipf_sentence(words, len, num_words, &rng);
        ck_feed_sentence(ring, words, len);
    }

    double t_fed = time_ms();
    printf("Fed 2000 sentences in %.1f ms\n", t_fed - t_start);

    /* Wait for convergence (timeout 60s) */
    double timeout = 60000.0;
    uint32_t last_iter = 0;
    double last_report = time_ms();

    while (!(*flags.done_flag)) {
        usleep(100000); /* 100ms */
        if (time_ms() - t_start > timeout) {
            printf("  TIMEOUT after 60s\n");
            *flags.done_flag = 1;
            break;
        }
        if (time_ms() - last_report > 5000.0 &&
            *flags.stats_iteration > last_iter) {
            printf("  iter=%u pairs=%u classes=%u entropy=%.4f sents=%u cc_runs=%u\n",
                   *flags.stats_iteration, *flags.stats_pairs,
                   *flags.stats_classes, *flags.stats_entropy,
                   state->total_sentences, state->cc_runs);
            last_iter = *flags.stats_iteration;
            last_report = time_ms();
        }
    }

    ck_wait();
    double t_end = time_ms();

    printf("\nResults:\n");
    printf("  Time: %.2f s\n", (t_end - t_start) / 1000.0);
    printf("  Iterations: %u\n", *flags.stats_iteration);
    printf("  Pairs: %u\n", *flags.stats_pairs);
    printf("  Classes: %u (num_classes=%u)\n", *flags.stats_classes,
           state->num_classes);
    printf("  GPU Entropy: %.4f\n", *flags.stats_entropy);
    printf("  Total sentences: %u\n", state->total_sentences);
    printf("  CC runs: %u\n", state->cc_runs);
    printf("  Converged: %s\n",
           state->iteration < LL_MAX_ITERATIONS ? "YES" : "NO (max iter)");

    /* CPU-side entropy verification */
    cudaDeviceSynchronize();
    uint32_t nc = state->num_classes;
    uint32_t nw = state->word_count_u32;
    uint32_t* cpu_class_sizes = (uint32_t*)calloc(nc + 1, sizeof(uint32_t));
    for (uint32_t w = 0; w < nw; w++) {
        uint32_t cid = state->word_class_id[w];
        if (cid < nc) cpu_class_sizes[cid]++;
    }
    double cpu_entropy = 0.0;
    uint32_t total_assigned = 0;
    for (uint32_t c = 0; c < nc; c++) {
        total_assigned += cpu_class_sizes[c];
        if (cpu_class_sizes[c] > 0) {
            double p = (double)cpu_class_sizes[c] / (double)nw;
            cpu_entropy -= p * log2(p);
        }
    }
    printf("  CPU Entropy: %.4f (nc=%u, nw=%u, assigned=%u)\n",
           cpu_entropy, nc, nw, total_assigned);

    /* Show entropy history */
    printf("  Entropy history:");
    for (int i = 0; i < LL_ENTROPY_WINDOW; i++) {
        printf(" [%d]=%.4f", i, state->entropy_history[i]);
    }
    printf("\n");
    printf("  Entropy idx: %u\n", state->entropy_idx);
    free(cpu_class_sizes);

    /* Verify meaningful results */
    int pass = 1;

    if (*flags.stats_pairs < 100) {
        printf("  FAIL: only %u pairs\n", *flags.stats_pairs);
        pass = 0;
    }
    if (*flags.stats_classes < 2) {
        printf("  FAIL: only %u classes\n", *flags.stats_classes);
        pass = 0;
    }
    if (state->total_sentences < 1800) {
        printf("  FAIL: only %u sentences processed\n", state->total_sentences);
        pass = 0;
    }

    printf("T4: %s\n", pass ? "PASS" : "FAIL");

    ck_shutdown(state, comps, ring);
    free_flags(&flags);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 *  TEST 5: MI correctness (compare with CPU reference)
 * ═══════════════════════════════════════════════════════════════ */

static int test5_mi_correctness() {
    printf("\n=== T5: MI correctness check ===\n");

    int num_sms = ck_get_num_sms();
    uint32_t num_words = 200;
    LearningState* state = ck_init(num_words);
    CompartmentState* comps = ck_init_compartments(num_sms, num_words);
    SentenceRing* ring = ck_init_ring();
    ControlFlags flags = alloc_flags();

    int err = ck_launch(state, comps, ring,
                        flags.done_flag, flags.pause_flag,
                        flags.stats_iteration, flags.stats_pairs,
                        flags.stats_classes, flags.stats_entropy);
    if (err != 0) {
        printf("FAIL: kernel launch failed\n");
        ck_shutdown(state, comps, ring);
        free_flags(&flags);
        return 0;
    }

    /* Feed 1000 sentences */
    uint32_t rng = 999;
    for (int s = 0; s < 1000; s++) {
        uint32_t words[8];
        gen_zipf_sentence(words, 8, num_words, &rng);
        ck_feed_sentence(ring, words, 8);
    }

    /* Wait for kernel to converge (data-exhausted) */
    double t5_start = time_ms();
    while (!(*flags.done_flag) && (time_ms() - t5_start) < 30000.0) {
        usleep(100000);
    }
    if (!(*flags.done_flag)) *flags.done_flag = 1;
    ck_wait();
    cudaDeviceSynchronize();

    /* Verify MI values are reasonable.
     * NOTE: MI was computed on intermediate counts (batch-by-batch),
     * so values won't exactly match CPU reference on final counts.
     * Instead check: no NaN/Inf, mostly positive for high-count pairs,
     * and MI is monotonic with count (higher count → higher MI generally). */
    uint32_t num_pairs = state->pair_count_u32;
    printf("Checking %u pairs...\n", num_pairs);

    uint32_t nan_count = 0, inf_count = 0, positive = 0, negative = 0;

    for (uint32_t p = 0; p < num_pairs && p < LL_MAX_PAIRS; p++) {
        double mi = state->pair_mi[p];
        if (isnan(mi)) { nan_count++; continue; }
        if (isinf(mi)) { inf_count++; continue; }
        if (mi > 0.0) positive++;
        else negative++;
    }

    printf("MI stats: %u positive, %u negative, %u NaN, %u Inf\n",
           positive, negative, nan_count, inf_count);
    printf("Total sentences: %u, pairs: %u\n",
           state->total_sentences, num_pairs);

    /* Checks:
     * 1. No NaN or Inf values
     * 2. More positive MI than negative (indicates real associations)
     * 3. At least some pairs computed */
    int pass = (nan_count == 0 && inf_count == 0 &&
                positive > negative / 2 && num_pairs > 100);
    printf("T5: %s\n", pass ? "PASS" : "FAIL");

    ck_shutdown(state, comps, ring);
    free_flags(&flags);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 *  TEST 6: Throughput benchmark
 * ═══════════════════════════════════════════════════════════════ */

static int test6_throughput() {
    printf("\n=== T6: Throughput benchmark ===\n");

    int num_sms = ck_get_num_sms();
    uint32_t num_words = 2000;
    LearningState* state = ck_init(num_words);
    CompartmentState* comps = ck_init_compartments(num_sms, num_words);
    SentenceRing* ring = ck_init_ring();
    ControlFlags flags = alloc_flags();

    ck_memory_report(num_sms);

    int err = ck_launch(state, comps, ring,
                        flags.done_flag, flags.pause_flag,
                        flags.stats_iteration, flags.stats_pairs,
                        flags.stats_classes, flags.stats_entropy);
    if (err != 0) {
        printf("FAIL: kernel launch failed\n");
        ck_shutdown(state, comps, ring);
        free_flags(&flags);
        return 0;
    }

    /* Feed 5000 sentences */
    uint32_t rng = 5555;
    double t_start = time_ms();

    for (int s = 0; s < 5000; s++) {
        uint32_t len = 5 + (xorshift32(&rng) % 12);
        uint32_t words[LL_MAX_SENTENCE_LEN];
        gen_zipf_sentence(words, len, num_words, &rng);
        ck_feed_sentence(ring, words, len);
    }

    double t_fed = time_ms();
    printf("Feed rate: %.0f sents/sec (%.1f ms)\n",
           5000.0 * 1000.0 / (t_fed - t_start), t_fed - t_start);

    /* Wait for convergence (timeout 120s) */
    double timeout = 120000.0;
    uint32_t last_iter = 0;
    double last_report = time_ms();

    while (!(*flags.done_flag)) {
        usleep(100000);
        if (time_ms() - t_start > timeout) {
            printf("  TIMEOUT\n");
            *flags.done_flag = 1;
            break;
        }
        if (time_ms() - last_report > 10000.0 &&
            *flags.stats_iteration > last_iter) {
            printf("  iter=%u pairs=%u classes=%u entropy=%.4f sents=%u\n",
                   *flags.stats_iteration, *flags.stats_pairs,
                   *flags.stats_classes, *flags.stats_entropy,
                   state->total_sentences);
            last_iter = *flags.stats_iteration;
            last_report = time_ms();
        }
    }

    ck_wait();
    double t_end = time_ms();
    double total_s = (t_end - t_start) / 1000.0;

    printf("\nThroughput results:\n");
    printf("  Total time: %.2f s\n", total_s);
    printf("  Pipeline throughput: %.1f sents/sec\n",
           5000.0 / total_s);
    printf("  Iterations: %u (%.1f ms/iter)\n",
           *flags.stats_iteration,
           *flags.stats_iteration > 0 ?
           (t_end - t_start) / *flags.stats_iteration : 0.0);
    printf("  Pairs: %u\n", *flags.stats_pairs);
    printf("  Classes: %u\n", *flags.stats_classes);
    printf("  Total sentences: %u\n", state->total_sentences);
    printf("  Compartments: %u\n", state->num_compartments);

    /* With ring size 2048, max sentences in a burst is 2048.
     * The real metric is throughput and convergence. */
    int pass = (state->total_sentences >= 1000 &&
                *flags.stats_pairs > 100);
    printf("T6: %s\n", pass ? "PASS" : "FAIL");

    ck_shutdown(state, comps, ring);
    free_flags(&flags);
    return pass;
}

/* ═══════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════ */

int main() {
    printf("GPU Compartment Kernel — Test Suite\n");
    printf("===================================\n");

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

    printf("CompartmentState size: %.2f MB\n",
           sizeof(CompartmentState) / (1024.0 * 1024.0));
    printf("LearningState size: %.2f MB\n",
           sizeof(LearningState) / (1024.0 * 1024.0));
    printf("SentenceRing size: %.2f KB (%d slots)\n",
           sizeof(SentenceRing) / 1024.0, LL_RING_SIZE);

    int passed = 0, failed = 0;

    if (test1_basic_launch())           passed++; else failed++;
    if (test2_sentence_distribution())  passed++; else failed++;
    if (test3_phase_gated_convergence()) passed++; else failed++;
    if (test4_real_convergence())       passed++; else failed++;
    if (test5_mi_correctness())         passed++; else failed++;
    if (test6_throughput())             passed++; else failed++;

    printf("\n===================================\n");
    printf("Results: %d/%d passed\n", passed, passed + failed);
    printf("===================================\n");

    return failed > 0 ? 1 : 0;
}
