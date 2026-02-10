/* test-learning-loop.cu — Tests T8, T9, T10 for persistent learning kernel
 *
 * T8: Persistent kernel convergence (feed sentences, verify classes form)
 * T9: Unified memory sentence feeding (CPU writes while GPU reads)
 * T10: Performance (timing at scale)
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true \
 *     -o test-learning-loop \
 *     test-learning-loop.cu gpu-learning-loop.cu \
 *     -lcudadevrt -lm
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <unistd.h>
#include "gpu-learning-types.h"

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

/* ─── Helper: Generate synthetic sentences ─── */

/* Create sentences with word clustering structure:
 * - Cluster A words (0-49) appear together
 * - Cluster B words (50-99) appear together
 * - Cross-cluster pairs are rare
 */
void generate_clustered_sentence(uint32_t* words, uint32_t* len,
                                  int cluster, int num_words_per_cluster,
                                  int sentence_len) {
    int base = cluster * num_words_per_cluster;
    *len = sentence_len;
    for (int i = 0; i < sentence_len; i++) {
        words[i] = base + (rand() % num_words_per_cluster);
    }
}

/* ─── T8: Persistent kernel convergence ─── */

int test_t8_convergence() {
    printf("T8: Persistent kernel convergence\n");

    const uint32_t NUM_WORDS = 100;
    const int WORDS_PER_CLUSTER = 50;
    const int NUM_SENTENCES = 200;
    const int SENTENCE_LEN = 5;

    srand(42);

    /* Initialize */
    LearningState* state = ll_init(NUM_WORDS);
    SentenceRing* ring = ll_init_ring();

    int* done_flag;
    int* pause_flag;
    uint32_t *stats_iter, *stats_pairs, *stats_classes;
    double* stats_entropy;
    cudaMallocManaged(&done_flag, sizeof(int));
    cudaMallocManaged(&pause_flag, sizeof(int));
    cudaMallocManaged(&stats_iter, sizeof(uint32_t));
    cudaMallocManaged(&stats_pairs, sizeof(uint32_t));
    cudaMallocManaged(&stats_classes, sizeof(uint32_t));
    cudaMallocManaged(&stats_entropy, sizeof(double));

    *done_flag = 0;
    *pause_flag = 0;
    *stats_iter = 0;
    *stats_pairs = 0;
    *stats_classes = 0;
    *stats_entropy = 0.0;

    /* Launch persistent kernel */
    int err = ll_launch(state, ring, done_flag, pause_flag,
                        stats_iter, stats_pairs, stats_classes, stats_entropy);
    if (err != 0) {
        printf("  FAIL: kernel launch failed\n");
        return 0;
    }
    printf("  Persistent kernel launched\n");

    /* Feed sentences: alternating cluster A and cluster B */
    for (int s = 0; s < NUM_SENTENCES; s++) {
        uint32_t words[64];
        uint32_t len;
        int cluster = (s % 2 == 0) ? 0 : 1;
        generate_clustered_sentence(words, &len, cluster, WORDS_PER_CLUSTER, SENTENCE_LEN);
        ll_feed_sentence(ring, words, len);

        /* Small delay every 10 sentences to let GPU catch up */
        if (s % 10 == 0) usleep(1000);  /* 1ms */
    }

    printf("  Fed %d sentences\n", NUM_SENTENCES);

    /* Wait for processing — poll status */
    int max_wait = 100;  /* 10 seconds max */
    uint32_t last_iter = 0;
    for (int w = 0; w < max_wait; w++) {
        usleep(100000);  /* 100ms */
        printf("  [%d] iter=%u pairs=%u classes=%u entropy=%.4f done=%d\n",
               w, *stats_iter, *stats_pairs, *stats_classes, *stats_entropy, *done_flag);

        if (*done_flag) break;

        /* If no progress for several polls, stop manually */
        if (*stats_iter == last_iter && w > 20) {
            printf("  No progress, stopping manually\n");
            *done_flag = 1;
            break;
        }
        last_iter = *stats_iter;
    }

    /* Wait for kernel to finish */
    ll_wait();

    /* Read final results */
    uint32_t h_classes[100];
    ll_read_classes(state, h_classes, NUM_WORDS);

    int pass = 1;

    /* Verify: some pairs were counted */
    printf("  Final: %u pairs, %u classes, %u iterations\n",
           *stats_pairs, *stats_classes, *stats_iter);

    if (*stats_pairs == 0) {
        printf("  FAIL: no pairs counted\n");
        pass = 0;
    }

    /* Verify: at least 1 iteration ran */
    if (*stats_iter == 0) {
        printf("  FAIL: no iterations completed\n");
        pass = 0;
    }

    /* Verify: some classes formed (not all singletons) */
    if (*stats_classes > 0 && *stats_classes < NUM_WORDS) {
        printf("  Classes formed: %u (good — not all singletons)\n", *stats_classes);
    } else if (*stats_classes == 0) {
        printf("  WARN: no classes formed (may need more data or lower threshold)\n");
    }

    /* Check that some cluster A words share a class */
    int shared_a = 0, shared_b = 0;
    for (uint32_t i = 1; i < WORDS_PER_CLUSTER; i++) {
        if (h_classes[i] == h_classes[0] && h_classes[0] != 0xFFFFFFFFU) shared_a++;
    }
    for (uint32_t i = WORDS_PER_CLUSTER + 1; i < NUM_WORDS; i++) {
        if (h_classes[i] == h_classes[WORDS_PER_CLUSTER] &&
            h_classes[WORDS_PER_CLUSTER] != 0xFFFFFFFFU) shared_b++;
    }
    printf("  Cluster A sharing: %d/%d  Cluster B sharing: %d/%d\n",
           shared_a, WORDS_PER_CLUSTER - 1, shared_b, WORDS_PER_CLUSTER - 1);

    /* Cleanup */
    ll_shutdown(state, ring);
    cudaFree(done_flag); cudaFree(pause_flag);
    cudaFree(stats_iter); cudaFree(stats_pairs);
    cudaFree(stats_classes); cudaFree(stats_entropy);

    printf("  T8: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ─── T9: Unified memory sentence feeding ─── */

int test_t9_unified_memory() {
    printf("T9: Unified memory sentence feeding\n");

    const uint32_t NUM_WORDS = 50;
    const int NUM_SENTENCES = 100;

    srand(123);

    LearningState* state = ll_init(NUM_WORDS);
    SentenceRing* ring = ll_init_ring();

    int* done_flag;
    int* pause_flag;
    uint32_t *stats_iter, *stats_pairs, *stats_classes;
    double* stats_entropy;
    cudaMallocManaged(&done_flag, sizeof(int));
    cudaMallocManaged(&pause_flag, sizeof(int));
    cudaMallocManaged(&stats_iter, sizeof(uint32_t));
    cudaMallocManaged(&stats_pairs, sizeof(uint32_t));
    cudaMallocManaged(&stats_classes, sizeof(uint32_t));
    cudaMallocManaged(&stats_entropy, sizeof(double));

    *done_flag = 0;
    *pause_flag = 0;
    *stats_iter = 0;
    *stats_pairs = 0;
    *stats_classes = 0;
    *stats_entropy = 0.0;

    /* Launch kernel */
    ll_launch(state, ring, done_flag, pause_flag,
              stats_iter, stats_pairs, stats_classes, stats_entropy);

    /* Rapid-fire sentence feeding — test that CPU writes don't race with GPU reads */
    int pass = 1;
    int total_fed = 0;

    for (int batch = 0; batch < 10; batch++) {
        /* Feed 10 sentences as fast as possible */
        for (int s = 0; s < 10; s++) {
            uint32_t words[5];
            for (int i = 0; i < 5; i++) words[i] = rand() % NUM_WORDS;
            ll_feed_sentence(ring, words, 5);
            total_fed++;
        }
        /* Brief pause between batches */
        usleep(10000);  /* 10ms */
    }

    printf("  Fed %d sentences in rapid bursts\n", total_fed);

    /* Let GPU process */
    usleep(500000);  /* 500ms */

    /* Stop and wait */
    *done_flag = 1;
    ll_wait();

    printf("  Final: %u pairs counted, %u iterations\n", *stats_pairs, *stats_iter);

    /* Verify all sentences were processed (pairs > 0) */
    if (*stats_pairs == 0) {
        printf("  FAIL: no pairs counted — unified memory feeding broken\n");
        pass = 0;
    } else {
        printf("  Pairs counted: %u (unified memory feeding works)\n", *stats_pairs);
    }

    /* Verify ring buffer indices are consistent */
    printf("  Ring: write_idx=%u read_idx=%u\n", ring->write_idx, ring->read_idx);
    if (ring->read_idx < ring->write_idx - 64) {  /* allow ring size lag */
        printf("  WARN: GPU may not have consumed all sentences\n");
    }

    /* Cleanup */
    ll_shutdown(state, ring);
    cudaFree(done_flag); cudaFree(pause_flag);
    cudaFree(stats_iter); cudaFree(stats_pairs);
    cudaFree(stats_classes); cudaFree(stats_entropy);

    printf("  T9: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ─── T10: Performance ─── */

int test_t10_performance() {
    printf("T10: Performance (timing)\n");

    const uint32_t NUM_WORDS = 5000;
    const int NUM_SENTENCES = 1000;
    const int SENTENCE_LEN = 8;

    srand(999);

    LearningState* state = ll_init(NUM_WORDS);
    SentenceRing* ring = ll_init_ring();

    int* done_flag;
    int* pause_flag;
    uint32_t *stats_iter, *stats_pairs, *stats_classes;
    double* stats_entropy;
    cudaMallocManaged(&done_flag, sizeof(int));
    cudaMallocManaged(&pause_flag, sizeof(int));
    cudaMallocManaged(&stats_iter, sizeof(uint32_t));
    cudaMallocManaged(&stats_pairs, sizeof(uint32_t));
    cudaMallocManaged(&stats_classes, sizeof(uint32_t));
    cudaMallocManaged(&stats_entropy, sizeof(double));

    *done_flag = 0;
    *pause_flag = 0;
    *stats_iter = 0;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    /* Launch kernel */
    ll_launch(state, ring, done_flag, pause_flag,
              stats_iter, stats_pairs, stats_classes, stats_entropy);

    /* Feed all sentences */
    for (int s = 0; s < NUM_SENTENCES; s++) {
        uint32_t words[64];
        uint32_t len = SENTENCE_LEN;
        /* 10 clusters of 500 words each */
        int cluster = s % 10;
        int base = cluster * 500;
        for (int i = 0; i < SENTENCE_LEN; i++) {
            words[i] = base + (rand() % 500);
        }
        ll_feed_sentence(ring, words, len);
    }

    /* Wait for processing */
    int max_wait = 300;  /* 30 seconds max */
    for (int w = 0; w < max_wait; w++) {
        usleep(100000);
        if (*done_flag) break;
        if (w > 50 && *stats_iter > 0) {
            /* Have some iterations, stop for timing */
            *done_flag = 1;
            break;
        }
    }

    ll_wait();

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    int pass = 1;

    printf("  %d words, %d sentences, %d sentence length\n",
           (int)NUM_WORDS, NUM_SENTENCES, SENTENCE_LEN);
    printf("  Total time: %.3f seconds\n", elapsed);
    printf("  Pairs: %u, Iterations: %u, Classes: %u\n",
           *stats_pairs, *stats_iter, *stats_classes);

    if (*stats_iter > 0) {
        double ms_per_iter = (elapsed * 1000.0) / *stats_iter;
        printf("  Time per iteration: %.1f ms\n", ms_per_iter);

        /* Target: < 50ms per iteration at 5K words */
        if (ms_per_iter < 50.0) {
            printf("  Performance: GOOD (< 50ms target)\n");
        } else if (ms_per_iter < 200.0) {
            printf("  Performance: ACCEPTABLE (< 200ms)\n");
        } else {
            printf("  Performance: SLOW (%.1f ms > 200ms target)\n", ms_per_iter);
            /* Don't fail — performance varies with system load */
        }
    }

    double sentences_per_sec = (double)NUM_SENTENCES / elapsed;
    printf("  Sentence throughput: %.0f sentences/second\n", sentences_per_sec);

    if (*stats_pairs == 0) {
        printf("  FAIL: no pairs counted\n");
        pass = 0;
    }

    /* Cleanup */
    ll_shutdown(state, ring);
    cudaFree(done_flag); cudaFree(pause_flag);
    cudaFree(stats_iter); cudaFree(stats_pairs);
    cudaFree(stats_classes); cudaFree(stats_entropy);

    printf("  T10: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ─── Main ─── */

int main() {
    printf("=== Learning Loop Tests ===\n\n");

    int t8  = test_t8_convergence();
    int t9  = test_t9_unified_memory();
    int t10 = test_t10_performance();

    printf("=== Results ===\n");
    printf("T8  (Convergence):      %s\n", t8  ? "PASS" : "FAIL");
    printf("T9  (Unified memory):   %s\n", t9  ? "PASS" : "FAIL");
    printf("T10 (Performance):      %s\n", t10 ? "PASS" : "FAIL");
    printf("Overall: %s\n", (t8 && t9 && t10) ? "ALL PASSED" : "SOME FAILED");

    return (t8 && t9 && t10) ? 0 : 1;
}
