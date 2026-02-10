/* bench-corpus-ck.cu — Real corpus validation for COMPARTMENT kernel
 *
 * Feeds actual Gutenberg text through the CUDA compartment kernel
 * (1 SM = 1 compartment) and validates:
 *   - Classes are formed (non-trivial clustering)
 *   - MI values are reasonable (positive, finite)
 *   - Convergence behavior matches expectations
 *   - Shows actual word clusters from real English text
 *
 * Input: Binary corpus file from bench-corpus-feeder
 *
 * Build:
 *   gcc -O2 -o bench-corpus-feeder bench-corpus-feeder.c -lm
 *   ./bench-corpus-feeder /path/to/corpus/*.txt > corpus.bin
 *   nvcc -O2 -arch=sm_75 -rdc=true -o bench-corpus-ck \
 *     bench-corpus-ck.cu gpu-compartment-kernel.cu -lcudadevrt -lm
 *   ./bench-corpus-ck corpus.bin
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

/* ─── Forward declarations (compartment kernel API) ─── */

extern "C" {
    LearningState*    ck_init(uint32_t num_words);
    CompartmentState* ck_init_compartments(uint32_t num_compartments,
                                           uint32_t num_words);
    SentenceRing*     ck_init_ring();
    void              ck_feed_sentence(SentenceRing* ring,
                                       uint32_t* words, uint32_t length);
    int               ck_launch(LearningState*, CompartmentState*,
                                SentenceRing*, int* done, int* pause,
                                uint32_t* iter, uint32_t* pairs,
                                uint32_t* classes, double* entropy);
    void              ck_wait();
    void              ck_read_classes(LearningState*, uint32_t* out, uint32_t n);
    void              ck_shutdown(LearningState*, CompartmentState*,
                                  SentenceRing*);
    int               ck_get_num_sms();
    void              ck_memory_report(int num_compartments);
}

/* ─── Helpers ─── */

static double time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

/* ─── Corpus loader ─── */

typedef struct {
    uint32_t* words;
    uint32_t  length;
} Sentence;

typedef struct {
    Sentence* sentences;
    uint32_t  num_sentences;
    uint32_t  num_words;
} Corpus;

static Corpus load_corpus(const char* filename) {
    Corpus c;
    memset(&c, 0, sizeof(c));

    FILE* f = fopen(filename, "rb");
    if (!f) {
        printf("ERROR: Cannot open %s\n", filename);
        return c;
    }

    fread(&c.num_sentences, sizeof(uint32_t), 1, f);
    fread(&c.num_words, sizeof(uint32_t), 1, f);

    c.sentences = (Sentence*)malloc(c.num_sentences * sizeof(Sentence));

    for (uint32_t i = 0; i < c.num_sentences; i++) {
        fread(&c.sentences[i].length, sizeof(uint32_t), 1, f);
        c.sentences[i].words =
            (uint32_t*)malloc(c.sentences[i].length * sizeof(uint32_t));
        fread(c.sentences[i].words, sizeof(uint32_t),
              c.sentences[i].length, f);
    }

    fclose(f);
    return c;
}

static void free_corpus(Corpus* c) {
    for (uint32_t i = 0; i < c->num_sentences; i++)
        free(c->sentences[i].words);
    free(c->sentences);
}

/* ─── Vocabulary loader (for readable output) ─── */

typedef struct {
    char     word[64];
    uint32_t id;
} VocabEntry;

static VocabEntry* vocab_table = NULL;
static uint32_t    vocab_count = 0;

static void load_vocab(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) return;

    vocab_table = (VocabEntry*)malloc(LL_MAX_WORDS * sizeof(VocabEntry));
    vocab_count = 0;

    while (fscanf(f, "%u\t%63s", &vocab_table[vocab_count].id,
                  vocab_table[vocab_count].word) == 2) {
        vocab_count++;
        if (vocab_count >= LL_MAX_WORDS) break;
    }
    fclose(f);
}

static const char* word_name(uint32_t id) {
    if (!vocab_table) return "?";
    for (uint32_t i = 0; i < vocab_count; i++) {
        if (vocab_table[i].id == id) return vocab_table[i].word;
    }
    return "?";
}

/* ═══════════════════════════════════════════════════════════════
 *  MAIN — REAL CORPUS VALIDATION
 * ═══════════════════════════════════════════════════════════════ */

int main(int argc, char** argv) {
    const char* corpus_file = "corpus.bin";
    if (argc > 1) corpus_file = argv[1];

    printf("Compartment Kernel — Real Corpus Validation\n");
    printf("============================================\n");

    /* GPU info */
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (SM %d.%d, %d SMs, %d MB)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           (int)(prop.totalGlobalMem / (1024*1024)));

    int num_sms = ck_get_num_sms();
    printf("Compartments: %d (1 per SM)\n\n", num_sms);

    /* Load corpus */
    Corpus corpus = load_corpus(corpus_file);
    if (corpus.num_sentences == 0) return 1;

    uint32_t num_words = corpus.num_words;
    if (num_words > LL_MAX_WORDS) {
        printf("Vocabulary %u > LL_MAX_WORDS %d, capping\n",
               num_words, LL_MAX_WORDS);
        num_words = LL_MAX_WORDS;
    }

    /* Corpus stats */
    uint32_t total_sents = corpus.num_sentences;
    double avg_len = 0;
    uint32_t min_len = UINT32_MAX, max_len = 0;
    uint32_t usable = 0;

    for (uint32_t i = 0; i < total_sents; i++) {
        /* Count usable words in this sentence */
        uint32_t len = 0;
        for (uint32_t w = 0; w < corpus.sentences[i].length; w++) {
            if (corpus.sentences[i].words[w] < num_words) len++;
        }
        if (len >= 2) {
            usable++;
            avg_len += len;
            if (len < min_len) min_len = len;
            if (len > max_len) max_len = len;
        }
    }
    avg_len /= (usable > 0 ? usable : 1);
    printf("Corpus: %u total sentences, %u usable (len>=2)\n",
           total_sents, usable);
    printf("Vocabulary: %u words\n", num_words);
    printf("Sentence length: avg=%.1f min=%u max=%u\n\n",
           avg_len, min_len, max_len > LL_MAX_SENTENCE_LEN ?
           LL_MAX_SENTENCE_LEN : max_len);

    /* Load vocabulary for readable output */
    load_vocab("corpus-vocab.txt");
    if (vocab_table) {
        printf("Vocabulary loaded (%u entries)\n\n", vocab_count);
    }

    /* ─── Init GPU ─── */

    LearningState*    state = ck_init(num_words);
    CompartmentState* comps = ck_init_compartments(num_sms, num_words);
    SentenceRing*     ring  = ck_init_ring();

    ck_memory_report(num_sms);
    printf("\n");

    int*      done_flag;
    int*      pause_flag;
    uint32_t* stats_iter;
    uint32_t* stats_pairs;
    uint32_t* stats_classes;
    double*   stats_entropy;

    cudaMallocManaged(&done_flag, sizeof(int));
    cudaMallocManaged(&pause_flag, sizeof(int));
    cudaMallocManaged(&stats_iter, sizeof(uint32_t));
    cudaMallocManaged(&stats_pairs, sizeof(uint32_t));
    cudaMallocManaged(&stats_classes, sizeof(uint32_t));
    cudaMallocManaged(&stats_entropy, sizeof(double));

    *done_flag     = 0;
    *pause_flag    = 0;
    *stats_iter    = 0;
    *stats_pairs   = 0;
    *stats_classes = 0;
    *stats_entropy = 0.0;

    /* ─── Launch kernel ─── */

    printf("Launching compartment kernel (%d SMs)...\n", num_sms);
    double t_start = time_ms();

    int err = ck_launch(state, comps, ring, done_flag, pause_flag,
                        stats_iter, stats_pairs, stats_classes,
                        stats_entropy);
    if (err != 0) {
        printf("FAILED to launch kernel\n");
        ck_shutdown(state, comps, ring);
        free_corpus(&corpus);
        return 1;
    }

    /* ─── Feed with backpressure ─── */

    printf("Feeding %u sentences (ring=%d slots)...\n",
           usable, LL_RING_SIZE);
    double t_feed_start = time_ms();

    uint32_t fed = 0;
    uint32_t skipped = 0;
    uint32_t stalls = 0;

    for (uint32_t s = 0; s < total_sents && !(*done_flag); s++) {
        /* Filter words beyond vocabulary */
        uint32_t filtered[LL_MAX_SENTENCE_LEN];
        uint32_t flen = 0;
        for (uint32_t w = 0;
             w < corpus.sentences[s].length && flen < LL_MAX_SENTENCE_LEN;
             w++) {
            if (corpus.sentences[s].words[w] < num_words) {
                filtered[flen++] = corpus.sentences[s].words[w];
            }
        }
        if (flen < 2) { skipped++; continue; }

        /* Backpressure: wait if ring > 75% full */
        while (ring->write_idx - ring->read_idx >
               (LL_RING_SIZE * 3 / 4)) {
            usleep(100);  /* 0.1ms */
            stalls++;
            if (*done_flag) break;
        }

        ck_feed_sentence(ring, filtered, flen);
        fed++;

        /* Progress every 10K sentences */
        if (fed % 10000 == 0) {
            printf("  fed %u/%u, iter=%u pairs=%u classes=%u "
                   "entropy=%.4f stalls=%u\n",
                   fed, usable, *stats_iter, *stats_pairs,
                   *stats_classes, *stats_entropy, stalls);
        }
    }

    double t_feed_end = time_ms();
    double feed_time = t_feed_end - t_feed_start;
    printf("Feeding done: %u fed, %u skipped, %u stalls, %.1f ms "
           "(%.0f sents/sec)\n",
           fed, skipped, stalls, feed_time,
           fed * 1000.0 / (feed_time > 0 ? feed_time : 1));

    /* ─── Wait for convergence ─── */

    printf("\nWaiting for convergence (timeout 120s)...\n");
    double timeout = 120000.0;
    uint32_t last_iter = 0;
    double last_report = time_ms();

    while (!(*done_flag)) {
        usleep(100000); /* 100ms poll */

        if (time_ms() - t_start > timeout) {
            printf("  TIMEOUT after %.0fs\n", timeout / 1000.0);
            break;
        }

        if (time_ms() - last_report > 5000.0 &&
            *stats_iter > last_iter) {
            printf("  iter=%u pairs=%u classes=%u entropy=%.4f\n",
                   *stats_iter, *stats_pairs, *stats_classes,
                   *stats_entropy);
            last_iter = *stats_iter;
            last_report = time_ms();
        }
    }

    ck_wait();
    double t_end = time_ms();
    double total_time = t_end - t_start;

    /* ═══════════════════════════════════════════════════════════════
     *  RESULTS
     * ═══════════════════════════════════════════════════════════════ */

    printf("\n=== Results ===\n");
    printf("Total time:       %.2f seconds\n", total_time / 1000.0);
    printf("Sentences fed:    %u\n", fed);
    printf("Iterations:       %u\n", *stats_iter);
    printf("Pairs discovered: %u\n", *stats_pairs);
    printf("Classes formed:   %u\n", *stats_classes);
    printf("Final entropy:    %.4f\n", *stats_entropy);
    printf("Pipeline:         %.1f sents/sec\n",
           fed * 1000.0 / total_time);
    printf("Converged:        %s\n",
           *done_flag ? "YES" : "NO (timeout)");

    /* ─── MI analysis ─── */

    printf("\n=== MI Analysis ===\n");
    cudaDeviceSynchronize();

    uint32_t actual_pairs = state->pair_count_u32;
    uint32_t mi_pos = 0, mi_neg = 0, mi_nan = 0, mi_inf = 0;
    double mi_sum = 0, mi_max = -1e30;
    uint32_t mi_max_idx = 0;

    for (uint32_t i = 0; i < actual_pairs && i < LL_MAX_PAIRS; i++) {
        double mi = state->pair_mi[i];
        if (isnan(mi)) mi_nan++;
        else if (isinf(mi)) mi_inf++;
        else if (mi > 0.0) {
            mi_pos++;
            mi_sum += mi;
            if (mi > mi_max) { mi_max = mi; mi_max_idx = i; }
        } else mi_neg++;
    }

    printf("Total pairs: %u\n", actual_pairs);
    printf("MI positive: %u, negative: %u, nan: %u, inf: %u\n",
           mi_pos, mi_neg, mi_nan, mi_inf);
    if (mi_pos > 0) {
        printf("Mean positive MI: %.4f\n", mi_sum / mi_pos);
        uint32_t wa = state->pair_word_a[mi_max_idx];
        uint32_t wb = state->pair_word_b[mi_max_idx];
        printf("Highest MI pair: %s — %s (MI=%.4f)\n",
               word_name(wa), word_name(wb), mi_max);
    }

    /* Top 20 MI pairs */
    printf("\nTop 20 MI pairs:\n");
    double* mi_copy = (double*)malloc(actual_pairs * sizeof(double));
    uint32_t* mi_order = (uint32_t*)malloc(actual_pairs * sizeof(uint32_t));
    for (uint32_t i = 0; i < actual_pairs; i++) {
        mi_copy[i] = state->pair_mi[i];
        mi_order[i] = i;
    }
    /* Simple selection of top 20 */
    for (int rank = 0; rank < 20 && rank < (int)actual_pairs; rank++) {
        uint32_t best = rank;
        for (uint32_t j = rank + 1; j < actual_pairs; j++) {
            if (mi_copy[j] > mi_copy[best]) best = j;
        }
        /* Swap */
        double tmp_d = mi_copy[rank]; mi_copy[rank] = mi_copy[best]; mi_copy[best] = tmp_d;
        uint32_t tmp_u = mi_order[rank]; mi_order[rank] = mi_order[best]; mi_order[best] = tmp_u;

        uint32_t idx = mi_order[rank];
        uint32_t wa = state->pair_word_a[idx];
        uint32_t wb = state->pair_word_b[idx];
        printf("  %2d. %-15s — %-15s  MI=%.4f  count=%.0f\n",
               rank + 1, word_name(wa), word_name(wb),
               state->pair_mi[idx], state->pair_count[idx]);
    }
    free(mi_copy);
    free(mi_order);

    /* ─── Class analysis ─── */

    printf("\n=== Class Analysis ===\n");
    uint32_t* class_ids = (uint32_t*)malloc(num_words * sizeof(uint32_t));
    ck_read_classes(state, class_ids, num_words);

    /* Count unique classes and class sizes */
    uint32_t max_class = 0;
    for (uint32_t i = 0; i < num_words; i++) {
        if (class_ids[i] > max_class) max_class = class_ids[i];
    }
    uint32_t num_classes_actual = max_class + 1;
    uint32_t* class_sz = (uint32_t*)calloc(num_classes_actual, sizeof(uint32_t));
    for (uint32_t i = 0; i < num_words; i++) {
        class_sz[class_ids[i]]++;
    }

    /* Count non-singleton classes */
    uint32_t singletons = 0, multi = 0;
    for (uint32_t c = 0; c < num_classes_actual; c++) {
        if (class_sz[c] == 1) singletons++;
        else if (class_sz[c] > 1) multi++;
    }
    printf("Classes: %u total, %u multi-word, %u singletons\n",
           num_classes_actual, multi, singletons);

    /* Show top 15 largest classes with words */
    printf("\nTop 15 classes by size:\n");
    for (int rank = 0; rank < 15; rank++) {
        uint32_t best_c = 0, best_sz = 0;
        for (uint32_t c = 0; c < num_classes_actual; c++) {
            if (class_sz[c] > best_sz) {
                best_sz = class_sz[c];
                best_c = c;
            }
        }
        if (best_sz <= 1) break;

        printf("  Class %3u (size %3u): ", best_c, best_sz);
        int printed = 0;
        for (uint32_t w = 0; w < num_words && printed < 12; w++) {
            if (class_ids[w] == best_c) {
                printf("%s ", word_name(w));
                printed++;
            }
        }
        if (best_sz > 12) printf("(+%u more)", best_sz - 12);
        printf("\n");

        class_sz[best_c] = 0;
    }

    /* ─── Entropy history ─── */

    printf("\nEntropy history:\n");
    for (int i = 0; i < LL_ENTROPY_WINDOW; i++) {
        printf("  [%d] = %.6f\n", i, state->entropy_history[i]);
    }

    /* ─── Convergence details ─── */

    printf("\nConvergence details:\n");
    printf("  total_sentences: %u\n", state->total_sentences);
    printf("  cc_runs:         %u\n", state->cc_runs);
    printf("  idle_rounds:     %u\n", state->idle_rounds);
    printf("  iteration:       %u\n", state->iteration);
    printf("  pair_count:      %u\n", state->pair_count_u32);
    printf("  dirty_count:     %u\n", state->dirty_count);

    /* ─── Validation summary ─── */

    printf("\n=== Validation ===\n");
    int pass = 1;

    if (*stats_pairs < 1000) {
        printf("FAIL: only %u pairs (expected >=1000 for real corpus)\n",
               *stats_pairs);
        pass = 0;
    } else {
        printf("PASS: %u pairs discovered\n", *stats_pairs);
    }

    if (mi_nan > 0 || mi_inf > 0) {
        printf("FAIL: %u NaN + %u Inf MI values\n", mi_nan, mi_inf);
        pass = 0;
    } else {
        printf("PASS: no NaN/Inf MI values\n");
    }

    if (*stats_entropy <= 0.0) {
        printf("FAIL: entropy = %.4f\n", *stats_entropy);
        pass = 0;
    } else {
        printf("PASS: entropy = %.4f\n", *stats_entropy);
    }

    if (multi < 2) {
        printf("FAIL: only %u multi-word classes\n", multi);
        pass = 0;
    } else {
        printf("PASS: %u multi-word classes\n", multi);
    }

    if (!(*done_flag)) {
        printf("FAIL: did not converge (timeout)\n");
        pass = 0;
    } else {
        printf("PASS: converged\n");
    }

    printf("\n=== %s ===\n", pass ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED");

    /* Cleanup */
    free(class_ids);
    free(class_sz);
    if (vocab_table) free(vocab_table);
    ck_shutdown(state, comps, ring);
    free_corpus(&corpus);
    cudaFree(done_flag); cudaFree(pause_flag);
    cudaFree(stats_iter); cudaFree(stats_pairs);
    cudaFree(stats_classes); cudaFree(stats_entropy);

    return pass ? 0 : 1;
}
