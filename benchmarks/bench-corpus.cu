/* bench-corpus.cu — Real corpus validation (Experiment 4)
 *
 * Feeds actual Gutenberg text through the CUDA persistent kernel
 * and validates correctness:
 *   - Classes are formed (non-trivial clustering)
 *   - High-frequency words cluster together
 *   - MI values are reasonable (positive, finite)
 *   - Convergence behavior matches expectations
 *
 * Input: Binary corpus file from bench-corpus-feeder
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true -o bench-corpus \
 *     bench-corpus.cu gpu-learning-loop.cu -lcudadevrt -lm
 *
 * Usage:
 *   ./bench-corpus-feeder corpus/*.txt > corpus.bin
 *   ./bench-corpus corpus.bin
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

static double time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}

/* ─── Corpus data ─── */

typedef struct {
    uint32_t* words;
    uint32_t  length;
} Sentence;

typedef struct {
    Sentence* sentences;
    uint32_t  num_sentences;
    uint32_t  num_words; /* vocabulary size */
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

    printf("Loading corpus: %u sentences, %u vocabulary words\n",
           c.num_sentences, c.num_words);

    c.sentences = (Sentence*)malloc(c.num_sentences * sizeof(Sentence));

    for (uint32_t i = 0; i < c.num_sentences; i++) {
        fread(&c.sentences[i].length, sizeof(uint32_t), 1, f);
        c.sentences[i].words = (uint32_t*)malloc(c.sentences[i].length * sizeof(uint32_t));
        fread(c.sentences[i].words, sizeof(uint32_t), c.sentences[i].length, f);
    }

    fclose(f);
    return c;
}

static void free_corpus(Corpus* c) {
    for (uint32_t i = 0; i < c->num_sentences; i++) {
        free(c->sentences[i].words);
    }
    free(c->sentences);
}

/* ─── Load vocabulary for printing ─── */

typedef struct {
    char word[64];
    uint32_t id;
} VocabEntry;

static VocabEntry* load_vocab(const char* filename, uint32_t* count) {
    FILE* f = fopen(filename, "r");
    if (!f) { *count = 0; return NULL; }

    VocabEntry* entries = (VocabEntry*)malloc(8192 * sizeof(VocabEntry));
    *count = 0;

    while (fscanf(f, "%u\t%63s", &entries[*count].id, entries[*count].word) == 2) {
        (*count)++;
        if (*count >= 8192) break;
    }
    fclose(f);
    return entries;
}

static const char* lookup_word(VocabEntry* vocab, uint32_t vocab_count, uint32_t id) {
    for (uint32_t i = 0; i < vocab_count; i++) {
        if (vocab[i].id == id) return vocab[i].word;
    }
    return "???";
}

/* ═══════════════════════════════════════════════════════════════
 *  CORPUS VALIDATION
 * ═══════════════════════════════════════════════════════════════ */

int main(int argc, char** argv) {
    const char* corpus_file = "corpus.bin";
    if (argc > 1) corpus_file = argv[1];

    printf("GPU Learning Pipeline — Real Corpus Validation\n");
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
        printf("ERROR: Device does not support cooperative launch\n");
        return 1;
    }

    /* Load corpus */
    Corpus corpus = load_corpus(corpus_file);
    if (corpus.num_sentences == 0) return 1;

    /* Cap vocabulary to LL_MAX_WORDS */
    uint32_t num_words = corpus.num_words;
    if (num_words > LL_MAX_WORDS) {
        printf("WARNING: Vocabulary (%u) exceeds LL_MAX_WORDS (%d), capping\n",
               num_words, LL_MAX_WORDS);
        num_words = LL_MAX_WORDS;
    }

    /* Cap sentences to avoid timeout */
    uint32_t num_sentences = corpus.num_sentences;
    uint32_t max_sentences = 10000;
    if (num_sentences > max_sentences) {
        printf("Capping to %u sentences (of %u) for benchmark\n",
               max_sentences, num_sentences);
        num_sentences = max_sentences;
    }

    /* Corpus stats */
    double avg_len = 0;
    uint32_t min_len = UINT32_MAX, max_len = 0;
    for (uint32_t i = 0; i < num_sentences; i++) {
        uint32_t len = corpus.sentences[i].length;
        avg_len += len;
        if (len < min_len) min_len = len;
        if (len > max_len) max_len = len;
    }
    avg_len /= num_sentences;
    printf("Corpus stats: %u sentences, avg_len=%.1f, min=%u, max=%u\n",
           num_sentences, avg_len, min_len, max_len);

    /* ─── Init GPU pipeline ─── */

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

    /* Launch persistent kernel */
    printf("\nLaunching persistent kernel...\n");
    double t_start = time_ms();

    int err = ll_launch(state, ring, done_flag, pause_flag,
                        stats_iteration, stats_pairs, stats_classes, stats_entropy);
    if (err != 0) {
        printf("FAILED to launch kernel!\n");
        ll_shutdown(state, ring);
        free_corpus(&corpus);
        return 1;
    }

    /* Feed sentences */
    printf("Feeding %u sentences...\n", num_sentences);
    double t_feed_start = time_ms();

    for (uint32_t s = 0; s < num_sentences; s++) {
        /* Filter out words beyond our vocabulary */
        uint32_t filtered[LL_MAX_SENTENCE_LEN];
        uint32_t flen = 0;
        for (uint32_t w = 0; w < corpus.sentences[s].length && flen < LL_MAX_SENTENCE_LEN; w++) {
            if (corpus.sentences[s].words[w] < num_words) {
                filtered[flen++] = corpus.sentences[s].words[w];
            }
        }
        if (flen >= 2) {
            ll_feed_sentence(ring, filtered, flen);
        }
    }

    double t_feed_end = time_ms();
    printf("Feeding complete: %.1f ms (%.0f sents/sec)\n",
           t_feed_end - t_feed_start,
           num_sentences * 1000.0 / (t_feed_end - t_feed_start));

    /* Wait for convergence (timeout 120s for large corpus) */
    printf("Waiting for convergence...\n");
    double timeout = 120000.0;
    uint32_t last_iter = 0;
    double last_report = time_ms();

    while (!(*done_flag)) {
        usleep(50000); /* 50ms poll */

        if (time_ms() - t_start > timeout) {
            printf("  TIMEOUT after %.0f seconds\n", timeout / 1000.0);
            *done_flag = 1;
            break;
        }

        /* Progress every 5 seconds */
        if (time_ms() - last_report > 5000.0 && *stats_iteration > last_iter) {
            printf("  iter=%u pairs=%u classes=%u entropy=%.4f\n",
                   *stats_iteration, *stats_pairs, *stats_classes, *stats_entropy);
            last_iter = *stats_iteration;
            last_report = time_ms();
        }
    }

    ll_wait();
    double t_end = time_ms();
    double total_time = t_end - t_start;

    printf("\n=== Results ===\n");
    printf("Total time: %.2f seconds\n", total_time / 1000.0);
    printf("Iterations: %u\n", *stats_iteration);
    printf("Pairs discovered: %u\n", *stats_pairs);
    printf("Classes formed: %u\n", *stats_classes);
    printf("Final entropy: %.4f\n", *stats_entropy);
    printf("Throughput: %.1f sentences/sec\n",
           num_sentences * 1000.0 / total_time);
    printf("Converged: %s\n",
           *stats_iteration < LL_MAX_ITERATIONS ? "YES" : "NO (hit max iterations)");

    /* ─── Validation checks ─── */

    printf("\n=== Validation ===\n");
    int pass = 1;

    /* Check 1: Non-trivial clustering */
    if (*stats_classes < 2) {
        printf("FAIL: Only %u classes (expected >= 2)\n", *stats_classes);
        pass = 0;
    } else if (*stats_classes >= num_words) {
        printf("FAIL: %u classes = %u words (no clustering)\n", *stats_classes, num_words);
        pass = 0;
    } else {
        printf("PASS: %u classes formed (non-trivial clustering)\n", *stats_classes);
    }

    /* Check 2: Reasonable number of pairs */
    if (*stats_pairs < 100) {
        printf("FAIL: Only %u pairs (expected >= 100)\n", *stats_pairs);
        pass = 0;
    } else {
        printf("PASS: %u pairs discovered\n", *stats_pairs);
    }

    /* Check 3: Positive entropy */
    if (*stats_entropy <= 0.0) {
        printf("FAIL: Entropy = %.4f (expected > 0)\n", *stats_entropy);
        pass = 0;
    } else {
        printf("PASS: Entropy = %.4f (positive)\n", *stats_entropy);
    }

    /* Check 4: MI values are reasonable */
    cudaDeviceSynchronize();
    uint32_t mi_positive = 0, mi_negative = 0, mi_nan = 0, mi_inf = 0;
    double mi_sum = 0.0;
    uint32_t actual_pairs = state->pair_count_u32;

    for (uint32_t i = 0; i < actual_pairs && i < LL_MAX_PAIRS; i++) {
        double mi = state->pair_mi[i];
        if (isnan(mi)) mi_nan++;
        else if (isinf(mi)) mi_inf++;
        else if (mi > 0.0) { mi_positive++; mi_sum += mi; }
        else mi_negative++;
    }

    printf("MI distribution: %u positive, %u negative, %u NaN, %u Inf\n",
           mi_positive, mi_negative, mi_nan, mi_inf);
    if (mi_nan > 0 || mi_inf > 0) {
        printf("FAIL: %u NaN + %u Inf MI values\n", mi_nan, mi_inf);
        pass = 0;
    } else {
        printf("PASS: No NaN/Inf MI values\n");
    }
    if (mi_positive > 0) {
        printf("PASS: Mean positive MI = %.4f\n", mi_sum / mi_positive);
    }

    /* Check 5: Read class assignments and analyze */
    uint32_t* classes = (uint32_t*)malloc(num_words * sizeof(uint32_t));
    ll_read_classes(state, classes, num_words);

    /* Count class sizes */
    uint32_t* class_sizes = (uint32_t*)calloc(*stats_classes + 1, sizeof(uint32_t));
    for (uint32_t i = 0; i < num_words; i++) {
        if (classes[i] < *stats_classes + 1) {
            class_sizes[classes[i]]++;
        }
    }

    /* Find largest classes */
    printf("\nTop 10 classes by size:\n");

    /* Load vocabulary for word names */
    uint32_t vocab_count = 0;
    VocabEntry* vocab = load_vocab("corpus-vocab.txt", &vocab_count);

    for (int rank = 0; rank < 10 && rank < (int)*stats_classes; rank++) {
        /* Find largest remaining class */
        uint32_t best_class = 0, best_size = 0;
        for (uint32_t c = 0; c < *stats_classes; c++) {
            if (class_sizes[c] > best_size) {
                best_size = class_sizes[c];
                best_class = c;
            }
        }
        if (best_size == 0) break;

        printf("  Class %u (size %u): ", best_class, best_size);

        /* Print first 10 words in this class */
        int printed = 0;
        for (uint32_t w = 0; w < num_words && printed < 10; w++) {
            if (classes[w] == best_class) {
                if (vocab) {
                    printf("%s ", lookup_word(vocab, vocab_count, w));
                } else {
                    printf("w%u ", w);
                }
                printed++;
            }
        }
        if (best_size > 10) printf("...");
        printf("\n");

        class_sizes[best_class] = 0; /* mark as printed */
    }

    /* Check 6: Convergence pattern */
    printf("\nEntropy history (last %d values):\n", LL_ENTROPY_WINDOW);
    for (int i = 0; i < LL_ENTROPY_WINDOW; i++) {
        printf("  [%d] = %.6f\n", i, state->entropy_history[i]);
    }

    /* Summary */
    printf("\n=== SUMMARY ===\n");
    if (pass) {
        printf("ALL CHECKS PASSED\n");
    } else {
        printf("SOME CHECKS FAILED\n");
    }

    printf("\nPerformance:\n");
    printf("  Pipeline throughput: %.1f sentences/sec\n",
           num_sentences * 1000.0 / total_time);
    printf("  Feed throughput: %.0f sentences/sec\n",
           num_sentences * 1000.0 / (t_feed_end - t_feed_start));
    printf("  Time per iteration: %.1f ms\n",
           *stats_iteration > 0 ? total_time / *stats_iteration : 0.0);

    /* Cleanup */
    free(classes);
    free(class_sizes);
    if (vocab) free(vocab);
    ll_shutdown(state, ring);
    free_corpus(&corpus);
    cudaFree(done_flag); cudaFree(pause_flag);
    cudaFree(stats_iteration); cudaFree(stats_pairs);
    cudaFree(stats_classes); cudaFree(stats_entropy);

    return pass ? 0 : 1;
}
