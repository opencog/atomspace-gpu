/* bench-corpus-feeder.c — Sentence tokenizer for corpus validation
 *
 * Reads Gutenberg text files, tokenizes into word-ID sequences,
 * outputs binary format for bench-corpus to consume.
 *
 * Output format (.corpus.bin):
 *   uint32_t num_sentences
 *   uint32_t num_words (vocabulary size)
 *   For each sentence:
 *     uint32_t length
 *     uint32_t word_ids[length]
 *
 * Build:
 *   gcc -O2 -o bench-corpus-feeder bench-corpus-feeder.c -lm
 *
 * Usage:
 *   ./bench-corpus-feeder corpus/*.txt > corpus.bin
 *   ./bench-corpus [corpus.bin]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>

#define MAX_VOCAB      8192
#define MAX_WORD_LEN   64
#define MAX_SENT_LEN   64
#define MAX_SENTENCES  100000
#define MAX_LINE_LEN   4096

/* Simple hash table for word → ID mapping */

typedef struct {
    char word[MAX_WORD_LEN];
    uint32_t id;
} VocabEntry;

static VocabEntry vocab[MAX_VOCAB];
static uint32_t vocab_size = 0;

/* djb2 hash */
static uint32_t hash_word(const char* s) {
    uint32_t h = 5381;
    while (*s) {
        h = ((h << 5) + h) + (unsigned char)*s;
        s++;
    }
    return h;
}

static uint32_t get_or_create_word_id(const char* word) {
    uint32_t h = hash_word(word) % MAX_VOCAB;
    int probe = 0;

    while (probe < MAX_VOCAB) {
        uint32_t idx = (h + probe) % MAX_VOCAB;
        if (vocab[idx].word[0] == '\0') {
            /* Empty slot — new word */
            if (vocab_size >= MAX_VOCAB - 100) return UINT32_MAX; /* near-full */
            strncpy(vocab[idx].word, word, MAX_WORD_LEN - 1);
            vocab[idx].id = vocab_size;
            vocab_size++;
            return vocab[idx].id;
        }
        if (strcmp(vocab[idx].word, word) == 0) {
            return vocab[idx].id;
        }
        probe++;
    }
    return UINT32_MAX;
}

/* Sentence storage */

typedef struct {
    uint32_t words[MAX_SENT_LEN];
    uint32_t length;
} Sentence;

static Sentence sentences[MAX_SENTENCES];
static uint32_t num_sentences = 0;

/* Detect Gutenberg header/footer lines */
static int is_gutenberg_meta(const char* line) {
    if (strstr(line, "*** START OF") != NULL) return 1;
    if (strstr(line, "*** END OF") != NULL) return 1;
    if (strstr(line, "Project Gutenberg") != NULL) return 1;
    if (strstr(line, "This eBook") != NULL) return 1;
    return 0;
}

/* Tokenize a line into words, splitting on spaces and punctuation */
static void tokenize_line(const char* line) {
    char word[MAX_WORD_LEN];
    int wlen = 0;

    Sentence* sent = NULL;

    for (const char* p = line; ; p++) {
        char c = *p;

        if (isalpha(c) || c == '\'') {
            /* Build word */
            if (wlen < MAX_WORD_LEN - 1) {
                word[wlen++] = tolower(c);
            }
        } else {
            if (wlen > 0) {
                word[wlen] = '\0';
                wlen = 0;

                uint32_t id = get_or_create_word_id(word);
                if (id == UINT32_MAX) continue;

                if (sent == NULL) {
                    if (num_sentences >= MAX_SENTENCES) return;
                    sent = &sentences[num_sentences];
                    sent->length = 0;
                }

                if (sent->length < MAX_SENT_LEN) {
                    sent->words[sent->length++] = id;
                }
            }

            /* Sentence boundary: period, !, ? */
            if (c == '.' || c == '!' || c == '?') {
                if (sent != NULL && sent->length >= 2) {
                    num_sentences++;
                }
                sent = NULL;
            }

            if (c == '\0') break;
        }
    }

    /* Line end: commit partial sentence if it has enough words */
    if (sent != NULL && sent->length >= 2) {
        num_sentences++;
    }
}

/* Process one file */
static void process_file(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open: %s\n", filename);
        return;
    }

    char line[MAX_LINE_LEN];
    int past_header = 0;

    while (fgets(line, sizeof(line), f)) {
        /* Skip Gutenberg header */
        if (!past_header) {
            if (strstr(line, "*** START OF") != NULL) {
                past_header = 1;
            }
            continue;
        }

        /* Stop at Gutenberg footer */
        if (strstr(line, "*** END OF") != NULL) break;

        /* Skip empty lines */
        int has_alpha = 0;
        for (const char* p = line; *p; p++) {
            if (isalpha(*p)) { has_alpha = 1; break; }
        }
        if (!has_alpha) continue;

        /* Skip chapter headings (all caps) */
        int all_upper = 1;
        for (const char* p = line; *p && *p != '\n'; p++) {
            if (isalpha(*p) && islower(*p)) { all_upper = 0; break; }
        }
        if (all_upper && strlen(line) > 5) continue;

        tokenize_line(line);
    }

    fclose(f);
    fprintf(stderr, "Processed %s: %u sentences, %u words in vocab\n",
            filename, num_sentences, vocab_size);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <file1.txt> [file2.txt ...]\n", argv[0]);
        return 1;
    }

    memset(vocab, 0, sizeof(vocab));

    /* Process all input files */
    for (int i = 1; i < argc; i++) {
        process_file(argv[i]);
    }

    fprintf(stderr, "Total: %u sentences, %u vocabulary words\n",
            num_sentences, vocab_size);

    /* Write binary output to stdout */
    fwrite(&num_sentences, sizeof(uint32_t), 1, stdout);
    fwrite(&vocab_size, sizeof(uint32_t), 1, stdout);

    for (uint32_t i = 0; i < num_sentences; i++) {
        fwrite(&sentences[i].length, sizeof(uint32_t), 1, stdout);
        fwrite(sentences[i].words, sizeof(uint32_t), sentences[i].length, stdout);
    }

    /* Also write vocabulary (word → ID mapping) for debugging */
    char vocab_fname[256];
    snprintf(vocab_fname, sizeof(vocab_fname), "corpus-vocab.txt");
    FILE* vf = fopen(vocab_fname, "w");
    if (vf) {
        for (uint32_t i = 0; i < MAX_VOCAB; i++) {
            if (vocab[i].word[0] != '\0') {
                fprintf(vf, "%u\t%s\n", vocab[i].id, vocab[i].word);
            }
        }
        fclose(vf);
        fprintf(stderr, "Vocabulary written to %s\n", vocab_fname);
    }

    return 0;
}
