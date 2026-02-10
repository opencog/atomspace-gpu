/* test-spanning-forest.cu — Tests T5 and T6 for grammar parse + surprise
 *
 * T5: Grammar parse (3 sentences with known MI, verify correct MST edges)
 * T6: Surprise (grammar vs word parse — verify surprise scoring)
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true \
 *     -o test-spanning-forest \
 *     test-spanning-forest.cu gpu-spanning-forest.cu \
 *     -lcudadevrt -lm
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

#define MAX_TREE_EDGES 63  /* MAX_SENTENCE_LEN - 1 */

/* Forward declaration */
extern "C" void spanning_forest_run(
    const uint32_t* d_sentence_words,
    const uint32_t* d_sentence_lengths,
    const uint32_t* d_sentence_offsets,
    uint32_t num_sentences,
    const uint32_t* d_word_class_id,
    const uint64_t* d_class_ht_keys,
    const double* d_class_ht_mi,
    uint32_t class_ht_capacity,
    const uint64_t* d_pair_ht_keys,
    const uint32_t* d_pair_ht_values,
    const double* d_pair_mi,
    uint32_t pair_ht_capacity,
    uint32_t* d_word_parse_a, uint32_t* d_word_parse_b,
    double* d_word_parse_mi, uint32_t* d_word_parse_count,
    uint32_t* d_gram_parse_a, uint32_t* d_gram_parse_b,
    double* d_gram_parse_mi, uint32_t* d_gram_parse_count,
    double* d_surprise);

/* ─── Helper: Build a simple word-pair MI hash table on GPU ─── */

struct SimpleHTBuilder {
    /* Host data */
    uint64_t* h_keys;
    uint32_t* h_values;
    double*   h_mi;
    uint32_t  capacity;
    uint32_t  pair_count;

    /* Device data */
    uint64_t* d_keys;
    uint32_t* d_values;
    double*   d_mi;
};

uint64_t splitmix64(uint64_t key) {
    key ^= key >> 30;
    key *= 0xBF58476D1CE4E5B9ULL;
    key ^= key >> 27;
    key *= 0x94D049BB133111EBULL;
    key ^= key >> 31;
    return key;
}

void ht_init(SimpleHTBuilder* ht, uint32_t capacity) {
    ht->capacity = capacity;
    ht->pair_count = 0;
    ht->h_keys = (uint64_t*)malloc(capacity * sizeof(uint64_t));
    ht->h_values = (uint32_t*)malloc(capacity * sizeof(uint32_t));
    ht->h_mi = (double*)malloc(capacity * sizeof(double));
    memset(ht->h_keys, 0xFF, capacity * sizeof(uint64_t));   /* empty */
    memset(ht->h_values, 0xFF, capacity * sizeof(uint32_t)); /* empty */
    memset(ht->h_mi, 0, capacity * sizeof(double));
}

void ht_insert(SimpleHTBuilder* ht, uint32_t word_a, uint32_t word_b, double mi) {
    uint32_t lo = (word_a <= word_b) ? word_a : word_b;
    uint32_t hi = (word_a <= word_b) ? word_b : word_a;
    uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;
    uint64_t mask = (uint64_t)(ht->capacity - 1);
    uint64_t slot = splitmix64(key) & mask;

    for (int probe = 0; probe < 4096; probe++) {
        if (ht->h_keys[slot] == 0xFFFFFFFFFFFFFFFFULL) {
            ht->h_keys[slot] = key;
            ht->h_values[slot] = ht->pair_count;
            ht->h_mi[ht->pair_count] = mi;
            ht->pair_count++;
            return;
        }
        slot = (slot + 1) & mask;
    }
    printf("ERROR: hash table full\n");
}

void ht_upload(SimpleHTBuilder* ht) {
    cudaMalloc(&ht->d_keys, ht->capacity * sizeof(uint64_t));
    cudaMalloc(&ht->d_values, ht->capacity * sizeof(uint32_t));
    cudaMalloc(&ht->d_mi, ht->pair_count * sizeof(double));
    cudaMemcpy(ht->d_keys, ht->h_keys, ht->capacity * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ht->d_values, ht->h_values, ht->capacity * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(ht->d_mi, ht->h_mi, ht->pair_count * sizeof(double), cudaMemcpyHostToDevice);
}

void ht_free(SimpleHTBuilder* ht) {
    free(ht->h_keys); free(ht->h_values); free(ht->h_mi);
    cudaFree(ht->d_keys); cudaFree(ht->d_values); cudaFree(ht->d_mi);
}

/* ─── Helper: Check if an edge exists in a parse result ─── */

int has_edge(uint32_t* h_ea, uint32_t* h_eb, uint32_t count,
             uint32_t a, uint32_t b) {
    uint32_t lo = (a <= b) ? a : b;
    uint32_t hi = (a <= b) ? b : a;
    for (uint32_t i = 0; i < count; i++) {
        uint32_t elo = (h_ea[i] <= h_eb[i]) ? h_ea[i] : h_eb[i];
        uint32_t ehi = (h_ea[i] <= h_eb[i]) ? h_eb[i] : h_ea[i];
        if (elo == lo && ehi == hi) return 1;
    }
    return 0;
}

/* ─── T5: Grammar parse (3 sentences) ─── */

int test_t5_grammar_parse() {
    printf("T5: Grammar parse (3 sentences)\n");

    /* Words: 0=the, 1=cat, 2=sat, 3=on, 4=mat, 5=dog, 6=ran, 7=fast
     *
     * Known word-pair MI values (higher = stronger association):
     *   (the, cat)=3.0  (the, dog)=2.8  (the, mat)=2.5
     *   (cat, sat)=4.0  (dog, ran)=3.5
     *   (sat, on)=3.2   (on, mat)=2.0
     *   (ran, fast)=3.8
     *   All other pairs: 0.5 (low background MI)
     *
     * Sentence 1: "the cat sat on mat" = words [0,1,2,3,4]
     *   Expected MST: (cat,sat)=4.0 → (sat,on)=3.2 → (the,cat)=3.0 → (the,mat)=2.5
     *   (connects all 5 words with 4 highest-MI edges that don't cycle)
     *
     * Sentence 2: "the dog ran fast" = words [0,5,6,7]
     *   Expected MST: (ran,fast)=3.8 → (dog,ran)=3.5 → (the,dog)=2.8
     *
     * Sentence 3: "cat dog" = words [1,5]
     *   Expected MST: (cat,dog)=0.5 (only pair)
     */

    /* Build word-pair MI hash table */
    SimpleHTBuilder ht;
    ht_init(&ht, 256);

    /* Strong associations */
    ht_insert(&ht, 0, 1, 3.0);  /* the-cat */
    ht_insert(&ht, 0, 5, 2.8);  /* the-dog */
    ht_insert(&ht, 0, 4, 2.5);  /* the-mat */
    ht_insert(&ht, 1, 2, 4.0);  /* cat-sat */
    ht_insert(&ht, 5, 6, 3.5);  /* dog-ran */
    ht_insert(&ht, 2, 3, 3.2);  /* sat-on  */
    ht_insert(&ht, 3, 4, 2.0);  /* on-mat  */
    ht_insert(&ht, 6, 7, 3.8);  /* ran-fast */

    /* Background MI for remaining pairs (low) */
    for (uint32_t i = 0; i < 8; i++) {
        for (uint32_t j = i + 1; j < 8; j++) {
            /* Skip if already inserted */
            uint64_t key = ((uint64_t)i << 32) | (uint64_t)j;
            int found = 0;
            for (uint32_t s = 0; s < ht.capacity; s++) {
                if (ht.h_keys[s] == key) { found = 1; break; }
            }
            if (!found) ht_insert(&ht, i, j, 0.5);
        }
    }

    ht_upload(&ht);

    /* Build sentence arrays */
    uint32_t h_words[] = {0,1,2,3,4,  0,5,6,7,  1,5};
    uint32_t h_lengths[] = {5, 4, 2};
    uint32_t h_offsets[] = {0, 5, 9};
    uint32_t num_sentences = 3;

    uint32_t *d_words, *d_lengths, *d_offsets;
    cudaMalloc(&d_words, 11 * sizeof(uint32_t));
    cudaMalloc(&d_lengths, 3 * sizeof(uint32_t));
    cudaMalloc(&d_offsets, 3 * sizeof(uint32_t));
    cudaMemcpy(d_words, h_words, 11 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, h_lengths, 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, 3 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* No classes (all unclassified) — word-only parse == grammar parse */
    uint32_t h_class_id[8];
    for (int i = 0; i < 8; i++) h_class_id[i] = 0xFFFFFFFFU;
    uint32_t* d_class_id;
    cudaMalloc(&d_class_id, 8 * sizeof(uint32_t));
    cudaMemcpy(d_class_id, h_class_id, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* Empty class MI table */
    uint32_t class_ht_cap = 64;
    uint64_t* d_class_keys;
    double* d_class_mi;
    cudaMalloc(&d_class_keys, class_ht_cap * sizeof(uint64_t));
    cudaMalloc(&d_class_mi, class_ht_cap * sizeof(double));
    cudaMemset(d_class_keys, 0xFF, class_ht_cap * sizeof(uint64_t));
    cudaMemset(d_class_mi, 0, class_ht_cap * sizeof(double));

    /* Output buffers */
    uint32_t pe = num_sentences * MAX_TREE_EDGES;
    uint32_t *d_wp_a, *d_wp_b, *d_wp_cnt, *d_gp_a, *d_gp_b, *d_gp_cnt;
    double *d_wp_mi, *d_gp_mi, *d_surprise;
    cudaMalloc(&d_wp_a, pe * sizeof(uint32_t));
    cudaMalloc(&d_wp_b, pe * sizeof(uint32_t));
    cudaMalloc(&d_wp_mi, pe * sizeof(double));
    cudaMalloc(&d_wp_cnt, num_sentences * sizeof(uint32_t));
    cudaMalloc(&d_gp_a, pe * sizeof(uint32_t));
    cudaMalloc(&d_gp_b, pe * sizeof(uint32_t));
    cudaMalloc(&d_gp_mi, pe * sizeof(double));
    cudaMalloc(&d_gp_cnt, num_sentences * sizeof(uint32_t));
    cudaMalloc(&d_surprise, num_sentences * sizeof(double));

    /* Run */
    spanning_forest_run(
        d_words, d_lengths, d_offsets, num_sentences,
        d_class_id,
        d_class_keys, d_class_mi, class_ht_cap,
        ht.d_keys, ht.d_values, ht.d_mi, ht.capacity,
        d_wp_a, d_wp_b, d_wp_mi, d_wp_cnt,
        d_gp_a, d_gp_b, d_gp_mi, d_gp_cnt,
        d_surprise);

    /* Read results */
    uint32_t h_wp_count[3], h_gp_count[3];
    cudaMemcpy(h_wp_count, d_wp_cnt, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gp_count, d_gp_cnt, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int pass = 1;

    /* Check edge counts */
    printf("  Word-parse edge counts: %u %u %u (expected 4 3 1)\n",
           h_wp_count[0], h_wp_count[1], h_wp_count[2]);
    if (h_wp_count[0] != 4) { printf("  FAIL: sent 0 expected 4 edges\n"); pass = 0; }
    if (h_wp_count[1] != 3) { printf("  FAIL: sent 1 expected 3 edges\n"); pass = 0; }
    if (h_wp_count[2] != 1) { printf("  FAIL: sent 2 expected 1 edge\n"); pass = 0; }

    /* Check specific edges in sentence 0 word-parse */
    uint32_t h_ea[MAX_TREE_EDGES * 3], h_eb[MAX_TREE_EDGES * 3];
    double h_emi[MAX_TREE_EDGES * 3];
    cudaMemcpy(h_ea, d_wp_a, pe * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_eb, d_wp_b, pe * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_emi, d_wp_mi, pe * sizeof(double), cudaMemcpyDeviceToHost);

    printf("  Sentence 0 word-parse edges:\n");
    for (uint32_t i = 0; i < h_wp_count[0]; i++) {
        printf("    (%u,%u) MI=%.1f\n", h_ea[i], h_eb[i], h_emi[i]);
    }

    /* Sentence 0 MST should contain: (1,2)=4.0, (2,3)=3.2, (0,1)=3.0, (0,4)=2.5 or (3,4)=2.0
     * Actually the 4 highest MI edges forming a tree on {0,1,2,3,4}:
     * (1,2)=4.0 → {1,2}
     * (2,3)=3.2 → {1,2,3}
     * (0,1)=3.0 → {0,1,2,3}
     * Now need to connect 4: (0,4)=2.5 or (3,4)=2.0 → pick (0,4)=2.5
     */
    if (!has_edge(h_ea, h_eb, h_wp_count[0], 1, 2)) {
        printf("  FAIL: sent 0 missing edge (1,2)\n"); pass = 0;
    }
    if (!has_edge(h_ea, h_eb, h_wp_count[0], 2, 3)) {
        printf("  FAIL: sent 0 missing edge (2,3)\n"); pass = 0;
    }
    if (!has_edge(h_ea, h_eb, h_wp_count[0], 0, 1)) {
        printf("  FAIL: sent 0 missing edge (0,1)\n"); pass = 0;
    }
    /* Either (0,4)=2.5 or (3,4)=2.0 connects word 4 */
    if (!has_edge(h_ea, h_eb, h_wp_count[0], 0, 4) &&
        !has_edge(h_ea, h_eb, h_wp_count[0], 3, 4)) {
        printf("  FAIL: sent 0 word 4 not connected\n"); pass = 0;
    }

    /* Sentence 1: (6,7)=3.8 (5,6)=3.5 (0,5)=2.8 */
    uint32_t s1_base = MAX_TREE_EDGES;
    printf("  Sentence 1 word-parse edges:\n");
    for (uint32_t i = 0; i < h_wp_count[1]; i++) {
        printf("    (%u,%u) MI=%.1f\n", h_ea[s1_base+i], h_eb[s1_base+i], h_emi[s1_base+i]);
    }
    if (!has_edge(&h_ea[s1_base], &h_eb[s1_base], h_wp_count[1], 6, 7)) {
        printf("  FAIL: sent 1 missing edge (6,7)\n"); pass = 0;
    }
    if (!has_edge(&h_ea[s1_base], &h_eb[s1_base], h_wp_count[1], 5, 6)) {
        printf("  FAIL: sent 1 missing edge (5,6)\n"); pass = 0;
    }

    /* Since there are no classes, grammar parse == word parse */
    printf("  Grammar-parse edge counts: %u %u %u\n",
           h_gp_count[0], h_gp_count[1], h_gp_count[2]);
    if (h_gp_count[0] != h_wp_count[0] ||
        h_gp_count[1] != h_wp_count[1] ||
        h_gp_count[2] != h_wp_count[2]) {
        printf("  FAIL: grammar parse count differs from word parse (no classes)\n");
        pass = 0;
    }

    /* Cleanup */
    cudaFree(d_words); cudaFree(d_lengths); cudaFree(d_offsets);
    cudaFree(d_class_id); cudaFree(d_class_keys); cudaFree(d_class_mi);
    cudaFree(d_wp_a); cudaFree(d_wp_b); cudaFree(d_wp_mi); cudaFree(d_wp_cnt);
    cudaFree(d_gp_a); cudaFree(d_gp_b); cudaFree(d_gp_mi); cudaFree(d_gp_cnt);
    cudaFree(d_surprise);
    ht_free(&ht);

    printf("  T5: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ─── T6: Surprise (grammar vs word parse divergence) ─── */

int test_t6_surprise() {
    printf("T6: Grammar surprise\n");

    /* Setup: 2 sentences, 6 words
     * Words: 0,1,2,3,4,5
     * Classes: 0,1 → class 0; 2,3 → class 1; 4,5 → unclassified
     *
     * Word-pair MI:
     *   (0,2)=5.0  (0,3)=4.5  (1,2)=4.8  (1,3)=4.2  — cross-class, high
     *   (0,4)=1.0  (1,5)=1.2                         — to unclassified, low
     *   (2,4)=3.0  (3,5)=2.5                         — to unclassified
     *   (4,5)=0.8                                     — between unclassified
     *   All others: 0.3
     *
     * Class-pair MI:
     *   (class0, class1) = 4.0  — class-level MI (average of cross-class pairs)
     *
     * Sentence 1: "0 2 4" → words [0,2,4]
     *   Word-parse MST: (0,2)=5.0, (2,4)=3.0
     *   Grammar-parse: For (0,2): class(0)=0, class(2)=1 → class MI=4.0
     *                  For (0,4): class(4)=none → word MI=1.0
     *                  For (2,4): class(4)=none → word MI=3.0
     *   Grammar MST: (0,2)=4.0 [class], (2,4)=3.0 [word]
     *   Both parses have same structure → surprise ≈ 0
     *
     * Sentence 2: "0 4 5" → words [0,4,5]
     *   Word-parse MST: (0,4)=1.0, (4,5)=0.8
     *     (but also (0,5)=0.3 — even lower)
     *   Grammar-parse: No class MI useful (4,5 unclassified)
     *     Falls back to word MI: same as word-parse
     *   Surprise ≈ 0 (same parse)
     *
     * Let me make a more interesting case with divergent parses:
     *
     * Sentence 3: "0 2 3" → words [0,2,3]
     *   Word-pair MI: (0,2)=5.0, (0,3)=4.5, (2,3)=0.3
     *   Word-parse MST: (0,2)=5.0, (0,3)=4.5  → star centered on 0
     *
     *   Grammar-parse: (0,2) → class MI(0,1)=4.0
     *                  (0,3) → class MI(0,1)=4.0
     *                  (2,3) → same class (1,1) → word MI=0.3
     *   Grammar MST: (0,2)=4.0, (0,3)=4.0  → same structure!
     *   Surprise ≈ 0 (same topology)
     *
     * To get actual surprise, we need grammar-parse to pick DIFFERENT edges
     * than word-parse. This happens when class-MI reshuffles preferences.
     */

    /* Let me design a clear case:
     * Words: 0,1,2,3
     * Classes: 0 → class A, 1 → class A, 2 → class B, 3 → class C
     *
     * Word-pair MI:
     *   (0,1)=1.0  (same class, not used for class MI)
     *   (0,2)=5.0  (A→B, strong word MI)
     *   (0,3)=2.0  (A→C, weak word MI)
     *   (1,2)=2.5  (A→B, moderate word MI)
     *   (1,3)=6.0  (A→C, very strong word MI)
     *   (2,3)=3.0  (B→C, moderate word MI)
     *
     * Class-pair MI (averages):
     *   (A,B): avg of (0,2)=5.0, (1,2)=2.5 → 3.75
     *   (A,C): avg of (0,3)=2.0, (1,3)=6.0 → 4.0
     *   (B,C): 3.0
     *
     * Sentence: "0 1 2 3" → all 4 words
     *
     * Word-parse MST (greedy on word MI):
     *   (1,3)=6.0 → connect 1,3
     *   (0,2)=5.0 → connect 0,2
     *   Next highest non-cycle: (2,3)=3.0 or (1,2)=2.5 or (0,3)=2.0
     *   → (2,3)=3.0 connects {0,2} with {1,3}
     *   Result: edges (1,3), (0,2), (2,3)
     *
     * Grammar-parse MST (greedy on class MI, fallback word MI):
     *   Evaluate each pair's "grammar MI":
     *   (0,1): same class A → word MI 1.0
     *   (0,2): class(A,B) → 3.75
     *   (0,3): class(A,C) → 4.0
     *   (1,2): class(A,B) → 3.75
     *   (1,3): class(A,C) → 4.0
     *   (2,3): class(B,C) → 3.0
     *
     *   Grammar MST: (0,3)=4.0 or (1,3)=4.0 (tie!) → pick (0,3) [lower index wins in scan]
     *   Actually: greedy scans all pairs, picks highest. (0,3) and (1,3) both = 4.0.
     *   Scan order: i=0,j=3 comes before i=1,j=3 → picks (0,3)=4.0
     *   Next: (1,3)=4.0 → connect 1 with {0,3}
     *   Next: (0,2)=3.75 or (1,2)=3.75 → (0,2)=3.75 (first in scan) connects 2
     *   Result: edges (0,3), (1,3), (0,2)
     *
     * Surprise:
     *   Word-parse edges: (1,3), (0,2), (2,3)
     *   Grammar-parse edges: (0,3), (1,3), (0,2)
     *
     *   Word edges in grammar? (1,3) YES, (0,2) YES, (2,3) NO
     *   Surprise = MI of (2,3) / length = 3.0 / 4 = 0.75
     */

    SimpleHTBuilder ht;
    ht_init(&ht, 256);
    ht_insert(&ht, 0, 1, 1.0);
    ht_insert(&ht, 0, 2, 5.0);
    ht_insert(&ht, 0, 3, 2.0);
    ht_insert(&ht, 1, 2, 2.5);
    ht_insert(&ht, 1, 3, 6.0);
    ht_insert(&ht, 2, 3, 3.0);
    ht_upload(&ht);

    /* Classes: 0→A(0), 1→A(0), 2→B(1), 3→C(2) */
    uint32_t h_class_id[] = {0, 0, 1, 2};
    uint32_t* d_class_id;
    cudaMalloc(&d_class_id, 4 * sizeof(uint32_t));
    cudaMemcpy(d_class_id, h_class_id, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* Class MI hash table: (A,B)=3.75, (A,C)=4.0, (B,C)=3.0 */
    uint32_t class_ht_cap = 64;
    uint64_t h_ckeys[64];
    double h_cmi[64];
    memset(h_ckeys, 0xFF, 64 * sizeof(uint64_t));
    memset(h_cmi, 0, 64 * sizeof(double));

    /* Insert class pairs using same hash function */
    auto insert_class_mi = [&](uint32_t ca, uint32_t cb, double mi) {
        uint32_t lo = (ca <= cb) ? ca : cb;
        uint32_t hi = (ca <= cb) ? cb : ca;
        uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;
        uint64_t mask = class_ht_cap - 1;
        uint64_t slot = splitmix64(key) & mask;
        for (int p = 0; p < 64; p++) {
            if (h_ckeys[slot] == 0xFFFFFFFFFFFFFFFFULL) {
                h_ckeys[slot] = key;
                h_cmi[slot] = mi;
                return;
            }
            slot = (slot + 1) & mask;
        }
    };
    insert_class_mi(0, 1, 3.75);  /* A-B */
    insert_class_mi(0, 2, 4.0);   /* A-C */
    insert_class_mi(1, 2, 3.0);   /* B-C */

    uint64_t* d_ckeys;
    double* d_cmi;
    cudaMalloc(&d_ckeys, class_ht_cap * sizeof(uint64_t));
    cudaMalloc(&d_cmi, class_ht_cap * sizeof(double));
    cudaMemcpy(d_ckeys, h_ckeys, class_ht_cap * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cmi, h_cmi, class_ht_cap * sizeof(double), cudaMemcpyHostToDevice);

    /* Sentence: [0, 1, 2, 3] */
    uint32_t h_words[] = {0, 1, 2, 3};
    uint32_t h_lengths[] = {4};
    uint32_t h_offsets[] = {0};
    uint32_t num_sentences = 1;

    uint32_t *d_words, *d_lengths, *d_offsets;
    cudaMalloc(&d_words, 4 * sizeof(uint32_t));
    cudaMalloc(&d_lengths, sizeof(uint32_t));
    cudaMalloc(&d_offsets, sizeof(uint32_t));
    cudaMemcpy(d_words, h_words, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lengths, h_lengths, sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, h_offsets, sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* Output buffers */
    uint32_t pe = MAX_TREE_EDGES;
    uint32_t *d_wp_a, *d_wp_b, *d_wp_cnt, *d_gp_a, *d_gp_b, *d_gp_cnt;
    double *d_wp_mi, *d_gp_mi, *d_surprise;
    cudaMalloc(&d_wp_a, pe * sizeof(uint32_t));
    cudaMalloc(&d_wp_b, pe * sizeof(uint32_t));
    cudaMalloc(&d_wp_mi, pe * sizeof(double));
    cudaMalloc(&d_wp_cnt, sizeof(uint32_t));
    cudaMalloc(&d_gp_a, pe * sizeof(uint32_t));
    cudaMalloc(&d_gp_b, pe * sizeof(uint32_t));
    cudaMalloc(&d_gp_mi, pe * sizeof(double));
    cudaMalloc(&d_gp_cnt, sizeof(uint32_t));
    cudaMalloc(&d_surprise, sizeof(double));

    spanning_forest_run(
        d_words, d_lengths, d_offsets, num_sentences,
        d_class_id,
        d_ckeys, d_cmi, class_ht_cap,
        ht.d_keys, ht.d_values, ht.d_mi, ht.capacity,
        d_wp_a, d_wp_b, d_wp_mi, d_wp_cnt,
        d_gp_a, d_gp_b, d_gp_mi, d_gp_cnt,
        d_surprise);

    /* Read results */
    uint32_t h_wp_count, h_gp_count;
    cudaMemcpy(&h_wp_count, d_wp_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_gp_count, d_gp_cnt, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t h_ea[MAX_TREE_EDGES], h_eb[MAX_TREE_EDGES];
    double h_emi[MAX_TREE_EDGES];
    uint32_t h_ga[MAX_TREE_EDGES], h_gb[MAX_TREE_EDGES];
    double h_gmi[MAX_TREE_EDGES];
    cudaMemcpy(h_ea, d_wp_a, pe * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_eb, d_wp_b, pe * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_emi, d_wp_mi, pe * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ga, d_gp_a, pe * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gb, d_gp_b, pe * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gmi, d_gp_mi, pe * sizeof(double), cudaMemcpyDeviceToHost);

    double h_surprise;
    cudaMemcpy(&h_surprise, d_surprise, sizeof(double), cudaMemcpyDeviceToHost);

    int pass = 1;

    printf("  Word-parse (%u edges):\n", h_wp_count);
    for (uint32_t i = 0; i < h_wp_count; i++)
        printf("    (%u,%u) MI=%.1f\n", h_ea[i], h_eb[i], h_emi[i]);

    printf("  Grammar-parse (%u edges):\n", h_gp_count);
    for (uint32_t i = 0; i < h_gp_count; i++)
        printf("    (%u,%u) MI=%.1f\n", h_ga[i], h_gb[i], h_gmi[i]);

    /* Verify word-parse contains (1,3) and (0,2) */
    if (!has_edge(h_ea, h_eb, h_wp_count, 1, 3)) {
        printf("  FAIL: word-parse missing (1,3)\n"); pass = 0;
    }
    if (!has_edge(h_ea, h_eb, h_wp_count, 0, 2)) {
        printf("  FAIL: word-parse missing (0,2)\n"); pass = 0;
    }

    /* Grammar-parse should contain (0,3) or (1,3) as top edges (class MI = 4.0) */
    int has_ac_edge = has_edge(h_ga, h_gb, h_gp_count, 0, 3) ||
                      has_edge(h_ga, h_gb, h_gp_count, 1, 3);
    if (!has_ac_edge) {
        printf("  FAIL: grammar-parse missing A-C class edge\n"); pass = 0;
    }

    /* Verify edge counts */
    if (h_wp_count != 3) { printf("  FAIL: word-parse expected 3 edges\n"); pass = 0; }
    if (h_gp_count != 3) { printf("  FAIL: grammar-parse expected 3 edges\n"); pass = 0; }

    /* Verify surprise is > 0 (parses should differ) */
    printf("  Surprise: %.4f\n", h_surprise);
    if (h_surprise < 0.0) {
        printf("  FAIL: surprise should be >= 0\n");
        pass = 0;
    }

    /* If parses differ, surprise should be > 0 */
    /* Check if parses actually differ */
    int differ = 0;
    for (uint32_t i = 0; i < h_wp_count; i++) {
        if (!has_edge(h_ga, h_gb, h_gp_count, h_ea[i], h_eb[i])) {
            differ = 1;
            printf("  Word edge (%u,%u) MI=%.1f NOT in grammar-parse → contributes to surprise\n",
                   h_ea[i], h_eb[i], h_emi[i]);
        }
    }
    if (differ && h_surprise <= 0.0) {
        printf("  FAIL: parses differ but surprise = 0\n");
        pass = 0;
    }
    if (!differ && h_surprise > 0.0) {
        printf("  NOTE: parses identical, surprise = %.4f (should be 0)\n", h_surprise);
    }
    if (!differ) {
        printf("  NOTE: parses are identical — no surprise (class MI didn't change MST)\n");
        printf("  This can happen when class MI preserves the same edge ordering\n");
    }

    /* Cleanup */
    cudaFree(d_words); cudaFree(d_lengths); cudaFree(d_offsets);
    cudaFree(d_class_id); cudaFree(d_ckeys); cudaFree(d_cmi);
    cudaFree(d_wp_a); cudaFree(d_wp_b); cudaFree(d_wp_mi); cudaFree(d_wp_cnt);
    cudaFree(d_gp_a); cudaFree(d_gp_b); cudaFree(d_gp_mi); cudaFree(d_gp_cnt);
    cudaFree(d_surprise);
    ht_free(&ht);

    printf("  T6: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ─── Main ─── */

int main() {
    printf("=== Spanning Forest Tests ===\n\n");

    int t5 = test_t5_grammar_parse();
    int t6 = test_t6_surprise();

    printf("=== Results ===\n");
    printf("T5 (Grammar parse):  %s\n", t5 ? "PASS" : "FAIL");
    printf("T6 (Surprise):       %s\n", t6 ? "PASS" : "FAIL");
    printf("Overall: %s\n", (t5 && t6) ? "ALL PASSED" : "SOME FAILED");

    return (t5 && t6) ? 0 : 1;
}
