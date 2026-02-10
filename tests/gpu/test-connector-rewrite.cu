/* test-connector-rewrite.cu — Tests T3 and T4 for grammar costs + class MI
 *
 * T3: Grammar costs (100 sections with varying counts)
 * T4: Class MI (5 words, 2 classes, MI aggregation correctness)
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true \
 *     -o test-connector-rewrite \
 *     test-connector-rewrite.cu gpu-connector-rewrite.cu \
 *     -lcudadevrt -lm
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

/* Forward declarations */
extern "C" void grammar_pipeline_run(
    const double* d_sec_count, double* d_sec_cost, uint32_t num_sections,
    const uint32_t* d_pair_word_a, const uint32_t* d_pair_word_b,
    const double* d_pair_mi, const double* d_pair_count, uint32_t num_pairs,
    const uint32_t* d_word_class_id,
    uint64_t* d_ht_keys, double* d_ht_mi_sum, uint32_t* d_ht_count,
    uint32_t ht_capacity, double* d_max_count);

extern "C" double grammar_read_class_mi(
    uint32_t class_a, uint32_t class_b,
    uint64_t* d_ht_keys, double* d_ht_mi_sum, uint32_t ht_capacity);

extern "C" uint32_t grammar_class_mi_count(
    uint64_t* d_ht_keys, uint32_t ht_capacity);

/* ─── T3: Grammar costs (100 sections) ─── */

int test_t3_grammar_costs() {
    printf("T3: Grammar costs (100 sections)\n");

    const uint32_t NUM_SECTIONS = 100;

    /* Create section counts: exponential distribution
     * Section i has count = 100 * exp(-i/20)
     * So section 0 has count 100, section 20 has ~37, section 60 has ~5, etc.
     */
    double h_sec_count[100];
    for (int i = 0; i < (int)NUM_SECTIONS; i++) {
        h_sec_count[i] = 100.0 * exp(-(double)i / 20.0);
    }
    /* Add a few zero-count sections */
    h_sec_count[95] = 0.0;
    h_sec_count[96] = 0.0;
    h_sec_count[97] = 0.5;  /* below threshold */
    h_sec_count[98] = 0.0;
    h_sec_count[99] = 0.0;

    /* Upload */
    double *d_sec_count, *d_sec_cost, *d_max_count;
    cudaMalloc(&d_sec_count, NUM_SECTIONS * sizeof(double));
    cudaMalloc(&d_sec_cost, NUM_SECTIONS * sizeof(double));
    cudaMalloc(&d_max_count, sizeof(double));
    cudaMemcpy(d_sec_count, h_sec_count, NUM_SECTIONS * sizeof(double), cudaMemcpyHostToDevice);

    /* We need dummy pair/class data for the pipeline — but we can just
     * call the cost kernels directly. Let's use the pipeline with empty pairs. */
    uint32_t *d_pair_wa, *d_pair_wb, *d_word_class;
    double *d_pair_mi, *d_pair_count;
    uint64_t *d_ht_keys;
    double *d_ht_mi_sum;
    uint32_t *d_ht_count;
    uint32_t ht_cap = 64;  /* tiny, no pairs */

    cudaMalloc(&d_pair_wa, sizeof(uint32_t));
    cudaMalloc(&d_pair_wb, sizeof(uint32_t));
    cudaMalloc(&d_pair_mi, sizeof(double));
    cudaMalloc(&d_pair_count, sizeof(double));
    cudaMalloc(&d_word_class, sizeof(uint32_t));
    cudaMalloc(&d_ht_keys, ht_cap * sizeof(uint64_t));
    cudaMalloc(&d_ht_mi_sum, ht_cap * sizeof(double));
    cudaMalloc(&d_ht_count, ht_cap * sizeof(uint32_t));

    grammar_pipeline_run(
        d_sec_count, d_sec_cost, NUM_SECTIONS,
        d_pair_wa, d_pair_wb, d_pair_mi, d_pair_count, 0,
        d_word_class,
        d_ht_keys, d_ht_mi_sum, d_ht_count, ht_cap,
        d_max_count);

    /* Read back costs */
    double h_sec_cost[100];
    cudaMemcpy(h_sec_cost, d_sec_cost, NUM_SECTIONS * sizeof(double), cudaMemcpyDeviceToHost);

    int pass = 1;
    double max_count = h_sec_count[0];  /* 100.0 */

    printf("  Max section count: %.1f\n", max_count);
    printf("  Sample costs:\n");

    /* Verify a few specific costs */
    /* Section 0: count=100, cost = -0.5 * log2(100/100) + 0.1 = -0.5 * 0 + 0.1 = 0.1 */
    double expected_0 = 0.1;
    printf("    sec[0]:  count=%.1f  cost=%.4f  expected=%.4f\n",
           h_sec_count[0], h_sec_cost[0], expected_0);
    if (fabs(h_sec_cost[0] - expected_0) > 0.01) {
        printf("    FAIL: sec[0] cost mismatch\n");
        pass = 0;
    }

    /* Section 20: count=100*exp(-1)≈36.79, cost = -0.5 * log2(36.79/100) + 0.1
     * = -0.5 * log2(0.3679) + 0.1 = -0.5 * (-1.443) + 0.1 = 0.7215 + 0.1 = 0.8215 */
    double ratio_20 = h_sec_count[20] / max_count;
    double expected_20 = -0.5 * log2(ratio_20) + 0.1;
    printf("    sec[20]: count=%.1f  cost=%.4f  expected=%.4f\n",
           h_sec_count[20], h_sec_cost[20], expected_20);
    if (fabs(h_sec_cost[20] - expected_20) > 0.01) {
        printf("    FAIL: sec[20] cost mismatch\n");
        pass = 0;
    }

    /* Section 60: count≈4.98, cost should be higher */
    double ratio_60 = h_sec_count[60] / max_count;
    double expected_60 = -0.5 * log2(ratio_60) + 0.1;
    expected_60 = fmax(0.1, fmin(expected_60, 10.0));
    printf("    sec[60]: count=%.2f  cost=%.4f  expected=%.4f\n",
           h_sec_count[60], h_sec_cost[60], expected_60);
    if (fabs(h_sec_cost[60] - expected_60) > 0.05) {
        printf("    FAIL: sec[60] cost mismatch\n");
        pass = 0;
    }

    /* Zero-count sections should have cost 99.0 */
    if (h_sec_cost[95] != 99.0) {
        printf("    FAIL: sec[95] (count=0) cost=%.1f, expected 99.0\n", h_sec_cost[95]);
        pass = 0;
    }

    /* Verify monotonicity: cost should increase as count decreases */
    int monotonic = 1;
    for (int i = 1; i < 95; i++) {  /* skip zero-count tail */
        if (h_sec_cost[i] < h_sec_cost[i-1] - 0.001) {
            printf("    FAIL: non-monotonic at sec[%d]: cost=%.4f < sec[%d]: cost=%.4f\n",
                   i, h_sec_cost[i], i-1, h_sec_cost[i-1]);
            monotonic = 0;
            break;
        }
    }
    if (monotonic) printf("  Cost monotonicity: OK (cost increases as count decreases)\n");
    else pass = 0;

    /* Verify cost range: all costs in [0.1, 10.0] for sections with count >= 1 */
    for (int i = 0; i < (int)NUM_SECTIONS; i++) {
        if (h_sec_count[i] < 1.0) continue;  /* skip pruned sections */
        if (h_sec_cost[i] < 0.1 || h_sec_cost[i] > 10.0) {
            printf("    FAIL: sec[%d] cost=%.4f out of range [0.1, 10.0]\n", i, h_sec_cost[i]);
            pass = 0;
            break;
        }
    }
    /* Verify pruned sections get cost 99.0 */
    for (int i = 0; i < (int)NUM_SECTIONS; i++) {
        if (h_sec_count[i] < 1.0 && h_sec_cost[i] != 99.0) {
            printf("    FAIL: sec[%d] (count=%.2f) cost=%.1f, expected 99.0\n",
                   i, h_sec_count[i], h_sec_cost[i]);
            pass = 0;
            break;
        }
    }

    /* Cleanup */
    cudaFree(d_sec_count); cudaFree(d_sec_cost); cudaFree(d_max_count);
    cudaFree(d_pair_wa); cudaFree(d_pair_wb);
    cudaFree(d_pair_mi); cudaFree(d_pair_count);
    cudaFree(d_word_class);
    cudaFree(d_ht_keys); cudaFree(d_ht_mi_sum); cudaFree(d_ht_count);

    printf("  T3: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ─── T4: Class MI (5 words, 2 classes) ─── */

int test_t4_class_mi() {
    printf("T4: Class MI (5 words, 2 classes)\n");

    /* Setup:
     *   5 words: w0, w1, w2 in class 0; w3, w4 in class 1
     *   10 pairs (5 choose 2):
     *     Within class 0: (0,1)=3.0 (0,2)=2.5 (1,2)=2.8  → same class, skipped
     *     Within class 1: (3,4)=4.0                        → same class, skipped
     *     Cross-class: (0,3)=1.5 (0,4)=2.0 (1,3)=1.8 (1,4)=2.2 (2,3)=1.0 (2,4)=1.3
     *
     *   Class pair (0,1) should accumulate MI from cross-class pairs:
     *     Sum = 1.5 + 2.0 + 1.8 + 2.2 + 1.0 + 1.3 = 9.8
     *     Count = 6
     *     Average = 9.8 / 6 ≈ 1.6333
     */

    const uint32_t NUM_WORDS = 5;
    const uint32_t NUM_PAIRS = 10;

    /* Word pairs and MI values */
    uint32_t h_pair_wa[] = {0, 0, 0, 0, 0, 1, 1, 1, 2, 3};
    uint32_t h_pair_wb[] = {1, 2, 3, 4, 4, 2, 3, 4, 3, 4};
    /* Fix: canonical ordering (lo, hi) — already correct since wa < wb */

    /* Wait — pair (0,4) appears twice? Let me recount:
     * (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4)
     * That's 10 pairs. Let me fix the arrays. */
    uint32_t h_pair_wa_fixed[] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 3};
    uint32_t h_pair_wb_fixed[] = {1, 2, 3, 4, 2, 3, 4, 3, 4, 4};

    double h_pair_mi[] = {3.0, 2.5, 1.5, 2.0, 2.8, 1.8, 2.2, 1.0, 1.3, 4.0};
    double h_pair_count[10];
    for (int i = 0; i < 10; i++) h_pair_count[i] = 10.0;  /* all have count > 0 */

    /* Word classes: w0,w1,w2 → class 0; w3,w4 → class 1 */
    uint32_t h_word_class[] = {0, 0, 0, 1, 1};

    /* Upload */
    uint32_t *d_pair_wa, *d_pair_wb, *d_word_class;
    double *d_pair_mi, *d_pair_count;
    cudaMalloc(&d_pair_wa, NUM_PAIRS * sizeof(uint32_t));
    cudaMalloc(&d_pair_wb, NUM_PAIRS * sizeof(uint32_t));
    cudaMalloc(&d_pair_mi, NUM_PAIRS * sizeof(double));
    cudaMalloc(&d_pair_count, NUM_PAIRS * sizeof(double));
    cudaMalloc(&d_word_class, NUM_WORDS * sizeof(uint32_t));
    cudaMemcpy(d_pair_wa, h_pair_wa_fixed, NUM_PAIRS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pair_wb, h_pair_wb_fixed, NUM_PAIRS * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pair_mi, h_pair_mi, NUM_PAIRS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pair_count, h_pair_count, NUM_PAIRS * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_word_class, h_word_class, NUM_WORDS * sizeof(uint32_t), cudaMemcpyHostToDevice);

    /* Dummy section data (not testing costs here) */
    double *d_sec_count, *d_sec_cost, *d_max_count;
    cudaMalloc(&d_sec_count, sizeof(double));
    cudaMalloc(&d_sec_cost, sizeof(double));
    cudaMalloc(&d_max_count, sizeof(double));

    /* Class MI hash table */
    uint32_t ht_cap = 256;  /* power of 2, plenty for 1 class pair */
    uint64_t *d_ht_keys;
    double *d_ht_mi_sum;
    uint32_t *d_ht_count;
    cudaMalloc(&d_ht_keys, ht_cap * sizeof(uint64_t));
    cudaMalloc(&d_ht_mi_sum, ht_cap * sizeof(double));
    cudaMalloc(&d_ht_count, ht_cap * sizeof(uint32_t));

    /* Run pipeline */
    grammar_pipeline_run(
        d_sec_count, d_sec_cost, 0,  /* no sections */
        d_pair_wa, d_pair_wb, d_pair_mi, d_pair_count, NUM_PAIRS,
        d_word_class,
        d_ht_keys, d_ht_mi_sum, d_ht_count, ht_cap,
        d_max_count);

    int pass = 1;

    /* Check class MI count — should be exactly 1 class pair (0,1) */
    uint32_t mi_count = grammar_class_mi_count(d_ht_keys, ht_cap);
    printf("  Class MI entries: %u (expected 1)\n", mi_count);
    if (mi_count != 1) {
        printf("  FAIL: expected 1 class pair, got %u\n", mi_count);
        pass = 0;
    }

    /* Read class MI for pair (0,1) */
    double class_mi_01 = grammar_read_class_mi(0, 1, d_ht_keys, d_ht_mi_sum, ht_cap);

    /* Expected: avg of cross-class pairs
     * Cross pairs: (0,3)=1.5, (0,4)=2.0, (1,3)=1.8, (1,4)=2.2, (2,3)=1.0, (2,4)=1.3
     * Sum = 9.8, count = 6, avg = 1.6333... */
    double expected_mi = 9.8 / 6.0;
    printf("  Class MI (0,1): %.4f (expected %.4f)\n", class_mi_01, expected_mi);
    if (fabs(class_mi_01 - expected_mi) > 0.01) {
        printf("  FAIL: class MI mismatch\n");
        pass = 0;
    }

    /* Verify symmetric lookup: (1,0) should give same result */
    double class_mi_10 = grammar_read_class_mi(1, 0, d_ht_keys, d_ht_mi_sum, ht_cap);
    printf("  Class MI (1,0): %.4f (should equal (0,1))\n", class_mi_10);
    if (fabs(class_mi_10 - class_mi_01) > 0.001) {
        printf("  FAIL: asymmetric class MI lookup\n");
        pass = 0;
    }

    /* Verify non-existent class pair returns 0 */
    double class_mi_99 = grammar_read_class_mi(0, 99, d_ht_keys, d_ht_mi_sum, ht_cap);
    printf("  Class MI (0,99): %.4f (expected 0.0)\n", class_mi_99);
    if (class_mi_99 != 0.0) {
        printf("  FAIL: non-existent class pair returned non-zero\n");
        pass = 0;
    }

    /* Cleanup */
    cudaFree(d_pair_wa); cudaFree(d_pair_wb);
    cudaFree(d_pair_mi); cudaFree(d_pair_count);
    cudaFree(d_word_class);
    cudaFree(d_sec_count); cudaFree(d_sec_cost); cudaFree(d_max_count);
    cudaFree(d_ht_keys); cudaFree(d_ht_mi_sum); cudaFree(d_ht_count);

    printf("  T4: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ─── Main ─── */

int main() {
    printf("=== Connector Rewrite Tests ===\n\n");

    int t3 = test_t3_grammar_costs();
    int t4 = test_t4_class_mi();

    printf("=== Results ===\n");
    printf("T3 (Grammar costs):  %s\n", t3 ? "PASS" : "FAIL");
    printf("T4 (Class MI):       %s\n", t4 ? "PASS" : "FAIL");
    printf("Overall: %s\n", (t3 && t4) ? "ALL PASSED" : "SOME FAILED");

    return (t3 && t4) ? 0 : 1;
}
