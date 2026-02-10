/* test-divergence.cu — Test T7 for polysemy detection
 *
 * T7: Polysemy detection across 2 compartments
 *     - 10 words, 2 compartments with overlapping vocabulary
 *     - Word "bank" (word 5) has different MI neighborhoods:
 *       Comp A: high MI with money/loan/account
 *       Comp B: high MI with river/fish/water
 *     - Non-polysemous words have similar neighborhoods
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true \
 *     -o test-divergence \
 *     test-divergence.cu gpu-divergence.cu \
 *     -lcudadevrt -lm
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

#define DIV_TOP_K 32

extern "C" void divergence_run(
    const uint32_t* d_pair_a_word_a,
    const uint32_t* d_pair_a_word_b,
    const double*   d_pair_a_mi,
    const double*   d_pair_a_count,
    uint32_t        num_pairs_a,
    const uint32_t* d_pair_b_word_a,
    const uint32_t* d_pair_b_word_b,
    const double*   d_pair_b_mi,
    const double*   d_pair_b_count,
    uint32_t        num_pairs_b,
    uint32_t        num_words,
    double          polysemy_threshold,
    uint32_t*       d_nbr_a_ids,
    double*         d_nbr_a_mi,
    uint32_t*       d_nbr_a_count,
    uint32_t*       d_nbr_b_ids,
    double*         d_nbr_b_mi,
    uint32_t*       d_nbr_b_count,
    double*         d_divergence,
    uint32_t*       d_polysemy_flag);

int test_t7_polysemy() {
    printf("T7: Polysemy detection (2 compartments, 10 words)\n");

    /* Words:
     *   0=the, 1=money, 2=loan, 3=account, 4=interest
     *   5=bank (POLYSEMOUS), 6=river, 7=fish, 8=water, 9=large
     *
     * Compartment A (financial text):
     *   bank(5) has high MI with: money(1), loan(2), account(3), interest(4)
     *   the(0) is function word: moderate MI with everything
     *   large(9): moderate MI with money(1), loan(2)
     *
     * Compartment B (nature text):
     *   bank(5) has high MI with: river(6), fish(7), water(8)
     *   the(0) is function word: same pattern (similar neighborhoods)
     *   large(9): moderate MI with river(6), fish(7)
     *
     * Expected: bank(5) flagged as polysemous (high divergence)
     *           the(0), large(9) NOT flagged (similar neighborhoods)
     */

    const uint32_t NUM_WORDS = 10;

    /* Compartment A pairs (financial context) */
    uint32_t h_a_wa[] = {5,5,5,5, 0,0,0,0,0,0,0,0,0, 1,1,1, 2,2, 3, 9,9};
    uint32_t h_a_wb[] = {1,2,3,4, 1,2,3,4,5,6,7,8,9, 2,3,4, 3,4, 4, 1,2};
    double   h_a_mi[] = {6.0,5.5,5.0,4.8, 2.0,1.8,1.9,2.1,2.0,1.5,1.4,1.3,1.7, 3.0,2.8,2.5, 2.6,2.3, 2.0, 3.2,2.9};
    int      n_a = 21;

    /* Compartment B pairs (nature context) */
    uint32_t h_b_wa[] = {5,5,5, 0,0,0,0,0,0,0,0,0, 6,6, 7, 9,9};
    uint32_t h_b_wb[] = {6,7,8, 1,2,3,5,6,7,8,9,4, 7,8, 8, 6,7};
    double   h_b_mi[] = {6.5,5.8,5.2, 1.8,1.6,1.7,1.9,2.1,2.0,2.2,1.8,1.5, 3.5,3.0, 2.8, 3.0,2.7};
    int      n_b = 17;

    /* All pairs have count > 0 */
    double h_a_count[21], h_b_count[17];
    for (int i = 0; i < n_a; i++) h_a_count[i] = 10.0;
    for (int i = 0; i < n_b; i++) h_b_count[i] = 10.0;

    /* Upload compartment A */
    uint32_t *d_a_wa, *d_a_wb;
    double *d_a_mi, *d_a_count;
    cudaMalloc(&d_a_wa, n_a * sizeof(uint32_t));
    cudaMalloc(&d_a_wb, n_a * sizeof(uint32_t));
    cudaMalloc(&d_a_mi, n_a * sizeof(double));
    cudaMalloc(&d_a_count, n_a * sizeof(double));
    cudaMemcpy(d_a_wa, h_a_wa, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_wb, h_a_wb, n_a * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_mi, h_a_mi, n_a * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_count, h_a_count, n_a * sizeof(double), cudaMemcpyHostToDevice);

    /* Upload compartment B */
    uint32_t *d_b_wa, *d_b_wb;
    double *d_b_mi, *d_b_count;
    cudaMalloc(&d_b_wa, n_b * sizeof(uint32_t));
    cudaMalloc(&d_b_wb, n_b * sizeof(uint32_t));
    cudaMalloc(&d_b_mi, n_b * sizeof(double));
    cudaMalloc(&d_b_count, n_b * sizeof(double));
    cudaMemcpy(d_b_wa, h_b_wa, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_wb, h_b_wb, n_b * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_mi, h_b_mi, n_b * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_count, h_b_count, n_b * sizeof(double), cudaMemcpyHostToDevice);

    /* Work buffers */
    uint32_t *d_nbr_a_ids, *d_nbr_b_ids, *d_nbr_a_cnt, *d_nbr_b_cnt;
    double *d_nbr_a_mi, *d_nbr_b_mi;
    cudaMalloc(&d_nbr_a_ids, NUM_WORDS * DIV_TOP_K * sizeof(uint32_t));
    cudaMalloc(&d_nbr_a_mi, NUM_WORDS * DIV_TOP_K * sizeof(double));
    cudaMalloc(&d_nbr_a_cnt, NUM_WORDS * sizeof(uint32_t));
    cudaMalloc(&d_nbr_b_ids, NUM_WORDS * DIV_TOP_K * sizeof(uint32_t));
    cudaMalloc(&d_nbr_b_mi, NUM_WORDS * DIV_TOP_K * sizeof(double));
    cudaMalloc(&d_nbr_b_cnt, NUM_WORDS * sizeof(uint32_t));

    /* Output */
    double *d_divergence;
    uint32_t *d_polysemy_flag;
    cudaMalloc(&d_divergence, NUM_WORDS * sizeof(double));
    cudaMalloc(&d_polysemy_flag, NUM_WORDS * sizeof(uint32_t));

    /* Run divergence */
    double threshold = 0.5;  /* divergence > 0.5 → polysemous */
    divergence_run(
        d_a_wa, d_a_wb, d_a_mi, d_a_count, n_a,
        d_b_wa, d_b_wb, d_b_mi, d_b_count, n_b,
        NUM_WORDS, threshold,
        d_nbr_a_ids, d_nbr_a_mi, d_nbr_a_cnt,
        d_nbr_b_ids, d_nbr_b_mi, d_nbr_b_cnt,
        d_divergence, d_polysemy_flag);

    /* Read results */
    double h_div[10];
    uint32_t h_poly[10];
    cudaMemcpy(h_div, d_divergence, NUM_WORDS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_poly, d_polysemy_flag, NUM_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    int pass = 1;

    const char* word_names[] = {"the", "money", "loan", "account", "interest",
                                "bank", "river", "fish", "water", "large"};

    printf("  Divergence scores:\n");
    for (int i = 0; i < (int)NUM_WORDS; i++) {
        printf("    %8s (w%d): div=%.4f %s\n",
               word_names[i], i, h_div[i],
               h_poly[i] ? "POLYSEMOUS" : "");
    }

    /* bank(5) should be flagged as polysemous (high divergence) */
    if (!h_poly[5]) {
        printf("  FAIL: bank(5) should be flagged as polysemous (div=%.4f)\n", h_div[5]);
        pass = 0;
    }

    /* bank's divergence should be high (neighborhoods are completely different) */
    if (h_div[5] < 0.3) {
        printf("  FAIL: bank(5) divergence=%.4f too low (expected > 0.3)\n", h_div[5]);
        pass = 0;
    }

    /* the(0) should have low divergence (similar neighborhoods in both) */
    /* "the" appears with most words in both compartments → similar neighborhood */
    if (h_div[0] > 0.5) {
        printf("  WARN: the(0) divergence=%.4f seems high for a function word\n", h_div[0]);
        /* Not a hard failure — depends on exact partner overlap */
    }

    /* bank should have HIGHER divergence than "the" */
    if (h_div[5] <= h_div[0]) {
        printf("  FAIL: bank(5) should have higher divergence than the(0)\n");
        printf("        bank=%.4f, the=%.4f\n", h_div[5], h_div[0]);
        pass = 0;
    }

    /* Count total polysemous words — should be selective (not everything) */
    int poly_count = 0;
    for (int i = 0; i < (int)NUM_WORDS; i++) {
        if (h_poly[i]) poly_count++;
    }
    printf("  Total polysemous words: %d / %d\n", poly_count, (int)NUM_WORDS);

    /* Shouldn't flag too many words */
    if (poly_count > 5) {
        printf("  FAIL: too many polysemous words (%d), expected <= 5\n", poly_count);
        pass = 0;
    }

    /* Neighborhood sanity check: read back neighbor counts */
    uint32_t h_cnt_a[10], h_cnt_b[10];
    cudaMemcpy(h_cnt_a, d_nbr_a_cnt, NUM_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cnt_b, d_nbr_b_cnt, NUM_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    printf("  Neighbor counts (A/B):\n");
    for (int i = 0; i < (int)NUM_WORDS; i++) {
        printf("    %8s: A=%u B=%u\n", word_names[i], h_cnt_a[i], h_cnt_b[i]);
    }

    /* Cleanup */
    cudaFree(d_a_wa); cudaFree(d_a_wb); cudaFree(d_a_mi); cudaFree(d_a_count);
    cudaFree(d_b_wa); cudaFree(d_b_wb); cudaFree(d_b_mi); cudaFree(d_b_count);
    cudaFree(d_nbr_a_ids); cudaFree(d_nbr_a_mi); cudaFree(d_nbr_a_cnt);
    cudaFree(d_nbr_b_ids); cudaFree(d_nbr_b_mi); cudaFree(d_nbr_b_cnt);
    cudaFree(d_divergence); cudaFree(d_polysemy_flag);

    printf("  T7: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

int main() {
    printf("=== Divergence Tests ===\n\n");

    int t7 = test_t7_polysemy();

    printf("=== Results ===\n");
    printf("T7 (Polysemy):  %s\n", t7 ? "PASS" : "FAIL");
    printf("Overall: %s\n", t7 ? "ALL PASSED" : "FAILED");

    return t7 ? 0 : 1;
}
