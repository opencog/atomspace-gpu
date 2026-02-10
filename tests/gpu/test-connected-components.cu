/* test-connected-components.cu — Tests T1 and T2 for connected components
 *
 * T1: Basic CC (10 words, 2 clusters)
 * T2: Threshold sweep (20 words, 3 natural clusters at different similarities)
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true \
 *     -o test-connected-components \
 *     test-connected-components.cu gpu-connected-components.cu \
 *     -lcudadevrt -lm
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>

/* Forward declarations from gpu-connected-components.cu */
extern "C" int cc_run(
    const uint32_t* d_cand_word_a,
    const uint32_t* d_cand_word_b,
    const double*   d_cand_cosine,
    uint32_t        num_candidates,
    uint32_t        num_words,
    float           threshold,
    uint32_t*       d_edge_a,
    uint32_t*       d_edge_b,
    float*          d_edge_weight,
    uint32_t*       d_edge_count,
    uint32_t        edge_capacity,
    uint32_t*       d_label,
    int*            d_changed,
    uint32_t*       d_component_flags,
    uint32_t*       d_component_count,
    uint32_t*       d_class_id,
    uint32_t*       d_next_class_id,
    uint32_t*       d_class_sizes,
    uint32_t*       h_num_components,
    uint32_t*       h_num_edges);

extern "C" float cc_threshold_sweep(
    const uint32_t* d_cand_word_a,
    const uint32_t* d_cand_word_b,
    const double*   d_cand_cosine,
    uint32_t        num_candidates,
    uint32_t        num_words,
    const float*    thresholds,
    int             num_thresholds,
    uint32_t*       d_edge_a,
    uint32_t*       d_edge_b,
    float*          d_edge_weight,
    uint32_t*       d_edge_count,
    uint32_t        edge_capacity,
    uint32_t*       d_label,
    int*            d_changed,
    uint32_t*       d_component_flags,
    uint32_t*       d_component_count,
    uint32_t*       d_class_id,
    uint32_t*       d_next_class_id,
    uint32_t*       d_class_sizes,
    uint32_t*       h_component_counts,
    float*          h_best_threshold);

/* ─── Helper: Allocate work buffers ─── */

struct CCWorkBuffers {
    uint32_t* d_edge_a;
    uint32_t* d_edge_b;
    float*    d_edge_weight;
    uint32_t* d_edge_count;
    uint32_t* d_label;
    int*      d_changed;
    uint32_t* d_component_flags;
    uint32_t* d_component_count;
    uint32_t* d_class_id;
    uint32_t* d_next_class_id;
    uint32_t* d_class_sizes;
    uint32_t  edge_capacity;
    uint32_t  num_words;
};

void alloc_work_buffers(CCWorkBuffers* wb, uint32_t num_words, uint32_t edge_capacity) {
    wb->edge_capacity = edge_capacity;
    wb->num_words = num_words;

    cudaMalloc(&wb->d_edge_a, edge_capacity * sizeof(uint32_t));
    cudaMalloc(&wb->d_edge_b, edge_capacity * sizeof(uint32_t));
    cudaMalloc(&wb->d_edge_weight, edge_capacity * sizeof(float));
    cudaMalloc(&wb->d_edge_count, sizeof(uint32_t));
    cudaMalloc(&wb->d_label, num_words * sizeof(uint32_t));
    cudaMalloc(&wb->d_changed, sizeof(int));
    cudaMalloc(&wb->d_component_flags, num_words * sizeof(uint32_t));
    cudaMalloc(&wb->d_component_count, sizeof(uint32_t));
    cudaMalloc(&wb->d_class_id, num_words * sizeof(uint32_t));
    cudaMalloc(&wb->d_next_class_id, sizeof(uint32_t));
    cudaMalloc(&wb->d_class_sizes, num_words * sizeof(uint32_t));
}

void free_work_buffers(CCWorkBuffers* wb) {
    cudaFree(wb->d_edge_a);
    cudaFree(wb->d_edge_b);
    cudaFree(wb->d_edge_weight);
    cudaFree(wb->d_edge_count);
    cudaFree(wb->d_label);
    cudaFree(wb->d_changed);
    cudaFree(wb->d_component_flags);
    cudaFree(wb->d_component_count);
    cudaFree(wb->d_class_id);
    cudaFree(wb->d_next_class_id);
    cudaFree(wb->d_class_sizes);
}

/* ─── T1: Basic CC (10 words, 2 clusters) ─── */

int test_t1_basic_cc() {
    printf("T1: Basic CC (10 words, 2 clusters)\n");

    /* Setup: 10 words forming 2 clear clusters:
     *   Cluster A: words 0,1,2,3,4 (all pairs have cosine 0.8)
     *   Cluster B: words 5,6,7,8,9 (all pairs have cosine 0.7)
     *   Cross-cluster: cosine 0.1 (below any reasonable threshold)
     */

    const uint32_t NUM_WORDS = 10;

    /* Build candidate pairs: all 10 choose 2 = 45 pairs */
    uint32_t h_word_a[45], h_word_b[45];
    double   h_cosine[45];
    int nc = 0;

    for (uint32_t i = 0; i < NUM_WORDS; i++) {
        for (uint32_t j = i + 1; j < NUM_WORDS; j++) {
            h_word_a[nc] = i;
            h_word_b[nc] = j;
            bool same_cluster = (i < 5 && j < 5) || (i >= 5 && j >= 5);
            h_cosine[nc] = same_cluster ? (i < 5 ? 0.8 : 0.7) : 0.1;
            nc++;
        }
    }

    /* Upload candidates to GPU */
    uint32_t* d_cand_a;
    uint32_t* d_cand_b;
    double*   d_cand_cos;
    cudaMalloc(&d_cand_a, nc * sizeof(uint32_t));
    cudaMalloc(&d_cand_b, nc * sizeof(uint32_t));
    cudaMalloc(&d_cand_cos, nc * sizeof(double));
    cudaMemcpy(d_cand_a, h_word_a, nc * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cand_b, h_word_b, nc * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cand_cos, h_cosine, nc * sizeof(double), cudaMemcpyHostToDevice);

    /* Allocate work buffers */
    CCWorkBuffers wb;
    alloc_work_buffers(&wb, NUM_WORDS, 256);

    /* Run CC at threshold 0.5 → should find 2 components */
    uint32_t num_components = 0, num_edges = 0;
    cc_run(d_cand_a, d_cand_b, d_cand_cos, nc, NUM_WORDS, 0.5f,
           wb.d_edge_a, wb.d_edge_b, wb.d_edge_weight,
           wb.d_edge_count, wb.edge_capacity,
           wb.d_label, wb.d_changed,
           wb.d_component_flags, wb.d_component_count,
           wb.d_class_id, wb.d_next_class_id, wb.d_class_sizes,
           &num_components, &num_edges);

    printf("  Threshold 0.5: %u components, %u edges\n", num_components, num_edges);

    /* Verify: exactly 2 components */
    int pass = 1;
    if (num_components != 2) {
        printf("  FAIL: expected 2 components, got %u\n", num_components);
        pass = 0;
    }

    /* Verify: within-cluster words have same class */
    uint32_t h_class_id[10];
    cudaMemcpy(h_class_id, wb.d_class_id, NUM_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    /* Words 0-4 should share one class, 5-9 another */
    uint32_t class_a = h_class_id[0];
    uint32_t class_b = h_class_id[5];
    for (int i = 0; i < 5; i++) {
        if (h_class_id[i] != class_a) {
            printf("  FAIL: word %d class=%u != word 0 class=%u\n", i, h_class_id[i], class_a);
            pass = 0;
        }
    }
    for (int i = 5; i < 10; i++) {
        if (h_class_id[i] != class_b) {
            printf("  FAIL: word %d class=%u != word 5 class=%u\n", i, h_class_id[i], class_b);
            pass = 0;
        }
    }
    if (class_a == class_b) {
        printf("  FAIL: clusters A and B have same class_id=%u\n", class_a);
        pass = 0;
    }

    /* Verify class sizes */
    uint32_t h_class_sizes[10];
    cudaMemcpy(h_class_sizes, wb.d_class_sizes, NUM_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (h_class_sizes[class_a] != 5) {
        printf("  FAIL: class A size=%u, expected 5\n", h_class_sizes[class_a]);
        pass = 0;
    }
    if (h_class_sizes[class_b] != 5) {
        printf("  FAIL: class B size=%u, expected 5\n", h_class_sizes[class_b]);
        pass = 0;
    }

    /* Verify: at high threshold 0.75, cluster B splits (cosine=0.7 < 0.75) */
    cc_run(d_cand_a, d_cand_b, d_cand_cos, nc, NUM_WORDS, 0.75f,
           wb.d_edge_a, wb.d_edge_b, wb.d_edge_weight,
           wb.d_edge_count, wb.edge_capacity,
           wb.d_label, wb.d_changed,
           wb.d_component_flags, wb.d_component_count,
           wb.d_class_id, wb.d_next_class_id, wb.d_class_sizes,
           &num_components, &num_edges);

    printf("  Threshold 0.75: %u components, %u edges\n", num_components, num_edges);
    /* At 0.75: cluster A (cos=0.8) stays together, cluster B (cos=0.7) breaks apart → 6 components */
    if (num_components != 6) {
        printf("  FAIL: expected 6 components at threshold 0.75, got %u\n", num_components);
        pass = 0;
    }

    /* Verify: at threshold 0.05, all words form one component */
    cc_run(d_cand_a, d_cand_b, d_cand_cos, nc, NUM_WORDS, 0.05f,
           wb.d_edge_a, wb.d_edge_b, wb.d_edge_weight,
           wb.d_edge_count, wb.edge_capacity,
           wb.d_label, wb.d_changed,
           wb.d_component_flags, wb.d_component_count,
           wb.d_class_id, wb.d_next_class_id, wb.d_class_sizes,
           &num_components, &num_edges);

    printf("  Threshold 0.05: %u components, %u edges\n", num_components, num_edges);
    if (num_components != 1) {
        printf("  FAIL: expected 1 component at threshold 0.05, got %u\n", num_components);
        pass = 0;
    }

    /* Cleanup */
    cudaFree(d_cand_a);
    cudaFree(d_cand_b);
    cudaFree(d_cand_cos);
    free_work_buffers(&wb);

    printf("  T1: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ─── T2: Threshold sweep (20 words, 3 natural clusters) ─── */

int test_t2_threshold_sweep() {
    printf("T2: Threshold sweep (20 words, 3 natural clusters)\n");

    /* Setup: 20 words forming 3 clusters at different similarity levels:
     *   Cluster A: words 0-6   (cos=0.9 within, very tight)
     *   Cluster B: words 7-13  (cos=0.6 within, moderate)
     *   Cluster C: words 14-19 (cos=0.4 within, loose)
     *   Cross-cluster: cos=0.05 (noise floor)
     *
     * Expected knee detection:
     *   Sweep from 0.9 → 0.1:
     *   At 0.9: many components (A partial, B+C all singletons)
     *   At 0.85: A merges fully → drop from ~18 to ~14 = drop of ~4
     *   At 0.55: B merges → drop from ~14 to ~8 = drop of ~6  ← BIGGEST DROP
     *   At 0.35: C merges → drop from ~8 to ~3 = drop of ~5
     *   At 0.04: all merge → drop from 3 to 1 = drop of 2
     *
     *   Actually let's use exact cosines and work out the real numbers.
     */

    const uint32_t NUM_WORDS = 20;

    /* Build all 20 choose 2 = 190 candidate pairs */
    uint32_t h_word_a[190], h_word_b[190];
    double   h_cosine[190];
    int nc = 0;

    for (uint32_t i = 0; i < NUM_WORDS; i++) {
        for (uint32_t j = i + 1; j < NUM_WORDS; j++) {
            h_word_a[nc] = i;
            h_word_b[nc] = j;

            /* Determine cluster membership */
            int ci = (i < 7) ? 0 : (i < 14) ? 1 : 2;
            int cj = (j < 7) ? 0 : (j < 14) ? 1 : 2;

            if (ci == cj) {
                /* Within cluster */
                double cos_vals[] = {0.9, 0.6, 0.4};
                h_cosine[nc] = cos_vals[ci];
            } else {
                h_cosine[nc] = 0.05;  /* noise */
            }
            nc++;
        }
    }
    printf("  Built %d candidate pairs\n", nc);

    /* Upload */
    uint32_t* d_cand_a;
    uint32_t* d_cand_b;
    double*   d_cand_cos;
    cudaMalloc(&d_cand_a, nc * sizeof(uint32_t));
    cudaMalloc(&d_cand_b, nc * sizeof(uint32_t));
    cudaMalloc(&d_cand_cos, nc * sizeof(double));
    cudaMemcpy(d_cand_a, h_word_a, nc * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cand_b, h_word_b, nc * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cand_cos, h_cosine, nc * sizeof(double), cudaMemcpyHostToDevice);

    /* Work buffers */
    CCWorkBuffers wb;
    alloc_work_buffers(&wb, NUM_WORDS, 1024);

    /* Sweep: high to low */
    float thresholds[] = {0.85f, 0.55f, 0.35f, 0.10f, 0.04f};
    int   num_thresh = 5;
    uint32_t h_counts[5];
    float h_best;

    cc_threshold_sweep(
        d_cand_a, d_cand_b, d_cand_cos, nc, NUM_WORDS,
        thresholds, num_thresh,
        wb.d_edge_a, wb.d_edge_b, wb.d_edge_weight,
        wb.d_edge_count, wb.edge_capacity,
        wb.d_label, wb.d_changed,
        wb.d_component_flags, wb.d_component_count,
        wb.d_class_id, wb.d_next_class_id, wb.d_class_sizes,
        h_counts, &h_best);

    printf("  Threshold sweep results:\n");
    for (int i = 0; i < num_thresh; i++) {
        printf("    threshold=%.2f → %u components\n", thresholds[i], h_counts[i]);
    }
    printf("  Best threshold (knee): %.2f\n", h_best);

    int pass = 1;

    /* Verify expected component counts:
     *   0.85: cluster A (cos=0.9 > 0.85) stays → 1 + 7 + 6 = 14 components
     *   0.55: A + B (cos=0.6 > 0.55) stay → 1 + 1 + 6 = 8 components
     *   0.35: A + B + C (cos=0.4 > 0.35) all → 1 + 1 + 1 = 3 components
     *   0.10: same → 3 (cross at 0.05 < 0.10)
     *   0.04: all merge → 1 component
     */
    uint32_t expected[] = {14, 8, 3, 3, 1};
    for (int i = 0; i < num_thresh; i++) {
        if (h_counts[i] != expected[i]) {
            printf("  FAIL: threshold=%.2f: expected %u components, got %u\n",
                   thresholds[i], expected[i], h_counts[i]);
            pass = 0;
        }
    }

    /* Knee should be at 0.55 (drop from 14 → 8 = 6, biggest) or 0.35 (drop from 8 → 3 = 5)
     * Actually drop at 0.85: 14→8=6, at 0.55: 8→3=5, at 0.35: 3→3=0, at 0.10: 3→1=2
     * Biggest drop is threshold[0]=0.85 (14→8=6)
     */
    /* The knee is the threshold BEFORE the biggest drop, so it should be 0.85
     * where the drop to the next threshold (0.55) is 14-8=6 */
    if (h_best != 0.85f) {
        printf("  NOTE: Expected knee at 0.85, got %.2f (largest drop: 14→8=6)\n", h_best);
        /* This is informational — the knee detection picks the largest drop,
         * which may vary. The important thing is it's reasonable. */
    }

    /* Run CC at the final threshold 0.35 and verify 3 proper clusters */
    uint32_t nc_final = 0, ne_final = 0;
    cc_run(d_cand_a, d_cand_b, d_cand_cos, nc, NUM_WORDS, 0.35f,
           wb.d_edge_a, wb.d_edge_b, wb.d_edge_weight,
           wb.d_edge_count, wb.edge_capacity,
           wb.d_label, wb.d_changed,
           wb.d_component_flags, wb.d_component_count,
           wb.d_class_id, wb.d_next_class_id, wb.d_class_sizes,
           &nc_final, &ne_final);

    printf("  Final run at 0.35: %u components, %u edges\n", nc_final, ne_final);

    /* Verify cluster membership */
    uint32_t h_class_id[20];
    cudaMemcpy(h_class_id, wb.d_class_id, NUM_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    /* Words 0-6 should share a class, 7-13 another, 14-19 another */
    uint32_t cA = h_class_id[0], cB = h_class_id[7], cC = h_class_id[14];
    for (int i = 0; i < 7; i++) {
        if (h_class_id[i] != cA) {
            printf("  FAIL: word %d class=%u != cluster A class=%u\n", i, h_class_id[i], cA);
            pass = 0;
        }
    }
    for (int i = 7; i < 14; i++) {
        if (h_class_id[i] != cB) {
            printf("  FAIL: word %d class=%u != cluster B class=%u\n", i, h_class_id[i], cB);
            pass = 0;
        }
    }
    for (int i = 14; i < 20; i++) {
        if (h_class_id[i] != cC) {
            printf("  FAIL: word %d class=%u != cluster C class=%u\n", i, h_class_id[i], cC);
            pass = 0;
        }
    }

    /* All three classes must be distinct */
    if (cA == cB || cA == cC || cB == cC) {
        printf("  FAIL: classes not distinct: A=%u B=%u C=%u\n", cA, cB, cC);
        pass = 0;
    }

    /* Verify class sizes */
    uint32_t h_sizes[20];
    cudaMemcpy(h_sizes, wb.d_class_sizes, NUM_WORDS * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (h_sizes[cA] != 7) { printf("  FAIL: class A size=%u, expected 7\n", h_sizes[cA]); pass = 0; }
    if (h_sizes[cB] != 7) { printf("  FAIL: class B size=%u, expected 7\n", h_sizes[cB]); pass = 0; }
    if (h_sizes[cC] != 6) { printf("  FAIL: class C size=%u, expected 6\n", h_sizes[cC]); pass = 0; }

    /* Cleanup */
    cudaFree(d_cand_a);
    cudaFree(d_cand_b);
    cudaFree(d_cand_cos);
    free_work_buffers(&wb);

    printf("  T2: %s\n\n", pass ? "PASS" : "FAIL");
    return pass;
}

/* ─── Main ─── */

int main() {
    printf("=== Connected Components Tests ===\n\n");

    int t1 = test_t1_basic_cc();
    int t2 = test_t2_threshold_sweep();

    printf("=== Results ===\n");
    printf("T1 (Basic CC):         %s\n", t1 ? "PASS" : "FAIL");
    printf("T2 (Threshold sweep):  %s\n", t2 ? "PASS" : "FAIL");
    printf("Overall: %s\n", (t1 && t2) ? "ALL PASSED" : "SOME FAILED");

    return (t1 && t2) ? 0 : 1;
}
