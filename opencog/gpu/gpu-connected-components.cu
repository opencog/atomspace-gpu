/* gpu-connected-components.cu — Shiloach-Vishkin parallel connected components
 *
 * Replaces CPU agglomerative clustering with GPU label propagation on the
 * cosine similarity graph. Threshold sweep with knee detection finds
 * natural word classes.
 *
 * Data layout matches existing OpenCL SoA pools:
 *   - Candidates from gpu-cosine.cl filter_candidates output
 *   - Word pool indices for labels/classes
 *   - Pair key encoding: ((ulong)lo << 32) | (ulong)hi
 *
 * Build: nvcc -O2 -arch=sm_75 -rdc=true -c gpu-connected-components.cu
 */

#include <cstdint>
#include <cstdio>
#include <cfloat>

/* ─── Constants ─── */

#ifndef CC_MAX_WORDS
#define CC_MAX_WORDS      131072   /* matches WORD_CAPACITY */
#endif

#ifndef CC_MAX_EDGES
#define CC_MAX_EDGES      4194304  /* max edges after filtering */
#endif

#ifndef CC_MAX_ITERATIONS
#define CC_MAX_ITERATIONS 64       /* O(log n) convergence, 64 is generous */
#endif

/* Sentinel matching OpenCL conventions */
#define CC_NO_CLASS       0xFFFFFFFFU

/* ─── Device data structures (SoA) ─── */

struct CCEdgeList {
    uint32_t* edge_a;       /* word index (lo) */
    uint32_t* edge_b;       /* word index (hi) */
    float*    edge_weight;  /* cosine similarity */
    uint32_t* edge_count;   /* atomic counter: number of edges */
    uint32_t  capacity;     /* CC_MAX_EDGES */
};

struct CCLabels {
    uint32_t* label;        /* label[word_idx] = component root */
    uint32_t  num_words;    /* active word count */
};

struct CCResult {
    uint32_t* class_id;        /* class_id[word_idx] = sequential class id */
    uint32_t* component_count; /* number of components found */
    uint32_t* class_sizes;     /* size of each class (indexed by class_id) */
    uint32_t  max_classes;     /* max distinct classes */
};

/* ─── Kernel 1: Build edge list from cosine candidates ─── */

__global__ void cc_build_edge_list(
    /* Cosine filter output (from gpu-cosine.cl step 5) */
    const uint32_t* __restrict__ cand_word_a,
    const uint32_t* __restrict__ cand_word_b,
    const double*   __restrict__ cand_cosine,
    uint32_t        num_candidates,
    /* Threshold */
    float           min_cosine,
    /* Output edge list */
    uint32_t*       edge_a,
    uint32_t*       edge_b,
    float*          edge_weight,
    uint32_t*       edge_count,
    uint32_t        edge_capacity
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_candidates) return;

    float cos_val = (float)cand_cosine[tid];
    if (cos_val < min_cosine) return;

    uint32_t idx = atomicAdd(edge_count, 1U);
    if (idx >= edge_capacity) return;  /* safety: don't overflow */

    edge_a[idx]      = cand_word_a[tid];
    edge_b[idx]      = cand_word_b[tid];
    edge_weight[idx]  = cos_val;
}

/* ─── Kernel 2: Initialize labels ─── */

__global__ void cc_init_labels(
    uint32_t* label,
    uint32_t  num_words
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_words) return;
    label[tid] = tid;  /* each word is its own component */
}

/* ─── Kernel 3: Label propagation (hook) ─── */
/* For each edge (u,v): if labels differ, point max to min via atomicMin.
 * This is the "hook" step of Shiloach-Vishkin. */

__global__ void cc_propagate(
    const uint32_t* __restrict__ edge_a,
    const uint32_t* __restrict__ edge_b,
    uint32_t        num_edges,
    uint32_t*       label,
    int*            changed  /* set to 1 if any label changed */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_edges) return;

    uint32_t u = edge_a[tid];
    uint32_t v = edge_b[tid];

    uint32_t lu = label[u];
    uint32_t lv = label[v];

    if (lu == lv) return;

    /* Hook: point the root of the larger label to the root of the smaller */
    uint32_t hi = (lu > lv) ? lu : lv;
    uint32_t lo = (lu > lv) ? lv : lu;

    uint32_t old = atomicMin(&label[hi], lo);
    if (old != lo) {
        *changed = 1;
    }
}

/* ─── Kernel 4: Pointer jumping (compress) ─── */
/* Follow label chains to root: label[i] = label[label[i]] until fixed point.
 * This is the "compress" step — flattens the tree. */

__global__ void cc_compress(
    uint32_t* label,
    uint32_t  num_words
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_words) return;

    uint32_t l = label[tid];
    /* Path compression: follow chain to root */
    while (label[l] != l) {
        l = label[l];
    }
    label[tid] = l;
}

/* ─── Kernel 5: Count components ─── */
/* Each unique root label = one component. Count via atomicAdd on a flag array. */

__global__ void cc_count_components(
    const uint32_t* __restrict__ label,
    uint32_t        num_words,
    uint32_t*       component_flags, /* must be zeroed: size = num_words */
    uint32_t*       component_count  /* output: number of distinct components */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_words) return;

    uint32_t root = label[tid];
    /* Atomic flag: only first thread to mark this root counts it */
    uint32_t old = atomicExch(&component_flags[root], 1U);
    if (old == 0U) {
        atomicAdd(component_count, 1U);
    }
}

/* ─── Kernel 6: Extract classes ─── */
/* Convert root labels to sequential class IDs and compute class sizes.
 * Two-pass: first assign class IDs to roots, then map all words. */

__global__ void cc_assign_root_class_ids(
    const uint32_t* __restrict__ label,
    uint32_t        num_words,
    uint32_t*       class_id,      /* output: class_id[word] */
    uint32_t*       next_class_id  /* atomic counter, starts at 0 */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_words) return;

    /* Only roots assign themselves a new class ID */
    if (label[tid] == tid) {
        uint32_t cid = atomicAdd(next_class_id, 1U);
        class_id[tid] = cid;
    } else {
        class_id[tid] = CC_NO_CLASS;  /* will be filled in second pass */
    }
}

__global__ void cc_map_to_class_ids(
    const uint32_t* __restrict__ label,
    uint32_t        num_words,
    uint32_t*       class_id  /* in/out: roots have IDs, non-roots get mapped */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_words) return;

    if (class_id[tid] == CC_NO_CLASS) {
        /* Follow to root, copy its class ID */
        uint32_t root = label[tid];
        class_id[tid] = class_id[root];
    }
}

__global__ void cc_compute_class_sizes(
    const uint32_t* __restrict__ class_id,
    uint32_t        num_words,
    uint32_t*       class_sizes  /* output: size of each class, zeroed initially */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_words) return;

    uint32_t cid = class_id[tid];
    if (cid != CC_NO_CLASS) {
        atomicAdd(&class_sizes[cid], 1U);
    }
}

/* ─── Host API: Run full connected components ─── */

/* Run CC at a single threshold. Returns number of components found.
 * All buffers must be pre-allocated on device. */
extern "C"
int cc_run(
    /* Cosine candidates (GPU buffers from filter_candidates) */
    const uint32_t* d_cand_word_a,
    const uint32_t* d_cand_word_b,
    const double*   d_cand_cosine,
    uint32_t        num_candidates,
    uint32_t        num_words,
    float           threshold,
    /* Pre-allocated GPU buffers */
    uint32_t*       d_edge_a,
    uint32_t*       d_edge_b,
    float*          d_edge_weight,
    uint32_t*       d_edge_count,    /* single uint32, zeroed before call */
    uint32_t        edge_capacity,
    uint32_t*       d_label,
    int*            d_changed,       /* single int */
    uint32_t*       d_component_flags,  /* size = num_words, zeroed */
    uint32_t*       d_component_count,  /* single uint32, zeroed */
    uint32_t*       d_class_id,
    uint32_t*       d_next_class_id, /* single uint32, zeroed */
    uint32_t*       d_class_sizes,   /* size = num_words, zeroed */
    /* Output (host) */
    uint32_t*       h_num_components,
    uint32_t*       h_num_edges
) {
    int threads = 256;

    /* Step 1: Build edge list from candidates above threshold */
    cudaMemset(d_edge_count, 0, sizeof(uint32_t));
    int blocks_cand = (num_candidates + threads - 1) / threads;
    if (blocks_cand > 0) {
        cc_build_edge_list<<<blocks_cand, threads>>>(
            d_cand_word_a, d_cand_word_b, d_cand_cosine,
            num_candidates, threshold,
            d_edge_a, d_edge_b, d_edge_weight,
            d_edge_count, edge_capacity);
    }
    cudaDeviceSynchronize();

    uint32_t num_edges;
    cudaMemcpy(&num_edges, d_edge_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (h_num_edges) *h_num_edges = num_edges;

    if (num_edges == 0) {
        /* No edges above threshold → every word is its own component */
        if (h_num_components) *h_num_components = num_words;
        return (int)num_words;
    }

    /* Step 2: Init labels */
    int blocks_words = (num_words + threads - 1) / threads;
    cc_init_labels<<<blocks_words, threads>>>(d_label, num_words);
    cudaDeviceSynchronize();

    /* Step 3: Iterate propagate + compress until convergence */
    int blocks_edges = (num_edges + threads - 1) / threads;
    for (int iter = 0; iter < CC_MAX_ITERATIONS; iter++) {
        int h_changed = 0;
        cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice);

        cc_propagate<<<blocks_edges, threads>>>(
            d_edge_a, d_edge_b, num_edges, d_label, d_changed);
        cudaDeviceSynchronize();

        cc_compress<<<blocks_words, threads>>>(d_label, num_words);
        cudaDeviceSynchronize();

        cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        if (!h_changed) break;
    }

    /* Step 4: Count components */
    cudaMemset(d_component_flags, 0, num_words * sizeof(uint32_t));
    cudaMemset(d_component_count, 0, sizeof(uint32_t));
    cc_count_components<<<blocks_words, threads>>>(
        d_label, num_words, d_component_flags, d_component_count);
    cudaDeviceSynchronize();

    uint32_t num_components;
    cudaMemcpy(&num_components, d_component_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    /* Step 5: Extract class IDs */
    cudaMemset(d_next_class_id, 0, sizeof(uint32_t));
    cc_assign_root_class_ids<<<blocks_words, threads>>>(
        d_label, num_words, d_class_id, d_next_class_id);
    cudaDeviceSynchronize();

    cc_map_to_class_ids<<<blocks_words, threads>>>(
        d_label, num_words, d_class_id);
    cudaDeviceSynchronize();

    /* Step 6: Class sizes */
    cudaMemset(d_class_sizes, 0, num_words * sizeof(uint32_t));
    cc_compute_class_sizes<<<blocks_words, threads>>>(
        d_class_id, num_words, d_class_sizes);
    cudaDeviceSynchronize();

    if (h_num_components) *h_num_components = num_components;
    return (int)num_components;
}

/* ─── Host API: Threshold sweep with knee detection ─── */

/* Try CC at decreasing thresholds. Detect knee: sharpest drop in
 * component count relative to threshold step.
 * Returns best threshold (at the knee). */
extern "C"
float cc_threshold_sweep(
    const uint32_t* d_cand_word_a,
    const uint32_t* d_cand_word_b,
    const double*   d_cand_cosine,
    uint32_t        num_candidates,
    uint32_t        num_words,
    /* Sweep parameters */
    const float*    thresholds,      /* host array of thresholds to try */
    int             num_thresholds,
    /* Pre-allocated GPU work buffers */
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
    /* Output: component counts per threshold (host) */
    uint32_t*       h_component_counts,
    /* Output: best threshold */
    float*          h_best_threshold
) {
    if (num_thresholds < 2) {
        if (h_best_threshold) *h_best_threshold = thresholds[0];
        return thresholds[0];
    }

    /* Run CC at each threshold */
    for (int i = 0; i < num_thresholds; i++) {
        uint32_t nc = 0, ne = 0;
        cc_run(
            d_cand_word_a, d_cand_word_b, d_cand_cosine,
            num_candidates, num_words, thresholds[i],
            d_edge_a, d_edge_b, d_edge_weight,
            d_edge_count, edge_capacity,
            d_label, d_changed,
            d_component_flags, d_component_count,
            d_class_id, d_next_class_id, d_class_sizes,
            &nc, &ne);
        h_component_counts[i] = nc;
    }

    /* Find knee: largest drop in component count between consecutive thresholds.
     * Thresholds should be sorted descending (high → low).
     * As threshold decreases, more edges connect, fewer components remain.
     * The knee is where the steepest drop occurs. */
    float best_threshold = thresholds[0];
    int   max_drop = 0;
    for (int i = 0; i < num_thresholds - 1; i++) {
        int drop = (int)h_component_counts[i] - (int)h_component_counts[i + 1];
        if (drop > max_drop) {
            max_drop = drop;
            /* Use the higher threshold (just before the big merge) */
            best_threshold = thresholds[i];
        }
    }

    if (h_best_threshold) *h_best_threshold = best_threshold;
    return best_threshold;
}
