/* gpu-learning-bridge.cu — CUDA bridge for Scheme/C integration
 *
 * Provides a simplified C API that wraps the persistent learning kernel.
 * Manages unified memory allocation, kernel launch, sentence feeding,
 * and result readback.
 *
 * Designed to be compiled into libgpu-learning.so and loaded via
 * Guile's (load-extension) or dlopen.
 *
 * Build:
 *   nvcc -O2 -arch=sm_75 -rdc=true --shared -Xcompiler -fPIC \
 *     -o libgpu-learning.so gpu-learning-bridge.cu gpu-learning-loop.cu \
 *     -lcudadevrt -lm
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

/* ═══════════════════════════════════════════════════════════════
 *  BRIDGE: Manages the full lifecycle
 * ═══════════════════════════════════════════════════════════════ */

static LearningState* g_state = NULL;
static SentenceRing*  g_ring = NULL;
static int*           g_done_flag = NULL;
static int*           g_pause_flag = NULL;
static uint32_t*      g_stats_iteration = NULL;
static uint32_t*      g_stats_pairs = NULL;
static uint32_t*      g_stats_classes = NULL;
static double*        g_stats_entropy = NULL;
static int            g_launched = 0;

extern "C" {

/* Initialize the GPU learning system */
int gpu_cuda_init(uint32_t num_words) {
    if (g_state) {
        printf("gpu_cuda_init: already initialized\n");
        return -1;
    }

    /* Check CUDA device */
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("gpu_cuda_init: no CUDA devices found\n");
        return -1;
    }

    int dev = 0;
    int supports_coop = 0;
    cudaDeviceGetAttribute(&supports_coop, cudaDevAttrCooperativeLaunch, dev);
    if (!supports_coop) {
        printf("gpu_cuda_init: device does not support cooperative launch\n");
        return -1;
    }

    /* Allocate managed memory for control flags */
    cudaMallocManaged(&g_done_flag, sizeof(int));
    cudaMallocManaged(&g_pause_flag, sizeof(int));
    cudaMallocManaged(&g_stats_iteration, sizeof(uint32_t));
    cudaMallocManaged(&g_stats_pairs, sizeof(uint32_t));
    cudaMallocManaged(&g_stats_classes, sizeof(uint32_t));
    cudaMallocManaged(&g_stats_entropy, sizeof(double));

    *g_done_flag = 0;
    *g_pause_flag = 1;  /* start paused */
    *g_stats_iteration = 0;
    *g_stats_pairs = 0;
    *g_stats_classes = 0;
    *g_stats_entropy = 0.0;

    g_state = ll_init(num_words);
    g_ring = ll_init_ring();
    g_launched = 0;

    printf("gpu_cuda_init: initialized for %u words\n", num_words);
    return 0;
}

/* Launch the persistent learning kernel (runs in background) */
int gpu_cuda_launch_learning() {
    if (!g_state || !g_ring) {
        printf("gpu_cuda_launch: not initialized\n");
        return -1;
    }
    if (g_launched) {
        printf("gpu_cuda_launch: already running\n");
        return -1;
    }

    *g_done_flag = 0;
    *g_pause_flag = 0;  /* unpause */

    int err = ll_launch(g_state, g_ring,
                        g_done_flag, g_pause_flag,
                        g_stats_iteration, g_stats_pairs,
                        g_stats_classes, g_stats_entropy);
    if (err == 0) {
        g_launched = 1;
        printf("gpu_cuda_launch: persistent kernel started\n");
    }
    return err;
}

/* Feed a sentence to the GPU (non-blocking, CPU writes to unified memory) */
void gpu_cuda_feed_sentences(uint32_t* words, uint32_t length) {
    if (!g_ring) return;
    ll_feed_sentence(g_ring, words, length);
}

/* Pause/resume processing (kernel stays running but skips stages) */
void gpu_cuda_pause() {
    if (g_pause_flag) *g_pause_flag = 1;
}

void gpu_cuda_resume() {
    if (g_pause_flag) *g_pause_flag = 0;
}

/* Poll current status (non-blocking) */
int gpu_cuda_poll_status(
    uint32_t* out_iteration,
    uint32_t* out_pairs,
    uint32_t* out_classes,
    double*   out_entropy,
    int*      out_done
) {
    if (!g_state) return -1;

    if (out_iteration) *out_iteration = *g_stats_iteration;
    if (out_pairs)     *out_pairs     = *g_stats_pairs;
    if (out_classes)   *out_classes   = *g_stats_classes;
    if (out_entropy)   *out_entropy   = *g_stats_entropy;
    if (out_done)      *out_done      = *g_done_flag;

    return 0;
}

/* Signal the kernel to stop */
void gpu_cuda_stop() {
    if (g_done_flag) *g_done_flag = 1;
}

/* Wait for kernel to finish and read class assignments */
int gpu_cuda_read_classes(uint32_t* out_class_ids, uint32_t num_words) {
    if (!g_state) return -1;

    /* Wait for kernel */
    ll_wait();
    g_launched = 0;

    ll_read_classes(g_state, out_class_ids, num_words);
    return 0;
}

/* Read grammar costs (placeholder — actual grammar cost computation
 * is done by gpu-connector-rewrite.cu kernels, called after the
 * persistent kernel finishes) */
int gpu_cuda_read_grammar(double* out_costs, uint32_t num_sections) {
    /* Not implemented yet — requires running connector-rewrite pipeline
     * after persistent kernel produces classes */
    (void)out_costs;
    (void)num_sections;
    return -1;
}

/* Cleanup everything */
void gpu_cuda_shutdown() {
    if (g_launched) {
        gpu_cuda_stop();
        ll_wait();
        g_launched = 0;
    }

    if (g_state && g_ring) {
        ll_shutdown(g_state, g_ring);
    }
    g_state = NULL;
    g_ring = NULL;

    if (g_done_flag) cudaFree(g_done_flag);
    if (g_pause_flag) cudaFree(g_pause_flag);
    if (g_stats_iteration) cudaFree(g_stats_iteration);
    if (g_stats_pairs) cudaFree(g_stats_pairs);
    if (g_stats_classes) cudaFree(g_stats_classes);
    if (g_stats_entropy) cudaFree(g_stats_entropy);

    g_done_flag = NULL;
    g_pause_flag = NULL;
    g_stats_iteration = NULL;
    g_stats_pairs = NULL;
    g_stats_classes = NULL;
    g_stats_entropy = NULL;

    printf("gpu_cuda_shutdown: cleanup complete\n");
}

} /* extern "C" */
