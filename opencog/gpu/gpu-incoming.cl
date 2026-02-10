/*
 * gpu-incoming.cl -- Parallel incoming-set scan for pair pool
 *
 * Scans the pair pool to find all pairs that reference a given
 * word index (as either word_a or word_b). This implements the
 * GPU-side of fetchIncomingByType for GpuStorageNode.
 *
 * One thread per pair slot. Matching pairs write their index
 * into the output array via atomic increment on match_count.
 *
 * This file is concatenated after gpu-hashtable.cl and
 * gpu-atomspace.cl at load time.
 */

/*
 * Find all pairs where word_a == target OR word_b == target.
 *
 * Args:
 *   pair_word_a    - pair pool: word_a column
 *   pair_word_b    - pair pool: word_b column
 *   target_idx     - the word index to search for
 *   pool_count     - number of active pairs in pool
 *   match_indices  - output: pair indices that match
 *   match_count    - output: atomic counter of matches
 *   max_matches    - maximum output capacity
 */
__kernel void incoming_scan(
    __global const uint* pair_word_a,
    __global const uint* pair_word_b,
    uint target_idx,
    uint pool_count,
    __global uint* match_indices,
    __global volatile uint* match_count,
    uint max_matches)
{
    uint tid = get_global_id(0);
    if (tid >= pool_count) return;

    uint wa = pair_word_a[tid];
    uint wb = pair_word_b[tid];

    if (wa == target_idx || wb == target_idx)
    {
        uint pos = atomic_add(match_count, 1U);
        if (pos < max_matches)
            match_indices[pos] = tid;
    }
}
