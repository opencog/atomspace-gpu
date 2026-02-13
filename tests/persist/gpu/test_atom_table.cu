/*
 * tests/persist/gpu/test_atom_table.cu
 *
 * Standalone CUDA test for GpuAtomTable.
 *
 * Tests:
 *   1. Alloc table
 *   2. Store 3 nodes with names
 *   3. Store 1 binary link (refs 2 nodes)
 *   4. Store 1 ternary link (refs 3 nodes)
 *   5. Store 1 link containing a link (link-of-links)
 *   6. Fetch everything back, verify matches
 *   7. Clear table, verify zeros
 *   8. Free table
 *
 * Build standalone:
 *   nvcc -I../../.. test_atom_table.cu \
 *        ../../../opencog/persist/gpu/GpuAtomTable.cu \
 *        -o test_atom_table && ./test_atom_table
 *
 * Copyright (C) 2026 OpenCog Foundation
 * SPDX-License-Identifier: AGPL-3.0-or-later
 */

#include "opencog/persist/gpu/GpuAtomTable.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>

static int g_failures = 0;

#define CHECK(cond, msg) do {                                       \
    if (!(cond)) {                                                  \
        fprintf(stderr, "  FAIL: %s  [%s:%d]\n", msg, __FILE__, __LINE__); \
        g_failures++;                                               \
    }                                                               \
} while(0)

#define CHECK_EQ_U16(a, b, msg) CHECK((a) == (b), msg)
#define CHECK_EQ_U32(a, b, msg) CHECK((a) == (b), msg)
#define CHECK_EQ_U8(a, b, msg)  CHECK((a) == (b), msg)

#define CHECK_STR(buf, len, expected, msg) do {                     \
    CHECK((len) == (uint16_t)strlen(expected), msg " (length)");    \
    CHECK(memcmp(buf, expected, len) == 0, msg " (content)");       \
} while(0)

/* ---------------------------------------------------------------
 * Fake OpenCog type IDs -- just small integers for testing.
 * In real AtomSpace, these come from atom_types.h generation.
 * --------------------------------------------------------------- */
#define CONCEPT_NODE   1
#define PREDICATE_NODE 2
#define NUMBER_NODE    3
#define LIST_LINK      10
#define EVAL_LINK      11
#define MEMBER_LINK    12

int main()
{
    GpuAtomTable t;
    int rc;

    printf("=== GpuAtomTable CUDA test ===\n");

    /* ---- 1. Alloc ---- */
    printf("1. Allocating table...\n");
    rc = gpu_table_alloc(&t);
    CHECK(rc == 0, "gpu_table_alloc succeeded");
    if (rc != 0) {
        fprintf(stderr, "Cannot allocate GPU table. CUDA device required.\n");
        return 1;
    }
    CHECK_EQ_U32(t.atom_count, 0, "atom_count initially 0");
    CHECK_EQ_U32(t.name_pool_used, 0, "name_pool_used initially 0");
    CHECK_EQ_U32(t.out_pool_used, 0, "out_pool_used initially 0");

    /* ---- 2. Store 3 nodes ---- */
    printf("2. Storing 3 nodes...\n");

    /* Slot 0: ConceptNode "cat" */
    rc = gpu_store_node(&t, 0, CONCEPT_NODE, "cat", 3);
    CHECK(rc == 0, "store node 0 (cat)");

    /* Slot 1: ConceptNode "dog" */
    rc = gpu_store_node(&t, 1, CONCEPT_NODE, "dog", 3);
    CHECK(rc == 0, "store node 1 (dog)");

    /* Slot 2: PredicateNode "is-a" */
    rc = gpu_store_node(&t, 2, PREDICATE_NODE, "is-a", 4);
    CHECK(rc == 0, "store node 2 (is-a)");

    /* ---- 3. Store binary link: ListLink(cat, dog) at slot 3 ---- */
    printf("3. Storing binary link...\n");
    {
        uint32_t out2[2] = {0, 1};  /* cat, dog */
        rc = gpu_store_link(&t, 3, LIST_LINK, out2, 2);
        CHECK(rc == 0, "store link 3 (ListLink)");
    }

    /* ---- 4. Store ternary link: EvalLink(is-a, cat, dog) at slot 4 ---- */
    printf("4. Storing ternary link...\n");
    {
        uint32_t out3[3] = {2, 0, 1};  /* is-a, cat, dog */
        rc = gpu_store_link(&t, 4, EVAL_LINK, out3, 3);
        CHECK(rc == 0, "store link 4 (EvalLink)");
    }

    /* ---- 5. Store link-of-links: MemberLink(ListLink, EvalLink) at slot 5 ---- */
    printf("5. Storing link containing links...\n");
    {
        uint32_t out_links[2] = {3, 4};  /* slot 3 = ListLink, slot 4 = EvalLink */
        rc = gpu_store_link(&t, 5, MEMBER_LINK, out_links, 2);
        CHECK(rc == 0, "store link 5 (MemberLink of links)");
    }

    gpu_table_barrier(&t);

    /* ---- 6. Fetch everything back ---- */
    printf("6. Fetching and verifying...\n");

    /* Fetch node 0 (cat) */
    {
        uint16_t ty; char nbuf[64]; uint16_t nlen;
        rc = gpu_fetch_node(&t, 0, &ty, nbuf, &nlen);
        CHECK(rc == 0, "fetch node 0");
        CHECK_EQ_U16(ty, CONCEPT_NODE, "node 0 type == CONCEPT_NODE");
        CHECK_STR(nbuf, nlen, "cat", "node 0 name == cat");
    }

    /* Fetch node 1 (dog) */
    {
        uint16_t ty; char nbuf[64]; uint16_t nlen;
        rc = gpu_fetch_node(&t, 1, &ty, nbuf, &nlen);
        CHECK(rc == 0, "fetch node 1");
        CHECK_EQ_U16(ty, CONCEPT_NODE, "node 1 type == CONCEPT_NODE");
        CHECK_STR(nbuf, nlen, "dog", "node 1 name == dog");
    }

    /* Fetch node 2 (is-a) */
    {
        uint16_t ty; char nbuf[64]; uint16_t nlen;
        rc = gpu_fetch_node(&t, 2, &ty, nbuf, &nlen);
        CHECK(rc == 0, "fetch node 2");
        CHECK_EQ_U16(ty, PREDICATE_NODE, "node 2 type == PREDICATE_NODE");
        CHECK_STR(nbuf, nlen, "is-a", "node 2 name == is-a");
    }

    /* Fetch link 3 (ListLink(cat, dog)) */
    {
        uint16_t ty; uint32_t obuf[8]; uint16_t ar;
        rc = gpu_fetch_link(&t, 3, &ty, obuf, &ar);
        CHECK(rc == 0, "fetch link 3");
        CHECK_EQ_U16(ty, LIST_LINK, "link 3 type == LIST_LINK");
        CHECK_EQ_U16(ar, 2, "link 3 arity == 2");
        CHECK_EQ_U32(obuf[0], 0, "link 3 out[0] == 0 (cat)");
        CHECK_EQ_U32(obuf[1], 1, "link 3 out[1] == 1 (dog)");
    }

    /* Fetch link 4 (EvalLink(is-a, cat, dog)) */
    {
        uint16_t ty; uint32_t obuf[8]; uint16_t ar;
        rc = gpu_fetch_link(&t, 4, &ty, obuf, &ar);
        CHECK(rc == 0, "fetch link 4");
        CHECK_EQ_U16(ty, EVAL_LINK, "link 4 type == EVAL_LINK");
        CHECK_EQ_U16(ar, 3, "link 4 arity == 3");
        CHECK_EQ_U32(obuf[0], 2, "link 4 out[0] == 2 (is-a)");
        CHECK_EQ_U32(obuf[1], 0, "link 4 out[1] == 0 (cat)");
        CHECK_EQ_U32(obuf[2], 1, "link 4 out[2] == 1 (dog)");
    }

    /* Fetch link 5 (MemberLink(ListLink, EvalLink)) -- link-of-links */
    {
        uint16_t ty; uint32_t obuf[8]; uint16_t ar;
        rc = gpu_fetch_link(&t, 5, &ty, obuf, &ar);
        CHECK(rc == 0, "fetch link 5");
        CHECK_EQ_U16(ty, MEMBER_LINK, "link 5 type == MEMBER_LINK");
        CHECK_EQ_U16(ar, 2, "link 5 arity == 2");
        CHECK_EQ_U32(obuf[0], 3, "link 5 out[0] == 3 (ListLink)");
        CHECK_EQ_U32(obuf[1], 4, "link 5 out[1] == 4 (EvalLink)");
    }

    /* ---- 7. Clear and verify ---- */
    printf("7. Clearing table...\n");
    gpu_table_clear(&t);
    CHECK_EQ_U32(t.atom_count, 0, "atom_count after clear == 0");
    CHECK_EQ_U32(t.name_pool_used, 0, "name_pool_used after clear == 0");
    CHECK_EQ_U32(t.out_pool_used, 0, "out_pool_used after clear == 0");

    /* Verify slot 0 returns zeros after clear */
    {
        uint16_t ty = 0xFFFF; char nbuf[64]; uint16_t nlen = 0xFFFF;
        rc = gpu_fetch_node(&t, 0, &ty, nbuf, &nlen);
        CHECK(rc == 0, "fetch node 0 after clear");
        CHECK_EQ_U16(ty, 0, "node 0 type == 0 after clear");
        CHECK_EQ_U16(nlen, 0, "node 0 name_len == 0 after clear");
    }
    {
        uint16_t ty; uint32_t obuf[8]; uint16_t ar;
        rc = gpu_fetch_link(&t, 3, &ty, obuf, &ar);
        CHECK(rc == 0, "fetch link 3 after clear");
        CHECK_EQ_U16(ty, 0, "link 3 type == 0 after clear");
        CHECK_EQ_U16(ar, 0, "link 3 arity == 0 after clear");
    }

    /* ---- 8. Free ---- */
    printf("8. Freeing table...\n");
    gpu_table_free(&t);

    /* ---- Summary ---- */
    printf("\n");
    if (g_failures == 0) {
        printf("ALL TESTS PASSED\n");
        return 0;
    } else {
        printf("FAILED: %d check(s) failed\n", g_failures);
        return 1;
    }
}
