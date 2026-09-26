#include "hashset.h"

static u64 fnv1a(const char* s, u64 len) {
    u64 h = 1469598103934665603ull; /* FNV offset basis */
    for (u64 i = 0; i < len; i++) {
        h ^= (u8)s[i];
        h *= 1099511628211ull; /* FNV prime */
    }
    return h;
}

static u64 next_pow2(u64 n) {
    u64 p = 1;
    while (p < n) p <<= 1;
    return p;
}

StrHashSet strset_create(Allocator* alloc, u64 capacity_hint) {
    StrHashSet set;
    set.alloc = alloc;
    set.capacity = next_pow2(capacity_hint * 3 + 16);
    set.count = 0;
    set.keys = (const char**)allocator_alloc(alloc, set.capacity * sizeof(char*));
    set.key_lens = (u64*)allocator_alloc(alloc, set.capacity * sizeof(u64));
    for (u64 i = 0; i < set.capacity; i++) set.keys[i] = NULL_PTR;
    return set;
}

static bool32 keys_equal(const char* a, u64 alen, const char* b, u64 blen) {
    if (alen != blen) return FALSE;
    for (u64 i = 0; i < alen; i++) if (a[i] != b[i]) return FALSE;
    return TRUE;
}

void strset_insert(StrHashSet* set, const char* key, u64 key_len) {
    u64 mask = set->capacity - 1;
    u64 idx = fnv1a(key, key_len) & mask;

    while (set->keys[idx] != NULL_PTR) {
        if (keys_equal(set->keys[idx], set->key_lens[idx], key, key_len)) return; /* zaten var */
        idx = (idx + 1) & mask;
    }

    char* copy = (char*)allocator_alloc(set->alloc, key_len);
    for (u64 i = 0; i < key_len; i++) copy[i] = key[i];

    set->keys[idx] = copy;
    set->key_lens[idx] = key_len;
    set->count++;
}

bool32 strset_contains(const StrHashSet* set, const char* key, u64 key_len) {
    u64 mask = set->capacity - 1;
    u64 idx = fnv1a(key, key_len) & mask;

    while (set->keys[idx] != NULL_PTR) {
        if (keys_equal(set->keys[idx], set->key_lens[idx], key, key_len)) return TRUE;
        idx = (idx + 1) & mask;
    }
    return FALSE;
}
