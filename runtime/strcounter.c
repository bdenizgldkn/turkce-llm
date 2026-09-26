#include "strcounter.h"

static u64 fnv1a(const char* s, u64 len) {
    u64 h = 1469598103934665603ull;
    for (u64 i = 0; i < len; i++) { h ^= (u8)s[i]; h *= 1099511628211ull; }
    return h;
}

static u64 next_pow2(u64 n) {
    u64 p = 1;
    while (p < n) p <<= 1;
    return p;
}

static bool32 keys_equal(const char* a, u64 alen, const char* b, u64 blen) {
    if (alen != blen) return FALSE;
    for (u64 i = 0; i < alen; i++) if (a[i] != b[i]) return FALSE;
    return TRUE;
}

StrCounter strcounter_create(Allocator* alloc, u64 capacity_hint) {
    StrCounter c;
    c.alloc = alloc;
    c.capacity = next_pow2(capacity_hint * 3 + 16);
    c.num_keys = 0;
    c.keys = (const char**)allocator_alloc(alloc, c.capacity * sizeof(char*));
    c.key_lens = (u64*)allocator_alloc(alloc, c.capacity * sizeof(u64));
    c.counts = (u64*)allocator_alloc(alloc, c.capacity * sizeof(u64));
    for (u64 i = 0; i < c.capacity; i++) c.keys[i] = NULL_PTR;
    return c;
}

void strcounter_increment(StrCounter* c, const char* key, u64 key_len) {
    u64 mask = c->capacity - 1;
    u64 idx = fnv1a(key, key_len) & mask;

    while (c->keys[idx] != NULL_PTR) {
        if (keys_equal(c->keys[idx], c->key_lens[idx], key, key_len)) {
            c->counts[idx]++;
            return;
        }
        idx = (idx + 1) & mask;
    }

    char* copy = (char*)allocator_alloc(c->alloc, key_len);
    for (u64 i = 0; i < key_len; i++) copy[i] = key[i];

    c->keys[idx] = copy;
    c->key_lens[idx] = key_len;
    c->counts[idx] = 1;
    c->num_keys++;
}

void strcounter_set(StrCounter* c, const char* key, u64 key_len, u64 value) {
    u64 mask = c->capacity - 1;
    u64 idx = fnv1a(key, key_len) & mask;

    while (c->keys[idx] != NULL_PTR) {
        if (keys_equal(c->keys[idx], c->key_lens[idx], key, key_len)) {
            c->counts[idx] = value;
            return;
        }
        idx = (idx + 1) & mask;
    }

    char* copy = (char*)allocator_alloc(c->alloc, key_len);
    for (u64 i = 0; i < key_len; i++) copy[i] = key[i];

    c->keys[idx] = copy;
    c->key_lens[idx] = key_len;
    c->counts[idx] = value;
    c->num_keys++;
}

u64 strcounter_get(const StrCounter* c, const char* key, u64 key_len) {
    u64 mask = c->capacity - 1;
    u64 idx = fnv1a(key, key_len) & mask;

    while (c->keys[idx] != NULL_PTR) {
        if (keys_equal(c->keys[idx], c->key_lens[idx], key, key_len)) return c->counts[idx];
        idx = (idx + 1) & mask;
    }
    return 0;
}
