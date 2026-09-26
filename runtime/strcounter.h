/* Katman 0 (uzantisi) - String -> sayac hash tablosu.
 * Kelime frekans tablosu (korpüs istatistikleri) icin. hashset.h ile
 * ayni FNV-1a + acik adresleme yontemi, ama her anahtarin bir u64
 * sayaci vardir. */
#ifndef RUNTIME_STRCOUNTER_H
#define RUNTIME_STRCOUNTER_H

#include "types.h"
#include "memory.h"

typedef struct StrCounter {
    Allocator* alloc;
    const char** keys;  /* NULL_PTR = bos slot */
    u64* key_lens;
    u64* counts;
    u64 capacity;
    u64 num_keys;
} StrCounter;

StrCounter strcounter_create(Allocator* alloc, u64 capacity_hint);

/* key varsa sayacini 1 artirir, yoksa 1 sayaciyla ekler. */
void strcounter_increment(StrCounter* c, const char* key, u64 key_len);

/* key'in sayacini value'ya DOGRUDAN esitler (yukleme/deserializasyon icin). */
void strcounter_set(StrCounter* c, const char* key, u64 key_len, u64 value);

/* key yoksa 0 dondurur. */
u64 strcounter_get(const StrCounter* c, const char* key, u64 key_len);

#endif /* RUNTIME_STRCOUNTER_H */
