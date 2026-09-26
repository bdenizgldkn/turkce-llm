/* Katman 0 (uzantisi) - Genel amacli string hash kumesi.
 * Kok sozlugu (286K+ kelime) gibi buyuk kume uyeligi kontrolleri icin.
 * FNV-1a hash + acik adresleme (linear probing). Hicbir kutuphane
 * (hash map/set) kullanilmaz -- sifirdan yazilmistir. */
#ifndef RUNTIME_HASHSET_H
#define RUNTIME_HASHSET_H

#include "types.h"
#include "memory.h"

typedef struct StrHashSet {
    Allocator* alloc;   /* anahtar kopyalari ve slot dizisi buradan ayrilir */
    const char** keys;  /* NULL_PTR = bos slot */
    u64* key_lens;
    u64 capacity;       /* 2'nin kuvveti olmalidir */
    u64 count;
} StrHashSet;

/* capacity_hint: beklenen eleman sayisi; ic capacity buna gore (yaklasik
 * 3 kat, 2'nin kuvvetine yuvarlanarak) belirlenir. */
StrHashSet strset_create(Allocator* alloc, u64 capacity_hint);

void strset_insert(StrHashSet* set, const char* key, u64 key_len);
bool32 strset_contains(const StrHashSet* set, const char* key, u64 key_len);

#endif /* RUNTIME_HASHSET_H */
