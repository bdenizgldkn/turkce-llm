/* Katman 0 - Bellek yönetimi.
 *
 * malloc/free kullanılmıyor (dilin standart kütüphanesi yasak).
 * Bunun yerine iki seviyeli bir yapı:
 *
 *   1) Arena: VirtualAlloc ile ayrılan büyük bir bölgeden basit bir
 *      "bump pointer" (ileri kaydırmalı) ayırıcı. Tekil serbest bırakma
 *      (free) yoktur; ya arena_reset ile tamamı sıfırlanır ya da
 *      arena_destroy ile bölge tamamen bırakılır. Tensor/eğitim
 *      arabellekleri gibi ömrü belirli, toplu ayrılan/bırakılan
 *      veriler için idealdir.
 *
 *   2) Allocator: Bir Arena üzerine kurulu, genel amaçlı, tekil
 *      alloc/free destekleyen bir serbest-liste (free-list) ayırıcı.
 *      Tokenizer/morfolojik analiz gibi değişken ömürlü, düzensiz
 *      boyutlu ayırmalar için kullanılacak.
 */
#ifndef RUNTIME_MEMORY_H
#define RUNTIME_MEMORY_H

#include "types.h"

/* ---------------- Arena ---------------- */

typedef struct Arena {
    u8* base;        /* VirtualAlloc'tan gelen ayrılmış (reserved) bölgenin başı */
    u64 reserved;     /* toplam ayrılmış (reserved) bayt */
    u64 committed;    /* fiziksel olarak eşleştirilmiş (committed) bayt */
    u64 used;         /* kullanılan (bump pointer ofseti) bayt */
} Arena;

/* reserve_size: adres alanında ayrılacak (ama henüz committed olmayan) toplam boyut.
 * Bu ucuzdur (sadece adres alanı işaretlenir), bu yüzden cömertçe büyük seçilebilir. */
Arena arena_create(u64 reserve_size);

/* size baytlık, align hizalamalı bir blok ayırır (align 2'nin kuvveti olmalı). */
void* arena_alloc(Arena* arena, u64 size, u64 align);

/* used=0 yapar; committed bellek serbest bırakılmaz (yeniden kullanım için hızlı). */
void arena_reset(Arena* arena);

/* Tüm reserved bölgeyi işletim sistemine iade eder. */
void arena_destroy(Arena* arena);

/* ---------------- Genel amaçlı ayırıcı (malloc/free karşılığı) ---------------- */

typedef struct FreeBlock FreeBlock;

typedef struct Allocator {
    Arena backing;         /* ham bellek buradan gelir */
    FreeBlock* free_list;  /* serbest bloklar (adres sıralı değil, ekleme sıralı) */
    FreeBlock* last_phys;  /* fiziksel (adres) sırada en son oluşturulan blok;
                             * yeni bloğu zincire eklemek ve komşu birleştirme
                             * (coalescing) yapabilmek için gerekli. */
} Allocator;

Allocator allocator_create(u64 reserve_size);
void* allocator_alloc(Allocator* a, u64 size);
void  allocator_free(Allocator* a, void* ptr);
void  allocator_destroy(Allocator* a);

#endif /* RUNTIME_MEMORY_H */
