#include "memory.h"
#include "console.h"

/* KATMAN 19 - Coklu platform: Windows'ta VirtualAlloc (reserve/commit
 * ayrimiyla), Linux'ta (Debian VM, bkz. PROJE_PLANI.md Bolum 19) mmap
 * (anonim, ozel eslem). Linux'ta ayri bir "commit" adimi YOKTUR --
 * sayfalar sadece GERCEKTEN DOKUNULDUGUNDA fiziksel RAM'e baglanir
 * (tembel/lazy sayfalama), yani mmap'in kendisi zaten "ucuz rezerv +
 * gerektiginde otomatik commit" davranisini verir. Bu yuzden POSIX
 * tarafinda committed'i bastan reserved'e esitleyip arena_ensure_committed'i
 * dogal olarak no-op yapiyoruz -- ayri bir kod yolu gerekmiyor. */
#ifdef _WIN32
#include "win32_syscalls.h"
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

/* ================= Arena ================= */

#define ARENA_COMMIT_CHUNK (1ull << 20) /* 1 MiB adımlarla commit et (sadece Windows) */

static u64 align_up(u64 value, u64 align) {
    return (value + (align - 1)) & ~(align - 1);
}

Arena arena_create(u64 reserve_size) {
    Arena a;
#ifdef _WIN32
    a.base = (u8*)VirtualAlloc(NULL_PTR, reserve_size, W_MEM_RESERVE, W_PAGE_READWRITE);
    a.committed = 0;
#else
    void* p = mmap(NULL_PTR, reserve_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    a.base = (p == MAP_FAILED) ? NULL_PTR : (u8*)p;
    a.committed = reserve_size; /* bkz. yukaridaki not -- Linux'ta mmap zaten "commit" sayilir */
#endif
    a.reserved = reserve_size;
    a.used = 0;
    return a;
}

#ifdef _WIN32
static void arena_ensure_committed(Arena* arena, u64 needed) {
    if (needed <= arena->committed) return;

    u64 new_committed = align_up(needed, ARENA_COMMIT_CHUNK);
    if (new_committed > arena->reserved) {
        new_committed = arena->reserved; /* reserved sınırına sabitle */
    }

    u64 grow = new_committed - arena->committed;
    if (grow == 0) return;

    VirtualAlloc(arena->base + arena->committed, grow, W_MEM_COMMIT, W_PAGE_READWRITE);
    arena->committed = new_committed;
}
#else
static void arena_ensure_committed(Arena* arena, u64 needed) {
    (void)arena; (void)needed; /* Linux'ta gerekmez -- bkz. arena_create notu */
}
#endif

void* arena_alloc(Arena* arena, u64 size, u64 align) {
    u64 aligned_used = align_up(arena->used, align);
    u64 new_used = aligned_used + size;

    /* KOK NEDEN KORUMASI: arena_ensure_committed, new_used reserved'i
     * astiginda commit'i SESSIZCE reserved'e sabitliyordu -- ama bu
     * fonksiyon yine de new_used'a kadar bir 'used' ilerletip commit
     * EDILMEMIS (hatta bazen hic REZERVE EDILMEMIS) bellege isaretci
     * donduruyordu, bu da o isaretciye yazildiginda SESSIZCE SEGFAULT'A
     * yol aciyordu (gercek olcekte, batch=8 tensor-batching denemesinde
     * yakalandi -- bkz. PROJE_PLANI.md Bolum 18). Bunun yerine, acikca
     * ve hemen basarisiz ol: bu, cagiranin reserve_size'ini buyutmesi
     * gerektigini gosteren net bir tanilama mesajidir. */
    if (new_used > arena->reserved) {
        console_write("[BELLEK HATASI] Arena rezerv sinirini asti: ihtiyac=");
        console_write_u64(new_used);
        console_write(" bayt, rezerv=");
        console_write_u64(arena->reserved);
        console_write_line(" bayt. allocator_create/arena_create'e verilen reserve_size'i buyutun.");
#ifdef _WIN32
        ExitProcess(1);
#else
        _exit(1);
#endif
    }

    arena_ensure_committed(arena, new_used);

    void* ptr = arena->base + aligned_used;
    arena->used = new_used;
    return ptr;
}

void arena_reset(Arena* arena) {
    arena->used = 0;
}

void arena_destroy(Arena* arena) {
    if (arena->base) {
#ifdef _WIN32
        VirtualFree(arena->base, 0, W_MEM_RELEASE);
#else
        munmap(arena->base, arena->reserved);
#endif
    }
    arena->base = NULL_PTR;
    arena->reserved = 0;
    arena->committed = 0;
    arena->used = 0;
}

/* ================= Genel amaçlı ayırıcı (free-list) ================= */
/*
 * Her blok, kullanıcı verisinden hemen önce bir BlockHeader taşır.
 * Bloklar adres sırasına göre next_phys/prev_phys ile zincirlenir
 * (komşu serbest blokları birleştirebilmek/coalesce için).
 * Serbest bloklar ayrıca free_list üzerinde de zincirlenir.
 */

struct FreeBlock {
    u64 size;              /* kullanıcı verisi alanının boyutu (header hariç) */
    bool32 is_free;
    struct FreeBlock* next_phys; /* bellekte bir sonraki blok (adres sırası) */
    struct FreeBlock* prev_phys; /* bellekte bir önceki blok (adres sırası) */
    struct FreeBlock* next_free; /* sadece is_free==TRUE iken anlamlı */
    struct FreeBlock* prev_free;
};

#define BLOCK_ALIGN 16u
#define HEADER_SIZE (sizeof(FreeBlock))
#define MIN_SPLIT_REMAINDER (HEADER_SIZE + 32)

Allocator allocator_create(u64 reserve_size) {
    Allocator a;
    a.backing = arena_create(reserve_size);
    a.free_list = NULL_PTR;
    a.last_phys = NULL_PTR;
    return a;
}

static void free_list_remove(Allocator* a, FreeBlock* b) {
    if (b->prev_free) b->prev_free->next_free = b->next_free;
    else a->free_list = b->next_free;
    if (b->next_free) b->next_free->prev_free = b->prev_free;
    b->next_free = NULL_PTR;
    b->prev_free = NULL_PTR;
}

static void free_list_push(Allocator* a, FreeBlock* b) {
    b->is_free = TRUE;
    b->next_free = a->free_list;
    b->prev_free = NULL_PTR;
    if (a->free_list) a->free_list->prev_free = b;
    a->free_list = b;
}

void* allocator_alloc(Allocator* a, u64 size) {
    u64 want = align_up(size, BLOCK_ALIGN);

    /* 1) Serbest listede ilk uygun (first-fit) bloğu ara. */
    FreeBlock* b = a->free_list;
    while (b) {
        if (b->size >= want) {
            free_list_remove(a, b);

            /* Blok yeterince büyükse ikiye böl, artan parçayı serbest bırak. */
            if (b->size >= want + MIN_SPLIT_REMAINDER) {
                u8* data_start = (u8*)b + HEADER_SIZE;
                FreeBlock* rest = (FreeBlock*)(data_start + want);
                rest->size = b->size - want - HEADER_SIZE;
                rest->is_free = TRUE;
                rest->next_phys = b->next_phys;
                rest->prev_phys = b;
                if (b->next_phys) b->next_phys->prev_phys = rest;
                else a->last_phys = rest; /* b, fiziksel zincirin sonuydu; artık rest sonda */
                b->next_phys = rest;
                b->size = want;
                free_list_push(a, rest);
            }

            b->is_free = FALSE;
            return (u8*)b + HEADER_SIZE;
        }
        b = b->next_free;
    }

    /* 2) Serbest listede yer yoksa arena'dan taze blok al ve fiziksel
     *    (adres sıralı) zincirin sonuna ekle — bu, allocator_free'de
     *    komşu bloklarla birleştirme (coalescing) yapabilmek için gerekli. */
    FreeBlock* nb = (FreeBlock*)arena_alloc(&a->backing, HEADER_SIZE + want, BLOCK_ALIGN);
    nb->size = want;
    nb->is_free = FALSE;
    nb->next_free = NULL_PTR;
    nb->prev_free = NULL_PTR;
    nb->next_phys = NULL_PTR;
    nb->prev_phys = a->last_phys;
    if (a->last_phys) a->last_phys->next_phys = nb;
    a->last_phys = nb;

    return (u8*)nb + HEADER_SIZE;
}

/* b'yi fiziksel zincirden çıkarmadan, verilerini into'ya taşıyarak
 * b ile into'yu tek bir bloğa birleştirir (into fiziksel olarak b'nin
 * hemen komşusu olmalı). into'nun boyutunu günceller. */
static void merge_into(Allocator* a, FreeBlock* into, FreeBlock* consumed) {
    (void)a;
    into->size += HEADER_SIZE + consumed->size;
}

void allocator_free(Allocator* a, void* ptr) {
    if (!ptr) return;
    FreeBlock* b = (FreeBlock*)((u8*)ptr - HEADER_SIZE);
    b->is_free = TRUE;

    /* Sonraki komşu serbestse: b'yi büyüt, komşuyu zincirden ve
     * serbest listeden düşür. */
    FreeBlock* next = b->next_phys;
    if (next && next->is_free) {
        free_list_remove(a, next);
        merge_into(a, b, next);
        b->next_phys = next->next_phys;
        if (next->next_phys) next->next_phys->prev_phys = b;
        if (a->last_phys == next) a->last_phys = b;
    }

    /* Önceki komşu serbestse: önceki bloğu büyüt, b'yi ona yedir. */
    FreeBlock* prev = b->prev_phys;
    if (prev && prev->is_free) {
        free_list_remove(a, prev);
        merge_into(a, prev, b);
        prev->next_phys = b->next_phys;
        if (b->next_phys) b->next_phys->prev_phys = prev;
        if (a->last_phys == b) a->last_phys = prev;
        b = prev;
    }

    free_list_push(a, b);
}

void allocator_destroy(Allocator* a) {
    arena_destroy(&a->backing);
    a->free_list = NULL_PTR;
}
