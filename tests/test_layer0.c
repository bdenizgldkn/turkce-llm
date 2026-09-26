/* Katman 0 doğrulama testleri: Arena, Allocator (free-list + coalescing),
 * ve dosya I/O. Hiçbir standart kütüphane/test framework'ü kullanılmaz;
 * kendi console_write yardımcımızla PASS/FAIL raporlanır. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/file_io.h"
#include "../runtime/console.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; console_write("  [FAIL] "); console_write_line(msg); } \
} while (0)

static void test_arena_basic(void) {
    Arena a = arena_create(8ull * 1024 * 1024); /* 8 MiB reserve */

    u8* p1 = (u8*)arena_alloc(&a, 100, 16);
    for (i32 i = 0; i < 100; i++) p1[i] = (u8)i;
    for (i32 i = 0; i < 100; i++) CHECK(p1[i] == (u8)i, "arena: yazilan deger okunamadi");

    u8* p2 = (u8*)arena_alloc(&a, 16, 16);
    CHECK((u64)((u8*)p2 - (u8*)p1) >= 100, "arena: bloklar cakisiyor");
    CHECK(((u64)p2 & 15) == 0, "arena: hizalama (align=16) bozuk");

    /* commit sinirini asan buyuk bir blok: 1 MiB commit-chunk'i asan 2 MiB */
    u8* p3 = (u8*)arena_alloc(&a, 2ull * 1024 * 1024, 16);
    p3[0] = 0xAB;
    p3[2 * 1024 * 1024 - 1] = 0xCD;
    CHECK(p3[0] == 0xAB && p3[2 * 1024 * 1024 - 1] == 0xCD,
          "arena: commit siniri asilinca yazma basarisiz");

    u64 used_before_reset = a.used;
    arena_reset(&a);
    CHECK(a.used == 0, "arena: reset sonrasi used sifir degil");
    CHECK(used_before_reset > 0, "arena: reset oncesi used sifirdi (test gecersiz)");

    arena_destroy(&a);
    CHECK(a.base == NULL_PTR, "arena: destroy sonrasi base temizlenmedi");
}

static void test_allocator_reuse(void) {
    Allocator a = allocator_create(1024ull * 1024);

    void* x = allocator_alloc(&a, 64);
    void* y = allocator_alloc(&a, 64);
    void* z = allocator_alloc(&a, 64);
    CHECK(x && y && z, "allocator: temel alloc basarisiz");
    CHECK(x != y && y != z && x != z, "allocator: farkli bloklar ayni adresi verdi");

    allocator_free(&a, y);
    void* y2 = allocator_alloc(&a, 64);
    CHECK(y2 == y, "allocator: serbest kalan blok yeniden kullanilmadi (first-fit)");

    allocator_destroy(&a);
}

static void test_allocator_coalescing(void) {
    Allocator a = allocator_create(1024ull * 1024);

    void* x = allocator_alloc(&a, 100);
    void* y = allocator_alloc(&a, 100);
    void* z = allocator_alloc(&a, 100);
    (void)z;

    allocator_free(&a, x);
    allocator_free(&a, y);
    /* x ve y fizikte komsu ve ikisi de serbest -> birlesip tek blok olmali.
     * Birlesen blok boyutu >= 100+100+header oldugu icin, 150 baytlik bir
     * istegi x'in eski adresinden karsilayabilmeli (yeni arena buyumesi
     * olmadan, first-fit ile). */
    void* w = allocator_alloc(&a, 150);
    CHECK(w == x, "allocator: komsu serbest bloklar birlesmedi (coalescing basarisiz)");

    allocator_destroy(&a);
}

static void test_file_io_roundtrip(void) {
    const char* path = "layer0_test_output.tmp";
    const char* content = "Merhaba Turkce LLM projesi! CSGUOI ozel karakter testi.";
    u64 content_len = 0;
    while (content[content_len] != 0) content_len++;

    FileHandle wf = file_open_write(path);
    CHECK(wf.valid, "file_io: dosya yazma icin acilamadi");
    u64 written = file_write(&wf, content, content_len);
    CHECK(written == content_len, "file_io: yazilan bayt sayisi yanlis");
    file_close(&wf);

    Allocator a = allocator_create(1024 * 1024);
    u64 read_size = 0;
    void* data = file_read_entire(path, &a, &read_size);
    CHECK(data != NULL_PTR, "file_io: dosya okunamadi");
    CHECK(read_size == content_len, "file_io: okunan boyut yazilanla eslesmiyor");

    bool32 same = TRUE;
    for (u64 i = 0; i < content_len && i < read_size; i++) {
        if (((u8*)data)[i] != (u8)content[i]) { same = FALSE; break; }
    }
    CHECK(same, "file_io: okunan icerik yazilanla eslesmiyor");

    allocator_destroy(&a);
    file_delete(path);
}

int main(void) {
    console_write_line("=== Katman 0 Testleri ===");

    test_arena_basic();
    test_allocator_reuse();
    test_allocator_coalescing();
    test_file_io_roundtrip();

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
