/* Faz 4 veri hazirligi - OpenSubtitles (OPUS v2024) Turkce tek-dilli
 * dokumunden temiz, tekrarsiz gunluk-konusma korpusu cikarir.
 *
 * Girdi satir basina bir altyazi cumlesidir (filmler arka arkaya, film
 * siniri isaretsiz). Islem:
 *   1) Her satir subtitle_clean_line ile temizlenir (etiketler, diyalog
 *      tireleri, OCR 'I'->'l', sarki/harfsiz satirlar atilir).
 *   2) Ardisik ayni satirlar tek satira indirilir.
 *   3) Tekrar eden film surumleri: ayni film cok sayida altyazi surumuyle
 *      (farkli cevirmen/kare hizi) dokumde birden fazla kez bulunur.
 *      Girdi BLOCK_LINES satirlik bloklarla islenir; blogun "uzun"
 *      (>= DEDUP_MIN_BYTES) satirlarinin yarisindan fazlasi daha once
 *      gorulmusse blok bir kopya sayilip TAMAMEN atilir (satir satir
 *      atmak kisa satirlari baglamsiz birakirdi). Gorulen satirlar 64-bit
 *      FNV-1a ozetiyle acik adresli bir tabloda tutulur.
 *   4) Blok ornekleme: tutulan bloklardan KEEP_PERMILLE/1000'i yazilir
 *      (dokum boyunca DUZGUN dagilim -- sadece bastaki filmler degil).
 *
 * Cikti: data/raw/subtitle_corpus.txt (satir basina bir replik; ardisik
 * olmayan bloklar arasina bos satir). */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/file_io.h"
#include "../runtime/console.h"
#include "../runtime/prng.h"
#include "subtitle_clean.h"

#define IN_PATH  "data/raw/opensubtitles_tr_v2024.txt"
#define OUT_PATH "data/raw/subtitle_corpus.txt"

#define WINDOW_SIZE     (256ull * 1024 * 1024)
#define MAX_LINE        (64ull * 1024)          /* bundan uzun "satir" altyazi degildir, atilir */
#define BLOCK_LINES     200u
#define BLOCK_BUF_CAP   (BLOCK_LINES * MAX_LINE)
#define DEDUP_MIN_BYTES 24u
#define HASH_BITS       31u                     /* 2^31 yuva x 8 bayt = 16 GB (mmap tembel ayirir) */
#define KEEP_PERMILLE   290u   /* tekrarsiz ~3,84 GB -> ~1,1 GB (~500M token; Faz 4 ~233M altyazi token kullanir) */

static u64* g_table;
static u64  g_table_mask;
static u64  g_table_used;

static u64 fnv1a(const char* s, u64 n) {
    u64 h = 1469598103934665603ull;
    for (u64 i = 0; i < n; i++) { h ^= (u8)s[i]; h *= 1099511628211ull; }
    return h ? h : 1; /* 0 = bos yuva */
}

/* Tabloda varsa TRUE; yoksa ekler ve FALSE. */
static bool32 seen_or_insert(u64 h) {
    u64 i = h & g_table_mask;
    for (;;) {
        if (g_table[i] == h) return TRUE;
        if (g_table[i] == 0) { g_table[i] = h; g_table_used++; return FALSE; }
        i = (i + 1) & g_table_mask;
    }
}

static bool32 table_contains(u64 h) {
    u64 i = h & g_table_mask;
    for (;;) {
        if (g_table[i] == h) return TRUE;
        if (g_table[i] == 0) return FALSE;
        i = (i + 1) & g_table_mask;
    }
}

/* Blok durumu: temizlenmis satirlar block_buf'ta '\n' ile ayrilmis. */
static char* g_block;
static u64   g_block_len;
static u32   g_block_lines;
static u32   g_block_long;
static u32   g_block_long_dup;
static char* g_prev_line;
static u64   g_prev_len;

static FileHandle g_out;
static PCGState   g_rng;
static u64 g_stat_lines_in, g_stat_lines_kept, g_stat_blocks, g_stat_blocks_dup, g_stat_blocks_sampled_out, g_stat_bytes_out;

static void flush_block(void) {
    if (g_block_lines == 0) return;
    g_stat_blocks++;

    bool32 is_dup = (g_block_long > 0 && g_block_long_dup * 2 > g_block_long);
    if (is_dup) {
        g_stat_blocks_dup++;
    } else {
        /* Blogun uzun satirlarini "goruldu" olarak isaretle (ayni blok
         * icinde tekrar eden satirlar yukarida zaten sayildi). */
        u64 s = 0;
        for (u64 i = 0; i < g_block_len; i++) {
            if (g_block[i] == '\n') {
                if (i - s >= DEDUP_MIN_BYTES) seen_or_insert(fnv1a(g_block + s, i - s));
                s = i + 1;
            }
        }
        if (pcg_range_i64(&g_rng, 0, 1000) < (i64)KEEP_PERMILLE) {
            /* Bloklar arasina HER ZAMAN ikinci bir '\n' ekleniyor (blok
             * zaten kendi son satirindan gelen tek '\n' ile bitiyor) --
             * boylece blok siniri "\n\n" ile ic-blok satir sonlarindan
             * ("\n") ayirt edilebilir hale geliyor. tokenize_corpus bu
             * "\n\n" izini <eos> tokenine ceviriyor (bkz. PROJE_PLANI.md,
             * kod incelemesi bolumu, madde 4). Not: burada "blok" 200
             * girdi satirlik bir pencere, gercek film siniri degil (girdi
             * dokumunde film sinirlari isaretsiz) -- yani bu, gercek belge
             * sinirinin YAKLASIK bir vekili; hicbir isaretten iyidir. */
            if (g_stat_bytes_out > 0) file_write(&g_out, "\n", 1);
            file_write(&g_out, g_block, g_block_len);
            g_stat_bytes_out += g_block_len;
            g_stat_lines_kept += g_block_lines;
        } else {
            g_stat_blocks_sampled_out++;
        }
    }
    g_block_len = 0; g_block_lines = 0; g_block_long = 0; g_block_long_dup = 0;
}

static void process_line(const char* line, u64 len) {
    g_stat_lines_in++;
    if (len <= MAX_LINE) {
        u64 clen = subtitle_clean_line(line, len, g_block + g_block_len);
        char* cl = g_block + g_block_len;
        bool32 same_as_prev = (clen == g_prev_len);
        for (u64 k = 0; same_as_prev && k < clen; k++) if (cl[k] != g_prev_line[k]) same_as_prev = FALSE;

        if (clen > 0 && !same_as_prev) {
            if (clen >= DEDUP_MIN_BYTES) {
                g_block_long++;
                if (table_contains(fnv1a(cl, clen))) g_block_long_dup++;
            }
            for (u64 k = 0; k < clen; k++) g_prev_line[k] = cl[k];
            g_prev_len = clen;
            g_block_len += clen;
            g_block[g_block_len++] = '\n';
            g_block_lines++;
        }
    }
    /* Blok siniri GIRDI satir sayisina gore (temizlikten bagimsiz). */
    if (g_stat_lines_in % BLOCK_LINES == 0) flush_block();
}

int main(void) {
    console_write_line("=== Altyazi Korpusu Olusturma (OpenSubtitles v2024, tr) ===");

    Allocator alloc = allocator_create(20ull * 1024 * 1024 * 1024);
    Arena table_arena = arena_create((8ull << HASH_BITS) + 4096);
    g_table = (u64*)arena_alloc(&table_arena, 8ull << HASH_BITS, 64); /* mmap: sifir dolu gelir */
    g_table_mask = (1ull << HASH_BITS) - 1;

    FileHandle in = file_open_read(IN_PATH);
    if (!in.valid) { console_write("HATA: girdi acilamadi: "); console_write_line(IN_PATH); return 1; }
    g_out = file_open_write(OUT_PATH);
    if (!g_out.valid) { console_write("HATA: cikti acilamadi: "); console_write_line(OUT_PATH); return 1; }

    char* window = (char*)allocator_alloc(&alloc, WINDOW_SIZE + MAX_LINE + 1);
    g_block = (char*)allocator_alloc(&alloc, BLOCK_BUF_CAP + BLOCK_LINES);
    g_prev_line = (char*)allocator_alloc(&alloc, MAX_LINE);
    g_rng = pcg_seed(2027, 4);

    u64 carry = 0, total_read = 0;
    for (;;) {
        u64 got = file_read(&in, window + carry, WINDOW_SIZE);
        total_read += got;
        u64 total = carry + got;
        if (total == 0) break;

        u64 s = 0;
        for (u64 i = 0; i < total; i++) {
            if (window[i] == '\n') { process_line(window + s, i - s); s = i + 1; }
        }
        if (got == 0) { if (s < total) process_line(window + s, total - s); break; }

        carry = total - s;
        if (carry > MAX_LINE) { process_line(window + s, carry); carry = 0; } /* asiri uzun satir: process_line atar */
        for (u64 k = 0; k < carry; k++) window[k] = window[s + k];

        console_write("  okunan MB: "); console_write_u64(total_read >> 20);
        console_write(" | satir: "); console_write_u64(g_stat_lines_in);
        console_write(" | yazilan MB: "); console_write_u64(g_stat_bytes_out >> 20);
        console_write(" | kopya blok: "); console_write_u64(g_stat_blocks_dup);
        console_write("/"); console_write_u64(g_stat_blocks);
        console_write_line("");
    }
    flush_block();

    file_close(&in);
    file_close(&g_out);

    console_write_line("");
    console_write("Girdi satir: "); console_write_u64(g_stat_lines_in); console_write_line("");
    console_write("Blok: "); console_write_u64(g_stat_blocks);
    console_write(" | kopya (atilan): "); console_write_u64(g_stat_blocks_dup);
    console_write(" | orneklemede atilan: "); console_write_u64(g_stat_blocks_sampled_out); console_write_line("");
    console_write("Yazilan satir: "); console_write_u64(g_stat_lines_kept); console_write_line("");
    console_write("Yazilan bayt: "); console_write_u64(g_stat_bytes_out); console_write_line("");
    console_write("Ozet tablosu doluluk: "); console_write_u64(g_table_used); console_write_line("");

    arena_destroy(&table_arena);
    allocator_destroy(&alloc);
    return 0;
}
