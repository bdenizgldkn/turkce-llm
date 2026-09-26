/* Katman 6 dogrulama: istatistiksel belirsizlik giderme.
 * "gidiyorlar" gibi gercekten belirsiz kelimelerde, korpüs frekansi
 * daha yuksek olan kokun (git > gid) secildigini dogrular. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/strutil.h"
#include "../tokenizer/lexicon.h"
#include "../tokenizer/freqtable.h"
#include "../tokenizer/analyzer.h"
#include "../tokenizer/disambiguate.h"
#include "../tokenizer/suffixes.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

static void check_best_root(const StrHashSet* lex, const StrCounter* freq, const char* word, const char* expected_root) {
    u64 len = str_len(word);
    AnalysisResult r = analyze_word(lex, word, len);

    if (r.num_parses == 0) {
        g_fail++;
        console_write("  [FAIL] "); console_write(word); console_write_line(" -- hic ayristirma bulunamadi");
        return;
    }

    u32 best = disambiguate(freq, &r);
    const Parse* p = &r.parses[best];

    if (str_eq(p->root, expected_root)) {
        g_pass++;
    } else {
        g_fail++;
        console_write("  [FAIL] "); console_write(word);
        console_write(" beklenen kok='"); console_write(expected_root);
        console_write("' secilen='"); console_write(p->root);
        console_write("' (frekans="); console_write_u64(strcounter_get(freq, p->root, p->root_len));
        console_write_line(")");
    }
}

int main(void) {
    console_write_line("=== Katman 6 (Istatistiksel Belirsizlik Giderme) Testleri ===");

    Allocator alloc = allocator_create(3000ull * 1024 * 1024);

    console_write_line("Kok sozlugu yukleniyor...");
    u64 word_count = 0;
    StrHashSet lex = lexicon_load(&alloc, "data/raw/kok_adaylari.txt", &word_count);

    console_write_line("Kelime frekans tablosu yukleniyor...");
    StrCounter freq = freqtable_load(&alloc, "data/raw/kelime_frekans.txt");
    console_write("Frekans tablosu boyutu: "); console_write_u64(freq.num_keys); console_write_line("");

    console_write("git frekansi: "); console_write_u64(strcounter_get(&freq, "git", 3)); console_write_line("");
    console_write("gid frekansi: "); console_write_u64(strcounter_get(&freq, "gid", 3)); console_write_line("");

    /* Bilinen belirsiz durum: git (yaygin, standart) vs gid (nadir/lehçesel). */
    check_best_root(&lex, &freq, "gidiyorlar", "git");
    check_best_root(&lex, &freq, "gidiyor", "git");

    /* Belirsizligi olmayan basit durumlar da dogru calismali. */
    check_best_root(&lex, &freq, "kitap", "kitap");
    check_best_root(&lex, &freq, "arabası", "araba");
    check_best_root(&lex, &freq, "evlerde", "ev");

    allocator_destroy(&alloc);

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
