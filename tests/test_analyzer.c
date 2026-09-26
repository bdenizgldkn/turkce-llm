/* Katman 6 dogrulama: uctan uca morfolojik ayristirici testi.
 * Gercek kok sozlugu (Vikisozluk'ten cikarilan) + ek tablosu birlikte
 * kullanilarak, bilinen cekimli Turkce kelimelerin dogru kok+ek
 * dizisine ayristigi kontrol edilir. */
#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../runtime/console.h"
#include "../runtime/strutil.h"
#include "../tokenizer/lexicon.h"
#include "../tokenizer/analyzer.h"
#include "../tokenizer/suffixes.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

static void print_parse(const Parse* p) {
    console_write("["); console_write(p->root); console_write("]");
    for (u32 i = 0; i < p->num_suffixes; i++) {
        console_write("+"); console_write(g_suffix_table[p->suffix_indices[i]].gloss);
    }
}

static bool32 parse_matches(const Parse* p, const char* expected_root, const char** expected_glosses, u32 n_glosses) {
    if (!str_eq(p->root, expected_root)) return FALSE;
    if (p->num_suffixes != n_glosses) return FALSE;
    for (u32 i = 0; i < n_glosses; i++) {
        if (!str_eq(g_suffix_table[p->suffix_indices[i]].gloss, expected_glosses[i])) return FALSE;
    }
    return TRUE;
}

static void check_word(const StrHashSet* lex, const char* word,
                        const char* expected_root, const char** expected_glosses, u32 n_glosses,
                        const char* label) {
    u64 len = str_len(word);
    AnalysisResult r = analyze_word(lex, word, len);

    bool32 found = FALSE;
    for (u32 i = 0; i < r.num_parses; i++) {
        if (parse_matches(&r.parses[i], expected_root, expected_glosses, n_glosses)) { found = TRUE; break; }
    }

    if (found) { g_pass++; }
    else {
        g_fail++;
        console_write("  [FAIL] "); console_write(label);
        console_write(" (\""); console_write(word); console_write("\") -- ");
        console_write_u64(r.num_parses); console_write_line(" ayristirma bulundu:");
        for (u32 i = 0; i < r.num_parses; i++) {
            console_write("      "); print_parse(&r.parses[i]); console_write_line("");
        }
    }
}

int main(void) {
    console_write_line("=== Katman 6 (Uctan Uca Morfolojik Ayristirici) Testleri ===");

    Allocator alloc = allocator_create(2000ull * 1024 * 1024);
    u64 word_count = 0;
    console_write_line("Kok sozlugu yukleniyor...");
    StrHashSet lex = lexicon_load(&alloc, "data/raw/kok_adaylari.txt", &word_count);
    console_write("Yuklenen kelime (turetilmis fiil kokleri dahil): "); console_write_u64(word_count); console_write_line("");

    check_word(&lex, "kitap", "kitap", NULL_PTR, 0, "kitap (ek yok)");
    { const char* g[] = {"isimlestirme(-lık)"}; check_word(&lex, "gözlük", "göz", g, 1, "gözlük"); }
    { const char* g[] = {"iyelik-3tekil"}; check_word(&lex, "arabası", "araba", g, 1, "arabası"); }
    { const char* g[] = {"cogul", "bulunma-hali(lokatif)"}; check_word(&lex, "evlerde", "ev", g, 2, "evlerde"); }
    { const char* g[] = {"cogul", "ayrilma-hali(ablatif)"}; check_word(&lex, "kitaplardan", "kitap", g, 2, "kitaplardan"); }
    { const char* g[] = {"cogul", "iyelik-1cogul", "ayrilma-hali(ablatif)"}; check_word(&lex, "kitaplarımızdan", "kitap", g, 3, "kitaplarımızdan"); }
    { const char* g[] = {"simdiki-zaman", "cogul-3sahis(fiil)"}; check_word(&lex, "gidiyorlar", "git", g, 2, "gidiyorlar"); }

    /* REGRESYON: gercek buyuk-olcekli korpüs kosusunda bulunan bir
     * YIGIN TASMASI (stack buffer overflow) hatasi -- ANALYZER_MAX_ROOT_BYTES
     * (64 bayt) sinirini asan bozuk/tekrarlayan "kelimeler" (orn.
     * vandalizmden kalma "ğüğüğü..." gibi 80+ baytlik cop veri)
     * recurse() ic tamponlarini tasiriyordu. analyze_word artik boyle
     * girdileri guvenle 0 ayristirmayla reddetmeli (cokme YOK). */
    {
        char garbage[90];
        for (u64 i = 0; i < 88; i += 4) {
            garbage[i] = (char)0xC4; garbage[i+1] = (char)0x9F;   /* ğ */
            garbage[i+2] = (char)0xC3; garbage[i+3] = (char)0xBC; /* ü */
        }
        u64 glen = 88;
        AnalysisResult r = analyze_word(&lex, garbage, glen);
        if (r.num_parses == 0) { g_pass++; }
        else {
            g_fail++;
            console_write_line("  [FAIL] guvenlik: 64+ bayt cop veri cokmeden 0 ayristirma dondurmeli");
        }
    }

    allocator_destroy(&alloc);

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
