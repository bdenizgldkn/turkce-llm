/* Katman 6 dogrulama: Ek tablosundaki (suffixes.c) somut girdilerin
 * gercek Turkce kelimelerle eslesip eslesmedigini kontrol eder.
 * (turkish_phon motorunun kendisi test_turkish_phon.c'de zaten
 * genel olarak dogrulanmisti; burada TABLODAKI VERININ dogrulugu
 * test edilir.) */
#include "../runtime/types.h"
#include "../runtime/console.h"
#include "../runtime/strutil.h"
#include "../tokenizer/turkish_phon.h"
#include "../tokenizer/suffixes.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

static const Suffix* find_suffix(const char* gloss) {
    for (u32 i = 0; i < g_suffix_table_count; i++) {
        if (str_eq(g_suffix_table[i].gloss, gloss)) return &g_suffix_table[i];
    }
    return NULL_PTR;
}

static void check(const char* root, const char* gloss, const char* expected) {
    const Suffix* s = find_suffix(gloss);
    if (!s) {
        g_fail++;
        console_write("  [FAIL] ek bulunamadi: "); console_write_line(gloss);
        return;
    }

    u64 root_len = str_len(root);
    WordEnding end = tr_analyze_ending(root, root_len);

    char suf_buf[32];
    u32 softened_cp = 0;
    u64 suf_len = tr_resolve_suffix(s->template_str, end, s->buffer_consonant,
                                     s->triggers_softening, s->drop_initial_vowel_after_vowel,
                                     suf_buf, &softened_cp);

    char combined[128];
    u64 combined_len;
    if (softened_cp != 0) combined_len = tr_apply_root_change(root, root_len, softened_cp, combined);
    else { for (u64 i = 0; i < root_len; i++) combined[i] = root[i]; combined_len = root_len; }
    for (u64 i = 0; i < suf_len; i++) combined[combined_len + i] = suf_buf[i];
    combined_len += suf_len;
    combined[combined_len] = 0;

    if (str_eq(combined, expected)) { g_pass++; }
    else {
        g_fail++;
        console_write("  [FAIL] "); console_write(root); console_write("+"); console_write(gloss);
        console_write(" beklenen='"); console_write(expected);
        console_write("' bulunan='"); console_write(combined); console_write_line("'");
    }
}

int main(void) {
    console_write_line("=== Katman 6 (Ek Tablosu) Testleri ===");

    check("ev", "cogul", "evler");
    check("araba", "iyelik-1tekil", "arabam");
    check("kitap", "iyelik-1tekil", "kitabım");
    check("ev", "iyelik-3tekil", "evi");
    check("araba", "iyelik-3tekil", "arabası");
    check("kitap", "belirtme-hali(akuzatif)", "kitabı");
    check("ev", "yonelme-hali(datif)", "eve");
    check("kitap", "yonelme-hali(datif)", "kitaba");
    check("ev", "bulunma-hali(lokatif)", "evde");
    check("kitap", "ayrilma-hali(ablatif)", "kitaptan");
    check("ev", "tamlayan-hali(genitif)", "evin");
    check("araba", "tamlayan-hali(genitif)", "arabanın");
    check("göz", "isimlestirme(-lık)", "gözlük");
    check("tuz", "yoksunluk(-sız)", "tuzsuz");
    check("tuz", "sahiplik(-lı)", "tuzlu");
    check("süt", "meslek/ugras(-cı)", "sütçü");
    check("git", "mastar(infinitif)", "gitmek");
    check("gel", "olumsuzluk", "gelme");
    check("oku", "simdiki-zaman", "okuyor");
    check("git", "simdiki-zaman", "gidiyor");
    check("gel", "digecmis-zaman", "geldi");
    check("yap", "digecmis-zaman", "yaptı");
    check("oku", "yeterlilik(-ebil)", "okuyabil");
    check("gel", "yeterlilik(-ebil)", "gelebil");

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
