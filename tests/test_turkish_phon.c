/* Katman 6 dogrulama: Turkce fonoloji kurallari (unlu uyumu, unsuz
 * yumusamasi/sertlesmesi, kaynastirma unsuzleri) gercek, bilinen
 * Turkce cekim ornekleriyle karsilastirilir. */
#include "../runtime/types.h"
#include "../runtime/console.h"
#include "../runtime/strutil.h"
#include "../tokenizer/turkish_phon.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

/* root + ek sablonunu birlestirip beklenen sonucla karsilastirir. */
static void check_inflect(const char* root, const char* template_str,
                           char buffer_cons, bool32 softening, bool32 drop_vowel,
                           const char* expected, const char* label) {
    u64 root_len = str_len(root);
    WordEnding end = tr_analyze_ending(root, root_len);

    char suf_buf[32];
    u32 softened_cp = 0;
    u64 suf_len = tr_resolve_suffix(template_str, end, buffer_cons, softening, drop_vowel, suf_buf, &softened_cp);

    char combined[128];
    u64 combined_len;

    if (softened_cp != 0) {
        combined_len = tr_apply_root_change(root, root_len, softened_cp, combined);
    } else {
        for (u64 i = 0; i < root_len; i++) combined[i] = root[i];
        combined_len = root_len;
    }
    for (u64 i = 0; i < suf_len; i++) combined[combined_len + i] = suf_buf[i];
    combined_len += suf_len;
    combined[combined_len] = 0;

    if (str_eq(combined, expected)) {
        g_pass++;
    } else {
        g_fail++;
        console_write("  [FAIL] "); console_write(label);
        console_write(" beklenen='"); console_write(expected);
        console_write("' bulunan='"); console_write(combined);
        console_write_line("'");
    }
}

int main(void) {
    console_write_line("=== Katman 6 (Turkce Fonoloji) Testleri ===");

    /* Dativ (-A), kaynastirma 'y', yumusama yok */
    check_inflect("ev", "A", 0, FALSE, FALSE, "eve", "ev+dativ (on unlu, unsuzle biter)");
    check_inflect("araba", "A", 'y', FALSE, FALSE, "arabaya", "araba+dativ (arka unlu, unluyle biter -> y tamponu)");
    check_inflect("okul", "A", 0, FALSE, FALSE, "okula", "okul+dativ");

    /* Akuzatif (-I), kaynastirma 'y', yumusama VAR */
    check_inflect("kitap", "I", 'y', TRUE, FALSE, "kitabı", "kitap+akuzatif (p->b yumusamasi)");
    check_inflect("ağaç", "I", 'y', TRUE, FALSE, "ağacı", "ağaç+akuzatif (ç->c yumusamasi)");
    check_inflect("araba", "I", 'y', TRUE, FALSE, "arabayı", "araba+akuzatif (unluyle biter -> y tamponu, yumusama yok)");
    check_inflect("göz", "I", 'y', TRUE, FALSE, "gözü", "göz+akuzatif (z yumusamaz)");

    /* Lokatif (-DA): D devoicing + A harmony, yumusama YOK (ilk harf unsuz) */
    check_inflect("ev", "DA", 0, FALSE, FALSE, "evde", "ev+lokatif (v sedali -> d)");
    check_inflect("kitap", "DA", 0, FALSE, FALSE, "kitapta", "kitap+lokatif (p sedasiz -> t)");

    /* Ablatif (-DAn) */
    check_inflect("ev", "DAn", 0, FALSE, FALSE, "evden", "ev+ablatif");
    check_inflect("kitap", "DAn", 0, FALSE, FALSE, "kitaptan", "kitap+ablatif");

    /* Meslek eki (-CI): C devoicing + I harmony (4 yollu) */
    check_inflect("göz", "CI", 0, FALSE, FALSE, "gözcü", "göz+meslek-eki (on-yuvarlak -> ü, z sedali -> c)");
    check_inflect("süt", "CI", 0, FALSE, FALSE, "sütçü", "süt+meslek-eki (on-yuvarlak -> ü, t sedasiz -> ç)");
    check_inflect("kapı", "CI", 0, FALSE, FALSE, "kapıcı", "kapı+meslek-eki (arka-duz -> ı, p... son harf ı unlu)");

    /* Cogul (-lAr) */
    check_inflect("ev", "lAr", 0, FALSE, FALSE, "evler", "ev+cogul (on unlu)");
    check_inflect("kitap", "lAr", 0, FALSE, FALSE, "kitaplar", "kitap+cogul (arka unlu)");

    /* 1. tekil sahis iyelik -(I)m: unluyle bitince I DUSER (kaynastirma degil) */
    check_inflect("araba", "Im", 0, FALSE, TRUE, "arabam", "araba+iyelik1sg (unlu duser)");
    check_inflect("kitap", "Im", 0, TRUE, TRUE, "kitabım", "kitap+iyelik1sg (yumusama + I)");
    check_inflect("ev", "Im", 0, FALSE, TRUE, "evim", "ev+iyelik1sg (unsuzle biter, I dusmez)");

    /* Simdiki zaman -(I)yor: unluyle bitince I duser, yumusama olabilir */
    check_inflect("oku", "Iyor", 0, FALSE, TRUE, "okuyor", "oku+simdikizaman (unlu duser)");
    check_inflect("git", "Iyor", 0, TRUE, TRUE, "gidiyor", "git+simdikizaman (t->d yumusamasi)");

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
