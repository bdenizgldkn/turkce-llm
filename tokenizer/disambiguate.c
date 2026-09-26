#include "disambiguate.h"

/* a parse'i b parse'inden daha iyi mi? Once kok frekansi (yuksek olan
 * kazanir), esitlikte daha az ek iceren (daha sade) ayristirma kazanir.
 *
 * Not: Erken bir surumde, kisa/rastgele bir alt-dizinin (orn. tek
 * harfli "g") METIN ICINDE kisaltma olarak sik gecmesi nedeniyle uzun
 * ama DILBILGISEL OLARAK GECERSIZ bir ek zinciriyle (isim+fiil
 * eklerini karistirarak) kazanmasi sorunu yasandi. Bu, frekans
 * skoruna yama yapmak yerine KAYNAGINDA -- analyzer.c'deki
 * isim/fiil kategori tutarlilik kuraliyla -- cozuldu; artik gecersiz
 * karisik-kategori zincirleri hic uretilmiyor, bu yuzden burada saf
 * frekans karsilastirmasi yeterli ve dogru. */
static bool32 is_better(const StrCounter* freq, const Parse* a, const Parse* b) {
    u64 fa = strcounter_get(freq, a->root, a->root_len);
    u64 fb = strcounter_get(freq, b->root, b->root_len);

    if (fa != fb) return fa > fb;
    return a->num_suffixes < b->num_suffixes;
}

u32 disambiguate(const StrCounter* freq, const AnalysisResult* result) {
    u32 best = 0;
    for (u32 i = 1; i < result->num_parses; i++) {
        if (is_better(freq, &result->parses[i], &result->parses[best])) {
            best = i;
        }
    }
    return best;
}
