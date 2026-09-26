#include "analyzer.h"
#include "turkish_phon.h"
#include "suffixes.h"

/* NOT: Ilk buyuk-olcekli korpüs kosusunda 200000'lik butce, bazi
 * (muhtemelen tekrarlayan karakterli/gecersiz) parcalarda asiri yavas
 * calismaya (milisaniyeler yerine saniyeler/parca) yol acti. Gercek
 * Turkce kelimeler icin arama cok daha hizli yakinsar; bu daha kucuk
 * butce, patolojik girdilerde bile parca basina en kotu durum
 * gecikmesini siki bir sekilde sinirlar. */
#define STEP_BUDGET 5000

typedef struct WorkingParse {
    u32 suffix_indices[ANALYZER_MAX_SUFFIXES]; /* disaridan (en son eklenen) ice dogru toplanir */
    u32 num_suffixes;
    bool32 has_noun_infl; /* zincirde simdiye kadar bir SFX_NOUN_INFLECTION eki var mi */
    bool32 has_verb_infl; /* zincirde simdiye kadar bir SFX_VERB_INFLECTION eki var mi */
    i32 min_noun_slot_seen; /* KOK-COGUL-IYELIK-HAL sirasi/tekillik kontrolu icin (bkz. suffixes.h) */
} WorkingParse;

/* Basit morfotaktik tutarlilik kurali: bir kelime ayni anda hem isim
 * hem fiil gibi cekimlenemez (orn. iyelik EKI + gecmis-zaman EKI ayni
 * zincirde olamaz). Turetim (SFX_DERIVATIONAL) ekleri kategori-notrdur
 * (koprulenebilir) ve serbesttir. Bu kural, morfotaktik sirayi TAM
 * ZORLAMAZ (bkz. PROJE_PLANI.md Riskler) ama en bariz gecersiz
 * kombinasyonlari (ve bunlarin istatistiksel modeli yanlis yonlendirmesini)
 * engeller. */
static bool32 class_is_compatible(const WorkingParse* wp, SuffixClass klass) {
    if (klass == SFX_NOUN_INFLECTION && wp->has_verb_infl) return FALSE;
    if (klass == SFX_VERB_INFLECTION && wp->has_noun_infl) return FALSE;
    return TRUE;
}

/* KOK-COGUL-IYELIK-HAL sirasini ve her yuvanin en fazla bir kez
 * kullanilmasini zorlar. wp icinde SOYULMA sirasiyla (surface->kok,
 * yani uygulama sirasinin tersi) cagirilir: once HAL, sonra IYELIK,
 * sonra COGUL beklenir -- yani slot degerleri KESIN AZALAN olmalidir. */
static bool32 noun_slot_is_compatible(const WorkingParse* wp, NounSlot slot) {
    if (slot == NOUN_SLOT_NONE) return TRUE;
    return (i32)slot < wp->min_noun_slot_seen;
}

static bool32 is_utf8_continuation(u8 b) {
    return (b & 0xC0) == 0x80;
}

static void record_parse(AnalysisResult* result, const char* root, u64 root_len, const WorkingParse* wp) {
    if (result->num_parses >= ANALYZER_MAX_PARSES) return;
    if (root_len >= ANALYZER_MAX_ROOT_BYTES) return;

    Parse* p = &result->parses[result->num_parses];
    for (u64 i = 0; i < root_len; i++) p->root[i] = root[i];
    p->root[root_len] = 0;
    p->root_len = root_len;

    /* wp->suffix_indices disaridan (soyulma sirasiyla) toplandi;
     * uygulama sirasi (koktin disariya) icin TERSINE cevir. */
    p->num_suffixes = wp->num_suffixes;
    for (u32 i = 0; i < wp->num_suffixes; i++) {
        p->suffix_indices[i] = wp->suffix_indices[wp->num_suffixes - 1 - i];
    }

    result->num_parses++;
}

static void recurse(const StrHashSet* lex, const char* s, u64 len,
                     WorkingParse* wp, AnalysisResult* result,
                     u32 depth, u64* budget) {
    if (*budget == 0) { result->truncated_by_budget = TRUE; return; }
    (*budget)--;

    if (result->num_parses >= ANALYZER_MAX_PARSES) return;

    /* Taban durum: s'nin tamami sozlukte bir kok mu? */
    if (strset_contains(lex, s, len)) {
        record_parse(result, s, len, wp);
    }

    if (depth >= ANALYZER_MAX_SUFFIXES) return;
    if (len < 2) return;

    for (u32 si = 0; si < g_suffix_table_count; si++) {
        const Suffix* suf = &g_suffix_table[si];
        if (!class_is_compatible(wp, suf->klass)) continue;
        if (!noun_slot_is_compatible(wp, suf->noun_slot)) continue;

        for (u64 stem_len = len - 1; stem_len >= 1; stem_len--) {
            if (is_utf8_continuation((u8)s[stem_len])) continue; /* karakter ortasi, gecersiz bolme */

            WordEnding ending = tr_analyze_ending(s, stem_len);

            char pred_suf[32];
            u32 softened_cp = 0;
            u64 pred_suf_len = tr_resolve_suffix(suf->template_str, ending, suf->buffer_consonant,
                                                  suf->triggers_softening, suf->drop_initial_vowel_after_vowel,
                                                  pred_suf, &softened_cp);

            char pred_stem[ANALYZER_MAX_ROOT_BYTES];
            u64 pred_stem_len;
            if (softened_cp != 0) {
                pred_stem_len = tr_apply_root_change(s, stem_len, softened_cp, pred_stem);
            } else {
                for (u64 i = 0; i < stem_len; i++) pred_stem[i] = s[i];
                pred_stem_len = stem_len;
            }

            if (pred_stem_len + pred_suf_len != len) {
                if (stem_len == 1) break; /* u64 alt tasmasini onlemek icin donguden cik */
                continue;
            }

            bool32 match = TRUE;
            for (u64 i = 0; i < pred_stem_len && match; i++) if (pred_stem[i] != s[i]) match = FALSE;
            for (u64 i = 0; i < pred_suf_len && match; i++) if (pred_suf[i] != s[pred_stem_len + i]) match = FALSE;

            if (match) {
                bool32 old_noun = wp->has_noun_infl;
                bool32 old_verb = wp->has_verb_infl;
                i32 old_slot = wp->min_noun_slot_seen;
                if (suf->klass == SFX_NOUN_INFLECTION) wp->has_noun_infl = TRUE;
                if (suf->klass == SFX_VERB_INFLECTION) wp->has_verb_infl = TRUE;
                if (suf->noun_slot != NOUN_SLOT_NONE) wp->min_noun_slot_seen = (i32)suf->noun_slot;

                wp->suffix_indices[wp->num_suffixes++] = si;
                recurse(lex, s, stem_len, wp, result, depth + 1, budget);

                /* Yumusama tersine cevrilebilir mi? Yuzeyde "kitab" gibi
                 * yumusamis bir kok gorunuyor olabilir; gercek sozluk
                 * kokü "kitap" olabilir. Ikisi de ayni yuzeyi uretir --
                 * hangisinin gercek oldugunu sozluk (asagida recurse
                 * icindeki taban durum) belirler. */
                if (suf->triggers_softening && !ending.ends_in_vowel) {
                    u32 unsoft = tr_unsoften(ending.last_cp);
                    if (unsoft != 0) {
                        char alt_buf[ANALYZER_MAX_ROOT_BYTES];
                        u64 alt_len = tr_apply_root_change(s, stem_len, unsoft, alt_buf);
                        recurse(lex, alt_buf, alt_len, wp, result, depth + 1, budget);
                    }
                }

                wp->num_suffixes--;
                wp->has_noun_infl = old_noun;
                wp->has_verb_infl = old_verb;
                wp->min_noun_slot_seen = old_slot;
            }

            if (stem_len == 1) break; /* u64 dongu degiskeni 0'in altina inemez */
        }
    }
}

AnalysisResult analyze_word(const StrHashSet* lexicon, const char* word, u64 len) {
    AnalysisResult result;
    result.num_parses = 0;
    result.truncated_by_budget = FALSE;

    /* GUVENLIK: recurse() ic tamponlari (pred_stem, alt_buf --
     * ANALYZER_MAX_ROOT_BYTES boyutunda) icin stem_len < len < bu sinir
     * olmasi GEREKIR. Gercek Turkce kelimeler bunu asla asmaz, ama
     * korpüsteki bozuk/vandalizm kaynakli cop veri (orn. tekrarlayan
     * "ğüğüğü..." gibi 80+ baytlik "kelimeler") asabilir -- bu, gercek
     * buyuk-olcekli kosuda YIGIN TASMASI (stack buffer overflow) ve
     * SEGFAULT'A yol acan bir hataydi. Bu kadar uzun bir parca zaten
     * gercekci bir Turkce kok/ek zinciri olamaz; guvenle "analiz
     * edilemedi" (BPE'ye dusecek) olarak isaretliyoruz. */
    if (len >= ANALYZER_MAX_ROOT_BYTES) {
        return result;
    }

    WorkingParse wp;
    wp.num_suffixes = 0;
    wp.has_noun_infl = FALSE;
    wp.has_verb_infl = FALSE;
    wp.min_noun_slot_seen = (i32)NOUN_SLOT_CASE + 1; /* baslangicta her yuva serbest */

    u64 budget = STEP_BUDGET;
    recurse(lexicon, word, len, &wp, &result, 0, &budget);

    return result;
}
