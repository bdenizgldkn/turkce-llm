#include "suffixes.h"

/* Sablon isaretleri: A=2 yollu unlu, I=4 yollu unlu, D=d/t, C=c/ç.
 * triggers_softening, unlu-baslangicli (A/I ile baslayan) neredeyse
 * tum eklerde TRUE olmalidir -- yumusama ekten degil, genel Turkce
 * fonotaktiginden kaynaklanir (kok sonu p/ç/t/k + unlu = yumusama). */
const Suffix g_suffix_table[] = {
    /* --- Isim cekim ekleri (KOK-COGUL-IYELIK-HAL sirasi zorlanir, bkz. suffixes.h) --- */
    { "lAr",  "cogul",                    0,   FALSE, FALSE, SFX_NOUN_INFLECTION, NOUN_SLOT_PLURAL },
    { "Im",   "iyelik-1tekil",            0,   TRUE,  TRUE,  SFX_NOUN_INFLECTION, NOUN_SLOT_POSSESSIVE },
    { "In",   "iyelik-2tekil",            0,   TRUE,  TRUE,  SFX_NOUN_INFLECTION, NOUN_SLOT_POSSESSIVE },
    { "I",    "iyelik-3tekil",            's', TRUE,  FALSE, SFX_NOUN_INFLECTION, NOUN_SLOT_POSSESSIVE },
    { "ImIz", "iyelik-1cogul",            0,   TRUE,  TRUE,  SFX_NOUN_INFLECTION, NOUN_SLOT_POSSESSIVE },
    { "InIz", "iyelik-2cogul",            0,   TRUE,  TRUE,  SFX_NOUN_INFLECTION, NOUN_SLOT_POSSESSIVE },
    { "lArI", "iyelik-3cogul",            0,   FALSE, FALSE, SFX_NOUN_INFLECTION, NOUN_SLOT_POSSESSIVE },
    { "I",    "belirtme-hali(akuzatif)",  'y', TRUE,  FALSE, SFX_NOUN_INFLECTION, NOUN_SLOT_CASE },
    { "A",    "yonelme-hali(datif)",      'y', TRUE,  FALSE, SFX_NOUN_INFLECTION, NOUN_SLOT_CASE },
    { "DA",   "bulunma-hali(lokatif)",    0,   FALSE, FALSE, SFX_NOUN_INFLECTION, NOUN_SLOT_CASE },
    { "DAn",  "ayrilma-hali(ablatif)",    0,   FALSE, FALSE, SFX_NOUN_INFLECTION, NOUN_SLOT_CASE },
    { "In",   "tamlayan-hali(genitif)",   'n', TRUE,  FALSE, SFX_NOUN_INFLECTION, NOUN_SLOT_CASE },

    /* --- Yapim ekleri --- */
    { "lIk",  "isimlestirme(-lık)",       0,   FALSE, FALSE, SFX_DERIVATIONAL, NOUN_SLOT_NONE },
    { "sIz",  "yoksunluk(-sız)",          0,   FALSE, FALSE, SFX_DERIVATIONAL, NOUN_SLOT_NONE },
    { "lI",   "sahiplik(-lı)",            0,   FALSE, FALSE, SFX_DERIVATIONAL, NOUN_SLOT_NONE },
    { "CI",   "meslek/ugras(-cı)",        0,   FALSE, FALSE, SFX_DERIVATIONAL, NOUN_SLOT_NONE },

    /* --- Fiil cekim ekleri --- */
    /* "-lAr" hem isim cogulu hem de fiil 3. cogul sahis ekidir (ayni
     * fonoloji, farkli kategori) -- morfotaktik tutarlilik kontrolunun
     * (bkz. analyzer.c) her baglamda doğru calismasi icin ayri girdi. */
    { "lAr",  "cogul-3sahis(fiil)",       0,   FALSE, FALSE, SFX_VERB_INFLECTION, NOUN_SLOT_NONE },
    { "mAk",  "mastar(infinitif)",        0,   FALSE, FALSE, SFX_VERB_INFLECTION, NOUN_SLOT_NONE },
    { "mA",   "olumsuzluk",               0,   FALSE, FALSE, SFX_VERB_INFLECTION, NOUN_SLOT_NONE },
    { "Iyor", "simdiki-zaman",            0,   TRUE,  TRUE,  SFX_VERB_INFLECTION, NOUN_SLOT_NONE },
    { "DI",   "digecmis-zaman",           0,   FALSE, FALSE, SFX_VERB_INFLECTION, NOUN_SLOT_NONE },
    { "Abil", "yeterlilik(-ebil)",        'y', TRUE,  FALSE, SFX_VERB_INFLECTION, NOUN_SLOT_NONE },
};

const u32 g_suffix_table_count = sizeof(g_suffix_table) / sizeof(g_suffix_table[0]);
