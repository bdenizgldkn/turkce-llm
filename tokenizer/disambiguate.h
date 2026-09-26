/* Katman 6 - Istatistiksel Belirsizlik Giderme.
 *
 * V1 yontemi: KOK UNIGRAM FREKANSI (korpüsten -- Vikisozluk Turkce
 * bolumlerinin duz metninden cikarilmis kelime sayimlari). Birden
 * fazla gecerli ayristirma arasinda, koku korpüste EN SIK GECEN
 * ayristirma tercih edilir; frekans esitse (orn. ikisi de 0/bilinmiyor)
 * DAHA AZ EK iceren (daha basit) ayristirma tercih edilir.
 *
 * BILINEN SINIRLAMA: Bu, PROJE_PLANI.md'de hedeflenen tam "n-gram/HMM
 * ile BAGLAMA gore" (comsu kelimelere bakan) modelden daha basittir --
 * kok bazinda bir UNIGRAM modeldir, cumle baglamini kullanmaz. Tam
 * bağlamsal (bigram/HMM) model, etiketlenmis/ayristirilmis bir egitim
 * korpüsü gerektirir (yumurta-tavuk sorunu: once bir ayristirici lazim);
 * ileride EM/Baum-Welch ile genisletilmesi degerlendirilecektir. */
#ifndef TOKENIZER_DISAMBIGUATE_H
#define TOKENIZER_DISAMBIGUATE_H

#include "../runtime/types.h"
#include "../runtime/strcounter.h"
#include "analyzer.h"

/* result->num_parses > 0 olmalidir. En iyi ayristirmanin result->parses
 * icindeki indeksini dondurur. */
u32 disambiguate(const StrCounter* freq, const AnalysisResult* result);

#endif /* TOKENIZER_DISAMBIGUATE_H */
