/* Katman 8 - Checkpoint (kaydetme/yukleme) sistemi.
 * Kendi ikili dosya formatimiz, Katman 0'in dosya I/O'su uzerinden.
 * Uzun suren egitimlerin kesintiye ugramasi durumunda kaldigi yerden
 * devam edebilmek icin gereklidir (bkz. PROJE_PLANI.md Riskler).
 *
 * Format (kucuk-endian, x86_64 yerlisi):
 *   [4 bayt magic "TRLM"] [u32 versiyon=1] [u32 num_params] [u32 has_opt_state]
 *   her parametre icin: [u32 ndim] [ndim x u64 shape] [numel x f32 deger]
 *   has_opt_state ise, her parametre icin ek olarak:
 *                       [numel x f32 momentum(m)] [numel x f32 ikinci-moment(v)]
 *   has_opt_state ise sonda: [u64 adam.t]
 */
#ifndef TRAINING_CHECKPOINT_H
#define TRAINING_CHECKPOINT_H

#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../autograd/node.h"
#include "adam.h"

/* checkpoint_load'un tek seferde dogrulayabilecegi en fazla parametre
 * tensoru (LM_MAX_LAYERS*12+2'den genis tutuldu). */
#define CKPT_MAX_PARAMS 1024u

/* Basariliysa TRUE; dosya acilamazsa (orn. klasor yok, izin yok) ya da
 * herhangi bir yazma eksik kalirsa (orn. disk dolu) FALSE dondurur --
 * cagiran taraf bunu MUTLAKA kontrol etmeli, aksi halde uzun bir egitimde
 * checkpoint'ler sessizce kaybolabilir. */
bool32 checkpoint_save(const char* path, Node** params, u32 num_params, const AdamOptimizer* opt);

/* opt NULL_PTR ise sadece parametre degerleri yuklenir (optimizer
 * durumu atlanir); opt verildiyse dosyada optimizer durumu OLMALI.
 * Basariliysa TRUE, dosya/format/boyut uyusmuyorsa FALSE dondurur --
 * FALSE durumunda params/opt'a HIC dokunulmamistir (once tum dosya
 * dogrulanir, sonra kopyalanir). params[i]->value.shape'in dosyadakiyle AYNI olmasi
 * varsayilir (mimari degismedi, sadece degerler/egitim durumu
 * yukleniyor). */
bool32 checkpoint_load(Allocator* scratch, const char* path, Node** params, u32 num_params, AdamOptimizer* opt);

#endif /* TRAINING_CHECKPOINT_H */
