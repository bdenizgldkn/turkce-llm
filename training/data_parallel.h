/* Katman 16 - Veri-Paralel (Data-Parallel) Cok-Thread'li Egitim Adimi.
 *
 * Tek bir egitim adiminda NUM_WORKERS BAGIMSIZ diziyi NUM_WORKERS OS
 * thread'ine (bkz. runtime/thread.h) dagitip paralel ileri+geri yayilim
 * yapar, sonra gradyanlari toplayip ortalamasini alir. Her thread
 * modelin agirlik DEGERLERINI SADECE OKUR (paylasilan, sabit); kendi
 * BAGIMSIZ (taze) gradyan tamponuna yazar -- bu yuzden kilit gerekmez.
 * Paylasilan CUDA baglami (model/gpu_ops.c) kendi mutex'iyle korunur. */
#ifndef TRAINING_DATA_PARALLEL_H
#define TRAINING_DATA_PARALLEL_H

#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../model/lm_model.h"

#define DATA_PARALLEL_MAX_WORKERS 64

/* orig'in HER agirlik Node'unu, deger paylasan ama gradyani taze bir
 * govde (shadow) ile degistirir (bkz. node_leaf). alloc icinde kurulur. */
void build_shadow_model(Allocator* alloc, const LMModel* orig, LMModel* shadow);

/* params[0..num_params) onceden SIFIRLANMIS olmalidir (adam_zero_grad).
 * num_workers BAGIMSIZ rastgele SEQ_LEN'lik pencere orneklenir (tokens
 * dizisinden, seed_base+worker_index ile tohumlanir), paralel islenir,
 * gradyanlari params[]'a TOPLANIR ve num_workers'a bolunur (ortalama).
 * Donen deger: worker'larin ortalama kaybi. */
/* use_gpu_layers/use_gpu_output: bkz. model/lm_model.h. Cok thread'li
 * kosuda GPU'nun (tek, paylasilan baglam) cekismesini (contention)
 * azaltmak icin katman-ici carpimlari CPU'da birakip sadece cikis
 * projeksiyonunu GPU'ya tasimak (0,1) faydali olabilir -- bkz.
 * PROJE_PLANI.md Bolum 17. */
f32 data_parallel_step(const LMModel* orig_model, Node** params, u32 num_params,
                        const u32* tokens, u64 num_tokens, u64 seq_len,
                        u64 step_arena_size, u32 num_workers, u64 seed_base,
                        i32 use_gpu_layers, i32 use_gpu_output);

#endif /* TRAINING_DATA_PARALLEL_H */
