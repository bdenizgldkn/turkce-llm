/* Katman 7-8 - Tam Dil Modeli (Language Model).
 * Embedding (agirlik baglantili/tied cikis projeksiyonuyla) + N adet
 * Transformer bloğu (RMSNorm+RoPE'li coklu-baslik dikkat+SwiGLU) +
 * son RMSNorm. */
#ifndef MODEL_LM_MODEL_H
#define MODEL_LM_MODEL_H

#include "../autograd/node.h"
#include "../runtime/prng.h"
#include "transformer_block.h"

#define LM_MAX_LAYERS 32 /* Faz 5 derinlestirme: 12->24 katman (bkz. PROJE_PLANI.md Bolum 19/25) */

typedef struct LMModel {
    Node* embed_table; /* [vocab_size, d_model] */
    BlockWeights blocks[LM_MAX_LAYERS];
    Node* final_norm_w;
    u32 num_layers;
    u32 num_heads;
    u32 d_model;
    u32 vocab_size;
    f32 eps;
} LMModel;

LMModel lm_init(Allocator* alloc, PCGState* rng, u32 vocab_size, u32 d_model,
                 u32 num_heads, u32 num_layers, u32 d_ff, f32 eps);

/* token_ids[0..batch_size*seq_len): batch_size BAGIMSIZ dizi, ARKA ARKAYA
 * (dizi 0'nin token'lari, sonra dizi 1'in token'lari, ...). Sonuc: logits
 * [batch_size*seq_len, vocab_size] (bkz. PROJE_PLANI.md Bolum 18: gercek
 * tensor-batching). batch_size=1 -> ORIJINAL (batch'siz) davranisla
 * BIREBIR AYNI (mevcut tum cagiran kod, testler dahil, degismeden calisir).
 *
 * use_gpu_layers != 0 ise HER Transformer katmanindaki projeksiyon/FFN
 * matris carpimlari GPU'da hesaplanir (batch_size>1 oldugunda TUM batch
 * icin TEK cagrida -- GPU cagri sayisini batch_size kati azaltir).
 * use_gpu_output != 0 ise buyuk vocab boyutu yuzunden FLOP'larin
 * cogunlugunu olusturan bagli (tied) cikis projeksiyonu GPU'da hesaplanir.
 * Ikisi BAGIMSIZDIR -- once gpu_ops_init() cagrilmis olmalidir (herhangi
 * biri TRUE ise). */
Node* lm_forward(Allocator* alloc, LMModel* m, const u32* token_ids,
                  u32 batch_size, u64 seq_len,
                  Node* cos_leaf, Node* sin_leaf, Node* causal_mask,
                  i32 use_gpu_layers, i32 use_gpu_output);

/* Tum egitilebilir parametreleri (Node*) out_params dizisine toplar,
 * toplam sayiyi dondurur. out_params en az LM_MAX_LAYERS*12+2 eleman almalidir. */
u32 lm_collect_params(LMModel* m, Node** out_params);

#endif /* MODEL_LM_MODEL_H */
