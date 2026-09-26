/* Katman 7 (devam) - Batch destegi.
 *
 * Tasarim notu: Tensor/otograd cekirdegini (Katman 3-4) batched N-boyutlu
 * matmul'a genisletmek yerine, ayni AGIRLIKLARI PAYLASARAK (weight sharing)
 * her dizi (sequence) icin AYRI bir alt-graf kurup transformer_block'u
 * B kez cagiriyoruz. Otograd motorumuz, paylasilan bir Node'un birden
 * fazla yerde kullanilmasini gradyan BIRIKTIRME ile zaten dogru
 * destekliyor (bkz. test_autograd.c DAG testleri) -- bu yuzden agirlik
 * gradyanlari otomatik olarak TUM batch uzerinden dogru sekilde toplanir.
 *
 * Bu, CPU'da basit ve dogrudur; performans (batch'ler arasi paralellik/
 * vektorlestirme) GPU entegrasyonu asamasinda ele alinacaktir. */
#ifndef MODEL_BATCH_H
#define MODEL_BATCH_H

#include "../autograd/node.h"
#include "transformer_block.h"

/* x_batch[0..batch_size): her biri [seq_len, d_model] sekilli bagimsiz
 * bir dizi. out_batch[0..batch_size): karsilik gelen ciktilar (cagiran
 * tarafindan onceden ayrilmis bir diziye yazilir). Ayni agirliklar (w)
 * her dizi icin paylasilir. */
void transformer_block_batch(Allocator* alloc, Node** x_batch, u32 batch_size,
                              const BlockWeights* w, u32 num_heads,
                              Node* cos_leaf, Node* sin_leaf, Node* causal_mask, f32 eps,
                              i32 use_gpu, Node** out_batch);

/* losses[0..batch_size) skaler (shape=[1]) node'larini ORTALAMASINI
 * alan tek bir skaler kayip node'u uretir (batch buyuklugunden
 * bagimsiz bir ogrenme orani secebilmek icin toplam yerine ortalama). */
Node* batch_mean_loss(Allocator* alloc, Node** losses, u32 batch_size);

#endif /* MODEL_BATCH_H */
