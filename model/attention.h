/* Katman 7 - Coklu-baslikli nedensel (causal) oz-dikkat (self-attention).
 * RoPE (pozisyon kodlama) ve nedensel maskeleme (gelecegi gormeme)
 * icerir. Tek bir dizi (batch boyutu yok, batch V1'de disariya birakildi
 * -- bkz. PROJE_PLANI.md) uzerinde calisir: x:[seq_len, d_model]. */
#ifndef MODEL_ATTENTION_H
#define MODEL_ATTENTION_H

#include "../autograd/node.h"

/* KATMAN 17 - FUZYONLU (fused) Q/K/V projeksiyonu: wq/wk/wv tek bir
 * [d_model, 3*d_model] matriste birlestirilir (w_qkv), tek bir matris
 * carpimiyla hesaplanip sonra Q/K/V'ye bolunur (node_slice_cols).
 * Amac: cok-thread'li veri-paralel egitimde (bkz. PROJE_PLANI.md
 * Bolum 16-17) paylasilan CUDA baglaminin GPU cagri sayisini (ve
 * dolayisiyla kilit cekismesini/contention) katman basina 3 cagridan
 * 1'e dusurmek -- matematiksel olarak AYNI sonucu 3 KAT DAHA AZ GPU
 * gidis-gelisiyle uretir. */
typedef struct AttentionWeights {
    Node* w_qkv; Node* b_qkv; /* [d_model, 3*d_model], [3*d_model] */
    Node* wo; Node* bo;
} AttentionWeights;

/* x: [batch_size*seq_len, d_model] -- batch_size BAGIMSIZ dizi, SATIR
 * boyunca ust uste yiginlanmis (bkz. PROJE_PLANI.md Bolum 18: gercek
 * tensor-batching). batch_size=1 ise bu, ORIJINAL (batch'siz) davranisla
 * BIREBIR AYNIDIR -- geriye-donuk uyumluluk garantidir (mevcut tum
 * testler bunu dogrular).
 *
 * cos_leaf/sin_leaf: rope_build_tables ile doldurulmus [seq_len,head_dim/2]
 * sabit tablolar (requires_grad=FALSE node_leaf). causal_mask:
 * [seq_len,seq_len] sabit tablo, alt ucgen (kendisi dahil) 0, ust ucgen -inf.
 * use_gpu != 0 ise buyuk Q/K/V/O projeksiyon matris carpimlari GPU'da
 * hesaplanir (bkz. model/gpu_ops.h) -- ve batch_size>1 oldugunda bu
 * carpimlar TUM batch icin TEK bir cagrida yapilir (GPU cagri sayisini
 * batch_size kati azaltir). Basina-dikkat (per-head, per-batch-ogesi)
 * kucuk skor/agirlikli-toplam carpimlari CPU'da kalir (dizi sinirlarini
 * korumak icin -- bir dizi digerinin token'ina asla bakmamali). */
Node* causal_self_attention(Allocator* alloc, Node* x, const AttentionWeights* w,
                             u32 num_heads, u32 batch_size, u64 seq_len,
                             Node* cos_leaf, Node* sin_leaf, Node* causal_mask,
                             i32 use_gpu);

/* [seq,seq] sabit nedensel maske tensoru olusturur (data icine yazar). */
void build_causal_mask(Tensor* mask_out);

#endif /* MODEL_ATTENTION_H */
