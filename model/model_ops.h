/* Katman 7 - Transformer'a ozgu otograd (fused) islemleri.
 * Genel amacli graf mekanigi autograd/node.h'de; burada Katman 3'teki
 * tensor.c/rope.c fonksiyonlarini saran, backward'i elle turetilmis
 * ozel dugumler var (RMSNorm, RoPE, bias ekleme, sutun dilimleme). */
#ifndef MODEL_MODEL_OPS_H
#define MODEL_MODEL_OPS_H

#include "../autograd/node.h"
#include "../tensor/tensor.h"

/* x:[rows,cols] + bias:[cols] (satir bazinda yayilarak). */
Node* node_add_bias(Allocator* alloc, Node* x, Node* bias);

/* a:[rows,cols] icinden [*, col_start:col_start+num_cols) VIEW'i (Katman 7'de
 * coklu-baslik/multi-head ayirma icin). */
Node* node_slice_cols(Allocator* alloc, Node* a, u64 col_start, u64 num_cols);

/* RMSNorm: y = x/sqrt(mean(x^2, son eksen)+eps) * weight. */
Node* node_rmsnorm(Allocator* alloc, Node* x, Node* weight, f32 eps);

/* RoPE: cos_leaf/sin_leaf, rope_build_tables ile doldurulmus, requires_grad=FALSE
 * bir node_leaf olarak sarilmis sabit tablolardir. */
Node* node_rope(Allocator* alloc, Node* x, Node* cos_leaf, Node* sin_leaf);

/* a:[rows,ca] ile b:[rows,cb] sutunlar boyunca birlestirir -> [rows,ca+cb]
 * (coklu-baslik/multi-head ciktilarini tek tensorde toplamak icin;
 * cok sayida basligi ikili zincirleme ile birlestirmek icin kullanilir). */
Node* node_concat_cols2(Allocator* alloc, Node* a, Node* b);

/* KATMAN 18 - Gercek tensor-batching icin (bkz. tensor_slice_rows):
 * a:[rows,cols] icinden [row_start:row_start+num_rows, *) VIEW'i. */
Node* node_slice_rows(Allocator* alloc, Node* a, u64 row_start, u64 num_rows);

/* a:[ra,cols] ile b:[rb,cols] satirlar boyunca birlestirir -> [ra+rb,cols]. */
Node* node_concat_rows2(Allocator* alloc, Node* a, Node* b);

#endif /* MODEL_MODEL_OPS_H */
