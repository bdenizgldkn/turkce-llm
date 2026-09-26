/* Katman 7 - RoPE (Rotary Positional Embedding).
 * Ogrenilebilir parametre yok; her pozisyon/frekans cifti icin sabit
 * bir dondurme (rotation) uygulanir. cos/sin tablolari mathlib.h'deki
 * kendi trigonometri fonksiyonlarimizla onceden hesaplanir. */
#ifndef MODEL_ROPE_H
#define MODEL_ROPE_H

#include "../runtime/types.h"
#include "../runtime/memory.h"
#include "../tensor/tensor.h"

/* cos_out/sin_out: onceden ayrilmis [seq_len, head_dim/2] sekilli tensorler.
 * base: tipik olarak 10000.0. */
void rope_build_tables(Tensor* cos_out, Tensor* sin_out, u64 seq_len, u64 head_dim, f32 base);

/* x: [seq_len, head_dim] (bitisik). cos_t/sin_t: [seq_len, head_dim/2] (bitisik). */
Tensor tensor_rope(Allocator* alloc, const Tensor* x, const Tensor* cos_t, const Tensor* sin_t);

#endif /* MODEL_ROPE_H */
