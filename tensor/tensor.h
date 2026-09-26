/* Katman 3 - Tensor / Cok Boyutlu Dizi Yapisi.
 *
 * CPU tarafinda, float32 veri uzerinde calisan, satir-majör (row-major)
 * bicimde duzenlenen, sekil (shape) + adim (stride) tabanli bir tensor.
 * Bellek, Katman 0'daki genel amacli Allocator uzerinden alinir.
 *
 * Transpoze gibi islemler veri kopyalamaz; sadece shape/stride'i
 * degistiren bir "view" (goruntu) dondurur -- ayni alttaki veriyi
 * paylasir. Bunu somutlastirmak (contiguous/bitisik hale getirmek)
 * icin tensor_contiguous kullanilir.
 */
#ifndef TENSOR_TENSOR_H
#define TENSOR_TENSOR_H

#include "../runtime/types.h"
#include "../runtime/memory.h"

#define TENSOR_MAX_DIMS 6

typedef struct Tensor {
    f32* data;              /* alttaki veriye isaretci (view'lerde paylasilir) */
    u64  shape[TENSOR_MAX_DIMS];
    i64  strides[TENSOR_MAX_DIMS]; /* eleman cinsinden (bayt degil) */
    u32  ndim;
    u64  numel;              /* toplam eleman sayisi (shape'in carpimi) */
    bool32 owns_data;        /* TRUE ise tensor_free veriyi serbest birakir */
} Tensor;

/* Bitisik (contiguous), satir-majör yeni bir tensor ayirir; verisi
 * sifirlanmaz (baslangic degeri tanimsizdir -- gerekiyorsa tensor_fill
 * ile doldurun). */
Tensor tensor_create(Allocator* alloc, const u64* shape, u32 ndim);

/* alloc==NULL_PTR ise (owns_data=FALSE oldugu icin) tensor_free hicbir
 * sey yapmaz -- view'ler icin kullanilir. */
void tensor_free(Allocator* alloc, Tensor* t);

bool32 tensor_is_contiguous(const Tensor* t);

void tensor_fill(Tensor* t, f32 value);
void tensor_copy_from(Tensor* dst, const Tensor* src); /* eleman eleman kopya (shape ayni olmali) */

/* --- Genel dizin yardimcilari (Katman 4 otograd tarafindan da kullanilir) --- */
/* flat (0..numel-1) duz indeksi, shape'e gore coklu-indekse cevirir. */
void tensor_flat_to_multi(u64 flat, const u64* shape, u32 ndim, u64* out_idx);
/* coklu-indeks + stride'lardan veri arabellegindeki (eleman cinsinden) ofseti hesaplar. */
u64 tensor_compute_offset(const i64* strides, const u64* idx, u32 ndim);

/* Coklu-indeks ile tekil eleman erisimi (idx dizisi t->ndim uzunlugunda). */
f32  tensor_get(const Tensor* t, const u64* idx);
void tensor_set(Tensor* t, const u64* idx, f32 value);

/* Reshape: sadece bitisik (contiguous) tensorlerde calisir, veri
 * kopyalamadan yeni bir shape ile ayni veriye bakan bir view dondurur.
 * numel esit olmalidir. */
Tensor tensor_reshape(const Tensor* src, const u64* new_shape, u32 new_ndim);

/* dim0 ve dim1 eksenlerini yer degistirir (view, veri kopyalanmaz). */
Tensor tensor_transpose(const Tensor* src, u32 dim0, u32 dim1);

/* src bitisik degilse, verisini bitisik bir duzene kopyalayan yeni bir
 * tensor dondurur (owns_data=TRUE); zaten bitisikse basit bir kopya
 * tensoru dondurur. */
Tensor tensor_contiguous(Allocator* alloc, const Tensor* src);

/* --- Elemanter islemler (V1: yayinlama/broadcasting yok, shape'ler esit olmali) --- */
Tensor tensor_add(Allocator* alloc, const Tensor* a, const Tensor* b);
Tensor tensor_sub(Allocator* alloc, const Tensor* a, const Tensor* b);
Tensor tensor_mul(Allocator* alloc, const Tensor* a, const Tensor* b); /* eleman-bazli carpim */
Tensor tensor_scale(Allocator* alloc, const Tensor* a, f32 scalar);

/* dst += src (yerinde, eleman-bazli; dst'nin shape'i uzerinden yinelenir,
 * src'nin stride'lari farkli/bitisik-olmayan olabilir -- ornegin bir
 * transpose view'i dogrudan biriktirmek icin kullanilabilir).
 * Otograd (Katman 4) gradyan biriktirme icin kullanir. */
void tensor_add_inplace(Tensor* dst, const Tensor* src);

/* --- 2B matris carpimi: a:[M,K] x b:[K,N] -> [M,N] (naif CPU implementasyonu) --- */
Tensor tensor_matmul2d(Allocator* alloc, const Tensor* a, const Tensor* b);

/* src:[rows,cols] icinden [*, col_start:col_start+num_cols) araligini
 * VIEW olarak doner (veri kopyalanmaz, Katman 7'de coklu-baslik/multi-head
 * dilimleme icin kullanilir). */
Tensor tensor_slice_cols(const Tensor* src, u64 col_start, u64 num_cols);

/* x:[rows,cols] + bias:[cols] (satir bazinda yayilarak/broadcast). */
Tensor tensor_add_bias_2d(Allocator* alloc, const Tensor* x, const Tensor* bias);

/* a:[rows,ca] ile b:[rows,cb] sutunlar boyunca birlestirir -> [rows,ca+cb]. */
Tensor tensor_concat_cols2(Allocator* alloc, const Tensor* a, const Tensor* b);

/* KATMAN 18 - Gercek tensor-batching icin: src:[rows,cols] icinden
 * [row_start:row_start+num_rows, *) araligini VIEW olarak doner (satir-
 * majör duzende satirlar BITISIK oldugu icin, tensor_slice_cols'tan
 * FARKLI OLARAK, bu her zaman zaten bitisik/contiguous bir view'dir --
 * ayrica bir tensor_contiguous() cagrisi gerekmez). Birden fazla
 * bagimsiz diziyi (batch) TEK bir buyuk [batch*seq, d_model] tensorde
 * birlestirip BUYUK matris carpimlarini (wqkv/wo/gate_up/w_down) TEK
 * seferde yapip, sonra dikkat hesaplamasi icin diziler-arasi sizintiyi
 * onlemek amaciyla tekrar dizi-bazinda ayirmak icin kullanilir. */
Tensor tensor_slice_rows(const Tensor* src, u64 row_start, u64 num_rows);

/* a:[ra,cols] ile b:[rb,cols] satirlar boyunca birlestirir -> [ra+rb,cols]. */
Tensor tensor_concat_rows2(Allocator* alloc, const Tensor* a, const Tensor* b);

/* --- Aktivasyonlar / indirgeme (mathlib.h uzerinden, Katman 4 otograd sarmalar) --- */
Tensor tensor_relu(Allocator* alloc, const Tensor* a);
Tensor tensor_sigmoid(Allocator* alloc, const Tensor* a);
Tensor tensor_tanh_op(Allocator* alloc, const Tensor* a);
/* Son eksen uzerinde softmax; a bitisik (contiguous) olmalidir. */
Tensor tensor_softmax_lastdim(Allocator* alloc, const Tensor* a);
/* Tum elemanlarin toplami; sonuc shape=[1] tek elemanli bir tensordur. */
Tensor tensor_sum_all(Allocator* alloc, const Tensor* a);

/* RMSNorm: son eksen boyunca y = x / sqrt(mean(x^2)+eps) * weight.
 * a bitisik (contiguous) olmalidir; weight.numel == a->shape[son eksen]. */
Tensor tensor_rmsnorm(Allocator* alloc, const Tensor* a, const Tensor* weight, f32 eps);

#endif /* TENSOR_TENSOR_H */
