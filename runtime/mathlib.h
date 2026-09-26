/* Katman 1 - Temel Matematik Kütüphanesi.
 *
 * math.h KULLANILMAZ. Tüm fonksiyonlar Taylor serisi / Newton-Raphson
 * iterasyonu ve IEEE-754 bit manipülasyonu (üstel/mantis çıkarma, ldexp
 * benzeri ölçekleme) ile sıfırdan yazılmıştır. Yöntem: DOĞRULUK
 * ÖNCELİKLİ (bkz. PROJE_PLANI.md, Açık Kararlar) — hız optimizasyonu
 * (bit-hack kısayolları, polinom yaklaşımları) profil sonrası ayrı
 * bir aşamada değerlendirilecek.
 *
 * İç hesaplamalar f64 (double) hassasiyetinde yapılır — bu, dilin
 * kendi yerleşik kayan nokta tipini (donanım FPU) kullanmaktır, bir
 * "kütüphane" değildir. Modelin kendisi Katman 7'den itibaren sadece
 * float32 kullanacaktır (bkz. Açık Kararlar); f32 sarmalayıcılar bu
 * yüzden burada f64 sonucu f32'ye yuvarlar.
 */
#ifndef RUNTIME_MATHLIB_H
#define RUNTIME_MATHLIB_H

#include "types.h"

/* --- Sabitler (bilinen matematiksel sabitlerin ondalık açılımı; bir
 *     kütüphaneden alınmıyor, sabit sayı yazmakla aynı kategoridedir) --- */
#define M_PI_VAL   3.14159265358979323846
#define M_E_VAL    2.71828182845904523536
#define M_LN2_VAL  0.69314718055994530942

/* --- IEEE-754 özel değerler (bit paterniyle inşa edilir) --- */
f64 m_inf(void);
f64 m_neg_inf(void);
f64 m_nan(void);
bool32 m_is_nan(f64 x);
bool32 m_is_inf(f64 x);

/* --- f64 (double hassasiyet) --- */
f64 m_sqrt(f64 x);
f64 m_exp(f64 x);
f64 m_log(f64 x);            /* doğal logaritma (ln) */
f64 m_pow(f64 base, f64 exponent);
f64 m_sin(f64 x);
f64 m_cos(f64 x);
f64 m_tanh(f64 x);

/* --- f32 sarmalayıcılar (model bunları kullanacak) --- */
f32 m_sqrtf(f32 x);
f32 m_expf(f32 x);
f32 m_logf(f32 x);
f32 m_powf(f32 base, f32 exponent);
f32 m_sinf(f32 x);
f32 m_cosf(f32 x);
f32 m_tanhf(f32 x);

#endif /* RUNTIME_MATHLIB_H */
