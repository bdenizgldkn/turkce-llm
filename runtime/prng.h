/* Katman 2 - Rastgele Sayi Ureteci (PRNG).
 *
 * Algoritma: PCG (Permuted Congruential Generator), O'Neill'in PCG32
 * ("pcg_setseq_64_xsh_rr_32") varyanti -- literatürden alinan, kamuya
 * acik (public domain) bir algoritma; kod satir satir bizim tarafimizdan
 * yazilmistir (bir kutuphane cagirilmiyor).
 *
 * Agirlik baslatma icin Gauss (normal) dagilimli sayilar, Box-Muller
 * donusumu ile PCG'nin uniform ciktisindan turetilir (mathlib.h'deki
 * kendi log/sqrt/sin/cos fonksiyonlarimiz kullanilir).
 */
#ifndef RUNTIME_PRNG_H
#define RUNTIME_PRNG_H

#include "types.h"

typedef struct PCGState {
    u64 state;
    u64 inc;
    bool32 has_cached_gaussian;
    f64 cached_gaussian;
} PCGState;

/* initseq, ayni initstate ile farkli bagimsiz akislar elde etmek icin
 * kullanilan bir "akis secici"dir (PCG'ye ozgu). */
PCGState pcg_seed(u64 initstate, u64 initseq);

u32 pcg_next_u32(PCGState* rng);

/* [0, 1) araliginda tekduze (uniform) dagilim, 32 bitlik cozunurluk. */
f64 pcg_uniform(PCGState* rng);

/* (0, 1] araliginda -- log(0) tekilligini onlemek icin Box-Muller'de kullanilir. */
f64 pcg_uniform_open(PCGState* rng);

/* [lo, hi) araliginda tam sayi (ornek: veri karistirma / shuffling icin). */
i64 pcg_range_i64(PCGState* rng, i64 lo, i64 hi);

/* Standart normal (ortalama 0, std sapma 1) -- Box-Muller, ikinci degeri
 * bir sonraki cagri icin onbellekler. */
f64 pcg_gaussian_std(PCGState* rng);

/* Genel Gauss(mean, stddev). */
f64 pcg_gaussian(PCGState* rng, f64 mean, f64 stddev);

/* Fisher-Yates ile bir u32 indeks dizisini yerinde karistirir (veri
 * karistirma / shuffling icin). */
void pcg_shuffle_u32(PCGState* rng, u32* array, u64 count);

#endif /* RUNTIME_PRNG_H */
