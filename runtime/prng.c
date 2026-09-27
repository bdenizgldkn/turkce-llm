#include "prng.h"
#include "mathlib.h"

#define PCG_MULT 6364136223846793005ull

static u32 pcg_rotr32(u32 value, u32 rot) {
    return (value >> rot) | (value << ((32u - rot) & 31u));
}

u32 pcg_next_u32(PCGState* rng) {
    u64 oldstate = rng->state;
    rng->state = oldstate * PCG_MULT + rng->inc;

    u32 xorshifted = (u32)(((oldstate >> 18u) ^ oldstate) >> 27u);
    u32 rot = (u32)(oldstate >> 59u);
    return pcg_rotr32(xorshifted, rot);
}

PCGState pcg_seed(u64 initstate, u64 initseq) {
    PCGState rng;
    rng.state = 0u;
    rng.inc = (initseq << 1u) | 1u;
    rng.has_cached_gaussian = FALSE;
    rng.cached_gaussian = 0.0;

    pcg_next_u32(&rng);
    rng.state += initstate;
    pcg_next_u32(&rng);

    return rng;
}

f64 pcg_uniform(PCGState* rng) {
    u32 v = pcg_next_u32(rng);
    return (f64)v * (1.0 / 4294967296.0); /* / 2^32 -> [0,1) */
}

f64 pcg_uniform_open(PCGState* rng) {
    /* 0 gelme olasiligi 2^-32; onu (0,1] icine iterek disliyoruz. */
    f64 u = pcg_uniform(rng);
    if (u <= 0.0) u = 1.0 / 4294967296.0;
    return u;
}

static u64 pcg_next_u64(PCGState* rng) {
    u64 hi_bits = (u64)pcg_next_u32(rng);
    u64 lo_bits = (u64)pcg_next_u32(rng);
    return (hi_bits << 32) | lo_bits;
}

i64 pcg_range_i64(PCGState* rng, i64 lo, i64 hi) {
    /* [lo, hi) -- yansiz (unbiased) aralik ornekleme. Onceki surum duz
     * "v % range" kullaniyordu: range, 2^32'yi tam bolmedigi zaman (ki
     * neredeyse hicbir zaman bolmez) [0, 2^32 mod range) bandindaki
     * degerler diger degerlerden daha sik secilir -- egitim veri
     * karistirmasinda gercek bir sapmaya yol acan bug budur (bkz.
     * PROJE_PLANI.md, kod incelemesi bolumu). Duzeltme: esik altindaki
     * degerleri reddederek (rejection sampling) kalan uzayin range'e
     * tam bolunmesini sagliyoruz -- 64-bit cekilerek gelecekte 4 milyar+
     * token'lik korpuslarda da (500M parametre hedefi icin planlanan
     * ~10 milyar token) dogru calisir. */
    u64 range = (u64)(hi - lo);
    if (range == 0) return lo;
    u64 threshold = (u64)(-(i64)range) % range; /* = (2^64 - range) % range, tasma guvenli */
    u64 v;
    do {
        v = pcg_next_u64(rng);
    } while (v < threshold);
    return lo + (i64)(v % range);
}

f64 pcg_gaussian_std(PCGState* rng) {
    if (rng->has_cached_gaussian) {
        rng->has_cached_gaussian = FALSE;
        return rng->cached_gaussian;
    }

    f64 u1 = pcg_uniform_open(rng); /* (0,1], log(0) tekilligini onler */
    f64 u2 = pcg_uniform(rng);      /* [0,1) */

    f64 r = m_sqrt(-2.0 * m_log(u1));
    f64 theta = 2.0 * M_PI_VAL * u2;

    f64 z0 = r * m_cos(theta);
    f64 z1 = r * m_sin(theta);

    rng->cached_gaussian = z1;
    rng->has_cached_gaussian = TRUE;
    return z0;
}

f64 pcg_gaussian(PCGState* rng, f64 mean, f64 stddev) {
    return mean + stddev * pcg_gaussian_std(rng);
}

void pcg_shuffle_u32(PCGState* rng, u32* array, u64 count) {
    /* Fisher-Yates: sondan basa dogru, her adimda [0, i] araliginda
     * rastgele bir indeksle takas et. */
    for (u64 i = count; i > 1; i--) {
        u64 j = (u64)pcg_range_i64(rng, 0, (i64)i);
        u32 tmp = array[i - 1];
        array[i - 1] = array[j];
        array[j] = tmp;
    }
}
