#include "mathlib.h"

/* ================= IEEE-754 bit erişimi ================= */

typedef union { f64 f; u64 u; } F64Bits;
typedef union { f32 f; u32 u; } F32Bits;

static u64 f64_bits(f64 x)      { F64Bits c; c.f = x; return c.u; }
static f64 f64_from_bits(u64 u) { F64Bits c; c.u = u; return c.f; }

#define F64_SIGN_MASK 0x8000000000000000ull
#define F64_EXP_MASK  0x7FF0000000000000ull
#define F64_MANT_MASK 0x000FFFFFFFFFFFFFull
#define F64_EXP_BIAS  1023

f64 m_inf(void)     { return f64_from_bits(0x7FF0000000000000ull); }
f64 m_neg_inf(void) { return f64_from_bits(0xFFF0000000000000ull); }
f64 m_nan(void)     { return f64_from_bits(0x7FF8000000000000ull); }

bool32 m_is_nan(f64 x) {
    u64 b = f64_bits(x);
    return ((b & F64_EXP_MASK) == F64_EXP_MASK) && ((b & F64_MANT_MASK) != 0);
}

bool32 m_is_inf(f64 x) {
    u64 b = f64_bits(x);
    return ((b & F64_EXP_MASK) == F64_EXP_MASK) && ((b & F64_MANT_MASK) == 0);
}

/* x = mantissa * 2^exp, mantissa in [1,2) (x > 0 varsayılır). */
static f64 decompose(f64 x, i32* out_exp) {
    u64 bits = f64_bits(x);
    i32 e = (i32)((bits & F64_EXP_MASK) >> 52) - F64_EXP_BIAS;
    u64 mant_bits = (bits & F64_MANT_MASK) | ((u64)F64_EXP_BIAS << 52);
    *out_exp = e;
    return f64_from_bits(mant_bits);
}

/* x * 2^e (üstel alanını doğrudan kaydırarak; taşma/alt taşma korumalı). */
static f64 ldexp2(f64 x, i32 e) {
    if (x == 0.0) return 0.0;
    u64 bits = f64_bits(x);
    u64 sign = bits & F64_SIGN_MASK;
    i32 cur_exp = (i32)((bits & F64_EXP_MASK) >> 52);
    i64 new_exp = (i64)cur_exp + e;

    if (new_exp >= 0x7FF) return (sign ? m_neg_inf() : m_inf());
    if (new_exp <= 0)     return (sign ? -0.0 : 0.0); /* alt taşma: sıfıra yuvarla (denormal desteklenmiyor) */

    u64 new_bits = sign | ((u64)new_exp << 52) | (bits & F64_MANT_MASK);
    return f64_from_bits(new_bits);
}

/* ================= sqrt ================= */

f64 m_sqrt(f64 x) {
    if (x < 0.0) return m_nan();
    if (x == 0.0 || m_is_nan(x)) return x;
    if (m_is_inf(x)) return x;

    i32 e;
    f64 m = decompose(x, &e); /* m in [1,2) */

    /* e'yi çift yap ki 2^e'nin tam sayı bir sqrt üsteli olsun. */
    if (e & 1) { m *= 2.0; e -= 1; } /* artik m in [1,4) */

    /* Newton-Raphson: y_{n+1} = 0.5*(y_n + m/y_n). Kaba baslangic: m/2+0.5.
     * Karekokte yakinsama KARESEL'dir (her iterasyonda dogru basamak
     * sayisi ikiye katlanir) -- 6 iterasyon, bu baslangic tahmininden
     * double'in tam 52-bit mantissa hassasiyetine fazlasiyla yeter
     * (test_mathlib.c'nin 1e-12 toleransli testleriyle dogrulandi).
     * Yakinsadiktan sonra y sabit noktaya ulasir (y=0.5*(y+y)=y), yani
     * fazladan iterasyon dogrulugu ARTIRMAZ, sadece zaman harcar --
     * Adam optimizer adiminda parametre sayisi kadar (milyonlarca)
     * cagrilan bu fonksiyon, egitim adim suresinin buyuk kismini
     * olusturuyordu (bkz. PROJE_PLANI.md Bolum 15/16). */
    f64 y = m * 0.5 + 0.5;
    for (i32 i = 0; i < 6; i++) {
        y = 0.5 * (y + m / y);
    }

    return ldexp2(y, e / 2);
}

/* ================= exp ================= */

f64 m_exp(f64 x) {
    if (m_is_nan(x)) return x;
    if (x > 709.782712893384) return m_inf();      /* double'da tasma siniri */
    if (x < -745.1332191019412) return 0.0;         /* alt tasma */

    f64 k_f = x / M_LN2_VAL;
    i64 k = (i64)(k_f + (k_f >= 0.0 ? 0.5 : -0.5)); /* en yakina yuvarla */
    f64 r = x - (f64)k * M_LN2_VAL;                  /* |r| <= ln2/2 civari */

    /* Taylor serisi: exp(r) = sum r^n / n! */
    f64 term = 1.0;
    f64 sum = 1.0;
    for (i32 n = 1; n < 40; n++) {
        term *= r / (f64)n;
        sum += term;
        if (term < 1e-18 && term > -1e-18) break;
    }

    return ldexp2(sum, (i32)k);
}

/* ================= log (dogal logaritma) ================= */

f64 m_log(f64 x) {
    if (m_is_nan(x) || x < 0.0) return m_nan();
    if (x == 0.0) return m_neg_inf();
    if (m_is_inf(x)) return x;

    i32 e;
    f64 m = decompose(x, &e); /* m in [1,2) */

    /* Yakinsamayi hizlandirmak icin m'yi 1'e daha yakin bir araliga getir. */
    const f64 SQRT2 = 1.4142135623730951;
    if (m > SQRT2) { m *= 0.5; e += 1; } /* artik m in [SQRT2/2, SQRT2) */

    f64 z = (m - 1.0) / (m + 1.0);
    f64 z2 = z * z;
    f64 term = z;
    f64 sum = z;
    for (i32 n = 3; n < 60; n += 2) {
        term *= z2;
        f64 add = term / (f64)n;
        sum += add;
        if (add < 1e-18 && add > -1e-18) break;
    }

    return 2.0 * sum + (f64)e * M_LN2_VAL;
}

/* ================= pow ================= */

static bool32 is_integer(f64 x) {
    return x == (f64)(i64)x;
}

f64 m_pow(f64 base, f64 exponent) {
    if (exponent == 0.0) return 1.0;
    if (base == 0.0) {
        if (exponent > 0.0) return 0.0;
        return m_inf();
    }
    if (base < 0.0) {
        if (!is_integer(exponent)) return m_nan();
        f64 mag = m_exp(exponent * m_log(-base));
        i64 ie = (i64)exponent;
        return (ie % 2 != 0) ? -mag : mag;
    }
    return m_exp(exponent * m_log(base));
}

/* ================= sin / cos ================= */

#define HALF_PI_VAL (M_PI_VAL / 2.0)
#define TWO_PI_VAL  (M_PI_VAL * 2.0)

/* |t| <= pi/4 icin Taylor serileri. */
static f64 sin_core(f64 t) {
    f64 t2 = t * t;
    f64 term = t;
    f64 sum = t;
    for (i32 n = 1; n < 20; n++) {
        term *= -t2 / (f64)((2 * n) * (2 * n + 1));
        sum += term;
        if (term < 1e-18 && term > -1e-18) break;
    }
    return sum;
}

static f64 cos_core(f64 t) {
    f64 t2 = t * t;
    f64 term = 1.0;
    f64 sum = 1.0;
    for (i32 n = 1; n < 20; n++) {
        term *= -t2 / (f64)((2 * n - 1) * (2 * n));
        sum += term;
        if (term < 1e-18 && term > -1e-18) break;
    }
    return sum;
}

/* x'i [-pi/4, pi/4] araligina indirger ve 0..3 kadran bilgisini dondurur. */
static f64 reduce_quadrant(f64 x, i32* out_quadrant) {
    f64 q_f = x / HALF_PI_VAL;
    i64 q = (i64)(q_f + (q_f >= 0.0 ? 0.5 : -0.5));
    f64 core = x - (f64)q * HALF_PI_VAL;
    i64 quadrant = q % 4;
    if (quadrant < 0) quadrant += 4;
    *out_quadrant = (i32)quadrant;
    return core;
}

f64 m_sin(f64 x) {
    if (m_is_nan(x) || m_is_inf(x)) return m_nan();
    i32 q;
    f64 core = reduce_quadrant(x, &q);
    switch (q) {
        case 0: return sin_core(core);
        case 1: return cos_core(core);
        case 2: return -sin_core(core);
        default: return -cos_core(core);
    }
}

f64 m_cos(f64 x) {
    if (m_is_nan(x) || m_is_inf(x)) return m_nan();
    i32 q;
    f64 core = reduce_quadrant(x, &q);
    switch (q) {
        case 0: return cos_core(core);
        case 1: return -sin_core(core);
        case 2: return -cos_core(core);
        default: return sin_core(core);
    }
}

/* ================= tanh ================= */

f64 m_tanh(f64 x) {
    if (m_is_nan(x)) return x;
    f64 sign = (x < 0.0) ? -1.0 : 1.0;
    f64 ax = (x < 0.0) ? -x : x;

    if (ax > 20.0) return sign; /* exp(2*ax) tasar/pratikte 1.0'a doyar */

    f64 t = m_exp(2.0 * ax);
    f64 result = (t - 1.0) / (t + 1.0);
    return sign * result;
}

/* ================= f32 sarmalayicilar ================= */

f32 m_sqrtf(f32 x) { return (f32)m_sqrt((f64)x); }
f32 m_expf(f32 x)  { return (f32)m_exp((f64)x); }
f32 m_logf(f32 x)  { return (f32)m_log((f64)x); }
f32 m_powf(f32 base, f32 exponent) { return (f32)m_pow((f64)base, (f64)exponent); }
f32 m_sinf(f32 x)  { return (f32)m_sin((f64)x); }
f32 m_cosf(f32 x)  { return (f32)m_cos((f64)x); }
f32 m_tanhf(f32 x) { return (f32)m_tanh((f64)x); }
