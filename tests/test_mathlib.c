/* Katman 1 dogrulama testleri: exp/log/sqrt/sin/cos/tanh/pow.
 * Bilinen matematiksel referans degerlerle (elle hesaplanmis/bilinen
 * sabitler) karsilastirma yapilir. Hicbir libm fonksiyonu cagrilmaz. */
#include "../runtime/types.h"
#include "../runtime/mathlib.h"
#include "../runtime/console.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

static f64 fabs_(f64 x) { return (x < 0.0) ? -x : x; }

/* Fark cok kucukse gecer; degilse ismini ve farki (1e12 ile olceklenmis
 * tam sayi olarak) yazdirir -- printf olmadan tanilama icin. */
static void check_close(const char* name, f64 got, f64 expected, f64 eps) {
    f64 diff = fabs_(got - expected);
    if (diff <= eps) {
        g_pass++;
    } else {
        g_fail++;
        console_write("  [FAIL] "); console_write(name);
        console_write(" | fark*1e12 = ");
        i64 scaled = (i64)(diff * 1e12);
        if (scaled < 0) { console_write("-"); scaled = -scaled; }
        console_write_u64((u64)scaled);
        console_write_line("");
    }
}

int main(void) {
    console_write_line("=== Katman 1 (Matematik Kutuphanesi) Testleri ===");

    /* sqrt */
    check_close("sqrt(4)",    m_sqrt(4.0),    2.0,                  1e-12);
    check_close("sqrt(2)",    m_sqrt(2.0),    1.4142135623730951,   1e-12);
    check_close("sqrt(0.25)", m_sqrt(0.25),   0.5,                  1e-12);
    check_close("sqrt(1e6)",  m_sqrt(1e6),    1000.0,               1e-9);
    check_close("sqrt(0)",    m_sqrt(0.0),    0.0,                  1e-15);

    /* exp */
    check_close("exp(0)",  m_exp(0.0),  1.0,                    1e-12);
    check_close("exp(1)",  m_exp(1.0),  2.718281828459045,      1e-12);
    check_close("exp(-1)", m_exp(-1.0), 0.36787944117144233,    1e-12);
    check_close("exp(10)", m_exp(10.0), 22026.465794806718,     1e-6);
    check_close("exp(-10)",m_exp(-10.0),0.0000453999297624848,  1e-15);

    /* log */
    check_close("log(1)",       m_log(1.0),        0.0,                 1e-12);
    check_close("log(e)",       m_log(M_E_VAL),    1.0,                 1e-12);
    check_close("log(2)",       m_log(2.0),        0.6931471805599453,  1e-12);
    check_close("log(100)",     m_log(100.0),      4.605170185988091,   1e-11);
    check_close("log(exp(3))",  m_log(m_exp(3.0)), 3.0,                 1e-10);

    /* pow */
    check_close("pow(2,10)",   m_pow(2.0, 10.0), 1024.0,               1e-9);
    check_close("pow(2,0.5)",  m_pow(2.0, 0.5),  1.4142135623730951,   1e-9);
    check_close("pow(-2,3)",   m_pow(-2.0, 3.0), -8.0,                 1e-9);
    check_close("pow(-2,2)",   m_pow(-2.0, 2.0), 4.0,                  1e-9);
    check_close("pow(5,0)",    m_pow(5.0, 0.0),  1.0,                  1e-12);

    /* sin / cos */
    check_close("sin(0)",     m_sin(0.0),               0.0, 1e-12);
    check_close("sin(pi/2)",  m_sin(M_PI_VAL / 2.0),    1.0, 1e-12);
    check_close("sin(pi/6)",  m_sin(M_PI_VAL / 6.0),    0.5, 1e-12);
    check_close("sin(pi)",    m_sin(M_PI_VAL),          0.0, 1e-11);
    check_close("cos(0)",     m_cos(0.0),               1.0, 1e-12);
    check_close("cos(pi/3)",  m_cos(M_PI_VAL / 3.0),    0.5, 1e-12);
    check_close("cos(pi)",    m_cos(M_PI_VAL),         -1.0, 1e-11);
    check_close("sin^2+cos^2 (x=1.234)",
                m_sin(1.234) * m_sin(1.234) + m_cos(1.234) * m_cos(1.234),
                1.0, 1e-12);

    /* tanh */
    check_close("tanh(0)",   m_tanh(0.0),  0.0,                 1e-12);
    check_close("tanh(1)",   m_tanh(1.0),  0.7615941559557649,  1e-12);
    check_close("tanh(-1)",  m_tanh(-1.0), -0.7615941559557649, 1e-12);
    check_close("tanh(50)",  m_tanh(50.0), 1.0,                 1e-12);
    check_close("tanh(-50)", m_tanh(-50.0),-1.0,                1e-12);

    /* f32 sarmalayicilar (daha gevsek epsilon, float32 hassasiyeti) */
    check_close("sqrtf(2)", (f64)m_sqrtf(2.0f), 1.4142135623730951, 1e-6);
    check_close("expf(1)",  (f64)m_expf(1.0f),  2.718281828459045,  1e-6);
    check_close("sinf(pi/6)", (f64)m_sinf((f32)(M_PI_VAL/6.0)), 0.5, 1e-6);

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
