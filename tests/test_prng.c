/* Katman 2 dogrulama testleri: PCG PRNG + Box-Muller Gauss uretimi.
 * Rastgelelik test edildigi icin kontroller istatistikseldir (buyuk
 * ornek boyutuyla, genis güvenlik payli toleranslarla). */
#include "../runtime/types.h"
#include "../runtime/prng.h"
#include "../runtime/console.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

static f64 fabs_(f64 x) { return (x < 0.0) ? -x : x; }

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; console_write("  [FAIL] "); console_write_line(msg); } \
} while (0)

static void check_close(const char* name, f64 got, f64 expected, f64 eps) {
    if (fabs_(got - expected) <= eps) { g_pass++; }
    else {
        g_fail++;
        console_write("  [FAIL] "); console_write(name);
        console_write(" got*1e6=");
        i64 g = (i64)(got * 1e6);
        console_write_u64((u64)(g < 0 ? -g : g));
        console_write_line(g < 0 ? " (negatif)" : "");
    }
}

static void test_reproducibility(void) {
    PCGState a = pcg_seed(42, 1);
    PCGState b = pcg_seed(42, 1);
    bool32 same = TRUE;
    for (i32 i = 0; i < 1000; i++) {
        if (pcg_next_u32(&a) != pcg_next_u32(&b)) { same = FALSE; break; }
    }
    CHECK(same, "prng: ayni seed ayni diziyi uretmiyor");

    PCGState c = pcg_seed(42, 2); /* farkli akis (initseq) */
    bool32 differs = FALSE;
    for (i32 i = 0; i < 20; i++) {
        if (pcg_next_u32(&a) != pcg_next_u32(&c)) { differs = TRUE; break; }
    }
    /* a artik 1000 adim ilerlemis durumda, c ise 0'dan basliyor; farkli
     * akis parametresiyle dizilerin ayni olmasi pratikte imkansiz. */
    CHECK(differs, "prng: farkli initseq ayni diziyi uretti (beklenmedik)");
}

static void test_uniform_range_and_moments(void) {
    PCGState rng = pcg_seed(12345, 99);
    const u64 N = 200000;
    f64 sum = 0.0;
    bool32 in_range = TRUE;

    for (u64 i = 0; i < N; i++) {
        f64 u = pcg_uniform(&rng);
        if (u < 0.0 || u >= 1.0) in_range = FALSE;
        sum += u;
    }

    CHECK(in_range, "prng: uniform ornekler [0,1) disina cikti");
    f64 mean = sum / (f64)N;
    /* Beklenen ortalama 0.5, teorik std hata ~0.00091; 0.02 toleransi
     * >20 sigma guvenlik payi birakir (pratikte hep gecmeli). */
    check_close("uniform ortalama ~0.5", mean, 0.5, 0.02);
}

static void test_gaussian_moments(void) {
    PCGState rng = pcg_seed(777, 5);
    const u64 N = 200000;
    f64 sum = 0.0;
    f64 sum_sq = 0.0;

    for (u64 i = 0; i < N; i++) {
        f64 z = pcg_gaussian_std(&rng);
        sum += z;
        sum_sq += z * z;
    }

    f64 mean = sum / (f64)N;
    f64 var = sum_sq / (f64)N - mean * mean;

    /* Standart normal: ortalama 0, varyans 1. Std hata (ortalama icin)
     * ~1/sqrt(N) = ~0.0022; genis pay birakalim. */
    check_close("gauss ortalama ~0", mean, 0.0, 0.02);
    check_close("gauss varyans ~1", var, 1.0, 0.05);
}

static void test_shuffle_is_permutation(void) {
    const u64 N = 1000;
    u32 arr[1000];
    for (u64 i = 0; i < N; i++) arr[i] = (u32)i;

    PCGState rng = pcg_seed(1, 1);
    pcg_shuffle_u32(&rng, arr, N);

    /* Her elemanin tam olarak bir kez gectigini dogrula (permutasyon testi). */
    u8 seen[1000];
    for (u64 i = 0; i < N; i++) seen[i] = 0;
    bool32 valid = TRUE;
    for (u64 i = 0; i < N; i++) {
        if (arr[i] >= N || seen[arr[i]]) { valid = FALSE; break; }
        seen[arr[i]] = 1;
    }
    CHECK(valid, "prng: shuffle sonrasi dizi gecerli bir permutasyon degil");

    bool32 changed = FALSE;
    for (u64 i = 0; i < N; i++) if (arr[i] != i) { changed = TRUE; break; }
    CHECK(changed, "prng: shuffle diziyi hic degistirmedi (kuskulu)");
}

int main(void) {
    console_write_line("=== Katman 2 (PRNG) Testleri ===");

    test_reproducibility();
    test_uniform_range_and_moments();
    test_gaussian_moments();
    test_shuffle_is_permutation();

    console_write("Sonuc: ");
    console_write_u64(g_pass);
    console_write(" basarili, ");
    console_write_u64(g_fail);
    console_write_line(" basarisiz.");

    return (g_fail == 0) ? 0 : 1;
}
