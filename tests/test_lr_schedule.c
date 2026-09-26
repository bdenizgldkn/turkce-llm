/* Katman 22 dogrulama: training/lr_schedule.c (isinma + kosinus). */
#include "../runtime/types.h"
#include "../runtime/console.h"
#include "../training/lr_schedule.h"

static u32 g_pass = 0;
static u32 g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; console_write("  [FAIL] "); console_write_line(msg); } \
} while (0)

static bool32 near(f32 a, f32 b) { f32 d = a - b; if (d < 0) d = -d; return d <= 1e-6f * (b < 0 ? -b : b) + 1e-12f; }

int main(void) {
    console_write_line("=== Katman 22: ogrenme hizi takvimi ===");
    const f32 peak = 6e-4f, mn = 6e-5f;
    const u64 W = 1000, TOT = 70000;

    CHECK(near(lr_warmup_cosine(0, W, TOT, peak, mn), peak / 1000.0f), "isinma: adim 0 = peak/W olmali");
    CHECK(near(lr_warmup_cosine(499, W, TOT, peak, mn), peak * 0.5f), "isinma: adim W/2-1 = peak/2 olmali");
    CHECK(near(lr_warmup_cosine(W - 1, W, TOT, peak, mn), peak), "isinma sonu = peak olmali");
    CHECK(near(lr_warmup_cosine(W, W, TOT, peak, mn), peak), "kosinus basi = peak olmali");
    CHECK(near(lr_warmup_cosine(W + (TOT - W) / 2, W, TOT, peak, mn), 0.5f * (peak + mn)), "kosinus ortasi = (peak+min)/2 olmali");
    CHECK(near(lr_warmup_cosine(TOT, W, TOT, peak, mn), mn), "son adim = min olmali");
    CHECK(near(lr_warmup_cosine(TOT + 5000, W, TOT, peak, mn), mn), "toplamdan sonra min'de kalmali");

    bool32 mono_up = TRUE, mono_down = TRUE;
    f32 prev = 0.0f;
    for (u64 s = 0; s < W; s++) { f32 v = lr_warmup_cosine(s, W, TOT, peak, mn); if (v <= prev) mono_up = FALSE; prev = v; }
    for (u64 s = W; s <= TOT; s++) { f32 v = lr_warmup_cosine(s, W, TOT, peak, mn); if (v > prev + 1e-12f) mono_down = FALSE; prev = v; }
    CHECK(mono_up, "isinma kesin artan olmali");
    CHECK(mono_down, "kosinus azalmayan olmamali (monoton azalan)");

    console_write("Sonuc: "); console_write_u64(g_pass); console_write(" basarili, ");
    console_write_u64(g_fail); console_write_line(" basarisiz.");
    return g_fail == 0 ? 0 : 1;
}
