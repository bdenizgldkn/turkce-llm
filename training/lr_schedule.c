#include "lr_schedule.h"
#include "../runtime/mathlib.h"

#define LR_PI 3.14159265358979323846

f32 lr_warmup_cosine(u64 step, u64 warmup_steps, u64 total_steps, f32 peak_lr, f32 min_lr) {
    if (warmup_steps > 0 && step < warmup_steps) {
        return peak_lr * (f32)(step + 1) / (f32)warmup_steps;
    }
    if (total_steps <= warmup_steps) return min_lr;
    f64 p = (f64)(step - warmup_steps) / (f64)(total_steps - warmup_steps);
    if (p > 1.0) p = 1.0;
    f64 c = 0.5 * (1.0 + m_cos(LR_PI * p));
    return (f32)((f64)min_lr + ((f64)peak_lr - (f64)min_lr) * c);
}
