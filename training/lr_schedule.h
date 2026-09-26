/* Katman 22 - Ogrenme hizi takvimi: dogrusal isinma + kosinus azaltma.
 *
 *   step <  warmup:  lr = peak * (step+1) / warmup
 *   step >= warmup:  lr = min + 0.5*(peak-min)*(1 + cos(pi * p)),
 *                    p = (step-warmup) / (total-warmup), 1'e kirpilir
 *
 * Sadece adim numarasinin fonksiyonudur -- checkpoint'ten devam eden
 * kosu kesintisiz kosuyla AYNI lr'yi kullanir. cos, runtime/mathlib.c'den. */
#ifndef TRAINING_LR_SCHEDULE_H
#define TRAINING_LR_SCHEDULE_H

#include "../runtime/types.h"

f32 lr_warmup_cosine(u64 step, u64 warmup_steps, u64 total_steps, f32 peak_lr, f32 min_lr);

#endif /* TRAINING_LR_SCHEDULE_H */
