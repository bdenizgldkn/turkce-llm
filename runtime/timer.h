/* Katman 0 (uzantisi) - Yuksek cozunurluklu zamanlayici.
 * Windows QueryPerformanceCounter uzerine ince bir sarmalayici
 * (CPU/GPU performans karsilastirmalari icin). */
#ifndef RUNTIME_TIMER_H
#define RUNTIME_TIMER_H

#include "types.h"

f64 timer_now_seconds(void);

#endif /* RUNTIME_TIMER_H */
