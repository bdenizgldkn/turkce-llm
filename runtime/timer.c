#include "timer.h"

/* KATMAN 19 - Coklu platform: Windows'ta QueryPerformanceCounter,
 * Linux'ta (Debian VM) clock_gettime(CLOCK_MONOTONIC) -- ikisi de
 * isletim sisteminin tekdüze (monotonic), yuksek cozunurluklu
 * saatidir (bkz. PROJE_PLANI.md Bolum 19). */
#ifdef _WIN32
#include "win32_syscalls.h"
#else
#include <time.h>
#endif

f64 timer_now_seconds(void) {
#ifdef _WIN32
    i64 count = 0, freq = 1;
    QueryPerformanceCounter(&count);
    QueryPerformanceFrequency(&freq);
    return (f64)count / (f64)freq;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (f64)ts.tv_sec + (f64)ts.tv_nsec / 1e9;
#endif
}
