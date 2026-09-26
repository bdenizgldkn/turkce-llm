/* Katman 0 (uzantisi) - Ham thread/mutex syscall'lari uzerine ince bir
 * sarmalayici: Windows'ta win32_syscalls.h'deki CreateThread/CreateMutexA,
 * Linux'ta (Debian VM, bkz. PROJE_PLANI.md Bolum 19) POSIX <pthread.h>
 * (glibc'nin dogrudan syscall sarmalayicisi -- NPTL -- <windows.h> ile
 * ayni felsefe: ihtiyacimiz olan minimum arayuz).
 *
 * Amac: cok cekirdekli CPU'nun (bkz. PROJE_PLANI.md Bolum 16) atil
 * kalan kapasitesini kullanmak -- egitim dongusu tek thread'de
 * calisirken cekirdeklerin buyuk kismi bos duruyordu.
 *
 * NOT: thread_create bir OUT-PARAMETRE alir (deger DONDURMEZ) --
 * POSIX tarafinda pthread_create'e verilen trampoline argumaninin
 * (fn+arg) STABIL bir adrese ihtiyaci var (thread calisirken tasinmamali);
 * bu yuzden cagiran, depolamayi (orn. threads[i]) KENDISI saglar. */
#ifndef RUNTIME_THREAD_H
#define RUNTIME_THREAD_H

#include "types.h"

#ifndef _WIN32
#include <pthread.h>
#endif

#define THREAD_MAX_HANDLES 64

typedef u32 (*ThreadFn)(void* arg);

typedef struct ThreadHandle {
#ifdef _WIN32
    void* os_handle;
#else
    pthread_t tid;
    ThreadFn fn;
    void* arg;
#endif
} ThreadHandle;

typedef struct MutexHandle {
#ifdef _WIN32
    void* os_handle;
#else
    pthread_mutex_t native;
#endif
} MutexHandle;

/* *out, cagiranin SAHIP OLDUGU ve thread bitene kadar TASINMAYACAK bir
 * depoda olmalidir (orn. bir dizinin elemani). */
void thread_create(ThreadHandle* out, ThreadFn fn, void* arg);
void thread_join(ThreadHandle* t);
/* count adet handle'in TUMUNUN bitmesini bekler. */
void thread_join_all(ThreadHandle* threads, u32 count);

MutexHandle mutex_create(void);
void mutex_lock(MutexHandle* m);
void mutex_unlock(MutexHandle* m);

#endif /* RUNTIME_THREAD_H */
