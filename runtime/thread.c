#include "thread.h"

#ifdef _WIN32
#include "win32_syscalls.h"
#endif

#ifdef _WIN32

void thread_create(ThreadHandle* out, ThreadFn fn, void* arg) {
    w_dword tid;
    out->os_handle = CreateThread(NULL_PTR, 0, (ThreadStartFn)fn, arg, 0, &tid);
}

void thread_join(ThreadHandle* t) {
    WaitForSingleObject(t->os_handle, W_INFINITE);
    CloseHandle(t->os_handle);
}

void thread_join_all(ThreadHandle* threads, u32 count) {
    w_handle handles[THREAD_MAX_HANDLES];
    for (u32 i = 0; i < count; i++) handles[i] = threads[i].os_handle;
    WaitForMultipleObjects(count, handles, TRUE, W_INFINITE);
    for (u32 i = 0; i < count; i++) CloseHandle(handles[i]);
}

MutexHandle mutex_create(void) {
    MutexHandle m;
    m.os_handle = CreateMutexA(NULL_PTR, FALSE, NULL_PTR);
    return m;
}

void mutex_lock(MutexHandle* m) {
    WaitForSingleObject(m->os_handle, W_INFINITE);
}

void mutex_unlock(MutexHandle* m) {
    ReleaseMutex(m->os_handle);
}

#else /* POSIX (Linux) */

/* pthread_create, "void* (*)(void*)" imzali bir fonksiyon bekler; bizim
 * ThreadFn'imiz "u32 (*)(void*)" -- bu kucuk trampoline ikisi arasinda
 * koprulyor. out (cagiranin KENDI, tasinmayacak deposu) hem sonucu hem
 * de trampoline'un okuyacagi {fn,arg} çiftini tasir. */
static void* posix_trampoline(void* p) {
    ThreadHandle* h = (ThreadHandle*)p;
    h->fn(h->arg);
    return NULL_PTR;
}

void thread_create(ThreadHandle* out, ThreadFn fn, void* arg) {
    out->fn = fn;
    out->arg = arg;
    pthread_create(&out->tid, NULL_PTR, posix_trampoline, out);
}

void thread_join(ThreadHandle* t) {
    pthread_join(t->tid, NULL_PTR);
}

void thread_join_all(ThreadHandle* threads, u32 count) {
    /* POSIX'te WaitForMultipleObjects'in dogrudan bir esdegeri yok --
     * ama HEPSINI beklemek (bWaitAll=TRUE) ile sirayla pthread_join
     * cagirmak DAVRANISSAL olarak ozdestir (sonuc: hepsi bitene kadar
     * blokla). */
    for (u32 i = 0; i < count; i++) pthread_join(threads[i].tid, NULL_PTR);
}

MutexHandle mutex_create(void) {
    MutexHandle m;
    pthread_mutex_init(&m.native, NULL_PTR);
    return m;
}

void mutex_lock(MutexHandle* m) {
    pthread_mutex_lock(&m->native);
}

void mutex_unlock(MutexHandle* m) {
    pthread_mutex_unlock(&m->native);
}

#endif
