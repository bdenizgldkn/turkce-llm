/* Katman 0 - Windows sistem çağrıları arayüzü.
 *
 * <windows.h> dahil edilmiyor (o başlık binlerce ilgisiz makro/typedef içerir).
 * Bunun yerine, sadece ihtiyacımız olan kernel32.dll fonksiyonlarının
 * imzalarını kendimiz bildiriyoruz. Bu fonksiyonlar Linux'taki
 * mmap/brk/open/read/write/close syscall'larının Windows karşılığıdır:
 * bir "kütüphane algoritması" değil, işletim sistemiyle aramızdaki
 * zorunlu geçiş noktasıdır (bkz. PROJE_PLANI.md, Bölüm 1).
 *
 * MinGW-w64, kernel32.dll için içe aktarma (import) kütüphanesini
 * (libkernel32.a) varsayılan olarak linker'a ekler; bu yüzden ayrı bir
 * -lkernel32 bayrağına gerek yoktur.
 */
#ifndef RUNTIME_WIN32_SYSCALLS_H
#define RUNTIME_WIN32_SYSCALLS_H

#include "types.h"

typedef void*    w_handle;
typedef u32      w_dword;
typedef i64      w_large_int;

#define W_INVALID_HANDLE_VALUE ((w_handle)(isize)-1)

/* --- Bellek: VirtualAlloc / VirtualFree (mmap/brk karşılığı) --- */
#define W_MEM_COMMIT   0x00001000u
#define W_MEM_RESERVE  0x00002000u
#define W_MEM_RELEASE  0x00008000u
#define W_MEM_DECOMMIT 0x00004000u
#define W_PAGE_READWRITE 0x04u

__declspec(dllimport) void*   __stdcall VirtualAlloc(void* lpAddress, u64 dwSize, w_dword flAllocationType, w_dword flProtect);
__declspec(dllimport) i32     __stdcall VirtualFree(void* lpAddress, u64 dwSize, w_dword dwFreeType);

/* --- Dosya I/O: CreateFile / ReadFile / WriteFile / CloseHandle (open/read/write/close karşılığı) --- */
#define W_GENERIC_READ  0x80000000u
#define W_GENERIC_WRITE 0x40000000u
#define W_FILE_SHARE_READ 0x00000001u
#define W_CREATE_ALWAYS 2u
#define W_OPEN_EXISTING 3u
#define W_OPEN_ALWAYS   4u
#define W_FILE_ATTRIBUTE_NORMAL 0x80u
#define W_FILE_BEGIN   0u
#define W_FILE_CURRENT 1u
#define W_FILE_END     2u

__declspec(dllimport) w_handle __stdcall CreateFileA(const char* lpFileName, w_dword dwDesiredAccess, w_dword dwShareMode, void* lpSecurityAttributes, w_dword dwCreationDisposition, w_dword dwFlagsAndAttributes, w_handle hTemplateFile);
__declspec(dllimport) i32      __stdcall ReadFile(w_handle hFile, void* lpBuffer, w_dword nNumberOfBytesToRead, w_dword* lpNumberOfBytesRead, void* lpOverlapped);
__declspec(dllimport) i32      __stdcall WriteFile(w_handle hFile, const void* lpBuffer, w_dword nNumberOfBytesToWrite, w_dword* lpNumberOfBytesWritten, void* lpOverlapped);
__declspec(dllimport) i32      __stdcall CloseHandle(w_handle hObject);
__declspec(dllimport) i32      __stdcall SetFilePointerEx(w_handle hFile, w_large_int liDistanceToMove, w_large_int* lpNewFilePointer, w_dword dwMoveMethod);
__declspec(dllimport) i32      __stdcall GetFileSizeEx(w_handle hFile, w_large_int* lpFileSize);
__declspec(dllimport) w_dword  __stdcall GetLastError(void);
__declspec(dllimport) i32      __stdcall DeleteFileA(const char* lpFileName);
#define W_MOVEFILE_REPLACE_EXISTING 0x1u
__declspec(dllimport) i32      __stdcall MoveFileExA(const char* lpExistingFileName, const char* lpNewFileName, w_dword dwFlags);

/* --- Yuksek cozunurluklu zamanlayici (performans olcumu icin) --- */
__declspec(dllimport) i32      __stdcall QueryPerformanceCounter(i64* lpPerformanceCount);
__declspec(dllimport) i32      __stdcall QueryPerformanceFrequency(i64* lpFrequency);

/* --- Isler/threadler: CreateThread/WaitForSingleObject/WaitForMultipleObjects
 * ve karsilikli dislama icin CreateMutexA/ReleaseMutex. Bunlar da (VirtualAlloc
 * gibi) isletim sistemi syscall'lari -- "eszamanli programlama kutuphanesi"
 * degil, cok cekirdekli donanima erisim kapisi (bkz. PROJE_PLANI.md Bolum 1).
 * Cok cekirdekli CPU'nun (bkz. Bolum 16) atil kalan kapasitesini kullanmak
 * icin: veri-paralel egitimde her thread kendi dizisini bagimsiz isler. */
typedef u32 (__stdcall *ThreadStartFn)(void* param);

#define W_INFINITE 0xFFFFFFFFu

__declspec(dllimport) w_handle __stdcall CreateThread(void* lpThreadAttributes, u64 dwStackSize, ThreadStartFn lpStartAddress, void* lpParameter, w_dword dwCreationFlags, w_dword* lpThreadId);
__declspec(dllimport) w_dword  __stdcall WaitForSingleObject(w_handle hHandle, w_dword dwMilliseconds);
__declspec(dllimport) w_dword  __stdcall WaitForMultipleObjects(w_dword nCount, const w_handle* lpHandles, i32 bWaitAll, w_dword dwMilliseconds);
__declspec(dllimport) w_handle __stdcall CreateMutexA(void* lpMutexAttributes, i32 bInitialOwner, const char* lpName);
__declspec(dllimport) i32      __stdcall ReleaseMutex(w_handle hMutex);
__declspec(dllimport) w_dword  __stdcall GetCurrentThreadId(void);

/* --- Çıkış / konsol (tanılama mesajları için minimum gereksinim) --- */
__declspec(dllimport) void     __stdcall ExitProcess(w_dword uExitCode);
__declspec(dllimport) w_handle __stdcall GetStdHandle(w_dword nStdHandle);
#define W_STD_INPUT_HANDLE  ((w_dword)-10)
#define W_STD_OUTPUT_HANDLE ((w_dword)-11)
#define W_STD_ERROR_HANDLE  ((w_dword)-12)

#endif /* RUNTIME_WIN32_SYSCALLS_H */
