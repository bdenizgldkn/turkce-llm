#include "console.h"

/* KATMAN 19 - Coklu platform: Windows'ta GetStdHandle+WriteFile/ReadFile,
 * Linux'ta (Debian VM) dogrudan POSIX write(1,...)/read(0,...) -- stdout/
 * stdin dosya betimleyicileri (fd) sabit oldugu icin "handle almaya"
 * bile gerek yok (bkz. PROJE_PLANI.md Bolum 19). */
#ifdef _WIN32
#include "win32_syscalls.h"
#else
#include <unistd.h>
#define POSIX_STDOUT_FD 1
#define POSIX_STDIN_FD  0
#endif

static u64 cstr_len(const char* s) {
    u64 n = 0;
    while (s[n] != 0) n++;
    return n;
}

void console_write(const char* text) {
    u32 len = (u32)cstr_len(text);
#ifdef _WIN32
    w_handle h = GetStdHandle(W_STD_OUTPUT_HANDLE);
    u32 written = 0;
    WriteFile(h, text, len, &written, NULL_PTR);
#else
    write(POSIX_STDOUT_FD, text, len);
#endif
}

void console_write_line(const char* text) {
    console_write(text);
    console_write("\n");
}

void console_write_u64(u64 value) {
    char buf[20]; /* u64 max 20 basamak */
    i32 i = 20;
    if (value == 0) {
        console_write("0");
        return;
    }
    while (value > 0 && i > 0) {
        i--;
        buf[i] = (char)('0' + (value % 10));
        value /= 10;
    }
    u32 len = (u32)(20 - i);
#ifdef _WIN32
    w_handle h = GetStdHandle(W_STD_OUTPUT_HANDLE);
    u32 written = 0;
    WriteFile(h, buf + i, len, &written, NULL_PTR);
#else
    write(POSIX_STDOUT_FD, buf + i, len);
#endif
}

u32 console_read_line(char* buf, u32 max_len) {
    u32 n = 0;
#ifdef _WIN32
    w_handle h = GetStdHandle(W_STD_INPUT_HANDLE);
#endif
    for (;;) {
        char c;
#ifdef _WIN32
        u32 got = 0;
        i32 ok = ReadFile(h, &c, 1, &got, NULL_PTR);
        if (!ok || got == 0) break; /* EOF ya da hata */
#else
        isize got = read(POSIX_STDIN_FD, &c, 1);
        if (got <= 0) break; /* EOF ya da hata */
#endif
        if (c == '\n') break;
        if (c == '\r') continue;
        if (n < max_len - 1) buf[n++] = c;
    }
    buf[n] = 0;
    return n;
}
