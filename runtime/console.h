/* Katman 0 - Minimum konsol çıktısı (printf yok; WriteFile syscall'ı
 * üzerine ince bir sarmalayıcı, sadece test/tanılama mesajları için). */
#ifndef RUNTIME_CONSOLE_H
#define RUNTIME_CONSOLE_H

#include "types.h"

void console_write(const char* text); /* sıfır sonlandırmalı (null-terminated) string yazar */
void console_write_u64(u64 value);    /* ondalık tabanda yazar */
void console_write_line(const char* text);

/* Standart girdiden (stdin) bir satır okur (Enter'a kadar, \r\n atilir).
 * buf'a en fazla max_len-1 bayt yazip sifir-sonlandirir. Okunan bayt
 * sayisini dondurur (0 = bos satir ya da EOF). */
u32 console_read_line(char* buf, u32 max_len);

#endif /* RUNTIME_CONSOLE_H */
