/* Katman 0 (uzantisi) - Temel string islemleri.
 * <string.h> KULLANILMAZ -- strlen/strstr/memcpy benzeri her sey
 * burada kendimiz yazilir. Bayt dizileri (UTF-8 dahil) uzerinde
 * calisir; Turkce'ye ozgu kod noktalarini bilmez, sadece ham bayt
 * karsilastirmasi yapar (bu yeterlidir: UTF-8'de bir ASCII baytin
 * baska bir karakterin parcasi olarak yanlislikla eslesmesi imkansizdir). */
#ifndef RUNTIME_STRUTIL_H
#define RUNTIME_STRUTIL_H

#include "types.h"

u64 str_len(const char* s);
bool32 str_eq(const char* a, const char* b);
bool32 str_eq_n(const char* a, const char* b, u64 n);
bool32 str_starts_with(const char* s, const char* prefix);

/* haystack icinde needle'i arar; bulunursa haystack icindeki isaretciyi,
 * bulunamazsa NULL_PTR dondurur. limit, haystack'in taranacak maksimum
 * uzunlugudur (buyuk dosya taramalarinda sonu bilinen arabellekler icin). */
const char* str_find(const char* haystack, u64 haystack_len, const char* needle);

void mem_copy(void* dst, const void* src, u64 n);
void mem_set(void* dst, u8 value, u64 n);

#endif /* RUNTIME_STRUTIL_H */
