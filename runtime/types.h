/* Katman 0 - Temel tipler.
 * <stdint.h>/<stddef.h> gibi standart başlıklara bağımlı olmamak için
 * derleyicinin yerleşik (built-in) tam sayı genişlikleri üzerinden
 * kendi sabit-genişlikli tiplerimizi tanımlıyoruz.
 */
#ifndef RUNTIME_TYPES_H
#define RUNTIME_TYPES_H

typedef unsigned char      u8;
typedef unsigned short     u16;
typedef unsigned int       u32;
typedef unsigned long long u64;

typedef signed char        i8;
typedef signed short       i16;
typedef signed int         i32;
typedef signed long long   i64;

typedef float  f32;
typedef double f64;

typedef u64 usize;   /* boyut/indeks türü (size_t yerine) */
typedef i64 isize;

typedef u32 bool32;
#define TRUE  1
#define FALSE 0

#define NULL_PTR ((void*)0)

#endif /* RUNTIME_TYPES_H */
