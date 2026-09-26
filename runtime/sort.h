/* Katman 0 (uzantisi) - Temel siralama.
 * <stdlib.h>'nin qsort'u KULLANILMAZ. Anahtar (key) dizisine gore
 * bir indeks dizisini siralayan basit bir quicksort. */
#ifndef RUNTIME_SORT_H
#define RUNTIME_SORT_H

#include "types.h"

/* indices[0..count) baslangicta 0,1,2,...,count-1 olmalidir (cagiran
 * doldurur). Siralama sonrasi: keys[indices[0]] >= keys[indices[1]] >= ...
 * (azalan sirada, BPE/frekans siralamasi icin). */
void sort_indices_by_u64_desc(const u64* keys, u32* indices, u64 count);

#endif /* RUNTIME_SORT_H */
