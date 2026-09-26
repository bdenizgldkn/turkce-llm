#include "sort.h"

static void swap_u32(u32* a, u32* b) {
    u32 t = *a; *a = *b; *b = t;
}

/* Ortanca-uc pivot secimi ile quicksort -- zaten sirali/ters-sirali
 * girdilerde en kotu durum performansini pratikte onler. */
static void quicksort_desc(const u64* keys, u32* idx, i64 lo, i64 hi) {
    while (lo < hi) {
        i64 mid = lo + (hi - lo) / 2;

        /* ortanca-uc: keys[idx[lo]], keys[idx[mid]], keys[idx[hi]] sirala,
         * ortanca degeri pivot olarak hi-1 konumuna tasi. */
        if (keys[idx[lo]] < keys[idx[mid]]) swap_u32(&idx[lo], &idx[mid]);
        if (keys[idx[lo]] < keys[idx[hi]]) swap_u32(&idx[lo], &idx[hi]);
        if (keys[idx[mid]] < keys[idx[hi]]) swap_u32(&idx[mid], &idx[hi]);
        swap_u32(&idx[mid], &idx[hi]);
        u64 pivot = keys[idx[hi]];

        i64 i = lo - 1;
        for (i64 j = lo; j < hi; j++) {
            if (keys[idx[j]] > pivot) { /* azalan sira */
                i++;
                swap_u32(&idx[i], &idx[j]);
            }
        }
        swap_u32(&idx[i + 1], &idx[hi]);
        i64 p = i + 1;

        /* Kucuk parcayi ozyinelemeli, buyuk parcayi dongude isle (yigin
         * derinligini O(log n) ile sinirlar). */
        if (p - lo < hi - p) {
            quicksort_desc(keys, idx, lo, p - 1);
            lo = p + 1;
        } else {
            quicksort_desc(keys, idx, p + 1, hi);
            hi = p - 1;
        }
    }
}

void sort_indices_by_u64_desc(const u64* keys, u32* indices, u64 count) {
    if (count < 2) return;
    quicksort_desc(keys, indices, 0, (i64)count - 1);
}
