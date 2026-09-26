/* Katman 0 - Dosya I/O.
 * CreateFileA/ReadFile/WriteFile/CloseHandle (win32_syscalls.h) üzerine
 * ince bir sarmalayıcı. Linux'taki open/read/write/close karşılığı. */
#ifndef RUNTIME_FILE_IO_H
#define RUNTIME_FILE_IO_H

#include "types.h"
#include "memory.h"

typedef struct FileHandle {
    void* os_handle;
    bool32 valid;
} FileHandle;

FileHandle file_open_read(const char* path);
FileHandle file_open_write(const char* path); /* varsa üzerine yazar (truncate) */
void       file_close(FileHandle* f);
/* Dosyayi siler (Windows: DeleteFileA, POSIX: unlink). Basarida TRUE. */
bool32     file_delete(const char* path);
/* from -> to olarak yeniden adlandirir; to varsa ATOMIK olarak degistirir
 * (Windows: MoveFileExA+REPLACE_EXISTING, POSIX: rename). Basarida TRUE.
 * "Gecici dosyaya yaz, sonra yeniden adlandir" deseniyle yarim yazilmis
 * dosyalarin asil dosyanin yerine gecmesini onler (bkz. checkpoint_save). */
bool32     file_rename(const char* from, const char* to);

u64  file_size(FileHandle* f);

/* count bayt okumayı dener, gerçekte okunan bayt sayısını döndürür. */
u64  file_read(FileHandle* f, void* buffer, u64 count);
/* count bayt yazmayı dener, gerçekte yazılan bayt sayısını döndürür. */
u64  file_write(FileHandle* f, const void* buffer, u64 count);

/* Dosyanın tamamını allocator üzerinden ayrılan bir arabelleğe okur.
 * out_size'a okunan bayt sayısını yazar. Hata durumunda NULL_PTR döner. */
void* file_read_entire(const char* path, Allocator* alloc, u64* out_size);

#endif /* RUNTIME_FILE_IO_H */
