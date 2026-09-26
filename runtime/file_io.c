#include "file_io.h"

/* KATMAN 19 - Coklu platform: Windows'ta CreateFileA/ReadFile/WriteFile,
 * Linux'ta (Debian VM) POSIX open/read/write/close (bkz. PROJE_PLANI.md
 * Bolum 19). FileHandle.os_handle, Windows'ta gercek bir HANDLE,
 * Linux'ta int fd'nin (void*)(isize) olarak sarilmis halidir -- iki
 * platformda da fonksiyon imzalari (file_open_read vb.) DEGISMEDEN
 * ayni kalir. */
#ifdef _WIN32
#include "win32_syscalls.h"
#else
#include <fcntl.h>
#include <unistd.h>
/* rename(2) POSIX'te <stdio.h>'de bildirilir; sadece bu tek syscall
 * sarmalayicisi icin stdio'yu dahil etmek yerine (win32_syscalls.h'deki
 * gibi) kendimiz bildiriyoruz. */
extern int rename(const char* oldpath, const char* newpath);
#endif

/* ReadFile/WriteFile tek seferde en fazla bir DWORD (u32) kadar bayt alır.
 * Büyük korpüs dosyalarını (GB seviyesinde) güvenle işleyebilmek için
 * bu sınırın altında parçalar halinde okuyup/yazıyoruz. */
#define IO_CHUNK (1u << 30) /* 1 GiB */

#ifdef _WIN32

static FileHandle make_handle(void* h) {
    FileHandle f;
    f.os_handle = h;
    f.valid = (h != W_INVALID_HANDLE_VALUE);
    return f;
}

FileHandle file_open_read(const char* path) {
    void* h = CreateFileA(path, W_GENERIC_READ, W_FILE_SHARE_READ, NULL_PTR,
                           W_OPEN_EXISTING, W_FILE_ATTRIBUTE_NORMAL, NULL_PTR);
    return make_handle(h);
}

FileHandle file_open_write(const char* path) {
    void* h = CreateFileA(path, W_GENERIC_WRITE, 0, NULL_PTR,
                           W_CREATE_ALWAYS, W_FILE_ATTRIBUTE_NORMAL, NULL_PTR);
    return make_handle(h);
}

void file_close(FileHandle* f) {
    if (f->valid) {
        CloseHandle(f->os_handle);
        f->valid = FALSE;
        f->os_handle = NULL_PTR;
    }
}

bool32 file_delete(const char* path) {
    return DeleteFileA(path) != 0;
}

bool32 file_rename(const char* from, const char* to) {
    return MoveFileExA(from, to, W_MOVEFILE_REPLACE_EXISTING) != 0;
}

u64 file_size(FileHandle* f) {
    if (!f->valid) return 0;
    i64 size = 0;
    GetFileSizeEx(f->os_handle, &size);
    return (u64)size;
}

u64 file_read(FileHandle* f, void* buffer, u64 count) {
    if (!f->valid) return 0;
    u8* dst = (u8*)buffer;
    u64 total_read = 0;

    while (total_read < count) {
        u64 remaining = count - total_read;
        u32 chunk = (remaining > IO_CHUNK) ? IO_CHUNK : (u32)remaining;
        u32 got = 0;

        i32 ok = ReadFile(f->os_handle, dst + total_read, chunk, &got, NULL_PTR);
        if (!ok || got == 0) break; /* hata ya da dosya sonu */

        total_read += got;
        if (got < chunk) break; /* kısmi okuma -> muhtemelen dosya sonu */
    }

    return total_read;
}

u64 file_write(FileHandle* f, const void* buffer, u64 count) {
    if (!f->valid) return 0;
    const u8* src = (const u8*)buffer;
    u64 total_written = 0;

    while (total_written < count) {
        u64 remaining = count - total_written;
        u32 chunk = (remaining > IO_CHUNK) ? IO_CHUNK : (u32)remaining;
        u32 wrote = 0;

        i32 ok = WriteFile(f->os_handle, src + total_written, chunk, &wrote, NULL_PTR);
        if (!ok) break;

        total_written += wrote;
        if (wrote < chunk) break;
    }

    return total_written;
}

#else /* POSIX (Linux) */

#define POSIX_INVALID_FD (-1)

static FileHandle make_handle(int fd) {
    FileHandle f;
    f.os_handle = (void*)(isize)fd;
    f.valid = (fd != POSIX_INVALID_FD);
    return f;
}

static int handle_fd(FileHandle* f) {
    return (int)(isize)f->os_handle;
}

FileHandle file_open_read(const char* path) {
    int fd = open(path, O_RDONLY);
    return make_handle(fd);
}

FileHandle file_open_write(const char* path) {
    int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    return make_handle(fd);
}

void file_close(FileHandle* f) {
    if (f->valid) {
        close(handle_fd(f));
        f->valid = FALSE;
        f->os_handle = NULL_PTR;
    }
}

bool32 file_delete(const char* path) {
    return unlink(path) == 0;
}

bool32 file_rename(const char* from, const char* to) {
    return rename(from, to) == 0;
}

u64 file_size(FileHandle* f) {
    if (!f->valid) return 0;
    int fd = handle_fd(f);
    isize cur = lseek(fd, 0, SEEK_CUR);
    isize end = lseek(fd, 0, SEEK_END);
    lseek(fd, cur, SEEK_SET);
    return (end > 0) ? (u64)end : 0;
}

u64 file_read(FileHandle* f, void* buffer, u64 count) {
    if (!f->valid) return 0;
    int fd = handle_fd(f);
    u8* dst = (u8*)buffer;
    u64 total_read = 0;

    while (total_read < count) {
        u64 remaining = count - total_read;
        u64 chunk = (remaining > IO_CHUNK) ? IO_CHUNK : remaining;

        isize got = read(fd, dst + total_read, chunk);
        if (got <= 0) break; /* hata (-1) ya da dosya sonu (0) */

        total_read += (u64)got;
        if ((u64)got < chunk) break; /* kismi okuma -> muhtemelen dosya sonu */
    }

    return total_read;
}

u64 file_write(FileHandle* f, const void* buffer, u64 count) {
    if (!f->valid) return 0;
    int fd = handle_fd(f);
    const u8* src = (const u8*)buffer;
    u64 total_written = 0;

    while (total_written < count) {
        u64 remaining = count - total_written;
        u64 chunk = (remaining > IO_CHUNK) ? IO_CHUNK : remaining;

        isize wrote = write(fd, src + total_written, chunk);
        if (wrote < 0) break;

        total_written += (u64)wrote;
        if ((u64)wrote < chunk) break;
    }

    return total_written;
}

#endif

void* file_read_entire(const char* path, Allocator* alloc, u64* out_size) {
    *out_size = 0;
    FileHandle f = file_open_read(path);
    if (!f.valid) return NULL_PTR;

    u64 size = file_size(&f);
    void* buffer = NULL_PTR;

    if (size > 0) {
        buffer = allocator_alloc(alloc, size);
        u64 got = file_read(&f, buffer, size);
        if (got != size) {
            allocator_free(alloc, buffer);
            buffer = NULL_PTR;
            size = 0;
        }
    }

    file_close(&f);
    *out_size = size;
    return buffer;
}
