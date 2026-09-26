#include "checkpoint.h"
#include "../runtime/file_io.h"
#include "../runtime/strutil.h"

/* Eksik yazmayi (orn. disk dolu) *ok'a isler; bir kez FALSE olunca
 * sonraki yazmalar atlanir. */
static void write_raw(FileHandle* f, const void* data, u64 bytes, bool32* ok) {
    if (!*ok) return;
    if (file_write(f, data, bytes) != bytes) *ok = FALSE;
}

#define CKPT_MAX_PATH 512

/* path + ".tmp" -> out. Sigmazsa FALSE. */
static bool32 make_tmp_path(const char* path, char* out) {
    const char suffix[] = ".tmp";
    u64 n = str_len(path);
    if (n + sizeof(suffix) > CKPT_MAX_PATH) return FALSE;
    mem_copy(out, path, n);
    mem_copy(out + n, suffix, sizeof(suffix)); /* sondaki '\0' dahil */
    return TRUE;
}

bool32 checkpoint_save(const char* path, Node** params, u32 num_params, const AdamOptimizer* opt) {
    /* Once path.tmp'ye yazilir, TAMAMI basariyla yazilinca path'in
     * uzerine ATOMIK olarak yeniden adlandirilir. Boylece kayit sirasinda
     * cokme/disk dolmasi olursa path'teki ONCEKI saglam checkpoint
     * korunur (egitime devam etme bu dosyaya dayaniyor). */
    char tmp_path[CKPT_MAX_PATH];
    if (!make_tmp_path(path, tmp_path)) return FALSE;

    FileHandle f = file_open_write(tmp_path);
    if (!f.valid) return FALSE;
    bool32 ok = TRUE;

    const char magic[4] = { 'T', 'R', 'L', 'M' };
    u32 version = 1;
    u32 has_opt = (opt != NULL_PTR) ? 1u : 0u;

    write_raw(&f, magic, 4, &ok);
    write_raw(&f, &version, sizeof(u32), &ok);
    write_raw(&f, &num_params, sizeof(u32), &ok);
    write_raw(&f, &has_opt, sizeof(u32), &ok);

    for (u32 i = 0; i < num_params; i++) {
        const Tensor* t = &params[i]->value;
        write_raw(&f, &t->ndim, sizeof(u32), &ok);
        for (u32 d = 0; d < t->ndim; d++) write_raw(&f, &t->shape[d], sizeof(u64), &ok);
        write_raw(&f, t->data, t->numel * sizeof(f32), &ok);
    }

    if (has_opt) {
        for (u32 i = 0; i < num_params; i++) {
            write_raw(&f, opt->m[i].data, opt->m[i].numel * sizeof(f32), &ok);
            write_raw(&f, opt->v[i].data, opt->v[i].numel * sizeof(f32), &ok);
        }
        write_raw(&f, &opt->t, sizeof(u64), &ok);
    }

    file_close(&f);

    if (ok) ok = file_rename(tmp_path, path);
    if (!ok) file_delete(tmp_path);
    return ok;
}

bool32 checkpoint_load(Allocator* scratch, const char* path, Node** params, u32 num_params, AdamOptimizer* opt) {
    u64 size = 0;
    u8* buf = (u8*)file_read_entire(path, scratch, &size);
    if (!buf) return FALSE;

    /* 1. GECIS -- sadece DOGRULA, hicbir seyi kopyalama. Basarisizlikta
     * params/opt'un YARIM yuklenmis halde kalmamasi icin (egitime devam
     * ederken bu, bozuk agirliklarla sessizce devam etmek demek olurdu).
     * Her okumadan once sinir kontrolu yapilir; beklenen toplam boyut
     * dosya boyutuyla BIREBIR eslesmeli (yarim yazilmis/kesik dosya
     * reddedilir). */
    u64 pos = 0;
    if (size < 16 || buf[0] != 'T' || buf[1] != 'R' || buf[2] != 'L' || buf[3] != 'M') return FALSE;
    pos += 4;

    u32 version, file_num_params, has_opt;
    mem_copy(&version, buf + pos, 4); pos += 4;
    mem_copy(&file_num_params, buf + pos, 4); pos += 4;
    mem_copy(&has_opt, buf + pos, 4); pos += 4;

    if (version != 1 || file_num_params != num_params) return FALSE;
    /* Optimizer durumu istendiyse dosyada OLMALI -- yoksa opt->t=0 ile
     * "devam etmek" sessizce yanlis olurdu. */
    if (opt != NULL_PTR && !has_opt) return FALSE;

    u64 param_offsets[CKPT_MAX_PARAMS];
    if (num_params > CKPT_MAX_PARAMS) return FALSE;

    u64 scalars_total = 0;
    for (u32 i = 0; i < num_params; i++) {
        u32 ndim;
        if (pos + 4 > size) return FALSE;
        mem_copy(&ndim, buf + pos, 4); pos += 4;
        if (ndim != params[i]->value.ndim) return FALSE;

        for (u32 d = 0; d < ndim; d++) {
            u64 dim_val;
            if (pos + 8 > size) return FALSE;
            mem_copy(&dim_val, buf + pos, 8); pos += 8;
            if (dim_val != params[i]->value.shape[d]) return FALSE;
        }

        u64 nbytes = params[i]->value.numel * sizeof(f32);
        if (pos + nbytes > size) return FALSE;
        param_offsets[i] = pos;
        pos += nbytes;
        scalars_total += params[i]->value.numel;
    }

    u64 opt_offset = pos;
    if (has_opt) pos += 2 * scalars_total * sizeof(f32) + sizeof(u64);
    if (pos != size) return FALSE;

    /* 2. GECIS -- dosya tamamen tutarli; simdi kopyala. */
    for (u32 i = 0; i < num_params; i++) {
        mem_copy(params[i]->value.data, buf + param_offsets[i], params[i]->value.numel * sizeof(f32));
    }

    if (opt != NULL_PTR) {
        pos = opt_offset;
        for (u32 i = 0; i < num_params; i++) {
            u64 nbytes = opt->m[i].numel * sizeof(f32);
            mem_copy(opt->m[i].data, buf + pos, nbytes); pos += nbytes;
            mem_copy(opt->v[i].data, buf + pos, nbytes); pos += nbytes;
        }
        mem_copy(&opt->t, buf + pos, 8);
    }

    return TRUE;
}
