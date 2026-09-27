#!/usr/bin/env bash
# Tum programlari ve testleri derler (Linux, GCC + NVIDIA surucusu).
#
# main() icermeyen her .c dosyasi bir kez nesne dosyasina derlenir; main()
# iceren her program/test bu nesnelerle baglanir. Cikti programlar
# kaynaklarinin yanina yazilir (orn. training/train_faz4, tests/test_gpu_train).
#
# Kullanim (depo kokunden):  ./scripts/build_all.sh
# GPU cekirdeklerini de yeniden derlemek icin (nvcc gerekir):  PTX=1 ./scripts/build_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

CFLAGS=(-Wall -Wextra -std=c11 -O2 -pthread -I.)
OBJ_DIR=build/obj
mkdir -p "$OBJ_DIR"

if [[ "${PTX:-0}" == "1" ]]; then
    ARCH=${ARCH:-sm_89}   # L4/RTX 40xx: sm_89, A100: sm_80, H100: sm_90
    echo "PTX derleniyor ($ARCH)..."
    nvcc -ptx -arch="$ARCH" cuda/kernels.cu -o cuda/kernels.ptx
    nvcc -ptx -arch="$ARCH" cuda/train_kernels.cu -o cuda/train_kernels.ptx
fi

mapfile -t ALL_C < <(git ls-files '*.c')
LIB=(); MAINS=()
for f in "${ALL_C[@]}"; do
    if grep -q 'int main' "$f"; then MAINS+=("$f"); else LIB+=("$f"); fi
done

OBJS=()
for f in "${LIB[@]}"; do
    o="$OBJ_DIR/$(echo "$f" | tr / _).o"
    gcc "${CFLAGS[@]}" -c "$f" -o "$o"
    OBJS+=("$o")
done
echo "Kutuphane: ${#LIB[@]} dosya derlendi."

for f in "${MAINS[@]}"; do
    gcc "${CFLAGS[@]}" "$f" "${OBJS[@]}" -o "${f%.c}" -lcuda
done
echo "Programlar: ${#MAINS[@]} adet baglandi."
