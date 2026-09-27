#!/usr/bin/env bash
# GitHub Releases'taki buyuk dosyalari (veri, checkpoint, log) indirip
# depodaki dogru klasorlere koyar ve SHA256 ile dogrular. `gh` (GitHub CLI,
# oturum acilmis) gerekir; depo gizli oldugu icin duz wget/curl calismaz.
#
# Kullanim (depo kokunden):
#   ./scripts/indir.sh            -> calismak icin gerekenler (Faz 3 + Faz 4 modelleri, token dosyalari)
#   ./scripts/indir.sh hepsi      -> ek olarak ham metinler, eski checkpoint'ler ve loglar
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw checkpoints training

MOD=${1:-temel}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

indir() {  # indir <release-etiketi> <klasor> <dosya...>
    local tag=$1 dir=$2; shift 2
    mkdir -p "$TMP/$tag"
    [[ -f "$TMP/$tag/SHA256SUMS" ]] || gh release download "$tag" -p SHA256SUMS -D "$TMP/$tag"
    for f in "$@"; do
        echo "[$tag] $f -> $dir/"
        gh release download "$tag" -p "$f" -D "$dir" --clobber
        awk -v f="$f" '$2 == f' "$TMP/$tag/SHA256SUMS" | (cd "$dir" && sha256sum -c -)
    done
}

# Faz 3 (Vikipedi) -- faz3-final
indir faz3-final checkpoints lm_wiki_faz3_best.bin lm_wiki_faz3_best_val.bin
indir faz3-final data/raw wikipedia_tokens.bin
# Faz 4 (altyazi, adim 1500'de durduruldu) -- faz4-adim1500
indir faz4-adim1500 checkpoints lm_faz4_best.bin lm_faz4_best_val.bin lm_faz4_latest.bin
indir faz4-adim1500 data/raw subtitle_tokens.bin

if [[ "$MOD" == "hepsi" ]]; then
    indir faz3-final checkpoints lm_wiki_faz3_final.bin lm_wiki_faz3_best_ctx128.bin lm_wiki_faz3_best_ctx128_val.bin lm_wiki_faz2_latest.bin
    indir faz3-final data/raw wikipedia_corpus.txt tokenize_log.txt
    indir faz3-final training faz3_log.txt train_log.txt train_log_28w_once.txt train_log_cpu_28w.txt
    indir faz4-adim1500 data/raw subtitle_corpus.txt subtitle_build_log.txt subtitle_tokenize_log.txt
    indir faz4-adim1500 training faz4_log.txt
fi
echo "Tamam."
