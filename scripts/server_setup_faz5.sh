#!/usr/bin/env bash
# Faz 5 sunucu kurulumu (Debian 13, G2, 1xL4). GH_TOKEN ortam degiskeni
# ile cagrilmali: GH_TOKEN=xxx ./server_setup_faz5.sh
set -euo pipefail

echo "=== Disk boyutu kontrolu / gerekirse buyutme ==="
df -h /
sudo growpart /dev/sda 1 || true
sudo resize2fs /dev/sda1 || true
df -h /

echo "=== Paketler ==="
sudo apt-get update -qq
sudo apt-get install -y -qq tmux bzip2 wget

if ! command -v gh >/dev/null 2>&1; then
    echo "=== gh CLI kuruluyor ==="
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq gh
fi

echo "${GH_TOKEN}" | gh auth login --with-token
gh auth setup-git

if [[ ! -d ~/proje ]]; then
    git clone https://github.com/bdenizgldkn/turkce-llm.git ~/proje
fi
cd ~/proje
git pull

echo "=== Butun programlar derleniyor ==="
./scripts/build_all.sh

mkdir -p checkpoints data/raw

echo "=== Vikipedi ham dokumu indiriliyor (veri-2026-09-26 release) ==="
if [[ ! -f data/raw/trwiki-latest-pages-articles.xml ]]; then
    gh release download veri-2026-09-26 -p "trwiki-latest-pages-articles.xml.bz2" -D data/raw --clobber
    bunzip2 -k data/raw/trwiki-latest-pages-articles.xml.bz2
fi

echo "=== Vikipedi korpusu yeniden uretiliyor (nbsp duzeltmesiyle) ==="
./tokenizer/build_wiki_corpus
echo "=== Vikipedi tokenize ediliyor (eos + orneklem duzeltmesiyle) ==="
./tokenizer/tokenize_corpus data/raw/wikipedia_corpus.txt data/raw/wikipedia_tokens.bin

echo "=== OpenSubtitles ham arsivi indiriliyor ==="
if [[ ! -f data/raw/opensubtitles_tr_v2024.txt ]]; then
    wget -q --show-progress -O data/raw/opensubtitles_tr_v2024.txt.gz \
        https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/tr.txt.gz
    gunzip data/raw/opensubtitles_tr_v2024.txt.gz
fi

echo "=== Altyazi korpusu yeniden uretiliyor (\\n\\n blok siniri duzeltmesiyle) ==="
./tokenizer/build_subtitle_corpus
echo "=== Altyazi tokenize ediliyor (eos + orneklem duzeltmesiyle) ==="
./tokenizer/tokenize_corpus data/raw/subtitle_corpus.txt data/raw/subtitle_tokens.bin

echo "=== HAZIR. checkpoints/lm_faz5_init.bin'in scp ile yuklenmesini bekliyor. ==="
ls -la data/raw/*.bin checkpoints/ 2>/dev/null
echo "Sonraki adim: tmux new -d -s faz5 './training/train_faz5 2>&1 | tee -a training/faz5_log.txt'"
