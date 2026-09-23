#!/data/data/com.termux/files/usr/bin/bash
# Instalação única dentro do Termux (Android 64-bit ARM).
set -euo pipefail
cd "$(dirname "$0")/.."

pkg update -y
# llama-cpp traz o llama-server; o Python só precisa da biblioteca padrão.
pkg install -y llama-cpp python curl

if ! llama-server --help 2>&1 | grep -q -- "--pooling"; then
  echo "erro: este llama-server não tem a opção --pooling; rode 'pkg upgrade llama-cpp'" >&2
  exit 1
fi

mkdir -p models
cat <<'EOF'

Instalação concluída. Agora copie para open-jev-mobile/models/ os dois
arquivos gerados pelo convert/build.py no PC:

    open-jev-2b-Q5_K_M.gguf   (~1,4 GB)
    head.json

O jeito mais fácil: coloque os dois na pasta Download do tablet e rode

    termux-setup-storage          # autoriza o acesso (só na primeira vez)
    cp ~/storage/downloads/open-jev-2b-Q5_K_M.gguf ~/storage/downloads/head.json models/

Depois é só iniciar:  ./termux/start.sh
EOF
