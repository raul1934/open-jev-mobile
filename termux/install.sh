#!/data/data/com.termux/files/usr/bin/bash
# Instalação única dentro do Termux (Android 64-bit ARM).
# Baixa o modelo do release model-v1 do GitHub; para usar arquivos convertidos
# por você (convert/build.py), copie-os para models/ antes de rodar este script.
set -euo pipefail
cd "$(dirname "$0")/.."

RELEASE=https://github.com/raul1934/open-jev-mobile/releases/download/model-v1
FILES=(
  "head.json 8b9282bf0f5fcae0e55f91778c067bbc2e20f1b3bba6dc82dd5179644297a5b9"
  "open-jev-2b-Q5_K_M.gguf 5140cc4c3ef79c4097880694999c60fffeda98289d72ca7660115abf5795aa66"
)

pkg update -y
# llama-cpp traz o llama-server; o Python só precisa da biblioteca padrão.
pkg install -y llama-cpp python curl

if ! llama-server --help 2>&1 | grep -q -- "--pooling"; then
  echo "erro: este llama-server não tem a opção --pooling; rode 'pkg upgrade llama-cpp'" >&2
  exit 1
fi

mkdir -p models
for entry in "${FILES[@]}"; do
  read -r name sha <<<"$entry"
  if [ -f "models/$name" ] && echo "$sha  models/$name" | sha256sum -c --status; then
    echo "ok: models/$name"
    continue
  fi
  echo "baixando $name (o modelo tem ~1,4 GB; prefira Wi-Fi)..."
  # -C - continua um download interrompido.
  curl -fL -C - --retry 5 -o "models/$name" "$RELEASE/$name"
  if ! echo "$sha  models/$name" | sha256sum -c --status; then
    echo "erro: models/$name está corrompido (SHA-256 diferente). Apague e rode de novo." >&2
    exit 1
  fi
done

echo
echo "Instalação concluída. Para iniciar:  ./termux/start.sh"
