#!/data/data/com.termux/files/usr/bin/bash
# Sobe o llama-server (backbone GGUF) e a API do Open-Jev em cima dele.
#
# Variáveis opcionais:
#   JEV_GGUF     modelo (padrão models/open-jev-2b-Q5_K_M.gguf)
#   JEV_CTX      maior prompt aceito, em tokens (padrão 2048; o Open-Jev aceita até 4096)
#   JEV_THREADS  threads da CPU (padrão 4 = os núcleos rápidos da maioria dos chips)
#   JEV_HOST     0.0.0.0 para aceitar pedidos de outros aparelhos na mesma rede Wi-Fi
#   JEV_PORT     porta da API do Open-Jev (padrão 8791)
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${JEV_GGUF:-models/open-jev-2b-Q5_K_M.gguf}
CTX=${JEV_CTX:-2048}
THREADS=${JEV_THREADS:-4}
HOST=${JEV_HOST:-127.0.0.1}
PORT=${JEV_PORT:-8791}
LLAMA_PORT=${JEV_LLAMA_PORT:-8792}

for file in "$MODEL" models/head.json; do
  if [ ! -f "$file" ]; then
    echo "erro: falta $file — veja o passo 3 do README" >&2
    exit 1
  fi
done

# Evita que o Android pause o Termux com a tela desligada.
command -v termux-wake-lock >/dev/null && termux-wake-lock || true

# --embeddings --pooling last: devolve o estado oculto final do último token,
# que é a entrada da cabeça de decisão. Um prompt inteiro precisa caber em um
# micro-lote, por isso -b e -ub são iguais ao contexto.
llama-server -m "$MODEL" --embeddings --pooling last \
  -c "$CTX" -b "$CTX" -ub "$CTX" -np 1 -t "$THREADS" \
  --host 127.0.0.1 --port "$LLAMA_PORT" > models/llama-server.log 2>&1 &
LLAMA_PID=$!
trap 'kill $LLAMA_PID 2>/dev/null || true' EXIT

echo "carregando o modelo..."
until curl -sf "http://127.0.0.1:$LLAMA_PORT/health" >/dev/null; do
  if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
    echo "erro: o llama-server parou. Últimas linhas do log:" >&2
    tail -n 20 models/llama-server.log >&2
    exit 1
  fi
  sleep 1
done

python -m jev_mobile --llama-url "http://127.0.0.1:$LLAMA_PORT" serve --host "$HOST" --port "$PORT"
