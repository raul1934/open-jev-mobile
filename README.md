# open-jev-mobile

Roda o [Open-Jev-2B](https://github.com/Zefan-Cai/Open-Jev) **localmente num tablet Android com 4 GB de RAM**, sem GPU e sem internet, usando um modelo quantizado de ~1,4 GB (`Q5_K_M`, cerca de 5 bits por peso; também há `Q4_K_M`, de 4 bits).

O Open-Jev original precisa de PyTorch e, na prática, de uma GPU NVIDIA. Este projeto:

1. **uma vez, no PC:** junta o adaptador LoRA do Open-Jev-2B com os pesos do Qwen3.5-2B, converte para GGUF e quantiza (padrão `Q5_K_M`). O resultado está publicado no release [`model-v1`](https://github.com/raul1934/open-jev-mobile/releases/tag/model-v1), então você não precisa refazer isso;
2. **no aparelho:** roda esse GGUF com o llama.cpp e aplica a cabeça de decisão treinada do Open-Jev.

As perguntas são as mesmas do Open-Jev (`choice`, `score` e `noul`), com a mesma temperatura de calibração e o mesmo formato de resposta.

Há dois jeitos de usar no aparelho:

- **App Android (APK)** — o mais simples. Instale, toque em *Baixar modelo* e use. Veja abaixo.
- **Termux** — dá a **mesma API HTTP do Open-Jev** (`POST /v1/inference`) no aparelho, para outros apps chamarem. Usa cópias sem alteração dos módulos do Open-Jev ([jev/VENDORED.md](jev/VENDORED.md)). Veja as seções "Termux" abaixo.

## App Android (APK)

1. Baixe o `open-jev-mobile.apk` na página de [Releases](https://github.com/raul1934/open-jev-mobile/releases) e instale. O Android vai pedir para permitir a instalação de apps de fora da Play Store.
2. Abra o app e toque em **Baixar modelo**. O app baixa `open-jev-2b-Q5_K_M.gguf` (1,4 GB) e `head.json` do release [`model-v1`](https://github.com/raul1934/open-jev-mobile/releases/tag/model-v1) e confere o SHA-256 dos dois. Prefira Wi-Fi. O download continua mesmo com o app em segundo plano.
   - Sem internet no aparelho? Copie os dois arquivos para a pasta Download e use **Importar arquivos**.
3. Escolha um dos **prompts de teste** (em português e inglês, os mesmos de [`examples/`](examples/)), ou cole o seu próprio pedido em JSON no formato do Open-Jev, e toque em **Rodar**.

Depois do download, tudo roda no aparelho, sem internet. Em **Ajustes** dá para mudar o número de threads e o tamanho máximo do prompt.

Por dentro, o app ([`android/`](android/)) compila o mesmo llama.cpp `v0.4.1` usado no Termux e na verificação, e reimplementa em Kotlin a montagem dos prompts, a cabeça de decisão, a calibração e o formato da resposta. Os testes em [`ParityTest.kt`](android/app/src/test/java/com/openjev/mobile/jev/ParityTest.kt) comparam esse código com o Open-Jev original em Python ([`android/tools/make_fixtures.py`](android/tools/make_fixtures.py)): prompts idênticos byte a byte, probabilidades iguais até 1e-12 e as mesmas mensagens de erro.

**Testado num emulador Android** (API 35, x86_64, 4 núcleos, 2,5 GB de RAM, ou seja, menos que um tablet de 4 GB) com os 6 prompts de teste. Detalhes em [`android/DEVICE_TEST.md`](android/DEVICE_TEST.md):

- a resposta principal foi **igual à do modelo original (PyTorch) em 15 de 15 perguntas**;
- as probabilidades ficaram tão perto do original quanto as do `llama-server` no PC (média de 3,2 pontos de diferença contra 3,4 no PC). Quase toda a diferença vem da quantização, não do app;
- memória do app durante o uso: ~1,6 GB (PSS);
- de 15 a 67 s por prompt no emulador, que emula a CPU. Num aparelho de verdade, o tempo depende do chip. Ainda não medi num tablet físico.

### Compilar o APK

Precisa do Android SDK com NDK `27.1.12297006` e CMake `3.22.1` (o Android Studio instala) e de um JDK 17 ou mais novo:

```bash
cd android
./gradlew :app:testReleaseUnitTest   # testes de paridade com o Python
./gradlew :app:assembleRelease       # APK em app/build/outputs/apk/release/
```

A compilação baixa o código do llama.cpp `v0.4.1` (hash fixado em [`CMakeLists.txt`](android/app/src/main/cpp/CMakeLists.txt)). Use `-PllamaSrc=/caminho/llama.cpp` para usar uma cópia local. No Windows, se der erro de caminho longo, use `-PcxxDir=C:/cxx`. O APK de release é assinado com a chave de debug local, o que basta para instalar manualmente. Para publicar na Play Store, configure uma chave própria.

## Fidelidade: quanto a quantização muda o resultado

Medido com [`convert/verify.py`](convert/verify.py), comparando com o modelo original em PyTorch (bf16), exatamente como o Open-Jev o carrega, em 12 arquivos de exemplo do Open-Jev (25 perguntas, 49 candidatos). A diferença é a maior mudança de probabilidade de uma pergunta, em pontos percentuais:

| Versão | Arquivo | Pior caso | Média | Mesma resposta que o original |
|---|---|---|---|---|
| F16 (sem quantizar) | 3,8 GB | 1,1 | 0,3 | 25/25 |
| Q8_0 | 2,0 GB | 4,1 | 0,7 | 25/25 |
| Q6_K | 1,6 GB | 5,9 | 1,2 | 25/25 |
| **Q5_K_M (padrão)** | **1,4 GB** | **7,8** | **2,0** | **25/25** |
| Q4_K_M (4 bits) | 1,3 GB | 23,6 | 4,1 | 25/25 |

O F16 praticamente coincide com o original, o que mostra que a conversão em si está correta (os IDs de token também foram idênticos em todos os prompts). O que muda nas outras linhas é só o efeito da quantização.

**Por que o padrão é Q5_K_M e não 4 bits:** o `Q4_K_M` economiza só 130 MB de arquivo, mas erra três vezes mais e, no PC medido, **usou mais RAM** (tabela abaixo): o llama.cpp cria uma cópia reorganizada dos pesos Q4 para acelerar o cálculo, e isso não acontece com o Q5. Para usar mesmo assim: `python convert/build.py --types Q4_K_M` e `JEV_GGUF=models/open-jev-2b-Q4_K_M.gguf ./termux/start.sh`.

Relatório completo: [convert/verify-report.json](convert/verify-report.json).

**Nos 6 prompts de teste do app (4 em português)**, a diferença do Q5_K_M foi maior: no pior caso, 15,5 pontos (média de 3,4), mas a resposta principal continuou igual em 15 de 15 perguntas. As maiores diferenças aparecem quando duas opções estão quase empatadas (ex.: sentimento "misto" 52% contra "positivo" 41%). Veja [`android/DEVICE_TEST.md`](android/DEVICE_TEST.md). O próprio Open-Jev avisa que quantizar muda os pesos em cima dos quais a cabeça de decisão foi treinada, então os números de qualidade publicados por eles **não valem automaticamente** para esta versão. 25 perguntas são um teste de sanidade, não um benchmark: se o seu uso for sensível a probabilidades exatas, rode o `verify.py` com os seus próprios exemplos.

## Memória

Pico de RAM do `llama-server` durante uma requisição (o exemplo `examples/01-en-support-routing.json`, com 7 candidatos), medido num PC (Ryzen 7 7735HS, Windows, 4 threads):

| Versão | `JEV_CTX=2048` | `JEV_CTX=4096` | Tempo da requisição |
|---|---|---|---|
| **Q5_K_M** | **1,7 GB** | **1,8 GB** | 18 s |
| Q6_K | 1,8 GB | 1,9 GB | 18 s |
| Q4_K_M | 2,2 GB | 2,3 GB | 13 s |

O processo Python usa mais uns 30 MB. Num aparelho de 4 GB, o Android costuma deixar livres de 1,5 a 2,5 GB, então feche outros apps antes. Os pesos são lidos do arquivo via *mmap*, e o sistema pode descartar e reler essas páginas quando falta memória: fica mais lento, mas não trava.

**Ainda não medido num tablet de verdade.** Um chip de celular de entrada é várias vezes mais lento que esse PC, então espere de 1 a 2 minutos por uma requisição como a do exemplo. O tempo cresce com o número de candidatos e com o tamanho do `state`.

## Requisitos

- **Tablet ou celular Android 64-bit** (ARM64) com 4 GB de RAM ou mais e ~2 GB livres de armazenamento.
- Para o jeito Termux: **Termux instalado pelo [F-Droid](https://f-droid.org/packages/com.termux/)**. A versão da Play Store está desatualizada e não recebe pacotes novos.
- Só para converter o modelo você mesmo (opcional): um PC com Python 3.10+, ~15 GB livres em disco e 8 GB de RAM. Não precisa de GPU.

## Termux — Passo 1: instalar

```bash
pkg install -y git
git clone https://github.com/raul1934/open-jev-mobile.git
cd open-jev-mobile
./termux/install.sh
```

O `install.sh` instala o `llama-cpp` e o Python do Termux, baixa o modelo do release [`model-v1`](https://github.com/raul1934/open-jev-mobile/releases/tag/model-v1) e confere os SHA-256. Se você converteu o modelo no PC (seção abaixo), copie os dois arquivos para `models/` antes de rodar o script: com os hashes certos, ele não baixa nada.

## Termux — Passo 2: usar

```bash
./termux/start.sh
```

Com o servidor no ar, abra **outra sessão** do Termux (deslize da borda esquerda e toque em *New session*):

```bash
cd open-jev-mobile
python -m jev_mobile predict examples/01-en-support-routing.json
```

Ou chame a API HTTP, igual ao Open-Jev:

```bash
curl -s http://127.0.0.1:8791/v1/inference -H 'Content-Type: application/json' -d @examples/01-en-support-routing.json
```

Para chamar a partir de outro aparelho na mesma rede Wi-Fi, use `JEV_HOST=0.0.0.0 ./termux/start.sh` e troque `127.0.0.1` pelo IP do tablet. Não faça isso em redes públicas: a API não tem senha.

### Ajustes (variáveis de ambiente do `start.sh`)

| Variável | Padrão | Para quê |
|---|---|---|
| `JEV_CTX` | `2048` | maior prompt aceito, em tokens. Suba para `4096` (limite do Open-Jev) se tiver memória sobrando |
| `JEV_THREADS` | `4` | threads da CPU. 4 costuma ser o número de núcleos rápidos |
| `JEV_GGUF` | `models/open-jev-2b-Q5_K_M.gguf` | para usar outro arquivo, por exemplo o `Q4_K_M` |
| `JEV_HOST` / `JEV_PORT` | `127.0.0.1` / `8791` | endereço da API |

## Converter o modelo você mesmo (opcional, no PC)

O release `model-v1` já tem o modelo pronto. Esta seção só serve para reproduzir a conversão ou gerar outra quantização.

```bash
git clone https://github.com/raul1934/open-jev-mobile.git
cd open-jev-mobile
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r convert/requirements.txt
```

Você também precisa do `llama-quantize`:

- **Windows:** baixe `llama-b<número>-bin-win-cpu-x64.zip` nos [releases do llama.cpp](https://github.com/ggml-org/llama.cpp/releases) e descompacte;
- **Mac/Linux:** `brew install llama.cpp`.

Depois rode (troque o caminho do `--quantize`):

```bash
python convert/build.py --quantize C:/caminho/llama-quantize.exe
```

O script baixa as revisões fixadas do Open-Jev-2B e do Qwen3.5-2B, confere os hashes do pacote, junta o LoRA, converte e quantiza. No fim, `models/` tem:

- `open-jev-2b-Q5_K_M.gguf` (~1,4 GB)
- `head.json` (cabeça de decisão, temperatura de calibração e formato do prompt)

Opcional, para medir a fidelidade no seu PC:

```bash
git clone https://github.com/Zefan-Cai/Open-Jev.git ../Open-Jev
python convert/verify.py --open-jev ../Open-Jev --llama-server C:/caminho/llama-server.exe models/open-jev-2b-Q5_K_M.gguf
```

## Problemas comuns

- **O Termux fecha sozinho ou o servidor morre:** no Android 12 ou mais novo, o sistema mata processos em segundo plano. Mantenha o Termux aberto, desative a otimização de bateria para ele e, se precisar, siga as [instruções do Termux sobre o *phantom process killer*](https://github.com/termux/termux-app/issues/2366).
- **`Input length ... exceeds max_length`:** o prompt é maior que `JEV_CTX`. Aumente o valor ou encurte o `state`.
- **Lento:** cada candidato é um prompt separado, então 10 opções custam 10 vezes mais que 1. Diminua o número de candidatos ou o tamanho do `state`.

## Como funciona

O Open-Jev dá uma nota a cada candidato com `head(h)`, em que `h` é o estado oculto final (depois da última normalização) do último token do prompt, e `head` é uma camada linear 2048 → 1. O `llama-server` com `--embeddings --pooling last` devolve exatamente esse `h`; `embd_normalize: -1` desliga a normalização L2. O [`jev_mobile/scorer.py`](jev_mobile/scorer.py) faz o produto escalar com a cabeça e entrega as notas ao `Predictor` do próprio Open-Jev, que aplica a temperatura e monta a resposta.

A tokenização é feita pelo llama.cpp. O `verify.py` confere que os IDs de token são idênticos aos do tokenizer do Hugging Face.

## Não coberto

- **iPhone/iOS:** não dá para rodar Python com llama-server no iOS. Seria preciso um app nativo em Swift.
- **Open-Jev-9B e 27B:** não cabem em 4 GB.
- **Prefix caching** do Open-Jev (`--prefix-cache`): não implementado. Cada candidato é processado do zero.

## Licenças

Código deste repositório: MIT ([LICENSE](LICENSE)). Os arquivos em `jev/` vêm do Open-Jev (MIT, [jev/LICENSE](jev/LICENSE)). Os pesos gerados derivam do Qwen3.5-2B (Apache-2.0) e do Open-Jev-2B; confira as licenças nos repositórios originais antes de redistribuir o GGUF.
