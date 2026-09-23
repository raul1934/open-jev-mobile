# App test in the Android emulator (2026-09-23)

**Setup:** release APK (`com.openjev.mobile` 0.1.0), `open-jev-2b-Q5_K_M.gguf` + `head.json`
copied into the app folder with `adb push` (the app checked the SHA-256 on startup),
default settings (4 threads, context 2048).

**Emulator:** Android 15 (API 35) `google_apis` x86_64, 4 vCPUs on a Ryzen 7 7735HS host,
2.5 GB of RAM seen by the system (less than a 4 GB tablet). The `libggml-cpu-haswell.so`
backend was selected. The emulator emulates the CPU, so the times below **are not** the times
on a real phone or tablet. No physical tablet has been tested yet.

Each prompt ran with `adb shell am start -n com.openjev.mobile/.MainActivity --es run <file>`.

## Compared with the original model

"Original" = the released Open-Jev-2B checkpoint in PyTorch (bf16), loaded exactly as upstream
does (`jev.model.DecisionModel.load`), on the same prompts. "PC" = the same Q5_K_M GGUF on the
Windows `llama-server` (the `jev_mobile`/Termux path). Differences are the largest change in
probability for a question, in percentage points.

| Prompt | Time (emulator) | App vs original | PC vs original | App vs PC |
|---|---|---|---|---|
| 01-en-support-routing | 52.4 s (first run, includes loading pages) | 0.01 | 0.33 | 0.33 |
| 02-pt-atendimento | 66.7 s | 10.53 | 8.42 | 2.19 |
| 03-pt-avaliacao | 30.1 s | 14.19 | 15.55 | 6.11 |
| 04-pt-golpe-sms | 18.8 s | 1.67 | 1.60 | 0.24 |
| 05-en-rag-filter | 28.0 s | 7.91 | 10.68 | 2.78 |
| 06-pt-intencao-assistente | 15.0 s | 0.48 | 0.85 | 0.37 |
| **All 15 questions** | | **max 14.19, mean 3.16** | max 15.55, mean 3.40 | |

- Top answer equal to the original model: **15/15** questions.
- The app is as close to the original as `llama-server` on the PC. The app-vs-PC difference comes
  from the CPU kernels of each build, not from the app code:
  running prompt 03 with a build configured exactly like `llama-server --embeddings --pooling last`
  (all tokens as outputs, pooling LAST) gave **bit-for-bit identical** probabilities to the
  final app, but took 106 s instead of 30 s.
- Most of the difference from the original comes from the Q5_K_M quantization, and it is largest
  when two options are nearly tied (prompt 03: "mixed" 52% vs "positive" 41% in the original).

## Memory

- Compute buffer: 200 MB (the app computes the model head for the last token only; the
  `llama-server` configuration reserves ~2 GB of address space for logits of every token).
- App PSS during the 6 prompts: ~1.6 GB (of which ~0.13 GB in swap), including the model's
  mmap-ed pages.
- No crash or kill by `lmkd` in the 6 runs.
