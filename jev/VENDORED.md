# Vendored Open-Jev modules

These files are unmodified copies from
[Zefan-Cai/Open-Jev](https://github.com/Zefan-Cai/Open-Jev) at commit
`3308a15ccd7eea1df7a37d6ddc39b023b801ba16` (MIT License, see `LICENSE` here):

- `__init__.py`
- `api.py` — request compilation, candidate prompts, typed response formatting
- `metrics.py` — softmax/calibration helpers (standard library only)
- `serving.py` — `Predictor` and candidate batching
- `server.py` — the local HTTP service (`/v1/inference`, `/health`, `/v1/models`)

They only use the Python standard library at import time, so they run in
Termux without PyTorch. Keeping them byte-identical means prompts, batching,
calibration and the response format are exactly the upstream ones; only the
scorer (the part that runs the network) is replaced by `jev_mobile`.

To check they are still identical:

```bash
for f in __init__ api metrics serving server; do
  curl -s https://raw.githubusercontent.com/Zefan-Cai/Open-Jev/3308a15ccd7eea1df7a37d6ddc39b023b801ba16/jev/$f.py | cmp - jev/$f.py && echo "$f ok"
done
```
