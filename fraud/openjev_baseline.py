#!/usr/bin/env python3
"""Score the hand-written Brazilian test set with Open-Jev-2B (GGUF via llama-server).

One noul question per message; its probability is compared with the labels
the same way as the small models in train.py.

    llama-server -m models/open-jev-2b-Q5_K_M.gguf --embeddings --pooling last -c 2048 -b 2048 -ub 2048 --port 8792
    python fraud/openjev_baseline.py --head models/head.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from jev.api import compile_request  # noqa: E402
from jev.metrics import softmax  # noqa: E402
from jev_mobile.scorer import LlamaServerScorer, load_head  # noqa: E402
from train import jsonl, metrics  # noqa: E402

QUESTION = {
    "golpe": {
        "type": "noul",
        "instructions": "Esta mensagem é uma tentativa de golpe ou fraude?",
        "criteria": {
            "true": "Tenta enganar a pessoa para obter dinheiro, senha, código, dados pessoais ou bancários, "
                    "ou para ela clicar em um link ou instalar algo.",
            "false": "Mensagem legítima: aviso real, conversa pessoal, cobrança ou promoção verdadeira, "
                     "sem tentar enganar.",
        },
    }
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, required=True)
    ap.add_argument("--llama-url", default="http://127.0.0.1:8792")
    args = ap.parse_args()
    head = load_head(args.head)
    scorer = LlamaServerScorer(head, url=args.llama_url)
    scorer.check()
    rows = jsonl(HERE / "data" / "test.jsonl")
    probs = []
    for i, r in enumerate(rows):
        records = compile_request("Mensagem recebida: " + r["text"], QUESTION)
        logits, _ = scorer.score(records)
        probs.append(softmax(logits[0], head["temperature"])[1])
        print(f"{i + 1}/{len(rows)} label={r['label']} p={probs[-1]:.3f}", flush=True)
    p = np.array(probs)
    y = np.array([r["label"] for r in rows])
    report = metrics(y, p, 0.5)
    report["errors"] = [(r["text"], r["label"], round(float(q), 3)) for r, q in zip(rows, p) if (q >= 0.5) != bool(r["label"])]
    print(json.dumps(report, ensure_ascii=False, indent=1))
    (HERE / "model" / "openjev-baseline.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
