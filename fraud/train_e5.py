#!/usr/bin/env python3
"""Train the scam detector used by the app: multilingual-e5-small + logistic regression.

The encoder (intfloat/multilingual-e5-small, "query: " prefix, mean pooling,
L2-normalized) runs on the phone through llama.cpp (fraud/convert_encoder.py;
embeddings match sentence-transformers to cosine 0.9999 in Q8_0). Only the
384 weights and the bias of the classifier are trained here.

Same data and splits as train.py (synthetic Brazilian + MOZ-Smishing 60% +
your real.jsonl), evaluated on the hand-written Brazilian set and the held-out
real MOZ SMS.

    python fraud/train_e5.py          # writes fraud/model/scam-head.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import HERE, jsonl, load_moz, metrics, split  # noqa: E402

MODEL_ID = "intfloat/multilingual-e5-small"
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
PREFIX = "query: "
C = 1.0


def main():
    synthetic = jsonl(HERE / "data" / "train.jsonl")
    moz_train, moz_test = split(load_moz(HERE / "data" / "cache" / "moz-smishing.csv"), 0.6, seed=7)
    real = jsonl(HERE / "data" / "real.jsonl")
    real_train, real_test = split(real, 0.6, seed=7) if len(real) >= 10 else (real, [])
    br_test = jsonl(HERE / "data" / "test.jsonl")
    train = synthetic + moz_train + real_train

    encoder = SentenceTransformer(MODEL_ID, revision=REVISION, device="cpu")

    def embed(rows):
        return encoder.encode([PREFIX + r["text"] for r in rows], batch_size=64,
                              normalize_embeddings=True, show_progress_bar=False)

    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    clf.fit(embed(train), np.array([r["label"] for r in train]))

    report = {"C": C, "train_rows": {"synthetic": len(synthetic), "moz": len(moz_train), "real": len(real_train)}}
    for name, rows in (("brasil_escrito_a_mao", br_test), ("moz_smishing_real", moz_test), ("real_seus", real_test)):
        if rows:
            p = clf.predict_proba(embed(rows))[:, 1]
            report[name] = metrics(np.array([r["label"] for r in rows]), p, 0.5)
            report[name]["errors"] = [(r["text"], r["label"], round(float(q), 3))
                                      for r, q in zip(rows, p) if (q >= 0.5) != bool(r["label"])][:15]

    out = HERE / "model"
    out.mkdir(parents=True, exist_ok=True)
    head = {
        "type": "e5_small_logreg",
        "encoder": {"model": MODEL_ID, "revision": REVISION, "prefix": PREFIX, "pooling": "mean", "normalize": True,
                    "gguf": "e5-small-Q8_0.gguf"},
        "threshold": 0.5,
        "bias": float(clf.intercept_[0]),
        "weight": [float(w) for w in clf.coef_[0]],
        "report": report,
    }
    # Expected probabilities for the device check ("Testar com 80 mensagens" in the app).
    expected = [{"text": r["text"], "label": r["label"], "probability": float(q)}
                for r, q in zip(br_test, clf.predict_proba(embed(br_test))[:, 1])]
    app_assets = HERE.parent / "android" / "app" / "src" / "main" / "assets" / "detector"
    app_assets.mkdir(parents=True, exist_ok=True)
    for folder in (out, app_assets):
        (folder / "scam-head.json").write_text(json.dumps(head, ensure_ascii=False) + "\n", encoding="utf-8")
        (folder / "scam-expected.json").write_text(json.dumps(expected, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "errors"} if isinstance(v, dict) else v
                      for k, v in report.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
