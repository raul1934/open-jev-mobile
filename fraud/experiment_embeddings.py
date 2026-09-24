#!/usr/bin/env python3
"""Compare sentence-embedding encoders + logistic regression with the char n-gram baseline.

Same data and splits as train.py. Prints precision/recall/F1/AUC on the
hand-written Brazilian test set and on the held-out real MOZ-Smishing SMS.

    python fraud/experiment_embeddings.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import HERE, jsonl, load_moz, metrics, split  # noqa: E402

ENCODERS = {
    # name: (model id, text prefix the model was trained with)
    "e5-small": ("intfloat/multilingual-e5-small", "query: "),
    "minilm-l12": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
}


def main():
    synthetic = jsonl(HERE / "data" / "train.jsonl")
    moz_train, moz_test = split(load_moz(HERE / "data" / "cache" / "moz-smishing.csv"), 0.6, seed=7)
    br_test = jsonl(HERE / "data" / "test.jsonl")
    train = synthetic + moz_train
    y = np.array([r["label"] for r in train])
    results = {}
    for name, (model_id, prefix) in ENCODERS.items():
        enc = SentenceTransformer(model_id, device="cpu")
        embed = lambda rows: enc.encode([prefix + r["text"] for r in rows], batch_size=64,
                                        normalize_embeddings=True, show_progress_bar=False)
        X = embed(train)
        for c in (1, 4, 16):
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(X, y)
            row = {}
            for test_name, rows in (("brasil", br_test), ("moz", moz_test)):
                p = clf.predict_proba(embed(rows))[:, 1]
                m = metrics(np.array([r["label"] for r in rows]), p, 0.5)
                row[test_name] = {k: m[k] for k in ("precision", "recall", "f1", "auc", "false_positives", "false_negatives")}
            results[f"{name} C={c}"] = row
            print(name, "C=", c, json.dumps(row), flush=True)
    (HERE / "model" / "experiment-embeddings.json").write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
