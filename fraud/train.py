#!/usr/bin/env python3
"""Train the scam-message detector: character n-grams + logistic regression.

Features: scikit-learn HashingVectorizer (analyzer="char_wb", 2-5 grams,
2^18 buckets, lowercase, accents stripped, L2 norm). The hashing trick needs no
vocabulary file, so the app only needs the weight vector and reimplements the
featurizer in Kotlin (checked against fixtures from this script).

Data:
  fraud/data/train.jsonl   synthetic Brazilian messages (fraud/data/generate.py)
  MOZ-Smishing             2,561 real Portuguese SMS from Mozambique (downloaded;
                           OpenRAIL-M license), split 60/40 into train/test
  fraud/data/real.jsonl    optional: real messages you labeled ({"text", "label"})
  fraud/data/test.jsonl    hand-written Brazilian evaluation set (never trained on)

    python fraud/train.py
"""
import argparse
import csv
import json
import random
import struct
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

HERE = Path(__file__).resolve().parent
MOZ_URL = "https://huggingface.co/datasets/MOZNLP/MOZ-Smishing/resolve/main/test.csv"
N_FEATURES = 2 ** 18
NGRAMS = (2, 5)


def vectorizer():
    return HashingVectorizer(analyzer="char_wb", ngram_range=NGRAMS, n_features=N_FEATURES,
                             alternate_sign=False, lowercase=True, strip_accents="unicode", norm="l2")


def jsonl(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_moz(cache):
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MOZ_URL, cache)
    rows = list(csv.DictReader(cache.open(encoding="utf-8")))
    return [{"text": r["text"], "label": int(r["label"] == "Smishing"), "source": "moz"} for r in rows if r["text"].strip()]


def split(rows, fraction, seed):
    """Stratified split so both parts keep the scam/legitimate ratio."""
    rng = random.Random(seed)
    out_a, out_b = [], []
    for label in (0, 1):
        group = [r for r in rows if r["label"] == label]
        rng.shuffle(group)
        cut = int(len(group) * fraction)
        out_a += group[:cut]
        out_b += group[cut:]
    return out_a, out_b


def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": int(len(y)), "scams": int(y.sum()), "precision": round(precision_score(y, pred, zero_division=0), 4),
            "recall": round(recall_score(y, pred, zero_division=0), 4), "f1": round(f1_score(y, pred, zero_division=0), 4),
            "auc": round(roc_auc_score(y, p), 4) if 0 < y.sum() < len(y) else None,
            "false_positives": int(fp), "false_negatives": int(fn)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moz-cache", type=Path, default=HERE / "data" / "cache" / "moz-smishing.csv")
    ap.add_argument("--out", type=Path, default=HERE / "model")
    args = ap.parse_args()

    synthetic = jsonl(HERE / "data" / "train.jsonl")
    moz_train, moz_test = split(load_moz(args.moz_cache), 0.6, seed=7)
    real = jsonl(HERE / "data" / "real.jsonl")
    real_train, real_test = split(real, 0.6, seed=7) if len(real) >= 10 else (real, [])
    br_test = jsonl(HERE / "data" / "test.jsonl")
    train = synthetic + moz_train + real_train

    vec = vectorizer()
    X = vec.transform([r["text"] for r in train])
    y = np.array([r["label"] for r in train])

    # Pick the regularization on a validation slice of the training data.
    fit_rows, val_rows = split(train, 0.8, seed=11)
    Xf, yf = vec.transform([r["text"] for r in fit_rows]), np.array([r["label"] for r in fit_rows])
    Xv, yv = vec.transform([r["text"] for r in val_rows]), np.array([r["label"] for r in val_rows])
    best_c, best_f1 = None, -1
    for c in (1, 4, 16, 64):
        m = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(Xf, yf)
        f1 = f1_score(yv, m.predict(Xv))
        print(f"C={c}: validation F1 {f1:.4f}")
        if f1 > best_f1:
            best_c, best_f1 = c, f1

    model = LogisticRegression(C=best_c, class_weight="balanced", max_iter=5000).fit(X, y)
    weights = model.coef_[0].astype(np.float32)
    bias = float(model.intercept_[0])

    def probs(rows):
        return model.predict_proba(vec.transform([r["text"] for r in rows]))[:, 1]

    report = {"C": best_c, "train_rows": {"synthetic": len(synthetic), "moz": len(moz_train), "real": len(real_train)}}
    for name, rows in (("brasil_escrito_a_mao", br_test), ("moz_smishing_real", moz_test), ("real_seus", real_test)):
        if rows:
            p = probs(rows)
            report[name] = metrics(np.array([r["label"] for r in rows]), p, 0.5)
            wrong = [(r["text"], r["label"], round(float(q), 3)) for r, q in zip(rows, p) if (q >= 0.5) != bool(r["label"])]
            report[name]["errors"] = wrong[:15]

    # Weights: sparse (index, value) pairs, little-endian; the app rebuilds the 2^18 vector.
    args.out.mkdir(parents=True, exist_ok=True)
    nz = np.flatnonzero(weights)
    with (args.out / "weights.bin").open("wb") as f:
        f.write(struct.pack("<4sIIf", b"JEVF", N_FEATURES, len(nz), bias))
        f.write(np.stack([nz.astype(np.uint32).view(np.float32), weights[nz]], axis=1).astype("<f4").tobytes())
    (args.out / "model.json").write_text(json.dumps({
        "type": "char_wb_hashing_logreg", "n_features": N_FEATURES, "ngram_range": list(NGRAMS),
        "lowercase": True, "strip_accents": "unicode", "norm": "l2", "threshold": 0.5,
        "weights_file": "weights.bin", "nonzero_weights": int(len(nz)), "report": report,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Parity fixture for the Kotlin featurizer/scorer.
    samples = [r["text"] for r in br_test[:20]] + [
        "  espaços   múltiplos\te\ttabs  ", "EMOJI 🚨⚠️ e acentuação ÁÉÍÕÇ", "a", "", "NBSP aqui", "link: https://x.y/z?a=1"]
    fixture = [{"text": t, "probability": float(model.predict_proba(vec.transform([t]))[0, 1]),
                "features": {str(k): float(v) for k, v in zip(vec.transform([t]).indices, vec.transform([t]).data)}}
               for t in samples]
    (args.out / "parity.json").write_text(json.dumps(fixture, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items()}, ensure_ascii=False, indent=1))
    print(f"weights: {len(nz)} nonzero of {N_FEATURES} -> {(args.out / 'weights.bin').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
