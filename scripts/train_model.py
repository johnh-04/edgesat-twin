"""Train, quantise and evaluate the on-board scene classifier.

    python scripts/train_model.py            # ~2-4 min on a laptop CPU

Outputs
    edgesat/ai/model_int8.npz   int8 weights deployed on the simulated satellite
    edgesat/ai/model_card.json  accuracy (float vs int8), confusion matrix, size, MACs
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from edgesat.ai.cnn import FloatCNN, Int8CNN, macs, quantize, train  # noqa: E402
from edgesat.ai.scenes import CLASSES, sample_training_set  # noqa: E402


def confusion(y, p, n=len(CLASSES)):
    m = np.zeros((n, n), int)
    np.add.at(m, (y, p), 1)
    return m


def main(n_train=9000, n_val=2000, epochs=16, seed=7):
    t0 = time.time()
    print("generating synthetic scenes ...")
    X, y = sample_training_set(n_train, seed)
    Xv, yv = sample_training_set(n_val, seed + 1)
    print("class balance:", {c: int((y == i).sum()) for i, c in enumerate(CLASSES)})
    counts = np.bincount(y, minlength=len(CLASSES)).astype(float)
    cw = (counts.sum() / (len(CLASSES) * counts)) ** 0.5
    model = train(FloatCNN(seed), X, y, Xv, yv, epochs=epochs, seed=seed, class_weights=cw)

    pf = model.predict(Xv).argmax(1)
    q = quantize(model, X[:1500])
    qm = Int8CNN(q)
    pq = np.concatenate([qm.forward(Xv[i:i + 500]).argmax(1) for i in range(0, len(Xv), 500)])
    acc_f, acc_q = float((pf == yv).mean()), float((pq == yv).mean())
    cm = confusion(yv, pq)
    recall = (np.diag(cm) / np.maximum(cm.sum(1), 1)).tolist()
    precision = (np.diag(cm) / np.maximum(cm.sum(0), 1)).tolist()
    out = ROOT / "edgesat" / "ai"
    np.savez_compressed(out / "model_int8.npz", **q)
    card = {
        "classes": list(CLASSES), "architecture": "conv3x3(8)-pool-conv3x3(16)-pool-conv3x3(16)-[GAP|GMP]-dense(4)",
        "input": "32x32x4 (B,G,R,NIR) decimated tile, int8", "params": qm.n_params,
        "size_bytes_int8": qm.size_bytes, "macs_per_tile": macs(),
        "val_accuracy_float": acc_f, "val_accuracy_int8": acc_q,
        "recall_int8": dict(zip(CLASSES, recall)), "precision_int8": dict(zip(CLASSES, precision)),
        "confusion_int8": cm.tolist(), "n_train": n_train, "n_val": n_val, "epochs": epochs,
        "train_time_s": round(time.time() - t0, 1),
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2))
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
