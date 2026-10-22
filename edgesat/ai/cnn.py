"""TinyML scene classifier: a 3-layer CNN in pure NumPy, with post-training int8 quantisation.

Architecture (input 32x32x4, NHWC)
    conv3x3  4->8   ReLU  maxpool2   -> 16x16x8
    conv3x3  8->16  ReLU  maxpool2   -> 8x8x16
    conv3x3 16->16  ReLU             -> 8x8x16
    [global-average-pool | global-max-pool] -> 32
    dense 32->4 (logits)

The float model is trained here with Adam. Deployment uses int8 weights
(per-output-channel scales), int8 activations (per-tensor scales from a
calibration set) and int32 accumulators, i.e. the arithmetic of CMSIS-NN /
TFLite-Micro kernels. Requantisation multipliers are kept in float for
readability; on a microcontroller they become a fixed-point multiply + shift.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

CONV_SPECS = [(4, 8), (8, 16), (16, 16)]
POOL_AFTER = [True, True, False]
N_CLASSES = 4


def im2col(x: np.ndarray) -> np.ndarray:
    """(N,H,W,C) -> (N*H*W, C*9) for a 3x3 'same' convolution."""
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    win = sliding_window_view(xp, (3, 3), axis=(1, 2))   # N,H,W,C,3,3
    n, h, w, c = x.shape
    return win.reshape(n * h * w, c * 9)


def col2im(dcol: np.ndarray, shape) -> np.ndarray:
    n, h, w, c = shape
    d = dcol.reshape(n, h, w, c, 3, 3)
    dxp = np.zeros((n, h + 2, w + 2, c), dcol.dtype)
    for i in range(3):
        for j in range(3):
            dxp[:, i:i + h, j:j + w, :] += d[..., i, j]
    return dxp[:, 1:-1, 1:-1, :]


def maxpool2(x):
    n, h, w, c = x.shape
    xr = x.reshape(n, h // 2, 2, w // 2, 2, c)
    out = xr.max(axis=(2, 4))
    return out, xr


def maxpool2_back(dout, xr, out):
    mask = (xr == out[:, :, None, :, None, :])
    mask = mask / np.maximum(mask.sum(axis=(2, 4), keepdims=True), 1)
    d = mask * dout[:, :, None, :, None, :]
    n, h2, _, w2, _, c = xr.shape
    return d.reshape(n, h2 * 2, w2 * 2, c)


def macs() -> int:
    total, h = 0, 32
    for (cin, cout), pool in zip(CONV_SPECS, POOL_AFTER):
        total += h * h * cout * cin * 9
        if pool:
            h //= 2
    total += 2 * CONV_SPECS[-1][1] * N_CLASSES
    return total


class FloatCNN:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.p = {}
        for k, (cin, cout) in enumerate(CONV_SPECS):
            self.p[f"W{k}"] = (rng.normal(0, np.sqrt(2.0 / (cin * 9)), (cin * 9, cout))).astype(np.float32)
            self.p[f"b{k}"] = np.zeros(cout, np.float32)
        f = 2 * CONV_SPECS[-1][1]
        self.p["Wd"] = rng.normal(0, np.sqrt(1.0 / f), (f, N_CLASSES)).astype(np.float32)
        self.p["bd"] = np.zeros(N_CLASSES, np.float32)

    def forward(self, x, keep=False, record=None):
        cache = []
        for k in range(len(CONV_SPECS)):
            n, h, w, _ = x.shape
            col = im2col(x)
            z = col @ self.p[f"W{k}"] + self.p[f"b{k}"]
            a = np.maximum(z, 0).reshape(n, h, w, -1)
            if record is not None:
                record.append(float(np.percentile(a, 99.99)))
            entry = {"x_shape": x.shape, "col": col, "z": z}
            if POOL_AFTER[k]:
                out, xr = maxpool2(a)
                entry.update(xr=xr, pooled=out)
                a = out
            cache.append(entry)
            x = a
        gap = x.mean(axis=(1, 2))
        gmp = x.max(axis=(1, 2))
        feat = np.concatenate([gap, gmp], axis=1)
        logits = feat @ self.p["Wd"] + self.p["bd"]
        if keep:
            self._cache = (cache, x, feat)
        return logits

    def backward(self, dlogits):
        cache, x_last, feat = self._cache
        g = {"Wd": feat.T @ dlogits, "bd": dlogits.sum(0)}
        dfeat = dlogits @ self.p["Wd"].T
        c = x_last.shape[-1]
        n, h, w, _ = x_last.shape
        dgap = dfeat[:, :c]
        dgmp = dfeat[:, c:]
        dx = np.broadcast_to(dgap[:, None, None, :] / (h * w), x_last.shape).copy()
        mx = x_last.max(axis=(1, 2), keepdims=True)
        m = (x_last == mx)
        m = m / np.maximum(m.sum(axis=(1, 2), keepdims=True), 1)
        dx += m * dgmp[:, None, None, :]
        for k in reversed(range(len(CONV_SPECS))):
            e = cache[k]
            if POOL_AFTER[k]:
                dx = maxpool2_back(dx, e["xr"], e["pooled"])
            dz = dx.reshape(-1, dx.shape[-1]) * (e["z"] > 0)
            g[f"W{k}"] = e["col"].T @ dz
            g[f"b{k}"] = dz.sum(0)
            if k > 0:
                dx = col2im(dz @ self.p[f"W{k}"].T, e["x_shape"])
        return g

    def predict(self, x, batch=512):
        out = [self.forward(x[i:i + batch]) for i in range(0, len(x), batch)]
        return np.concatenate(out)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def train(model: FloatCNN, X, y, Xv, yv, epochs=14, batch=64, lr=3e-3, seed=0, log=print,
          class_weights=None):
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in model.p.items()}
    s = {k: np.zeros_like(v) for k, v in model.p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    step = 0
    cw = np.ones(N_CLASSES, np.float32) if class_weights is None else np.asarray(class_weights, np.float32)
    for ep in range(epochs):
        idx = rng.permutation(len(X))
        lr_ep = lr * 0.5 * (1 + np.cos(np.pi * ep / epochs))
        tot = 0.0
        for i in range(0, len(X), batch):
            bi = idx[i:i + batch]
            xb = X[bi]
            if rng.random() < 0.5:
                xb = xb[:, ::-1]
            if rng.random() < 0.5:
                xb = xb[:, :, ::-1]
            xb = np.rot90(xb, rng.integers(4), axes=(1, 2)).copy()
            logits = model.forward(xb, keep=True)
            p = softmax(logits)
            w = cw[y[bi]]
            tot += float(-(w * np.log(p[np.arange(len(bi)), y[bi]] + 1e-9)).sum())
            d = p.copy()
            d[np.arange(len(bi)), y[bi]] -= 1.0
            d *= (w / w.sum())[:, None]
            grads = model.backward(d.astype(np.float32))
            step += 1
            for k in model.p:
                gk = grads[k] + 1e-4 * model.p[k] * (k.startswith("W"))
                m[k] = b1 * m[k] + (1 - b1) * gk
                s[k] = b2 * s[k] + (1 - b2) * gk * gk
                mh = m[k] / (1 - b1 ** step)
                sh = s[k] / (1 - b2 ** step)
                model.p[k] -= (lr_ep * mh / (np.sqrt(sh) + eps)).astype(np.float32)
        acc = (model.predict(Xv).argmax(1) == yv).mean()
        log(f"epoch {ep + 1:2d}/{epochs}  loss {tot / len(X):.4f}  val acc {acc:.4f}")
    return model


# ----------------------------------------------------------------------------- int8 deployment


def quantize(model: FloatCNN, Xcal) -> dict:
    rec: list[float] = []
    model.forward(Xcal, record=rec)
    s_act = [max(r, 1e-6) / 127.0 for r in rec]
    q = {"s_in": np.float32(1.0 / 127.0)}
    s_prev = q["s_in"]
    for k in range(len(CONV_SPECS)):
        W = model.p[f"W{k}"]
        sw = np.maximum(np.abs(W).max(axis=0), 1e-8) / 127.0
        q[f"W{k}"] = np.clip(np.round(W / sw), -127, 127).astype(np.int8)
        q[f"b{k}"] = np.round(model.p[f"b{k}"] / (s_prev * sw)).astype(np.int32)
        q[f"M{k}"] = (s_prev * sw / s_act[k]).astype(np.float32)  # requant multipliers
        q[f"s{k}"] = np.float32(s_act[k])
        s_prev = np.float32(s_act[k])
    Wd = model.p["Wd"]
    swd = np.maximum(np.abs(Wd).max(axis=0), 1e-8) / 127.0
    q["Wd"] = np.clip(np.round(Wd / swd), -127, 127).astype(np.int8)
    q["bd"] = np.round(model.p["bd"] / (s_prev * swd)).astype(np.int32)
    q["sd"] = (s_prev * swd).astype(np.float32)
    return q


class Int8CNN:
    """Integer-arithmetic inference engine (what runs on board)."""

    def __init__(self, q: dict):
        self.q = q

    @classmethod
    def load(cls, path: str | Path) -> "Int8CNN":
        d = np.load(path)
        return cls({k: d[k] for k in d.files})

    def forward(self, x: np.ndarray) -> np.ndarray:
        q = self.q
        if x.ndim == 3:
            x = x[None]
        a = np.clip(np.round(x / q["s_in"]), -127, 127).astype(np.int32)
        for k in range(len(CONV_SPECS)):
            n, h, w, _ = a.shape
            acc = im2col(a) @ q[f"W{k}"].astype(np.int32) + q[f"b{k}"]          # int32
            out = np.clip(np.round(acc * q[f"M{k}"]), 0, 127).astype(np.int32)  # ReLU + int8
            a = out.reshape(n, h, w, -1)
            if POOL_AFTER[k]:
                a, _ = maxpool2(a)
        gap = np.round(a.mean(axis=(1, 2))).astype(np.int32)
        gmp = a.max(axis=(1, 2))
        feat = np.concatenate([gap, gmp], axis=1)
        acc = feat @ q["Wd"].astype(np.int32) + q["bd"]
        return acc * q["sd"]

    def predict_proba(self, x):
        return softmax(self.forward(x))

    @property
    def size_bytes(self) -> int:
        return int(sum(v.nbytes for k, v in self.q.items() if k[0] in "Wb"))

    @property
    def n_params(self) -> int:
        return int(sum(v.size for k, v in self.q.items() if k[0] in "Wb"))
