"""Procedural multispectral scene generator (B, G, R, NIR — top-of-atmosphere reflectance).

The simulator does not ship real imagery: for every acquired frame it builds a
32x32 decimated preview whose content follows the *truth* at the footprint
(land fraction from Natural Earth, cloud cover from the cloud field, active
wildfires from the fire scenario). The same generator, sampled randomly, is
the training distribution of the on-board model. Spectral signatures are
typical values for vegetation, soil, water, cloud, burn scars, smoke and
active flaming fronts.
"""
from __future__ import annotations

import numpy as np

CLASSES = ("CLOUD", "WATER", "LAND", "FIRE")
CLOUD, WATER, LAND, FIRE = range(4)
SIZE = 32

_VEG = np.array([0.030, 0.060, 0.040, 0.330])
_SOIL = np.array([0.100, 0.140, 0.185, 0.260])
_DRY = np.array([0.060, 0.090, 0.115, 0.240])
_URBAN = np.array([0.110, 0.120, 0.130, 0.170])
_WATER = np.array([0.050, 0.045, 0.030, 0.015])
_CLOUDR = np.array([0.560, 0.570, 0.575, 0.530])
_SCAR = np.array([0.030, 0.030, 0.035, 0.060])
_FLAME = np.array([0.070, 0.130, 0.380, 0.780])
_SMOKE = np.array([0.180, 0.140, 0.100, 0.030])


def _smooth(rng: np.random.Generator, coarse: int) -> np.ndarray:
    """Smooth random field on SIZE x SIZE by bicubic-ish upsampling of a coarse grid."""
    g = rng.random((coarse + 1, coarse + 1))
    x = np.linspace(0, coarse, SIZE, endpoint=False)
    i = x.astype(int)
    f = x - i
    f = f * f * (3 - 2 * f)
    rows = g[i] * (1 - f)[:, None] + g[i + 1] * f[:, None]
    return rows[:, i] * (1 - f)[None, :] + rows[:, i + 1] * f[None, :]


def _field(rng):
    return 0.6 * _smooth(rng, 3) + 0.3 * _smooth(rng, 6) + 0.1 * _smooth(rng, 12)


def value_of(cls: int, cloud: float) -> float:
    """Mission value of a frame (operational utility for the wildfire/land service)."""
    if cls == FIRE:
        return 10.0
    if cls == LAND:
        return float(max(0.0, 1.0 - cloud))
    if cls == WATER:
        return 0.05
    return 0.0


def synth(rng: np.random.Generator, land_frac: float, cloud: float, fire: float = 0.0,
          sun_elev_deg: float = 45.0, lat: float = 35.0):
    """Returns (image float32 [32,32,4], info dict with the frame truth)."""
    # surface
    lf = _field(rng)
    if land_frac <= 0.0:
        land = np.zeros((SIZE, SIZE), bool)
    elif land_frac >= 1.0:
        land = np.ones((SIZE, SIZE), bool)
    else:
        land = lf > np.quantile(lf, 1.0 - land_frac)
    mix = _field(rng)
    aridity = np.clip(0.5 + (abs(lat) - 30.0) / -40.0 + rng.normal(0, 0.25), 0.0, 1.0)
    veg_w = np.clip(mix * (1.2 - aridity), 0, 1)[..., None]
    soil = rng.choice([0, 1, 2], p=[0.5, 0.35, 0.15])
    base = (_SOIL, _DRY, _URBAN)[soil]
    surf = veg_w * _VEG + (1 - veg_w) * base
    surf = surf * (1.0 + 0.15 * (_smooth(rng, 10)[..., None] - 0.5))
    img = np.where(land[..., None], surf, _WATER * (1.0 + 0.1 * (_smooth(rng, 5)[..., None] - 0.5)))

    # wildfire: burn scar, smoke plume and active flame pixels (on land)
    flame_mask = np.zeros((SIZE, SIZE), bool)
    if fire > 0.0 and land.any():
        ys, xs = np.nonzero(land)
        k = rng.integers(len(ys))
        cy, cx = ys[k], xs[k]
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        rad = 2.0 + 4.0 * fire
        d = np.hypot(yy - cy, xx - cx)
        scar = (d < rad * (0.8 + 0.4 * _smooth(rng, 4))) & land
        img = np.where(scar[..., None], _SCAR, img)
        ang = rng.uniform(0, 2 * np.pi)
        u = (yy - cy) * np.sin(ang) + (xx - cx) * np.cos(ang)
        v = -(yy - cy) * np.cos(ang) + (xx - cx) * np.sin(ang)
        plume = np.exp(-(v / (1.5 + 0.12 * np.clip(u, 0, None))) ** 2) * np.exp(-np.clip(u, 0, None) / (10 + 10 * fire))
        plume *= (u > -1.0)
        img = img + (0.75 * fire * plume)[..., None] * _SMOKE
        front = (np.abs(d - rad) < 1.2) & land & (rng.random((SIZE, SIZE)) < 0.55)
        flame_mask = front
        img = np.where(flame_mask[..., None], _FLAME * (0.6 + 0.4 * fire), img)

    # clouds
    cf = _field(rng)
    if cloud <= 0.0:
        alpha = np.zeros((SIZE, SIZE))
    else:
        thr = np.quantile(cf, 1.0 - min(cloud, 0.999))
        alpha = np.clip((cf - thr) / 0.06, 0.0, 1.0)
        alpha *= 0.85 + 0.15 * _smooth(rng, 8)
    img = img * (1 - alpha[..., None]) + alpha[..., None] * _CLOUDR * (0.9 + 0.2 * _smooth(rng, 8)[..., None])

    # radiometry: illumination and sensor noise
    illum = np.sqrt(np.sin(np.radians(max(sun_elev_deg, 5.0))))
    img = img * illum + rng.normal(0, 0.008, img.shape)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    cloud_frac = float((alpha > 0.5).mean())
    land_actual = float(land.mean())
    fire_visible = bool((flame_mask & (alpha < 0.5)).sum() >= 2)
    if fire_visible:
        cls = FIRE
    elif cloud_frac >= 0.55:
        cls = CLOUD
    elif land_actual < 0.25:
        cls = WATER
    else:
        cls = LAND
    info = {"cls": cls, "cloud": cloud_frac, "land": land_actual, "fire_visible": fire_visible,
            "value": value_of(cls, cloud_frac)}
    return img, info


def to_rgb(img: np.ndarray) -> np.ndarray:
    """True-colour quicklook (uint8) with a simple stretch."""
    rgb = img[..., [2, 1, 0]] * 3.2
    return (np.clip(rgb, 0, 1) ** (1 / 1.6) * 255).astype(np.uint8)


def sample_training_set(n: int, seed: int):
    rng = np.random.default_rng(seed)
    X = np.empty((n, SIZE, SIZE, 4), np.float32)
    y = np.empty(n, np.int64)
    for i in range(n):
        u = rng.random()
        land = 0.0 if u < 0.22 else 1.0 if u < 0.55 else rng.random()
        cloud = 0.0 if rng.random() < 0.25 else rng.random() ** 0.8
        fire = rng.uniform(0.3, 1.0) if rng.random() < 0.3 else 0.0
        img, info = synth(rng, land, cloud, fire, rng.uniform(10, 75), rng.uniform(-45, 50))
        X[i], y[i] = img, info["cls"]
    return X, y
