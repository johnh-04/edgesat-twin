"""Space and Earth environment models.

* Sun ephemeris: low-precision analytical model (Vallado, Alg. 29), ~0.01 deg.
* Eclipse: conical umbra/penumbra with a smooth illumination fraction.
* Geomagnetic field: tilted dipole from IGRF-13 (2020) first-degree coefficients.
* Atmosphere: Vallado exponential model (Table 8-4) with a solar-activity scale.
* Earth surface: Natural Earth land raster, procedural cloud field, wildfire scenario.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import AU, DEG, R_EARTH, R_SUN

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ----------------------------------------------------------------------------- sun


def sun_position_eci(jd: float) -> np.ndarray:
    t = (jd - 2451545.0) / 36525.0
    lam_m = (280.460 + 36000.771 * t) % 360.0
    m = np.radians((357.5291092 + 35999.05034 * t) % 360.0)
    lam = np.radians(lam_m + 1.914666471 * np.sin(m) + 0.019994643 * np.sin(2 * m))
    r = (1.000140612 - 0.016708617 * np.cos(m) - 0.000139589 * np.cos(2 * m)) * AU
    eps = np.radians(23.439291 - 0.0130042 * t)
    return r * np.array([np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)])


def sun_right_ascension(jd: float) -> float:
    s = sun_position_eci(jd)
    return float(np.arctan2(s[1], s[0]))


def illumination_fraction(r_sat: np.ndarray, r_sun: np.ndarray) -> float:
    """Fraction of the solar disc visible from the satellite (0 = umbra, 1 = full sun)."""
    rel = r_sun - r_sat
    d_sun = np.linalg.norm(rel)
    d_earth = np.linalg.norm(r_sat)
    a = np.arcsin(min(1.0, R_SUN / d_sun))       # apparent radius of the Sun
    b = np.arcsin(min(1.0, R_EARTH / d_earth))   # apparent radius of the Earth
    cos_c = float(-r_sat @ rel / (d_earth * d_sun))
    c = np.arccos(np.clip(cos_c, -1.0, 1.0))      # angular separation
    if c >= a + b:
        return 1.0
    if c <= b - a:
        return 0.0
    # partial overlap of two discs (small-angle planar approximation)
    x = (c * c + a * a - b * b) / (2 * c)
    y = np.sqrt(max(a * a - x * x, 0.0))
    area = (a * a * np.arccos(np.clip(x / a, -1, 1)) + b * b * np.arccos(np.clip((c - x) / b, -1, 1))
            - c * y)
    return float(np.clip(1.0 - area / (np.pi * a * a), 0.0, 1.0))


def sun_elevation_at(lat: float, lon: float, sun_ecef_unit: np.ndarray) -> float:
    """Solar elevation [rad] at a ground point (spherical Earth)."""
    up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
    return float(np.arcsin(np.clip(up @ sun_ecef_unit, -1.0, 1.0)))


# ----------------------------------------------------------------------------- magnetic field

# IGRF-13 epoch 2020 degree-1 Gauss coefficients [nT]
_G10, _G11, _H11 = -29404.8, -1450.9, 4652.5
_DIPOLE_ECEF = np.array([_G11, _H11, _G10])


def magnetic_field_ecef(r_ecef: np.ndarray) -> np.ndarray:
    """Tilted-dipole field [T] in ECEF."""
    rn = np.linalg.norm(r_ecef)
    u = r_ecef / rn
    return (R_EARTH / rn) ** 3 * (3.0 * (_DIPOLE_ECEF @ u) * u - _DIPOLE_ECEF) * 1e-9


# ----------------------------------------------------------------------------- atmosphere

_ATM = np.array([  # h0 [km], rho0 [kg/m^3], H [km]   (Vallado 4th ed., Table 8-4)
    [0, 1.225, 7.249], [25, 3.899e-2, 6.349], [30, 1.774e-2, 6.682], [40, 3.972e-3, 7.554],
    [50, 1.057e-3, 8.382], [60, 3.206e-4, 7.714], [70, 8.770e-5, 6.549], [80, 1.905e-5, 5.799],
    [90, 3.396e-6, 5.382], [100, 5.297e-7, 5.877], [110, 9.661e-8, 7.263], [120, 2.438e-8, 9.473],
    [130, 8.484e-9, 12.636], [140, 3.845e-9, 16.149], [150, 2.070e-9, 22.523],
    [180, 5.464e-10, 29.740], [200, 2.789e-10, 37.105], [250, 7.248e-11, 45.546],
    [300, 2.418e-11, 53.628], [350, 9.518e-12, 53.298], [400, 3.725e-12, 58.515],
    [450, 1.585e-12, 60.828], [500, 6.967e-13, 63.822], [600, 1.454e-13, 71.835],
    [700, 3.614e-14, 88.667], [800, 1.170e-14, 124.64], [900, 5.245e-15, 181.05],
    [1000, 3.019e-15, 268.00],
])


def atmospheric_density(alt_m: float, scale: float = 1.0) -> float:
    h = alt_m / 1000.0
    idx = int(np.searchsorted(_ATM[:, 0], h, side="right") - 1)
    idx = max(0, min(idx, len(_ATM) - 1))
    h0, rho0, hs = _ATM[idx]
    return float(scale * rho0 * np.exp(-(h - h0) / hs))


# ----------------------------------------------------------------------------- land mask


class LandMask:
    def __init__(self, path: Path | None = None):
        path = path or DATA_DIR / "landmask_0p25.npz"
        d = np.load(path)
        shape = tuple(d["shape"])
        self.mask = np.unpackbits(d["mask"], axis=1)[:, : shape[1]].astype(bool)
        self.res = float(d["res_deg"])

    def is_land(self, lat_deg, lon_deg):
        lat = np.asarray(lat_deg)
        lon = (np.asarray(lon_deg) + 180.0) % 360.0
        i = np.clip(((90.0 - lat) / self.res).astype(int), 0, self.mask.shape[0] - 1)
        j = np.clip((lon / self.res).astype(int), 0, self.mask.shape[1] - 1)
        return self.mask[i, j]

    def land_fraction(self, lat_deg: float, lon_deg: float, half_size_km: float, n: int = 5) -> float:
        dlat = half_size_km / 111.0
        dlon = dlat / max(np.cos(np.radians(lat_deg)), 0.05)
        g = np.linspace(-1.0, 1.0, n)
        la, lo = np.meshgrid(lat_deg + g * dlat, lon_deg + g * dlon)
        return float(self.is_land(la.ravel(), lo.ravel()).mean())


# ----------------------------------------------------------------------------- clouds


class CloudField:
    """Procedural, time-evolving cloud-cover fraction.

    Two octaves of value noise on a lat/lon lattice, advected eastwards and
    blended between daily random states, shifted by a zonal climatology
    (cloudy ITCZ and storm tracks, clear subtropical deserts)."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed + 7)
        self.days = 64
        self.coarse = rng.random((self.days, 46, 91))    # 4 deg lattice
        self.fine = rng.random((self.days, 181, 361))   # 1 deg lattice
        for g in (self.coarse, self.fine):               # periodic in longitude
            g[:, :, -1] = g[:, :, 0]

    @staticmethod
    def _bilinear(grid, lat, lon, res):
        y = (90.0 - lat) / res
        x = ((lon + 180.0) % 360.0) / res
        i0, j0 = int(np.floor(y)), int(np.floor(x))
        fy, fx = y - i0, x - j0
        i1 = min(i0 + 1, grid.shape[0] - 1)
        j1 = j0 + 1
        i0 = min(max(i0, 0), grid.shape[0] - 1)
        sy, sx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        top = grid[i0, j0] * (1 - sx) + grid[i0, j1] * sx
        bot = grid[i1, j0] * (1 - sx) + grid[i1, j1] * sx
        return top * (1 - sy) + bot * sy

    @staticmethod
    def climatology(lat: float) -> float:
        a = abs(lat)
        itcz = 0.55 * np.exp(-((lat - 6.0) / 9.0) ** 2)
        subtropic = -0.35 * np.exp(-((a - 25.0) / 8.0) ** 2)
        storm = 0.25 * np.exp(-((a - 55.0) / 12.0) ** 2)
        return 0.45 + itcz + subtropic + storm

    def cover(self, lat: float, lon: float, t_s: float) -> float:
        day = t_s / 86400.0
        d0 = int(np.floor(day)) % self.days
        d1 = (d0 + 1) % self.days
        w = day - np.floor(day)
        w = w * w * (3 - 2 * w)
        drift = 8.0 * day  # deg of eastward advection per day
        lon_a = lon - drift
        v = 0.0
        for g, res, amp in ((self.coarse, 4.0, 0.65), (self.fine, 1.0, 0.35)):
            a = self._bilinear(g[d0], lat, lon_a, res)
            b = self._bilinear(g[d1], lat, lon_a, res)
            v += amp * ((1 - w) * a + w * b)
        x = 6.0 * (v - 0.5) + 4.0 * (self.climatology(lat) - 0.5)
        return float(1.0 / (1.0 + np.exp(-x)))


# ----------------------------------------------------------------------------- wildfires


@dataclass
class Fire:
    fid: int
    lat: float
    lon: float
    t_start: float
    t_end: float
    intensity: float
    zone: str
    imaged: bool = False
    first_acq_t: float | None = None
    alert_ground_t: float | None = None     # alert received on ground (AI)
    image_ground_t: float | None = None     # full image received on ground


class FireScenario:
    """Poisson wildfire births inside configured zones (land only)."""

    def __init__(self, zones: list[dict], land: LandMask, seed: int, horizon_s: float = 30 * 86400):
        rng = np.random.default_rng(seed + 99)
        self.fires: list[Fire] = []
        fid = 0
        for z in zones:
            rate = z["births_per_day"] / 86400.0
            t = -2.0 * 86400.0  # some fires already burning at epoch
            while True:
                t += rng.exponential(1.0 / rate)
                if t > horizon_s:
                    break
                for _ in range(50):
                    lat = rng.uniform(*z["lat"])
                    lon = rng.uniform(*z["lon"])
                    if land.is_land(lat, lon):
                        break
                else:
                    continue
                life = rng.uniform(*z["lifetime_h"]) * 3600.0
                if t + life < 0:
                    continue
                self.fires.append(Fire(fid, lat, lon, t, t + life, float(rng.uniform(0.4, 1.0)),
                                       z["name"]))
                fid += 1
        self._lat = np.array([f.lat for f in self.fires])
        self._lon = np.array([f.lon for f in self.fires])
        self._t0 = np.array([f.t_start for f in self.fires])
        self._t1 = np.array([f.t_end for f in self.fires])

    def active(self, t: float) -> np.ndarray:
        return np.nonzero((self._t0 <= t) & (self._t1 >= t))[0]

    def in_footprint(self, lat: float, lon: float, t: float, radius_km: float) -> list[Fire]:
        idx = self.active(t)
        if idx.size == 0:
            return []
        dlat = (self._lat[idx] - lat) * 111.0
        dlon = ((self._lon[idx] - lon + 180.0) % 360.0 - 180.0) * 111.0 * np.cos(np.radians(lat))
        hit = idx[np.hypot(dlat, dlon) < radius_km]
        return [self.fires[i] for i in hit]


def earth_angular_radius(r_norm: float) -> float:
    return float(np.arcsin(min(1.0, R_EARTH / r_norm)))


__all__ = ["sun_position_eci", "sun_right_ascension", "illumination_fraction", "sun_elevation_at",
           "magnetic_field_ecef", "atmospheric_density", "LandMask", "CloudField", "FireScenario",
           "Fire", "earth_angular_radius", "DEG"]
