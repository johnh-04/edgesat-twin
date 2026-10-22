"""Orbit mechanics: elements, numerical propagation (two-body + J2 + drag + thrust),
time and reference frames.

The truth orbit is integrated numerically (Cowell, RK4) rather than with an
analytical model, so that drag (which depends on the current attitude) and
thruster firings act on it directly.

Frames: ECI is treated as a true-of-date inertial frame and Earth rotation is
modelled with GMST only (no precession/nutation/polar motion). The resulting
errors (~km) are irrelevant for subsystem engineering and are documented.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .constants import (DEG, F_EARTH, J2, MU_EARTH, OMEGA_EARTH, R_EARTH,
                        SSO_RAAN_RATE)

# ----------------------------------------------------------------------------- time


def julian_date(dt: datetime) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    unix = dt.timestamp()
    return unix / 86400.0 + 2440587.5


def gmst(jd: float) -> float:
    """Greenwich mean sidereal time [rad] (IAU-82)."""
    t = (jd - 2451545.0) / 36525.0
    sec = (67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t
           + 0.093104 * t * t - 6.2e-6 * t ** 3)
    return np.mod(sec * (2.0 * np.pi / 86400.0), 2.0 * np.pi)


# ----------------------------------------------------------------------------- frames


def rot3(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def eci_to_ecef_matrix(jd: float) -> np.ndarray:
    return rot3(gmst(jd))


def geodetic_to_ecef(lat: float, lon: float, alt: float) -> np.ndarray:
    """lat/lon in radians, alt in metres (WGS-84)."""
    e2 = F_EARTH * (2.0 - F_EARTH)
    n = R_EARTH / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.array([(n + alt) * np.cos(lat) * np.cos(lon),
                     (n + alt) * np.cos(lat) * np.sin(lon),
                     (n * (1.0 - e2) + alt) * np.sin(lat)])


def ecef_to_geodetic(r: np.ndarray) -> tuple[float, float, float]:
    """Returns (lat [rad], lon [rad], alt [m]) — Bowring's method, two iterations."""
    x, y, z = r
    e2 = F_EARTH * (2.0 - F_EARTH)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))
    for _ in range(3):
        n = R_EARTH / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - e2 * n / (n + alt)))
    n = R_EARTH / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - n
    return float(lat), float(lon), float(alt)


def look_angles(r_sat_ecef: np.ndarray, r_gs_ecef: np.ndarray, lat: float, lon: float):
    """Elevation, azimuth [rad] and slant range [m] of the satellite from a station."""
    rho = r_sat_ecef - r_gs_ecef
    sl, cl, so, co = np.sin(lat), np.cos(lat), np.sin(lon), np.cos(lon)
    east = -so * rho[0] + co * rho[1]
    north = -sl * co * rho[0] - sl * so * rho[1] + cl * rho[2]
    up = cl * co * rho[0] + cl * so * rho[1] + sl * rho[2]
    rng = float(np.sqrt(east * east + north * north + up * up))
    return float(np.arcsin(up / rng)), float(np.arctan2(east, north)), rng


# ----------------------------------------------------------------------------- elements


def sso_inclination(a: float, e: float = 0.0) -> float:
    """Inclination [rad] that makes the J2 nodal drift equal to the Sun's mean motion."""
    p = a * (1.0 - e * e)
    n = np.sqrt(MU_EARTH / a ** 3)
    cos_i = -SSO_RAAN_RATE / (1.5 * n * J2 * (R_EARTH / p) ** 2)
    return float(np.arccos(cos_i))


def coe_to_rv(a, e, i, raan, argp, nu):
    p = a * (1.0 - e * e)
    r_pf = p / (1.0 + e * np.cos(nu)) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(MU_EARTH / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cO, sO, ci, si, cw, sw = (np.cos(raan), np.sin(raan), np.cos(i), np.sin(i),
                              np.cos(argp), np.sin(argp))
    R = np.array([
        [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
        [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
        [sw * si, cw * si, ci],
    ])
    return R @ r_pf, R @ v_pf


def rv_to_coe(r: np.ndarray, v: np.ndarray) -> dict:
    rn, vn = np.linalg.norm(r), np.linalg.norm(v)
    h = np.cross(r, v)
    hn = np.linalg.norm(h)
    k = np.array([0.0, 0.0, 1.0])
    nvec = np.cross(k, h)
    nn = np.linalg.norm(nvec)
    evec = ((vn * vn - MU_EARTH / rn) * r - (r @ v) * v) / MU_EARTH
    e = np.linalg.norm(evec)
    energy = 0.5 * vn * vn - MU_EARTH / rn
    a = -MU_EARTH / (2.0 * energy)
    i = np.arccos(h[2] / hn)
    raan = np.arctan2(nvec[1], nvec[0]) if nn > 1e-9 else 0.0
    return {"a": float(a), "e": float(e), "i": float(i), "raan": float(np.mod(raan, 2 * np.pi)),
            "period": float(2 * np.pi * np.sqrt(a ** 3 / MU_EARTH))}


# ----------------------------------------------------------------------------- dynamics


def gravity_accel(r: np.ndarray) -> np.ndarray:
    x, y, z = r
    r2 = x * x + y * y + z * z
    rn = np.sqrt(r2)
    a = -MU_EARTH / (r2 * rn) * r
    k = 1.5 * J2 * MU_EARTH * R_EARTH ** 2 / (r2 * r2 * rn)
    zz = 5.0 * z * z / r2
    a = a + k * np.array([x * (zz - 1.0), y * (zz - 1.0), z * (zz - 3.0)])
    return a


def gravity_accel_batch(r: np.ndarray) -> np.ndarray:
    """Vectorised version, r with shape (N, 3)."""
    r2 = np.einsum("ij,ij->i", r, r)
    rn = np.sqrt(r2)
    a = -(MU_EARTH / (r2 * rn))[:, None] * r
    k = 1.5 * J2 * MU_EARTH * R_EARTH ** 2 / (r2 * r2 * rn)
    zz = 5.0 * r[:, 2] ** 2 / r2
    corr = np.column_stack((r[:, 0] * (zz - 1.0), r[:, 1] * (zz - 1.0), r[:, 2] * (zz - 3.0)))
    return a + k[:, None] * corr


def relative_wind(r: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Velocity relative to a co-rotating atmosphere (ECI)."""
    return v - np.cross(np.array([0.0, 0.0, OMEGA_EARTH]), r)


class OrbitPropagator:
    """Cowell propagation with RK4. Perturbing (non-gravitational) acceleration is
    supplied per step by the caller and assumed constant over the step."""

    def __init__(self, r0: np.ndarray, v0: np.ndarray):
        self.r = np.array(r0, dtype=float)
        self.v = np.array(v0, dtype=float)

    def step(self, dt: float, a_pert: np.ndarray) -> None:
        r, v = self.r, self.v

        def f(rr, vv):
            return vv, gravity_accel(rr) + a_pert

        k1r, k1v = f(r, v)
        k2r, k2v = f(r + 0.5 * dt * k1r, v + 0.5 * dt * k1v)
        k3r, k3v = f(r + 0.5 * dt * k2r, v + 0.5 * dt * k2v)
        k4r, k4v = f(r + dt * k3r, v + dt * k3v)
        self.r = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        self.v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)


def predict_states(r0: np.ndarray, v0: np.ndarray, duration: float, step: float):
    """Fast look-ahead propagation (two-body + J2, RK4) used by the on-board planner.
    Returns times (N,), positions (N,3), velocities (N,3)."""
    n = int(duration / step) + 1
    rs = np.empty((n, 3))
    vs = np.empty((n, 3))
    r, v = np.array(r0, float), np.array(v0, float)
    rs[0], vs[0] = r, v
    h = step
    for k in range(1, n):
        k1v = gravity_accel(r)
        k1r = v
        k2v = gravity_accel(r + 0.5 * h * k1r)
        k2r = v + 0.5 * h * k1v
        k3v = gravity_accel(r + 0.5 * h * k2r)
        k3r = v + 0.5 * h * k2v
        k4v = gravity_accel(r + h * k3r)
        k4r = v + h * k3v
        r = r + h / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + h / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        rs[k], vs[k] = r, v
    return np.arange(n) * step, rs, vs


def eci_to_latlon_batch(rs: np.ndarray, jds: np.ndarray):
    """Geocentric-to-geodetic (fast approximate, spherical latitude correction) for plotting
    and planning. Returns lat, lon in degrees."""
    th = gmst(jds) if np.ndim(jds) else np.full(len(rs), gmst(jds))
    c, s = np.cos(th), np.sin(th)
    x = c * rs[:, 0] + s * rs[:, 1]
    y = -s * rs[:, 0] + c * rs[:, 1]
    z = rs[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    e2 = F_EARTH * (2.0 - F_EARTH)
    lat = np.degrees(np.arctan2(z, np.hypot(x, y) * (1.0 - e2)))
    return lat, lon, np.column_stack((x, y, z))


def ltan_raan(jd: float, ltan_hours: float, sun_ra: float) -> float:
    """RAAN [rad] giving the requested local time of ascending node."""
    return float(np.mod(sun_ra + (ltan_hours - 12.0) * 15.0 * DEG, 2.0 * np.pi))
