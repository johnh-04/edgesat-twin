"""Attitude mathematics.

Conventions (Markley & Crassidis, *Fundamentals of Spacecraft Attitude
Determination and Control*, 2014):

* quaternions are scalar-last ``q = [x, y, z, w]``;
* ``q`` represents the attitude of the body frame B with respect to the
  inertial frame I through the direction-cosine matrix ``A(q)`` that maps
  inertial components to body components, ``v_B = A(q) v_I``;
* the product ``q ⊗ p`` is defined so that ``A(q ⊗ p) = A(q) A(p)``;
* kinematics: ``q_dot = 0.5 * [w; 0] ⊗ q`` with ``w`` the body angular rate.
"""
from __future__ import annotations

import numpy as np


def cross(a, b) -> np.ndarray:
    """Fast 3-vector cross product (np.cross has large per-call overhead)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def qmul(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Markley product q ⊗ p (scalar last)."""
    qv, qw = q[:3], q[3]
    pv, pw = p[:3], p[3]
    v = pw * qv + qw * pv - cross(qv, pv)
    w = qw * pw - qv @ pv
    return np.array([v[0], v[1], v[2], w])


def qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qnormalize(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    return q if q[3] >= 0.0 else -q


def dcm_from_quat(q: np.ndarray) -> np.ndarray:
    """A(q): inertial -> body."""
    x, y, z, w = q
    return np.array([
        [w*w + x*x - y*y - z*z, 2.0*(x*y + w*z),       2.0*(x*z - w*y)],
        [2.0*(x*y - w*z),       w*w - x*x + y*y - z*z, 2.0*(y*z + w*x)],
        [2.0*(x*z + w*y),       2.0*(y*z - w*x),       w*w - x*x - y*y + z*z],
    ])


def quat_from_dcm(A: np.ndarray) -> np.ndarray:
    """Inverse of :func:`dcm_from_quat` (Shepperd's method)."""
    tr = A[0, 0] + A[1, 1] + A[2, 2]
    cands = np.array([A[0, 0], A[1, 1], A[2, 2], tr])
    i = int(np.argmax(cands))
    if i == 3:
        w = 0.5 * np.sqrt(1.0 + tr)
        q = np.array([(A[1, 2] - A[2, 1]) / (4 * w),
                      (A[2, 0] - A[0, 2]) / (4 * w),
                      (A[0, 1] - A[1, 0]) / (4 * w), w])
    elif i == 0:
        x = 0.5 * np.sqrt(1.0 + 2 * A[0, 0] - tr)
        q = np.array([x, (A[0, 1] + A[1, 0]) / (4 * x),
                      (A[0, 2] + A[2, 0]) / (4 * x), (A[1, 2] - A[2, 1]) / (4 * x)])
    elif i == 1:
        y = 0.5 * np.sqrt(1.0 + 2 * A[1, 1] - tr)
        q = np.array([(A[0, 1] + A[1, 0]) / (4 * y), y,
                      (A[1, 2] + A[2, 1]) / (4 * y), (A[2, 0] - A[0, 2]) / (4 * y)])
    else:
        z = 0.5 * np.sqrt(1.0 + 2 * A[2, 2] - tr)
        q = np.array([(A[0, 2] + A[2, 0]) / (4 * z), (A[1, 2] + A[2, 1]) / (4 * z),
                      z, (A[0, 1] - A[1, 0]) / (4 * z)])
    return qnormalize(q)


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Quaternion of a frame rotation by ``angle`` about ``axis``."""
    n = np.linalg.norm(axis)
    if n < 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0])
    a = axis / n
    s = np.sin(0.5 * angle)
    return np.array([a[0] * s, a[1] * s, a[2] * s, np.cos(0.5 * angle)])


def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    return quat_from_axis_angle(rv, float(np.linalg.norm(rv)))


def propagate_quat(q: np.ndarray, w: np.ndarray, dt: float) -> np.ndarray:
    """Exact propagation for a constant body rate over dt."""
    return qnormalize(qmul(quat_from_rotvec(w * dt), q))


def quat_kinematics(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    return 0.5 * qmul(np.array([w[0], w[1], w[2], 0.0]), q)


def attitude_error(q: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """Small-angle error vector (rad) of q relative to q_ref, in body axes."""
    dq = qmul(q, qconj(q_ref))
    if dq[3] < 0.0:
        dq = -dq
    return 2.0 * dq[:3]


def angle_between(q: np.ndarray, p: np.ndarray) -> float:
    """Principal rotation angle between two attitudes [rad]."""
    d = abs(float(q @ p))
    return 2.0 * np.arccos(min(1.0, d))


def triad(b1: np.ndarray, b2: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> np.ndarray:
    """Attitude DCM A that maps r1 exactly to b1 and r2 as close as possible to b2."""
    def frame(u, v):
        u = u / np.linalg.norm(u)
        w = cross(u, v)
        nw = np.linalg.norm(w)
        if nw < 1e-9:  # degenerate secondary: pick any perpendicular
            alt = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            w = cross(u, alt)
            nw = np.linalg.norm(w)
        w = w / nw
        return np.column_stack((u, w, cross(u, w)))
    return frame(b1, b2) @ frame(r1, r2).T
