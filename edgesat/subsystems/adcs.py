"""Attitude Determination and Control Subsystem.

Truth model
    Rigid body with three orthogonal reaction wheels, three magnetorquers and
    environmental disturbance torques (gravity gradient, aerodynamic, solar
    radiation pressure, residual magnetic dipole, thruster misalignment).

Determination
    Multiplicative EKF (6 states: attitude error, gyro bias) propagated with
    the gyro and updated with star-tracker quaternions, coarse sun-sensor
    vectors and magnetometer vectors. When the star tracker is blinded or
    failed, the filter keeps running on Sun + magnetic field, and its
    covariance shows the loss of observability.

Control
    Rate-limited quaternion-feedback law with gyroscopic compensation,
    B-dot detumbling, magnetic momentum dumping and fault-tolerant actuator
    allocation (a failed wheel axis is taken over by the magnetorquers,
    limited to the torque component perpendicular to the field).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.attitude import (attitude_error, cross, dcm_from_quat, propagate_quat, qconj, qmul,
                             qnormalize, quat_from_dcm, quat_from_rotvec, skew, triad)
from ..core.constants import DEG, MU_EARTH, SRP_PRESSURE
from ..core.environment import earth_angular_radius

MODES = ("DETUMBLE", "SUN_POINT", "NADIR", "GS_TRACK", "DELTA_V", "FREE_DRIFT")

AX_X, AX_Y, AX_Z = np.eye(3)


@dataclass
class Face:
    name: str
    normal: np.ndarray
    area: float
    center: np.ndarray


def build_geometry(cfg: dict) -> list[Face]:
    dx, dy, dz = cfg["spacecraft"]["body_dims_m"]
    com = np.array(cfg["spacecraft"]["com_offset_m"])
    faces = []
    for k, (axis, size) in enumerate(((AX_X, (dy, dz)), (AX_Y, (dx, dz)), (AX_Z, (dx, dy)))):
        half = (dx, dy, dz)[k] / 2.0
        for sgn, tag in ((1.0, "+"), (-1.0, "-")):
            n = sgn * axis
            faces.append(Face(f"{tag}{'XYZ'[k]}", n, size[0] * size[1], n * half - com))
    wing_area = dx * dz
    for sx in (1.0, -1.0):
        c = np.array([sx * dx, dy / 2.0, 0.0]) - com
        faces.append(Face(f"wing{'+' if sx > 0 else '-'}X front", AX_Y.copy(), wing_area, c))
        faces.append(Face(f"wing{'+' if sx > 0 else '-'}X back", -AX_Y, wing_area, c))
    return faces


@dataclass
class MEKF:
    q: np.ndarray
    bias: np.ndarray
    P: np.ndarray
    sigma_v: float
    sigma_u: float
    st_used: bool = False
    ss_used: bool = False
    mag_used: bool = False

    def propagate(self, w_meas: np.ndarray, dt: float) -> np.ndarray:
        w = w_meas - self.bias
        self.q = propagate_quat(self.q, w, dt)
        F = np.zeros((6, 6))
        F[:3, :3] = -skew(w)
        F[:3, 3:] = -np.eye(3)
        Phi = np.eye(6) + F * dt + 0.5 * (F @ F) * dt * dt
        sv2, su2 = self.sigma_v ** 2, self.sigma_u ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3] = (sv2 * dt + su2 * dt ** 3 / 3.0) * np.eye(3)
        Q[:3, 3:] = Q[3:, :3] = -(su2 * dt * dt / 2.0) * np.eye(3)
        Q[3:, 3:] = su2 * dt * np.eye(3)
        self.P = Phi @ self.P @ Phi.T + Q
        return w

    def _update(self, dx: np.ndarray, H: np.ndarray, resid: np.ndarray, R: np.ndarray) -> np.ndarray:
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = dx + K @ (resid - H @ dx)
        IKH = np.eye(6) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T  # Joseph form
        return dx

    def update(self, q_st, sun_meas, sun_ref, mag_meas, mag_ref, sig_st, sig_ss, sig_mag):
        dx = np.zeros(6)
        A = dcm_from_quat(self.q)
        self.st_used = q_st is not None
        self.ss_used = sun_meas is not None
        self.mag_used = mag_meas is not None
        if q_st is not None:
            dq = qmul(q_st, qconj(self.q))
            if dq[3] < 0:
                dq = -dq
            H = np.hstack((np.eye(3), np.zeros((3, 3))))
            dx = self._update(dx, H, 2.0 * dq[:3], sig_st ** 2 * np.eye(3))
        for meas, ref, sig in ((sun_meas, sun_ref, sig_ss), (mag_meas, mag_ref, sig_mag)):
            if meas is None:
                continue
            b_hat = A @ ref
            H = np.hstack((skew(b_hat), np.zeros((3, 3))))
            dx = self._update(dx, H, meas - b_hat, sig ** 2 * np.eye(3))
        self.q = qnormalize(qmul(quat_from_rotvec(dx[:3]), self.q))
        self.bias = self.bias + dx[3:]
        self.P = 0.5 * (self.P + self.P.T)


@dataclass
class ADCSFaults:
    wheel_seized: list = field(default_factory=lambda: [False, False, False])
    wheel_isolated: list = field(default_factory=lambda: [False, False, False])
    star_tracker_failed: bool = False
    star_tracker_blind_until: float = -1.0


class ADCS:
    def __init__(self, cfg: dict, rng: np.random.Generator, q0: np.ndarray, w0: np.ndarray):
        a = cfg["adcs"]
        self.cfg = a
        self.rng = rng
        self.J = np.array(cfg["spacecraft"]["inertia_kgm2"], dtype=float)
        self.Jinv = np.linalg.inv(self.J)
        self.faces = build_geometry(cfg)
        self.cd = cfg["spacecraft"]["drag_cd"]
        self.rho_srp = cfg["spacecraft"]["srp_reflectivity"]
        self.m_res = np.array(cfg["spacecraft"]["residual_dipole_Am2"])
        # truth state
        self.q = qnormalize(np.array(q0, float))
        self.w = np.array(w0, float)
        self.h_w = np.zeros(3)
        self.gyro_bias = np.array(a["gyro_initial_bias_deg_s"]) * DEG
        # actuators
        self.tau_max = a["wheel_max_torque_Nm"]
        self.h_max = a["wheel_max_momentum_Nms"]
        self.I_w = a["wheel_inertia_kgm2"]
        self.m_max = a["mtq_max_dipole_Am2"]
        # control gains
        wn, z = a["control_wn"], a["control_zeta"]
        self.k_w = 2.0 * z * wn
        self.k_q = wn / (2.0 * z)
        self.w_max = a["max_slew_rate_deg_s"] * DEG
        # sensors
        self.sig_st = a["star_tracker_noise_arcsec"] / 3600.0 * DEG
        self.sig_ss = a["sun_sensor_noise_deg"] * DEG
        self.mag_noise = a["magnetometer_noise_nT"] * 1e-9
        self.st_bore = np.array(a["star_tracker_boresight_body"], float)
        self.st_bore /= np.linalg.norm(self.st_bore)
        # estimator: start with a small knowledge error
        q_hat = qnormalize(qmul(quat_from_rotvec(rng.normal(0, 0.5 * DEG, 3)), self.q))
        P0 = np.diag([(1.0 * DEG) ** 2] * 3 + [(0.02 * DEG) ** 2] * 3)
        self.ekf = MEKF(q_hat, np.zeros(3), P0, a["gyro_arw_rad_sqrt_s"], a["gyro_rrw_rad_s_sqrt_s"])
        self.faults = ADCSFaults()
        self.mode = "SUN_POINT"
        self.loads = dict(cfg["eps"]["loads"])
        self.gs_eci: np.ndarray | None = None
        self.gs_eci_next: np.ndarray | None = None
        self.b_prev: np.ndarray | None = None
        self.h_expected = np.zeros(3)
        self.wheel_mismatch_s = np.zeros(3)
        self.tm: dict = {}
        self.q_ref = self.q.copy()

    # ------------------------------------------------------------------ references
    def reference_dcm(self, mode: str, r: np.ndarray, v: np.ndarray, sun_u: np.ndarray,
                      gs_eci: np.ndarray | None) -> np.ndarray:
        nadir = -r / np.linalg.norm(r)
        if mode == "NADIR":
            return triad(AX_Z, AX_Y, nadir, sun_u)
        if mode == "GS_TRACK" and gs_eci is not None:
            los = gs_eci - r
            return triad(AX_Z, AX_Y, los / np.linalg.norm(los), sun_u)
        if mode == "DELTA_V":
            return triad(AX_X, AX_Y, v / np.linalg.norm(v), sun_u)
        return triad(AX_Y, AX_Z, sun_u, nadir)  # SUN_POINT / SAFE

    # ------------------------------------------------------------------ disturbances
    def disturbances(self, A: np.ndarray, r: np.ndarray, v_rel: np.ndarray, rho: float,
                     sun_u: np.ndarray, illum: float, B_b: np.ndarray):
        rn = np.linalg.norm(r)
        rb = A @ (r / rn)
        t_gg = 3.0 * MU_EARTH / rn ** 3 * cross(rb, self.J @ rb)
        vb = A @ v_rel
        vn = np.linalg.norm(vb)
        vu = vb / vn
        s_b = A @ sun_u
        q_dyn = 0.5 * rho * vn * vn * self.cd
        t_aero = np.zeros(3)
        t_srp = np.zeros(3)
        f_aero = np.zeros(3)
        area_proj = 0.0
        for f in self.faces:
            c = f.normal @ vu
            if c > 0.0:
                F = -q_dyn * f.area * c * vu
                f_aero += F
                t_aero += cross(f.center, F)
                area_proj += f.area * c
            cs = f.normal @ s_b
            if cs > 0.0 and illum > 0.0:
                F = -SRP_PRESSURE * illum * f.area * cs * ((1 - self.rho_srp) * s_b
                                                          + 2 * self.rho_srp * cs * f.normal)
                t_srp += cross(f.center, F)
        t_mag = cross(self.m_res, B_b)
        return t_gg, t_aero, t_srp, t_mag, area_proj

    # ------------------------------------------------------------------ main step
    def step(self, t: float, dt: float, env, extra_torque: np.ndarray, enabled: bool = True) -> dict:
        A = dcm_from_quat(self.q)
        B_b = A @ env.B_eci
        t_gg, t_aero, t_srp, t_mag, area_proj = self.disturbances(
            A, env.r, env.v_rel, env.rho, env.sun_u, env.illum, B_b)
        tau_dist = t_gg + t_aero + t_srp + t_mag + extra_torque

        # ---------------- sensors
        sv = self.cfg["gyro_arw_rad_sqrt_s"]
        su = self.cfg["gyro_rrw_rad_s_sqrt_s"]
        self.gyro_bias = self.gyro_bias + su * np.sqrt(dt) * self.rng.normal(size=3)
        w_meas = self.w + self.gyro_bias + sv / np.sqrt(dt) * self.rng.normal(size=3)
        st_ok, st_reason = self._star_tracker_available(t, A, env)
        q_st = None
        if st_ok:
            q_st = qnormalize(qmul(quat_from_rotvec(self.rng.normal(0, self.sig_st, 3)), self.q))
        sun_meas = None
        if env.illum > 0.5:
            s = A @ env.sun_u + self.rng.normal(0, self.sig_ss, 3)
            sun_meas = s / np.linalg.norm(s)
        Bm = B_b + self.rng.normal(0, self.mag_noise, 3)
        Bn = np.linalg.norm(env.B_eci)
        mag_meas = Bm / np.linalg.norm(Bm)

        # ---------------- estimation
        w_hat = self.ekf.propagate(w_meas, dt)
        self.ekf.update(q_st, sun_meas, env.sun_u, mag_meas, env.B_eci / Bn,
                        self.sig_st, self.sig_ss, self.mag_noise / Bn)
        w_hat = w_meas - self.ekf.bias
        A_hat = dcm_from_quat(self.ekf.q)

        # ---------------- guidance
        A_ref = self.reference_dcm(self.mode, env.r, env.v, env.sun_u, self.gs_eci)
        r2, v2 = env.r + env.v * dt, env.v
        gs2 = self.gs_eci_next
        A_ref2 = self.reference_dcm(self.mode, r2, v2, env.sun_u, gs2)
        dA = A_ref2 @ A_ref.T
        w_ref = np.array([dA[1, 2] - dA[2, 1], dA[2, 0] - dA[0, 2], dA[0, 1] - dA[1, 0]]) / (2 * dt)
        self.q_ref = quat_from_dcm(A_ref)

        # ---------------- control
        m_cmd = np.zeros(3)
        tau_w_body = np.zeros(3)
        Bm_T = Bm
        if not enabled or self.mode == "FREE_DRIFT":
            pass
        elif self.mode == "DETUMBLE":
            if self.b_prev is not None:
                bdot = (Bm_T - self.b_prev) / dt
                m_cmd = np.clip(-self.cfg["bdot_gain"] * bdot, -self.m_max, self.m_max)
        else:
            d_theta = attitude_error(self.ekf.q, self.q_ref)
            w_ref_b = A_hat @ A_ref.T @ w_ref
            w_err = w_hat - w_ref_b
            w_des = -self.k_q * d_theta
            nd = np.linalg.norm(w_des)
            if nd > self.w_max:
                w_des *= self.w_max / nd
            tau_cmd = self.J @ (self.k_w * (w_des - w_err)) + cross(w_hat, self.J @ w_hat + self.h_w)
            m_cmd, tau_w_body = self._allocate(tau_cmd, Bm_T)
        self.b_prev = Bm_T
        tau_mtq = cross(m_cmd, B_b)

        tau_w_cmd = tau_w_body.copy()
        # wheels: seized wheels deliver nothing
        for i in range(3):
            if self.faults.wheel_seized[i]:
                tau_w_body[i] = 0.0
        # wheel momentum saturation
        for i in range(3):
            if abs(self.h_w[i]) >= self.h_max and np.sign(-tau_w_body[i]) == np.sign(self.h_w[i]):
                tau_w_body[i] = 0.0

        # errors at time t (before the dynamics step)
        point_err = np.degrees(np.linalg.norm(attitude_error(self.q, self.q_ref)))
        know_err = np.degrees(np.linalg.norm(attitude_error(self.q, self.ekf.q))) * 3600.0
        w_rate_err = self.w - A @ A_ref.T @ w_ref

        # ---------------- dynamics (RK4 on w, exact quaternion step on mean rate)
        tau_ext = tau_dist + tau_mtq

        def wdot(w, h):
            return self.Jinv @ (tau_ext + tau_w_body - cross(w, self.J @ w + h))
        h0 = self.h_w
        hdot = -tau_w_body
        k1 = wdot(self.w, h0)
        k2 = wdot(self.w + 0.5 * dt * k1, h0 + 0.5 * dt * hdot)
        k3 = wdot(self.w + 0.5 * dt * k2, h0 + 0.5 * dt * hdot)
        k4 = wdot(self.w + dt * k3, h0 + dt * hdot)
        w_new = self.w + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.q = propagate_quat(self.q, 0.5 * (self.w + w_new), dt)
        self.w = w_new
        self.h_w = np.clip(self.h_w + hdot * dt, -self.h_max, self.h_max)
        for i in range(3):
            if self.faults.wheel_seized[i]:
                self.h_w[i] = 0.0

        # ---------------- wheel health monitor (commanded vs measured momentum)
        self.h_expected = np.clip(self.h_expected - tau_w_cmd * dt, -self.h_max, self.h_max)
        # a healthy wheel follows its commanded momentum; a seized one also reports a stall
        # (zero speed with drive current) through its driver telemetry
        stalled = np.array(self.faults.wheel_seized) & (np.abs(tau_w_cmd) > 1e-7)
        mismatch = (np.abs(self.h_expected - self.h_w) > self.cfg.get("wheel_mismatch_Nms", 5e-5)) | stalled
        self.wheel_mismatch_s = np.where(mismatch, self.wheel_mismatch_s + dt, 0.0)
        if not mismatch.any():
            self.h_expected = self.h_w.copy()

        # ---------------- telemetry
        A_new = dcm_from_quat(self.q)
        P = self.ekf.P
        self.tm = {
            "mode": self.mode,
            "q": self.q.tolist(), "q_est": self.ekf.q.tolist(), "q_ref": self.q_ref.tolist(),
            "dcm": A_new.tolist(),
            "rate_deg_s": np.degrees(self.w).tolist(),
            "rate_norm_deg_s": float(np.degrees(np.linalg.norm(self.w))),
            "rate_err_rad_s": float(np.linalg.norm(w_rate_err)),
            "pointing_error_deg": float(point_err),
            "knowledge_error_arcsec": float(know_err),
            "bias_est_deg_h": (np.degrees(self.ekf.bias) * 3600).tolist(),
            "bias_true_deg_h": (np.degrees(self.gyro_bias) * 3600).tolist(),
            "sigma3_arcsec": (3 * np.sqrt(np.diag(P)[:3]) / DEG * 3600).tolist(),
            "cov_corr": _corr(P).tolist(),
            "cov_log10_diag": np.log10(np.diag(P)).tolist(),
            "wheel_h_mNms": (self.h_w * 1e3).tolist(),
            "wheel_rpm": (self.h_w / self.I_w * 60 / (2 * np.pi)).tolist(),
            "wheel_torque_mNm": (-tau_w_body * 1e3).tolist(),
            "mtq_dipole_Am2": m_cmd.tolist(),
            "torque_uNm": {
                "gravity_gradient": float(np.linalg.norm(t_gg) * 1e6),
                "aerodynamic": float(np.linalg.norm(t_aero) * 1e6),
                "solar_pressure": float(np.linalg.norm(t_srp) * 1e6),
                "magnetic": float(np.linalg.norm(t_mag) * 1e6),
                "thruster": float(np.linalg.norm(extra_torque) * 1e6),
            },
            "sensors": {"star_tracker": st_reason, "sun_sensor": sun_meas is not None,
                        "magnetometer": True, "gyro": True},
            "momentum_total_mNms": float(np.linalg.norm(self.J @ self.w + self.h_w) * 1e3),
            "B_body_nT": (B_b * 1e9).tolist(),
            "wheels_seized": list(self.faults.wheel_seized),
            "wheels_isolated": list(self.faults.wheel_isolated),
        }
        self._area_proj = area_proj
        self._power = self._power_draw(tau_w_body, m_cmd, st_reason)
        return self.tm

    # ------------------------------------------------------------------ helpers
    def _allocate(self, tau_cmd: np.ndarray, B_b: np.ndarray):
        """Split the commanded torque between wheels and magnetorquers."""
        b2 = B_b @ B_b
        avail = np.array([not (self.faults.wheel_isolated[i]) for i in range(3)])
        m = np.zeros(3)
        if b2 > 1e-14:
            m += -self.cfg["momentum_dump_gain"] * cross(B_b, self.h_w * avail) / b2
            tau_fail = np.where(avail, 0.0, tau_cmd)
            if np.any(tau_fail):
                m += cross(B_b, tau_fail) / b2
        m = np.clip(m, -self.m_max, self.m_max)
        tau_mtq = cross(m, B_b)
        tau_w = np.where(avail, tau_cmd - tau_mtq, 0.0)
        tau_w = np.clip(tau_w, -self.tau_max, self.tau_max)
        return m, tau_w

    def _star_tracker_available(self, t, A, env) -> tuple[bool, str]:
        if self.faults.star_tracker_failed:
            return False, "FAILED"
        if t < self.faults.star_tracker_blind_until:
            return False, "BLINDED"
        bore_i = A.T @ self.st_bore
        if np.degrees(np.arccos(np.clip(bore_i @ env.sun_u, -1, 1))) < self.cfg["star_tracker_sun_exclusion_deg"]:
            return False, "SUN_EXCL"
        nadir = -env.r / np.linalg.norm(env.r)
        ang_earth = np.degrees(np.arccos(np.clip(bore_i @ nadir, -1, 1)))
        if ang_earth < np.degrees(earth_angular_radius(np.linalg.norm(env.r))) + self.cfg["star_tracker_earth_margin_deg"]:
            return False, "EARTH_EXCL"
        if np.degrees(np.linalg.norm(self.w)) > self.cfg["star_tracker_max_rate_deg_s"]:
            return False, "RATE"
        return True, "TRACKING"

    def _power_draw(self, tau_w, m_cmd, st_reason) -> dict:
        L = self.loads
        wheel_speed = self.h_w / self.I_w
        p_wheels = 3 * L["wheel_idle"] + float(np.sum(np.abs(tau_w * wheel_speed))) / 0.5
        p_mtq = float(np.sum((m_cmd / self.m_max) ** 2) * L["mtq_max"])
        p_st = 0.0 if st_reason == "FAILED" else L["star_tracker"]
        return {"adcs_electronics": L["adcs_electronics"], "reaction_wheels": p_wheels,
                "magnetorquers": p_mtq, "star_tracker": p_st, "gyro": L["gyro"], "gnss": L["gnss"]}

    @property
    def area_proj(self) -> float:
        return getattr(self, "_area_proj", 0.1)

    @property
    def power(self) -> dict:
        return getattr(self, "_power", {})

    def seize_wheel(self, i: int) -> None:
        """A seized wheel stops abruptly: its momentum is transferred to the body."""
        self.faults.wheel_seized[i] = True
        h = np.zeros(3)
        h[i] = self.h_w[i]
        self.w = self.w + self.Jinv @ h
        self.h_w[i] = 0.0


def _corr(P: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(P))
    return P / np.outer(d, d)
