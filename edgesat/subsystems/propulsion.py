"""Propulsion: warm-gas thruster for orbit maintenance.

Burns are planned by the OBDH when the semi-major axis has decayed beyond a
dead-band; the thruster only fires once the ADCS has aligned the thrust axis
with the velocity vector. Mass flow follows the rocket equation, thruster
misalignment produces a disturbance torque, and a valve-leak failure drains
propellant until FDIR closes the latch valve.
"""
from __future__ import annotations

import numpy as np

from ..core.constants import G0


class Propulsion:
    def __init__(self, cfg: dict):
        p = cfg["propulsion"]
        self.cfg = p
        self.isp = p["isp_s"]
        self.thrust = p["thrust_N"]
        self.prop0 = p["propellant_kg"]
        self.prop = p["propellant_kg"]
        self.axis = np.array(p["thrust_axis_body"], float)
        self.lever = np.array(p["thrust_misalignment_m"], float)
        self.dry = cfg["spacecraft"]["dry_mass_kg"]
        self.burn_remaining = 0.0
        self.firing = False
        self.leaking = False
        self.latch_closed = False
        self.dv_used = 0.0
        self.burns: list[dict] = []
        self.tm: dict = {}

    @property
    def mass(self) -> float:
        return self.dry + self.prop

    @property
    def dv_remaining(self) -> float:
        return self.isp * G0 * np.log(self.mass / self.dry)

    def plan_burn(self, dv: float, t: float) -> float:
        dur = self.mass * dv / self.thrust
        self.burn_remaining = dur
        self.burns.append({"t_plan": t, "dv_cmd": dv, "duration": dur, "dv_done": 0.0})
        return dur

    def step(self, dt: float, aligned: bool, enabled: bool):
        """Returns (thrust force in body [N], torque [Nm], valve power [W])."""
        mdot = self.thrust / (self.isp * G0)
        force = np.zeros(3)
        torque = np.zeros(3)
        self.firing = False
        if self.burn_remaining > 0 and aligned and enabled and not self.latch_closed and self.prop > 0:
            self.firing = True
            burn = min(dt, self.burn_remaining)
            dm = min(mdot * burn, self.prop)
            dv = self.isp * G0 * np.log(self.mass / (self.mass - dm))
            self.prop -= dm
            self.dv_used += dv
            self.burn_remaining -= burn
            if self.burns:
                self.burns[-1]["dv_done"] += dv
            force = self.axis * self.thrust * (burn / dt)
            torque = np.cross(self.lever, force)
        if self.leaking and not self.latch_closed:
            self.prop = max(0.0, self.prop - 4.0e-6 * dt)
        valve_w = 1.0 if self.firing else 0.0
        self.tm = {"propellant_g": self.prop * 1000.0, "propellant_pct": 100.0 * self.prop / self.prop0,
                   "dv_used_m_s": self.dv_used, "dv_remaining_m_s": float(self.dv_remaining),
                   "firing": self.firing, "burn_remaining_s": self.burn_remaining,
                   "leaking": self.leaking, "latch_closed": self.latch_closed,
                   "mass_kg": self.mass, "burns": len(self.burns),
                   "burn_log": [{k: round(v, 4) for k, v in b.items()} for b in self.burns[-6:]]}
        return force, torque, valve_w
