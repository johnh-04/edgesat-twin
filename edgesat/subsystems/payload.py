"""Payload: multispectral camera + on-board Edge AI + mass memory data products.

For every frame the camera acquires, the simulator knows the *truth* of the
footprint (land fraction, cloud cover, visible wildfires, image smear from
the attitude motion) and renders the decimated preview the on-board model
sees. Two data-handling policies are compared:

``edge-AI``   the int8 CNN classifies each frame; cloudy/water/blurred frames
              are discarded, wildfire detections raise a tiny high-priority
              ALERT and a 512x512 ROI crop, the rest is ranked by expected
              mission value and downlinked best-first.
``baseline``  a conventional pipeline that stores every frame and downlinks
              first-in-first-out.

When the live dashboard runs, the baseline is a *shadow* store fed with the
very same frames and the very same downlink capacity, so the two policies can
be compared second by second.
"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..ai.cnn import Int8CNN, macs
from ..ai.scenes import CLASSES, FIRE, LAND, WATER, synth, to_rgb

PRIO_HK, PRIO_ALERT, PRIO_ROI, PRIO_FRAME = 0, 1, 2, 3


@dataclass
class Product:
    pid: int
    kind: str          # HK | ALERT | ROI | FRAME
    size: float
    prio: int
    score: float
    t: float
    truth_cls: int = -1
    truth_value: float = 0.0
    useful: bool = False
    fire_ids: tuple = ()
    sent: float = 0.0
    pending: bool = False

    @property
    def key(self):
        return (self.prio, -self.score, self.t, self.pid)


@dataclass
class DeliveryStats:
    bytes_by_kind: dict = field(default_factory=lambda: {"HK": 0.0, "ALERT": 0.0, "ROI": 0.0, "FRAME": 0.0})
    frames_delivered: int = 0
    useful_frames_delivered: int = 0
    useful_bytes: float = 0.0
    value_delivered: float = 0.0
    alerts_delivered: int = 0
    false_alerts_delivered: int = 0
    dropped_memory_full: int = 0
    evicted: int = 0
    fire_alert_ground_t: dict = field(default_factory=dict)
    fire_image_ground_t: dict = field(default_factory=dict)
    frame_age_s: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if not k.startswith("fire_") and k != "frame_age_s"}
        d["median_frame_age_h"] = float(np.median(self.frame_age_s) / 3600) if self.frame_age_s else None
        d["fires_alerted"] = len(self.fire_alert_ground_t)
        d["fires_image_on_ground"] = len(self.fire_image_ground_t)
        return d


class DataStore:
    """Mass memory with a priority downlink queue."""

    def __init__(self, capacity_bytes: float, policy: str, half_life_s: float = 86400.0):
        self.capacity = capacity_bytes
        self.policy = policy           # "ai" (value-ranked, ageing) or "fifo"
        self.tau = half_life_s / np.log(2.0)
        self.items: dict[int, Product] = {}
        self.used = 0.0
        self.stats = DeliveryStats()

    def _rank(self, p: Product, t: float):
        """Downlink order: priority class first; within a class the AI policy ranks by
        expected value decaying with age (fresh information is worth more), FIFO by age."""
        if self.policy == "ai":
            return (p.prio, -p.score * np.exp(-(t - p.t) / self.tau), p.t, p.pid)
        return (p.prio, p.t, p.pid)

    def free(self) -> float:
        return self.capacity - self.used

    def add(self, p: Product) -> bool:
        if p.size > self.free():
            if self.policy == "ai" and p.kind in ("FRAME", "ROI", "ALERT"):
                # evict the lowest-value frames if the newcomer is worth more
                age = lambda x: np.exp(-(p.t - x.t) / self.tau)  # noqa: E731
                victims = sorted((x for x in self.items.values() if x.kind == "FRAME" and not x.pending
                                  and x.sent == 0), key=lambda x: x.score * age(x))
                freed, chosen = 0.0, []
                for v in victims:
                    if v.score * age(v) >= p.score and p.kind == "FRAME":
                        break
                    chosen.append(v)
                    freed += v.size
                    if p.size <= self.free() + freed:
                        break
                if p.size <= self.free() + freed:
                    for v in chosen:
                        self.remove(v.pid)
                        self.stats.evicted += 1
                else:
                    self.stats.dropped_memory_full += 1
                    return False
            else:
                self.stats.dropped_memory_full += 1
                return False
        self.items[p.pid] = p
        self.used += p.size
        return True

    def remove(self, pid: int) -> None:
        p = self.items.pop(pid, None)
        if p is not None:
            self.used -= (p.size - p.sent)

    def has_data(self) -> bool:
        return any(not p.pending for p in self.items.values())

    def transmit(self, budget: float, t: float, fires_by_id: dict) -> float:
        sent_total = 0.0
        while budget > 1e-6:
            ready = [p for p in self.items.values() if not p.pending]
            if not ready:
                break
            p = min(ready, key=lambda x: self._rank(x, t))
            chunk = min(budget, p.size - p.sent)
            p.sent += chunk
            self.used -= chunk
            budget -= chunk
            sent_total += chunk
            self.stats.bytes_by_kind[p.kind] += chunk
            if p.useful and p.kind in ("FRAME", "ROI"):
                self.stats.useful_bytes += chunk
            if p.sent >= p.size - 1e-6:
                self.items.pop(p.pid)
                self._delivered(p, t)
        return sent_total

    def _delivered(self, p: Product, t: float) -> None:
        s = self.stats
        if p.kind == "FRAME":
            s.frames_delivered += 1
            s.frame_age_s.append(t - p.t)
            s.value_delivered += p.truth_value
            if p.useful:
                s.useful_frames_delivered += 1
        if p.kind == "ALERT":
            s.alerts_delivered += 1
            if not p.fire_ids:
                s.false_alerts_delivered += 1
            for f in p.fire_ids:
                s.fire_alert_ground_t.setdefault(f, t)
        if p.kind in ("FRAME", "ROI"):
            for f in p.fire_ids:
                s.fire_image_ground_t.setdefault(f, t)

    def summary(self) -> dict:
        kinds = {"HK": 0, "ALERT": 0, "ROI": 0, "FRAME": 0, "PENDING": 0}
        for p in self.items.values():
            kinds["PENDING" if p.pending else p.kind] += 1
        return {"used_gb": self.used / 1e9, "capacity_gb": self.capacity / 1e9,
                "fill_pct": 100.0 * self.used / self.capacity, "counts": kinds}


class Payload:
    def __init__(self, cfg: dict, rng: np.random.Generator, land, clouds, fires, root: Path):
        p = cfg["payload"]
        self.cfg = p
        self.ai_cfg = p["ai"]
        self.rng = rng
        self.land, self.clouds, self.fires = land, clouds, fires
        self.frame_bytes = p["frame_px"] ** 2 * p["bands"] * p["bits"] / 8.0
        self.swath_km = p["frame_px"] * p["gsd_m"] / 1000.0
        self.model = Int8CNN.load(root / self.ai_cfg["model_file"])
        self.t_infer = (self.ai_cfg["io_overhead_s"]
                        + self.ai_cfg["tiles_per_frame"] * 2 * macs() / (self.ai_cfg["effective_gops"] * 1e9))
        cap = p["mass_memory_GB"] * 1e9
        # a satellite flown without the AI uses the conventional FIFO policy for its mass memory
        policy = "ai" if self.ai_cfg["enabled"] else "fifo"
        self.store = DataStore(cap, policy, self.ai_cfg.get("value_half_life_h", 24.0) * 3600.0)
        self.shadow = DataStore(cap, "fifo")
        self.ai_enabled = bool(self.ai_cfg["enabled"])
        self.ai_hang_until = -1.0
        self.ai_powered = False
        self.queue: deque[tuple] = deque()
        self.next_frame_t = 0.0
        self.pid = 0
        self.imaging = False
        self.acq_count = 0
        self.truth_counts = [0, 0, 0, 0]
        self.blurred = 0
        self.confusion = np.zeros((4, 4), int)
        self.decisions = {"kept": 0, "alert": 0, "discard_cloud": 0, "discard_water": 0,
                          "discard_lowvalue": 0, "discard_blur": 0, "fallback_store": 0}
        self.ai_busy_s = 0.0
        self.ai_energy_j = 0.0
        self.recent: deque[dict] = deque(maxlen=10)
        self.alert_log: deque[dict] = deque(maxlen=30)
        self.busy_frac = 0.0
        self.tm: dict = {}
        self.skip_reason = ""

    # --------------------------------------------------------------- products
    def _new(self, **kw) -> Product:
        self.pid += 1
        return Product(pid=self.pid, **kw)

    def add_hk(self, nbytes: float, t: float) -> None:
        for st in (self.store, self.shadow):
            st.add(self._new(kind="HK", size=nbytes, prio=PRIO_HK, score=0.0, t=t))

    def ai_available(self, t: float) -> bool:
        return self.ai_enabled and self.ai_powered and t >= self.ai_hang_until

    # --------------------------------------------------------------- acquisition
    def step(self, t: float, dt: float, imaging_allowed: bool, ctx: dict) -> dict:
        """ctx: lat, lon, alt_m, sun_elev_deg, pointing_error_deg, rate_err_rad_s, in_aoi."""
        p = self.cfg
        self.imaging = imaging_allowed
        self.skip_reason = ""
        if imaging_allowed and t >= self.next_frame_t:
            if ctx["pointing_error_deg"] > p["max_pointing_error_deg"]:
                self.skip_reason = "POINTING"
            else:
                img, truth = self._acquire(ctx, t)
                self.shadow.add(self._frame_product(t, truth, 0.0))
                if self.ai_available(t):
                    # frames wait in the AI processor's RAM buffer; only what the model
                    # keeps is ever written to mass memory
                    if len(self.queue) < self.ai_cfg.get("ram_buffer_frames", 80):
                        self.queue.append((self._frame_product(t, truth, 0.1), img, truth, dict(ctx)))
                    else:
                        self.skip_reason = "AI_BUFFER_FULL"
                        self.store.stats.dropped_memory_full += 1
                else:
                    if self.store.add(self._frame_product(t, truth, 0.1)):
                        self.decisions["fallback_store"] += 1
                    else:
                        self.skip_reason = "MEMORY_FULL"
                    self._remember(img, truth, None, "STORED (no AI)", t, ctx)
                self.next_frame_t = t + p["frame_interval_s"]
        self._run_ai(t, dt, ctx)
        self.tm = self._telemetry(t)
        return self.tm

    def _acquire(self, ctx: dict, t: float):
        p = self.cfg
        half = 0.5 * self.swath_km
        land = self.land.land_fraction(ctx["lat"], ctx["lon"], half)
        cloud = float(np.clip(self.clouds.cover(ctx["lat"], ctx["lon"], t) + self.rng.normal(0, 0.06), 0, 1))
        fires = self.fires.in_footprint(ctx["lat"], ctx["lon"], t, half * 1.1)
        intensity = max((f.intensity for f in fires), default=0.0)
        img, info = synth(self.rng, land, cloud, intensity, ctx["sun_elev_deg"], ctx["lat"])
        smear = ctx["rate_err_rad_s"] * p["integration_time_s"] * ctx["alt_m"] / p["gsd_m"]
        blurred = smear > p["max_smear_px"]
        fire_ids = tuple(f.fid for f in fires) if info["fire_visible"] else ()
        for f in fires:
            if info["fire_visible"] and f.first_acq_t is None:
                f.first_acq_t = t
                f.imaged = True
        value = 0.0 if blurred else info["value"]
        truth = {**info, "smear_px": float(smear), "blurred": bool(blurred), "value": value,
                 "useful": value >= 0.3, "fire_ids": fire_ids}
        self.acq_count += 1
        self.truth_counts[info["cls"]] += 1
        self.blurred += int(blurred)
        return img, truth

    def _frame_product(self, t, truth, score) -> Product:
        return self._new(kind="FRAME", size=self.frame_bytes, prio=PRIO_FRAME, score=score, t=t,
                         truth_cls=truth["cls"], truth_value=truth["value"], useful=truth["useful"],
                         fire_ids=truth["fire_ids"])

    # --------------------------------------------------------------- edge AI
    def _run_ai(self, t: float, dt: float, ctx: dict) -> None:
        busy = 0.0
        if not self.ai_powered:
            self.busy_frac = 0.0
            return
        if t < self.ai_hang_until:
            self.busy_frac = 1.0     # hung processor spins at full power
            self.ai_busy_s += dt
            return
        while self.queue and busy + self.t_infer <= dt + 1e-9:
            prod, img, truth, actx = self.queue.popleft()
            busy += self.t_infer
            self._decide(prod, img, truth, t, actx)
        self.busy_frac = busy / dt
        self.ai_busy_s += busy

    def _decide(self, prod: Product, img: np.ndarray, truth: dict, t: float, ctx: dict) -> None:
        a = self.ai_cfg
        probs = self.model.predict_proba(img)[0]
        pred = int(np.argmax(probs))
        self.confusion[truth["cls"], pred] += 1
        ev = float(10.0 * probs[FIRE] + 0.8 * probs[LAND] + 0.05 * probs[WATER])
        decision = ""
        if truth["blurred"]:
            # the OBC knows the smear from ADCS telemetry: no need to look at the image
            self.decisions["discard_blur"] += 1
            decision = "DISCARD (smear)"
        elif probs[FIRE] >= a["fire_alert_confidence"]:
            alert = self._new(kind="ALERT", size=self.cfg["alert_bytes"], prio=PRIO_ALERT, score=ev, t=t,
                              truth_cls=truth["cls"], fire_ids=truth["fire_ids"])
            roi = self._new(kind="ROI", size=self.cfg["roi_crop_bytes"], prio=PRIO_ROI, score=ev, t=t,
                            truth_cls=truth["cls"], truth_value=truth["value"], useful=truth["useful"],
                            fire_ids=truth["fire_ids"])
            self.store.add(alert)
            self.store.add(roi)
            prod.score = ev
            self.store.add(prod)
            self.decisions["alert"] += 1
            decision = "FIRE ALERT"
            self.alert_log.appendleft({"t": t, "lat": round(ctx["lat"], 3), "lon": round(ctx["lon"], 3),
                                       "conf": round(float(probs[FIRE]), 3),
                                       "true": bool(truth["fire_ids"]),
                                       "delivered": False, "pid": alert.pid})
        elif ev >= a["keep_threshold"]:
            prod.score = ev
            kept = self.store.add(prod)
            self.decisions["kept"] += 1
            decision = "KEEP" if kept else "DISCARD (memory full)"
        else:
            key = {0: "discard_cloud", 1: "discard_water"}.get(pred, "discard_lowvalue")
            self.decisions[key] += 1
            decision = "DISCARD (" + CLASSES[pred].lower() + ")"
        self._remember(img, truth, probs, decision, t, ctx)

    def _remember(self, img, truth, probs, decision, t, ctx) -> None:
        rgb = to_rgb(img)
        self.recent.appendleft({
            "t": t, "lat": round(ctx["lat"], 2), "lon": round(ctx["lon"], 2),
            "rgb": base64.b64encode(rgb.tobytes()).decode(),
            "truth": CLASSES[truth["cls"]], "cloud": round(truth["cloud"], 2),
            "smear_px": round(truth["smear_px"], 3),
            "pred": None if probs is None else CLASSES[int(np.argmax(probs))],
            "probs": None if probs is None else [round(float(x), 3) for x in probs],
            "decision": decision,
        })

    def ai_power(self, loads: dict) -> float:
        if not self.ai_powered:
            return 0.0
        return loads["ai_idle"] + (loads["ai_active"] - loads["ai_idle"]) * min(self.busy_frac, 1.0)

    def _telemetry(self, t: float) -> dict:
        ai_store = self.store.stats.as_dict()
        sh_store = self.shadow.stats.as_dict()
        for a in self.alert_log:
            if not a["delivered"] and a["pid"] not in self.store.items:
                a["delivered"] = True
        return {
            "imaging": self.imaging, "skip_reason": self.skip_reason,
            "frames_acquired": self.acq_count, "truth_counts": dict(zip(CLASSES, self.truth_counts)),
            "blurred": self.blurred, "ai_enabled": self.ai_enabled, "ai_powered": self.ai_powered,
            "ai_hung": t < self.ai_hang_until, "ai_queue": len(self.queue), "ai_busy_frac": self.busy_frac,
            "t_infer_ms": self.t_infer * 1000.0, "decisions": dict(self.decisions),
            "confusion": self.confusion.tolist(), "memory": self.store.summary(),
            "shadow_memory": self.shadow.summary(), "delivery_ai": ai_store, "delivery_baseline": sh_store,
            "frame_mb": self.frame_bytes / 1e6,
            "alerts": list(self.alert_log)[:10],
        }
