"""Mission-control server: runs the simulation in real time (with time warp) and streams
telemetry to the web dashboard over Server-Sent Events.

    python -m server.app            # then open http://localhost:8000
    python -m server.app --scenario leop --port 8080
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from edgesat.sim import Simulation

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def dumps(obj) -> str:
    return json.dumps(obj, default=_default, separators=(",", ":"), allow_nan=False)


class Runner:
    """Steps the simulation on a background thread at `warp` x real time."""

    TICK = 0.05          # wall seconds per loop iteration
    SNAPSHOT_EVERY = 0.25

    def __init__(self, scenario: str = "nominal", warp: float = 20.0):
        self.sim = Simulation(scenario=scenario)
        self.lock = threading.Lock()
        self.warp = warp
        self.paused = False
        self.seq = 0
        self.snapshot = ""
        self.achieved_warp = 0.0
        self._acc = 0.0
        self._next_sample = 0.0
        self._stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.thread.start()

    def _loop(self):
        last_snap = 0.0
        while not self._stop:
            t0 = time.perf_counter()
            with self.lock:
                if not self.paused:
                    self._acc += self.warp * self.TICK / self.sim.sc.dt
                    n = int(self._acc)
                    self._acc -= n
                    deadline = t0 + self.TICK * 0.9
                    done = 0
                    sample_dt = max(self.sim.sc.dt, self.warp / 12.0)
                    while done < n and time.perf_counter() < deadline:
                        self.sim.sc.step()
                        done += 1
                        if self.sim.t >= self._next_sample:
                            self.sim.sample()
                            self._next_sample = self.sim.t + sample_dt
                    self._acc = 0.0 if done < n else self._acc
                    self.achieved_warp = 0.8 * self.achieved_warp + 0.2 * (done * self.sim.sc.dt / self.TICK)
                if t0 - last_snap >= self.SNAPSHOT_EVERY:
                    snap = self.sim.snapshot()
                    snap["series_rows"] = self.sim.pop_series()
                    snap["warp"] = self.warp
                    snap["achieved_warp"] = self.achieved_warp
                    snap["paused"] = self.paused
                    snap["ai_enabled"] = self.sim.sc.payload.ai_enabled
                    self.snapshot = dumps(snap)
                    self.seq += 1
                    last_snap = t0
            dt = time.perf_counter() - t0
            time.sleep(max(0.0, self.TICK - dt))

    def command(self, cmd: str, arg=None) -> str:
        with self.lock:
            if cmd == "warp":
                self.warp = float(max(0.1, min(float(arg), 2000.0)))
                return "ok"
            if cmd == "pause":
                self.paused = bool(arg)
                return "ok"
            if cmd == "reset":
                self.sim.reset(arg or None)
                self._next_sample = 0.0
                return "reset"
            return self.sim.command(cmd, arg)


runner: Runner | None = None


async def index(request: Request):
    return FileResponse(WEB / "index.html")


async def info(request: Request):
    with runner.lock:
        data = runner.sim.static_info()
    return Response(dumps(data), media_type="application/json")


async def land(request: Request):
    return FileResponse(ROOT / "edgesat" / "data" / "land_110m.json", media_type="application/json")


async def stream(request: Request):
    async def gen():
        last = -1
        yield "retry: 2000\n\n"
        while True:
            if await request.is_disconnected():
                break
            if runner.seq != last and runner.snapshot:
                last = runner.seq
                yield f"data: {runner.snapshot}\n\n"
            await asyncio.sleep(0.05)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def command(request: Request):
    body = await request.json()
    try:
        msg = runner.command(body.get("cmd"), body.get("arg"))
        return JSONResponse({"ok": True, "msg": msg})
    except (ValueError, KeyError, TypeError) as e:
        return JSONResponse({"ok": False, "msg": str(e)}, status_code=400)


def create_app(scenario: str = "nominal", warp: float = 20.0, start: bool = True) -> Starlette:
    global runner
    runner = Runner(scenario, warp)
    if start:
        runner.start()
    return Starlette(routes=[
        Route("/", index),
        Route("/api/info", info),
        Route("/api/land", land),
        Route("/api/stream", stream),
        Route("/api/command", command, methods=["POST"]),
        Mount("/static", StaticFiles(directory=WEB), name="static"),
    ])


def main():
    ap = argparse.ArgumentParser(description="EDGESAT-1 mission control")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--scenario", choices=["nominal", "leop"], default="nominal")
    ap.add_argument("--warp", type=float, default=20.0, help="initial time warp (sim s per wall s)")
    a = ap.parse_args()
    import uvicorn
    app = create_app(a.scenario, a.warp)
    print(f"EDGESAT-1 mission control on http://{a.host}:{a.port}")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
