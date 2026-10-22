# EDGESAT-1 — mission and spacecraft design

This document is the engineering companion of the simulator. It states what the
mission must do, how the spacecraft is sized to do it, and how every subsystem is
modelled. Numbers come from `config/mission.toml`, which is the single source of
truth for the simulation.

## 1. Mission statement

> Detect active wildfires over the Mediterranean and other fire-prone regions from a
> 6U CubeSat, and deliver actionable alerts to the ground within one ground-station
> pass, using on-board AI to overcome the downlink bottleneck of small satellites.

The business case behind it: a 6U Earth-observation CubeSat can acquire tens of
gigabytes per day but a single low-cost S-band ground station brings down roughly one
or two. About two thirds of the planet is covered by clouds at any time, so a
conventional satellite spends most of its scarce downlink on images nobody will use,
and time-critical information (a fire that started this morning) waits in a FIFO
queue behind them. Moving the first stage of the analysis on board changes what the
downlink carries, not how big it is.

## 2. Requirements

| ID | Requirement | Verified by |
|---|---|---|
| MR-1 | Image the configured areas of interest in daylight (Sun elevation ≥ 10°) at ≤ 10 m GSD in B, G, R, NIR | `payload` model, `config [payload]` |
| MR-2 | Deliver wildfire detections to the ground at the first ground-station pass after acquisition | `run_mission.py`, wildfire latency |
| MR-3 | Maximise the useful fraction of downlinked imagery | `run_mission.py`, AI vs baseline |
| SR-ADCS-1 | Pointing error < 2° during imaging, rate error giving < 1 px smear | ADCS telemetry, smear gate |
| SR-ADCS-2 | Attitude knowledge < 30″ (3σ) with star tracker | MEKF covariance, `test_estimator_accuracy` |
| SR-ADCS-3 | Detumble from 5°/s within 2 orbits using magnetorquers only | LEOP scenario, `TestLEOP` |
| SR-EPS-1 | Energy-positive in every orbit with nominal operations; SoC never below 30% | `min_soc_pct` in results |
| SR-TCS-1 | Battery 0…40 °C, payload 0…35 °C in all modes | TCS limits, thermostats |
| SR-COMMS-1 | Close the S-band link at ≥ 5° elevation with ≥ 3 dB margin | link budget, ACM |
| SR-OBDH-1 | Autonomous planning of imaging, downlink, charging and orbit maintenance | OBDH planner |
| SR-FDIR-1 | Detect, isolate and recover the faults in §8 without ground intervention | `TestFDIR` |
| SR-PROP-1 | Keep the mean semi-major axis within 150 m of its reference | orbit-maintenance planner |

## 3. Concept of operations

```
          ┌────────────── NOMINAL (autonomous) ──────────────┐
 LEOP ──► │ CHARGE (Sun pointing) ◄──► IMAGING (nadir)        │
 B-dot    │        ▲    │                 │  frames → Edge AI │
 Sun acq. │        │    ▼                 ▼                    │
          │ ORBIT_MAINT (Δv)    DOWNLINK (track Bari, ACM)    │
          └──────────────▲──────────────────────────────────────┘
                         │ autonomous recovery
          SAFE (Sun pointing, payload off) ◄── FDIR, low SoC
          REBOOT (OBC watchdog) ──► SAFE
```

The on-board planner re-propagates the orbit every 5 minutes (two-body + J2, 4 h
look-ahead) to know the next area-of-interest window and the next ground contact. It
starts slewing 150 s ahead of each so that the camera or the antenna is already on
target when the window opens. When the spacecraft is over an imaging area and Bari at
the same time it keeps imaging and downlinks through the nadir-facing patch at a lower
rate.

## 4. Orbit

| Parameter | Value | Rationale |
|---|---|---|
| Type | Sun-synchronous, near-circular | constant illumination geometry for change detection |
| Mean altitude | 525 km | ~5–7 year natural lifetime, acceptable drag for a 6U |
| Inclination | 97.50° (derived) | J2 nodal regression = 0.9856°/day |
| LTAN | 10:30 | morning pass: fires already active, fewer convective clouds |
| Epoch | 2027-07-15 04:00 UTC | peak Mediterranean fire season |

The initial osculating state is corrected so that the *mean* altitude is 525 km
(J2 short-period terms shift the osculating semi-major axis by ~10 km).

## 5. Spacecraft configuration

6U CubeSat, 226 × 100 × 366 mm, two deployable solar wings on the +Y side.

| Face | Hosts |
|---|---|
| +Z | camera aperture, S-band patch antenna (nadir in imaging) |
| +Y | deployable solar wings, 2 × 0.083 m² |
| −X | radiator, 0.025 m², white paint |
| −Y, tilted 35° to −Z | star-tracker boresight (keeps Sun and Earth out of the field of view) |
| +X axis | warm-gas thruster (prograde burns) |

### 5.1 Mass budget (estimate)

| Item | kg |
|---|---:|
| Structure, deployment mechanisms | 1.60 |
| Solar wings | 0.90 |
| EPS + 2S4P Li-ion battery | 0.90 |
| ADCS: 3 wheels, 3 MTQ, star tracker, Sun sensors, gyro, magnetometer | 1.30 |
| OBC + Edge AI processor | 0.35 |
| S-band transmitter + patch, GNSS | 0.45 |
| Multispectral camera (optics + detector) | 3.20 |
| Propulsion, dry | 0.90 |
| Harness, thermal hardware | 0.65 |
| **Subtotal** | **10.25** |
| System margin (13%) | 1.35 |
| **Dry mass** | **11.60** |
| Propellant (butane) | 0.25 |

### 5.2 Power budget

| Mode | Loads | Power |
|---|---|---:|
| Always on | OBC 0.8, ADCS electronics 0.7, wheels 0.45, star tracker 1.0, gyro 0.3, GNSS 0.6, receiver 0.4 | 4.25 W |
| Charge | always-on + magnetorquers + heaters (duty-cycled) | 4.5–7 W |
| Imaging | + camera 3.5, AI processor 3.5 idle / 9 W while inferring | 11.5–12 W |
| Downlink | + transmitter 6.7 W DC (2 W RF) | 11 W |
| Imaging + downlink | both of the above | 18–19 W |

Generation: two wings, 0.165 m² of 29.5% cells, about 55 W at normal incidence and
28 °C, about 42 W at the 75 °C the wings reach in sunlight. The battery is 99 Wh (2S4P
18650, 3.35 Ah cells). Eclipse lasts about 35 of the 95 minutes of an orbit.

### 5.3 Link budget (S-band, 2245 MHz, Bari 2.4 m dish, G/T 10.5 dB/K)

| Elevation | Range | Antenna | EIRP | FSPL | Atmosphere | C/N₀ | Rate (ACM) | Margin |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 5° | 2143 km | tracking | 7.5 dBW | 166.1 dB | 3.4 dB | 76.6 dBHz | 12 Mbit/s | 3.3 dB |
| 5° | 2143 km | nadir-pointed | 1.8 dBW | 166.1 dB | 3.4 dB | 70.9 dBHz | 2 Mbit/s | 5.4 dB |
| 10° | 1756 km | nadir-pointed | 2.2 dBW | 164.4 dB | 1.7 dB | 74.7 dBHz | 8 Mbit/s | 3.1 dB |
| 45° | 716 km | tracking | 7.5 dBW | 156.6 dB | 0.4 dB | 89.1 dBHz | 12 Mbit/s | 15.8 dB |
| 90° | 525 km | tracking | 7.5 dBW | 153.9 dB | 0.3 dB | 91.9 dBHz | 12 Mbit/s | 18.6 dB |

With ground-station tracking the link is limited by the 12 Mbit/s channel, not by the
budget; when the spacecraft keeps the camera on nadir and downlinks at the same time,
the patch antenna looks at Bari up to 65° off boresight and the adaptive rate drops.

Required Eb/N₀ is 2.5 dB (QPSK, LDPC r = 1/2) plus 3 dB margin. The rate ladder
(0.25 … 12 Mbit/s) is selected every second from the live geometry and attitude.

### 5.4 Data budget

One frame is 2048 × 2048 px × 4 bands × 12 bit = 25.2 MB, acquired every 3 s
(contiguous 20.5 km along-track coverage). In the 5-day reference run the camera
acquired about 325 frames, 8.2 GB, per day over the four areas of interest, while 4
Bari passes per day delivered about 2.4 GB (0.6 GB per pass). Acquisition exceeds the
downlink by a factor of 3.4 and the 16 GB mass memory fills in about two days, which is
the premise of the Edge AI payload: the question is not how much to downlink but what.

## 6. Subsystem models

### 6.1 Orbit and environment
- Cowell propagation, RK4, Δt = 1 s: two-body + J2 + drag (attitude-dependent
  projected area) + thrust. ECI ↔ ECEF through GMST.
- Sun: analytical ephemeris (~0.01°). Eclipse: conical umbra/penumbra.
- Geomagnetic field: tilted dipole, IGRF-13 degree-1 coefficients.
- Atmosphere: exponential model with a solar-activity scale factor (1.6).
- Earth surface: Natural Earth land raster, procedural cloud field with zonal
  climatology, Poisson wildfire births in fire-prone zones.

### 6.2 ADCS
- Dynamics: `J ω̇ = τ_ext + τ_wheels − ω × (Jω + h_w)`, RK4.
- Disturbances: gravity gradient `3μ/r³ (r̂ × J r̂)`, aerodynamic and solar-pressure
  forces on every face (centre of pressure vs centre of mass), residual dipole,
  thruster misalignment.
- Sensors: gyro with angle random walk and bias instability, star tracker (12″) with
  Sun/Earth exclusion and rate limit, coarse Sun sensors (0.5°), magnetometer (150 nT).
- Estimation: multiplicative EKF on attitude error and gyro bias, Joseph-form updates.
- Control: rate-limited quaternion feedback, `ω_des = −k_q δθ`, `τ = J k_ω (ω_des − ω_e) + ω × (Jω + h)`
  with ωn = 0.08 rad/s, ζ = 0.9, 1.5°/s slew limit; B-dot detumbling; magnetic
  momentum dumping `m = −k (B × h)/|B|²`; when a wheel fails its axis is handed to the
  magnetorquers.
- Guidance: Sun pointing, nadir with yaw steering towards the Sun, ground-station
  tracking, Δv alignment; reference rates by finite differences of the reference frame.

### 6.3 EPS
- Panel power `S·A·η(T)·cos θ·packing·degradation·health`, η falls 0.45%/K.
- MPPT with charge-current limit, curtailment at full charge, charge inhibit below 0 °C.
- Battery: OCV(SoC) curve, `R I² + OCV I − P = 0` for the current, coulombic efficiency.
- Load switches (LCLs) commanded by the OBDH for payload, AI unit, transmitter, heaters, valve.

### 6.4 TCS
Six lumped nodes (structure, solar array, battery, avionics, payload, comms) with
conductive links. External surfaces receive direct Sun, albedo and Earth IR using
their true orientation and radiate to space; MLI faces have effective α = ε = 0.02.
Every electrical load deposits its dissipation in its host node (transmitter: DC − RF).
Thermostatic heaters on battery and payload.

### 6.5 COMMS
Link budget recomputed every step (see 5.3), Doppler from range rate, pass logging,
adaptive data rate. The downlink capacity of each step is shared identically by the
Edge AI store and the shadow store-everything store.

### 6.6 OBDH
Mode manager, on-board planner (§3), housekeeping telemetry at 1 kbit/s, mass memory
(16 GB) with a priority downlink queue: housekeeping → fire alerts → ROI crops →
frames ranked by expected value.

### 6.7 Propulsion
Warm-gas thruster (25 mN, Isp 105 s, 250 g, ~22 m/s Δv). The OBDH averages the
osculating semi-major axis over one orbit and plans a prograde burn of
`Δv = v Δa / (2a)` when the mean SMA has decayed by 150 m; the burn starts only once
the thrust axis is aligned with the velocity.

### 6.8 Payload and Edge AI
- Each frame's truth (land fraction, cloud cover, visible fires, smear from the
  attitude rate error) drives a procedural 32 × 32 × 4 preview.
- On-board model: 3-layer CNN, 3 916 parameters, 4 KB in int8, 737 k MAC per tile,
  integer inference (int8 weights/activations, int32 accumulators).
- Cost model: 64 tiles per frame on a 12 GOPS int8 accelerator plus 140 ms of I/O,
  i.e. ~150 ms per frame, 9 W while busy, 3.5 W idle.
- Decisions: smear > 1 px → discard (the ADCS knows it without looking at the image);
  P(fire) ≥ 0.6 → 12 kB alert + 1.6 MB ROI crop + full frame; expected value ≥ 0.35 →
  keep, ranked; otherwise discard. When memory is full, the lowest-value frame is
  evicted if the new one is worth more.

## 7. Couplings that the simulator captures

- ADCS → EPS: panel incidence; wheel and magnetorquer power.
- ADCS → payload: pointing gate and image smear.
- ADCS → COMMS: antenna off-boresight angle, hence data rate.
- ADCS → orbit: projected drag area; Δv attitude.
- EPS → OBDH: load-shedding and safe mode on low SoC.
- Payload/AI, COMMS, EPS → TCS: dissipation in the hosting nodes; TCS → EPS: cell efficiency, cold-charge inhibit.
- OBDH → everything: modes, activities, switches.

## 8. FDIR

| Fault (injectable) | Detection | Autonomous response |
|---|---|---|
| Reaction wheel seizure | commanded vs measured momentum, 5 s | isolate wheel, axis to magnetorquers, safe mode, auto recovery |
| Star tracker blinded / failed | tracker status, 60 s | MEKF continues on Sun sensor + magnetometer; event |
| Gyro bias step | — | estimated by the MEKF bias states |
| Solar string loss | — | lower generation; load shedding if SoC drops |
| Battery string loss | — | lower capacity; SoC thresholds protect it |
| Transponder PA −6 dB | — | ACM selects lower rates |
| Battery heater stuck on | heater current while commanded off, 60 s | open upstream LCL |
| OBC single-event upset | watchdog | 90 s reboot, free drift, safe mode, auto recovery |
| Thruster valve leak | propellant decrease with valve closed, 20 s | close latch valve |
| AI processor hang | heartbeat watchdog, 30 s | power-cycle; meanwhile frames stored without AI |
| Low SoC | < 30% / < 20% | payload load-shed / safe mode |
| Tumbling | rate > 4°/s | back to B-dot detumbling |

## 9. Limitations (deliberate simplifications)

- ECI is treated as true-of-date; no precession, nutation or polar motion.
- Dipole magnetic field (truth and on-board model are the same; no model error).
- Lumped thermal network with no internal radiation; MLI as effective properties.
- No wheel friction, jitter or structural flexibility.
- Imagery is synthetic; the classifier is trained and evaluated on the same generator.
  Real-world accuracy would require training on Sentinel-2/Landsat data and in-orbit
  calibration; the generator is designed so that this swap only touches `edgesat/ai`.
- One ground station; no uplink or intersatellite links.
