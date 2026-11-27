# Doppler // Acoustic Aircraft Geolocator

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-06b6d4.svg)](https://awne8886.github.io/doppler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)

**Doppler** estimates where a microphone was by matching the acoustic Doppler "S-curves" of passing aircraft against ADS-B
trajectories. It is built to be honest: when the evidence does not pin down one location it says **inconclusive**, with a
machine-readable reason, instead of presenting the best grid cell as an answer.

🔗 **Live GitHub Pages Web App**: [https://awne8886.github.io/doppler/](https://awne8886.github.io/doppler/)
(browser-only mode: demo scenarios and *uploaded* tracks; no live ADS-B access).

---

## ✈️ Core Architecture & Physics

```
   ADS-B trajectories (uploaded / historical / live / demo, tagged)
        │   QNH correction · conditioning (resample + smooth + fuse reported velocity)
        ▼
   Kinematic + time-window pre-filter  ◄── constant-velocity flyover fit of each ridge
        │
   Recording ──► ridge discovery (spectrogram → contrast → bounded-slope Viterbi → harmonic groups)
        │
        ▼
   Joint shared-clock matcher over a grid of observer positions (wind, terrain, fitted f0/drift, robust loss)
        │   coarse-to-fine grid · Nelder-Mead refinement of local optima · sensitivity re-runs
        ▼
   Evidence checks ──► supported | inconclusive (+ reason_code)
```

### 1. Propagation and Doppler shift (with wind)
Let **R** be the vector from the aircraft at emission to the observer, **v** the aircraft's ground velocity and **W** a uniform
horizontal wind (the direction the air moves *toward*). Sound travels at *c* relative to the air, so the travel time *T* solves
`c T = |R − W T|`, which has the exact root

$$T = \frac{-\mathbf R\cdot\mathbf W + \sqrt{(\mathbf R\cdot\mathbf W)^2 + (c^2-|\mathbf W|^2)|\mathbf R|^2}}{c^2-|\mathbf W|^2} \;\approx\; \frac{|\mathbf R|}{c + \mathbf W\cdot\hat n}.$$

Differentiating with respect to emission time gives the received frequency

$$\frac{f}{f_0} = \frac{c + \mathbf W\cdot\hat k}{c + \mathbf W\cdot\hat k - \mathbf v\cdot\hat k},\qquad \hat k = \frac{\mathbf R - \mathbf W T}{cT},$$

where $\hat k$ is the **air-frame wave normal**, not the ground ray $\hat n$ ($\hat k \approx \hat n - \mathbf W_\perp/c$). The common
shortcut $c_\text{eff}/(c_\text{eff}+u_r)$ with $c_\text{eff} = c + \mathbf W\cdot\hat n$ drops a first-order term
$\mathbf v\cdot\mathbf W_\perp/c$: against an independent forward model (travel time by fixed-point iteration, frequency as
$f_0\,d\tau/dt$) it is ~24 Hz off at 820 Hz in a 15 m/s wind, while the exact form adds no error. With $\mathbf W = 0$ both reduce
to $T = |\mathbf R|/c$ and $f = f_0\,c/(c + u_r)$, $u_r$ the receding radial speed.

The speed of sound uses Cramer (1993) for humid air when temperature or humidity is given (default 343.2 m/s, 20 °C dry air).

### 2. Fitted rest frequency, robust loss and a shared clock
The rest frequency is never trusted from a hand estimate. For each event and clock hypothesis the matcher solves, in closed form,
for $f_0$ and a bounded linear source drift evaluated at **emission** time,
$f(t) = g(t)\,(b_0 + b_1 u(\tau))$ with $|b_1/b_0| \le 2\%$. Residuals use a pseudo-Huber loss (IRLS), every measured sample must
fall inside the trajectory's arrival-time support (no extrapolation, gaps longer than 20 s are holes), and **all events share one
recording clock**: a second flyover cannot get its own time shift. Ridges in one harmonic group must map to one flight.

### 3. Why one flyover is not enough
A straight, constant-velocity pass is mirror-symmetric about the vertical plane containing its track, and with an unknown clock the
observer can also slide along the track. A single flyover therefore yields a ridge of equally good locations; the UI draws the
mirror point and the ridges. A second, differently oriented flyover on the same clock breaks the symmetry.

---

## 📡 Where the trajectories come from (and how that is reported)

Every trajectory carries `data_source` and `altitude_basis`, and every result reports the source of the flights it used.

| `data_source` | When it is used |
|---|---|
| `uploaded` | Tracks you import (`/api/flights/import`) and send back as `trajectories`. Always wins when given. |
| `historical` | A configured historical provider covers the requested time window (OpenSky, see below). |
| `opensky_live` / `adsblol_live` | Live snapshot, **only** when no time window is given or it is within ~2 minutes of now. |
| `curated_demo` | Only with an explicit `scenario_id`, or `demo=true` near a curated scenario. |
| `synthetic` | Only with `demo=true`. Synthetic flights cannot locate a real recording. |

With no real data and no opt-in the API returns `status: "no_adsb_data"` (decision reason `no_adsb_data`) — nothing is invented.
The UI's "Allow demo / synthetic flights" box is off by default; the curated scenarios in the header pass `scenario_id` explicitly.

**ADS-B parsing.** Geometric (GNSS) altitude is preferred over barometric and the basis recorded; aircraft on the ground and states
missing position, speed, track, altitude or position time are skipped rather than defaulted.

**Live-snapshot limitations.** Live data is one state vector per aircraft projected linearly ±30 s around its own timestamp, the
nearest `DOPPLER_MAX_LIVE_AIRCRAFT` (default 8) aircraft only. It is only useful for a recording made in the last couple of minutes.

### Importing tracks (recommended for real recordings)
`POST /api/flights/import` (multipart `file`, optional `format`, `altitude_unit`, `speed_unit`, `vertical_rate_unit`, `timezone`,
`altitude_basis`) parses and validates — it does not store anything — and returns trajectories to pass as `trajectories` to
`/api/geolocate`. The same parsers run in the browser for the static build.

| Format | Notes |
|---|---|
| Generic CSV | Header row; `time`, `lat`, `lon`, altitude with its unit in the header (`alt_ft`, `alt_m`, `altitude (m)`), optional speed (`speed_kt`, `gs_mps`…), track, vertical rate, `icao24`/`flight_id`, `callsign`. An altitude column without a unit is rejected. |
| FlightRadar24 CSV | `Timestamp,UTC,Callsign,Position,Altitude,Speed,Direction` (feet, knots, barometric). |
| KML | `gx:Track` (`when` + `gx:coord`) or time-stamped Placemark points (FR24 KML). Metres. |
| GeoJSON | LineString with `properties.times`/`coordTimes`, or time-stamped Points grouped by `flight_id`/`icao24`. Metres. |
| readsb / tar1090 trace | `trace_full_<hex>.json` (gzip accepted); geometric altitude used when present. |

Times may be Unix seconds/ms or ISO 8601; ISO times **without an offset are rejected** unless `timezone` (`UTC`, `+02:00`) is
given. Coordinates, monotonic time, plausible altitude (catches feet-as-metres) and implied speed are checked, and a reported speed
that disagrees with the position-implied speed by more than 35 % is flagged as a likely kt-vs-m/s mix-up.
Fixtures for every format live in `tests/fixtures/tracks/`.

### Historical provider: OpenSky Network
Set `OPENSKY_CLIENT_ID` and `OPENSKY_CLIENT_SECRET` (OAuth2 client credentials — OpenSky no longer accepts basic auth). Then:
- windows within the last hour are sampled from `/states/all?time=` every 10 s (full state vectors, geometric altitude);
- windows up to 30 days old need `icao24_hints` (candidate aircraft) and use the experimental `/tracks/all` waypoints
  (barometric altitude, straight segments between waypoints);
- older windows are not available through OpenSky's REST API — import tracks instead.
Responses for past windows are cached on disk; HTTP 429 (credits exhausted) becomes a note, not a crash. Providers implement a small
interface (`app/adsb/providers.py`) so others can be added.

### Trajectory conditioning
Historical and uploaded tracks are resampled to 1 s with Gaussian-kernel **local quadratic** regression (smoothed position and
velocity in one step; little bias in turns), fused with reported ground speed/track/vertical rate (weight 0.7), and split at gaps
longer than 20 s (the support check still treats gaps as holes). On jittery 1 Hz tracks this halves velocity error from positions
alone and cuts it ~8× when reported velocity is available.

### Kinematic and time pre-filter
Before the search, flights are discarded only if they are impossible for **every** event group: no time overlap with a trusted
recording clock, or a reliable flyover fit (≥ 8 samples, RMSE ≤ 4 Hz, not at bounds) whose speed differs by more than
max(30 m/s, 25 %) or whose CPA slant range is shorter than the aircraft's lowest height above the observer minus
max(800 m, 60 %). Discards and their reasons are returned in `discarded_flights` and shown in the UI.

### Recording start time from file metadata
Uploads report `recording_metadata` (also `POST /api/audio/metadata`): BWF `bext` (`TimeReference` or `OriginationTime`), iXML, or
ffprobe `creation_time` (with an "end of recording" alternative, since many devices write the close time). BWF times have **no time
zone** and are read as UTC; the UI prefills the clock field and asks you to confirm. A trusted clock collapses the single-flyover
clock/position ambiguity.

---

## 🌬️ Atmosphere and altitude inputs

| Input | Effect |
|---|---|
| `temperature_c`, `relative_humidity_pct` | Speed of sound (Cramer 1993; RH defaults to 50 % when only temperature is given). |
| `wind_speed_mps`, `wind_from_deg` | Uniform horizontal wind in the propagation model (meteorological FROM direction). |
| `wind_uncertainty_mps` (default 5) | ± per-axis wind perturbation in the sensitivity analysis (covers unknown wind). |
| `qnh_hpa` | Converts barometric (pressure) altitudes to QNH altitudes with the ISA profile. |
| `observer_alt_m` | Microphone height in the trajectory altitude datum (above ground when terrain is used). |
| `observer_elevation_grid` / `dem_lookup` | Per-cell ground elevation from a user lat/lon grid, or an opt-in DEM lookup (OpenTopoData-compatible, cached, `DOPPLER_DEM_URL`); offline it falls back to flat ground. `elevation_datum_offset_m` adds e.g. the geoid undulation for GNSS (ellipsoidal) tracks. |

The sensitivity analysis widens its altitude perturbation with the altitude basis: ±150 m geometric, ±200 m barometric with QNH,
±300 m barometric without QNH or unknown basis. Regression tests show that ignoring wind, an unknown QNH or terrain biases the
location by 40–115 m on data generated with the effect, and that modelling it removes the bias.

---

## ⚡ Performance and long jobs

The matcher evaluates many (cell, flight, event, clock-start) rows in one vectorised pass, prunes clock starts that are already
impossible for an earlier event group, can evaluate the grid coarse-to-fine (default for grids ≥ 31), and runs grid chunks and the
independent refinements in a process pool (`DOPPLER_WORKERS`, default min(4, CPUs); results are bit-identical to serial).

`sfo_dual` at grid 41 on a 4-core machine: **18.8 s → 3.3 s** (5.7×), same location and decision.

| configuration | grid 25 | grid 41 |
|---|---|---|
| before (original code) | 10.2 s | 18.8 s |
| serial, full grid | 10.8 s | 17.1 s |
| serial, coarse-to-fine | 8.6 s | 10.4 s |
| 4 workers, full grid | 3.6 s | 5.3 s |
| 4 workers, coarse-to-fine | 2.7 s | 3.3 s |

Long requests can use the **jobs API**: `POST /api/jobs/geolocate` (same body as `/api/geolocate`) returns a `job_id`;
`GET /api/jobs/{id}` reports `status`, `stage` (`grid`, `refine`, `sensitivity`) and `progress`, and finally the same result
schema; `GET /api/jobs/{id}/events` streams progress as Server-Sent Events; `DELETE /api/jobs/{id}` cancels. Jobs live in memory
and expire after `DOPPLER_JOB_TTL_S`. In the browser-only build, discovery and matching run in a Web Worker.

---

## 📊 Benchmark and decision calibration

`scripts/benchmark.py` runs a seeded Monte Carlo over synthetic but physically independent scenarios: the true microphone up to
3 km from the search centre; 1–3 audible flyovers (random heading, CPA altitude 300–3000 m, CPA offset 0–3 km, 110–230 m/s,
climb/descent, 350–1200 Hz tone) plus 0–2 silent decoy aircraft; Doppler events from a **first-principles forward model**
(iterative travel time, frequency $= f_0\,d\tau/dt$; not the matcher's formula) with 0.2–2 Hz noise and, in half the runs, 3 %
gross outliers; a 0–10 m/s wind the matcher is **not** told; ADS-B altitude bias (σ 30 m geometric, 120 m barometric); air
temperature given to the matcher in only half the runs; a free clock or a trusted clock with σ 0.3 s error. Each run goes through
the same pre-filter, matcher, sensitivity analysis and decision as the API (grid 15 over 8 km).

```bash
python scripts/benchmark.py --runs 200 --seed 1 --out benchmarks/baseline_v1      # calibration set
python scripts/benchmark.py --rescore benchmarks/baseline_v1.json                  # re-score under the current policy
python scripts/benchmark.py --ci                                                   # the 6-run CI subset
```

**Results** (precision target 1000 m; a *false support* is a `supported` result more than 1000 m from the truth):

| set (200 runs each) | policy | supported | false supports | median error, all runs | supported: median / p95 / max error | truth within sensitivity radius |
|---|---|---|---|---|---|---|
| calibration (seed 1) | engineering-v1 (plain RMSE gate) | 22 (11 %) | 0 | 193 m | 74 / 368 / 446 m | 82 % |
| calibration (seed 1) | **engineering-v2 (robust RMSE gate)** | 52 (26 %) | **0** | 193 m | 59 / 342 / 755 m | 90 % |
| validation (seed 2, held out) | engineering-v1 | 17 (8 %) | 0 | 172 m | 74 / 343 / 380 m | 88 % |
| validation (seed 2, held out) | **engineering-v2** | 51 (26 %) | **0** | 172 m | 71 / 429 / 756 m | 88 % |

No single-flyover run is ever supported (mirror ambiguity). With two flyovers 42–43 % of runs are supported, with three 33–34 %;
a trusted clock helps (29–32 % vs 19–22 %). Full tables: `benchmarks/*.md`; raw runs: `benchmarks/*.json`.

**Calibration of `DecisionPolicy` (engineering-v2).** Thresholds were re-scored on the stored calibration runs and the one change
was then confirmed on the held-out validation set:

| threshold | value | evidence |
|---|---|---|
| residual gate | **robust RMSE ≤ 3 Hz** (was plain RMSE) | With the plain-RMSE gate, 105 of the 107 calibration runs containing 3 % gross outliers ended as `inconsistent_event_match` (only 1 was supported), and 46 of 93 runs without outliers did too. Gating on the pseudo-Huber-equivalent RMSE raised the supported fraction from 10 % to 26 % with **0 false supports in 400 runs** (95 % upper bound ≈ 2.9 %), the same as the plain gate. Plain RMSE remains available (`residual_metric: "rmse"`). |
| `max_rmse_hz` | 3.0 (unchanged) | 2 Hz loses supported runs; 4–6 Hz (plain) also showed no false supports here but was not adopted: the benchmark has no structured misfit (turns, reflections) that a looser gate would admit. |
| `min_independent_flyovers` | 2 (unchanged) | Allowing 1 produced false supports in both sets (4 of 29 and 1 of 24 supported runs). |
| `near_tie_hz`, `max_outlier_fraction` | 0.5 Hz, 0.25 (unchanged) | No effect anywhere in 0.25–2 Hz and 0.15–0.4 on these runs; kept as declared engineering values. |
| `precision_target_m` | 1000 m | Max supported error 756 m across both sets. |

The sensitivity radius contained the truth in 88–90 % of supported runs: it is an engineering envelope, **not** a calibrated
confidence interval. **These results apply only to the benchmarked conditions**; they say nothing about false-support rates on real
recordings with effects the benchmark does not model (see Limitations). The CI gate (`tests/test_benchmark_subset.py`) fails if the
false-support count of the fixed 6-run subset rises above `benchmarks/ci_baseline.json`.

---

## 🎙️ Live microphone mode

**Start live listening** asks for microphone access and then listens continuously. A microphone cannot be located from sound
alone. Live mode does what the rest of the app does with a recording, but continuously and with two advantages: the clock is known
(audio blocks are time-stamped as they arrive), and live ADS-B is collected for the whole session.

1. Audio is low-passed, resampled to 8 kHz and kept for the last 5 minutes. Each block is time-stamped from the AudioContext clock
   (`getOutputTimestamp`). A new anchor starts when audio was dropped, and slow audio-vs-system clock drift is followed.
2. Every ~15 s the latest 4 minutes are scanned for Doppler ridges in a Web Worker. A harmonic group is committed as a **flyover**
   once it has fully passed (ended ≥ 8 s before the window end, not clipped at its start), with UTC times. Later windows that see
   the same flyover again (time overlap and matching frequency) replace it only with a longer detection and never add a duplicate.
3. In server mode the live snapshot is polled every ~10 s around the search centre. The browser keeps each aircraft's *reported*
   state (not the ±30 s projection) and builds real multi-point tracks. They are sent as `trajectories`
   (`source_format: "live_poll"`), so the server validates and conditions them like uploaded tracks and reports them as `uploaded`.
4. After every new flyover, the most recent 8 flyovers are matched jointly on the shared clock (`recording_start_epoch` = the
   first flyover's start, `clock_tolerance_s` from the form). After a **supported** result, the search re-centres on the estimate
   and shrinks to 4× the sensitivity radius (min 2 km) for a finer grid, and runs once more. An inconclusive result widens the
   search back towards the original area, so a wrong lock-in can recover. Each estimate is listed and drawn as a trail on the map.

You still give a rough starting area (city level: *Use map centre*, shift-click the map, or the fields), because ADS-B has to be
fetched for *some* area. One flyover is ambiguous (mirror symmetry). Two or more from different directions are needed, and each
further flyover refines the estimate. Analysis and polling are driven by incoming audio blocks rather than timers, so they keep
running in a background tab, but the tab must stay open. **Export live data** saves the flyovers, collected tracks and estimates
as JSON. Microphone access needs HTTPS or `http://localhost`. The static GitHub Pages build has no live ADS-B: it still detects
and time-stamps flyovers, and can locate them against imported tracks.

## 🖥️ System Features

- **Audio**: drag-and-drop WAV/MP3/FLAC/OGG/M4A (non-WAV server-side decoding needs ffmpeg), STFT spectrogram, palettes, playback.
- **Ridge discovery**: whole-recording scan, stationary-tone removal, bounded-slope Viterbi, harmonic grouping, flyover fit per
  ridge; server or browser (Web Worker). Ridges are acoustic proposals, not verified aircraft.
- **Tracing**: mouse, touch and pen (Pointer Events), snap-to-peak, multiple events, save/load a trace session (events + settings +
  imported tracks) as JSON.
- **Results**: supported/inconclusive banner with every evidence check, per-event diagnostics, candidate-flight table with
  data-source badges, kinematic consistency and pre-filter reasons, heatmap, alternatives, sensitivity circle, single-flyover
  mirror ambiguity and ridges, and export as JSON or GeoJSON.

---

## 🚀 Quickstart

Python **3.12 or 3.13** (what CI tests; the pinned scientific stack needs ≥ 3.12).

```bash
git clone https://github.com/awne8886/doppler.git
cd doppler
pip install -r requirements.txt -c constraints.txt
python3 run.py            # http://localhost:8000
```

Docker (includes ffmpeg):

```bash
docker build -t doppler .
docker run -p 8000:8000 -v doppler-data:/data doppler
```

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `DOPPLER_DATA_DIR` | `./var` | Uploads and caches (never inside `app/static`). |
| `DOPPLER_MAX_UPLOAD_MB` | 50 | Upload size cap (HTTP 413 above it). |
| `DOPPLER_UPLOAD_TTL_S` | 21600 | Uploaded audio is deleted after this; served only by unguessable name. |
| `DOPPLER_CORS_ORIGINS` | `*` | Comma-separated origins; credentials are always off; empty disables CORS. |
| `DOPPLER_WORKERS` | min(4, CPUs) | Matcher worker processes (1 = serial). |
| `DOPPLER_MAX_JOBS`, `DOPPLER_JOB_TTL_S` | 2, 3600 | Concurrent async jobs; how long finished jobs are kept. |
| `DOPPLER_MAX_LIVE_AIRCRAFT` | 8 | Live-snapshot aircraft kept (nearest first). |
| `OPENSKY_CLIENT_ID`, `OPENSKY_CLIENT_SECRET` | – | Enables the OpenSky historical provider (OAuth2). |
| `DOPPLER_DEM_URL` | OpenTopoData SRTM 90 m | DEM service for `dem_lookup`; empty disables it. |
| `DOPPLER_CARTO_KEY` | – | Basemap key served via `/js/config.js` (restrict it by HTTP referrer). |

---

## 🧪 Development and testing

```bash
pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt
ruff check app tests scripts
pytest -q                                 # Python tests, incl. the benchmark subset; no network access
node --test "tests/js/*.test.mjs"         # Python/JS parity + UI helpers (no npm dependencies)
python scripts/build_pages.py --check     # docs/ must equal app/static
```

- Tests never touch the network: `tests/conftest.py` makes every `requests` call fail; providers are tested with mocks and
  recorded fixtures.
- **Parity**: `scripts/make_parity_fixtures.py` writes golden outputs from Python; the Node tests load the browser scripts in a `vm`
  context and check the flyover fit, `fit_event`, wind propagation, speed of sound, conditioning, pre-filter, decision layer, track
  import and an end-to-end match (every grid loss within 1e-6) against them.
- **GitHub Pages** serves `/docs`, a byte-identical copy of `app/static` produced by `python scripts/build_pages.py`; CI fails if
  it is stale.
- CI (`.github/workflows/ci.yml`): ruff + pytest on Python 3.12 and 3.13, the benchmark subset, the Node tests and the Pages check.

---

## 📡 API Reference

| Endpoint | Description |
|---|---|
| `GET /api/health` | Health check. |
| `GET /api/scenarios` | Curated demo scenarios and sample audio. |
| `POST /api/flights/search` | Candidate trajectories (`status`, `data_source`, `notes`, `limitations`; demo data only with `scenario_id` / `demo`). |
| `POST /api/flights/import` | Parse and validate uploaded tracks (CSV, FR24 CSV, KML, GeoJSON, readsb trace). |
| `POST /api/geolocate` | Synchronous joint matching, refinement, sensitivity and decision. |
| `POST /api/jobs/geolocate` · `GET/DELETE /api/jobs/{id}` · `GET /api/jobs/{id}/events` | Asynchronous geolocation with progress (polling or SSE). |
| `POST /api/audio/discover` | Automatic ridge discovery (`file`, `sample_id` or `saved_as`). |
| `POST /api/audio/upload` · `GET /api/audio/uploads/{name}` | Upload (size-capped, TTL) and fetch by unguessable name; returns `recording_metadata`. |
| `POST /api/audio/metadata` | Recording start time suggestion from file metadata. |
| `POST /api/flyover/fit` | Constant-velocity flyover fit of one traced curve. |

`/api/geolocate` accepts the legacy `traced_curve` + `f0` or `events: [{id, group_id, traced_curve, flight_id?}]`, plus
`trajectories`, `scenario_id`, `demo`, `icao24_hints`, `time_window_start/end`, `recording_start_epoch` + `clock_tolerance_s`,
`fit_f0`, the atmosphere/altitude inputs above, `grid_resolution`, `grid_strategy`, `sensitivity`, `condition_tracks`, `prefilter`,
`max_rmse_hz`, `min_independent_flyovers` and `precision_target_m`. Responses add `data_source`, `data_notes`,
`data_limitations`, `discarded_flights`, `atmosphere`, `decision`, `alternatives`, `sensitivity`, per-event `events`,
`flyover_fits` and `evaluated_flights` (with data source, altitude basis, kinematic consistency and conditioning).

---

## ⚠️ Limitations

- Results are **not calibrated probabilities**. Thresholds are engineering choices checked on the synthetic benchmark above; outside
  the benchmarked conditions (real recordings, reflections, turning or accelerating aircraft, wind shear, non-uniform temperature,
  ridge-tracking errors) false-support rates are unknown.
- The propagation model is straight-line in a uniform atmosphere with uniform horizontal wind; no refraction, ground reflection
  or wind shear.
- Live ADS-B is a single projected snapshot; OpenSky's REST API cannot list aircraft in an area more than one hour back, and its
  `/tracks` endpoint is experimental with sparse barometric waypoints.
- BWF metadata has no time zone; `creation_time` may mark the end of a recording. Metadata clocks always need confirmation.
- DEM heights are above mean sea level while GNSS altitudes are ellipsoidal; supply the geoid undulation when it matters.
- Jobs are in-memory and per process; uploads are deleted after their TTL.

---

## 🙏 Acknowledgements

The ridge discovery, fitted-source matching with a locked recording clock, multi-flyover evidence and inconclusive-outcome policy are inspired by the methodology documented in [DOPPLERGEIST](https://github.com/1337isnot1337/DOPPLERGEIST) by Cinna Davis, itself an independent reconstruction of an approach shown in a video by Colsto. DOPPLERGEIST is licensed under PolyForm Noncommercial 1.0.0; **no DOPPLERGEIST source code is included here**. The implementations in this repository were written independently from its published method descriptions and remain under this project's own license.
