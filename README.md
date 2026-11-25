# Doppler // Acoustic Aircraft Geolocator

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-06b6d4.svg)](https://awne8886.github.io/doppler/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.124+-green.svg)](https://fastapi.tiangolo.com/)

**Doppler** is a full-stack acoustic surveillance and inverse physics application that geolocates audio recordings of passing airplanes by matching their acoustic Doppler frequency curves with real-time and historical ADS-B flight tracking data.

🔗 **Live GitHub Pages Web App**: [https://awne8886.github.io/doppler/](https://awne8886.github.io/doppler/)

---

## ✈️ Core Architecture & Physics

```
   [ Aircraft Trajectory (t_emit, x, y, z, v) ]
                       \
                        \  3D Slant Distance d(t) = ||p_plane(t) - p_obs||
                         \  Acoustic Propagation Delay tau = d(t) / c
                          \
                           v
          [ Observer Microphone @ (lat, lon) ]
                           |
             Recorded Acoustic Audio (WAV/MP3)
                           |
                   [ Spectrogram STFT ]
                           |
            Traced Frequency S-Curve f(t_obs)
                           |
             [ Least-Squares Error Optimization ]
                           |
            [ Heatmap Geolocation Distribution ]
```

### 1. Doppler Frequency Shift
For an aircraft traveling with 3D velocity vector $\vec{v}(t)$ emitting sound at fundamental frequency $f_0$, the radial velocity $v_r$ towards a candidate observer at $\vec{p}_{obs}$ is:

$$v_r(t_{emit}) = -\frac{(\vec{p}_{plane}(t_{emit}) - \vec{p}_{obs}) \cdot \vec{v}(t_{emit})}{\|\vec{p}_{plane}(t_{emit}) - \vec{p}_{obs}\|}$$

The theoretical observed acoustic frequency at the observer is:

$$f(t_{obs}) = f_0 \left( \frac{c}{c - v_r(t_{emit})} \right)$$

where $c \approx 343.2 \text{ m/s}$ is the speed of sound in air.

### 2. Acoustic Propagation Delay
Sound takes time to travel through the atmosphere. Acoustic waves emitted at timestamp $t_{emit}$ arrive at the microphone at observer time $t_{obs}$:

$$t_{obs} = t_{emit} + \frac{d(t_{emit})}{c} = t_{emit} + \frac{\sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}}{c}$$

where $\Delta z = h_{plane} - h_{obs}$ represents the aircraft's 3D barometric/geometric altitude.

### 3. Least Squares Grid Search & Match Scoring
The candidate observer space is discretized into a 2D spatial grid over the user's geographic search radius ($R_{km}$). For each candidate location:

$$\text{MSE} = \min_{\Delta \tau} \frac{1}{M} \sum_{i=1}^M \left( f_{model}(t_i + \Delta \tau) - f_{traced, i} \right)^2$$

$$\text{Score} = 100 \times \exp\left( -\frac{\text{RMSE}^2}{2\sigma^2} \right)$$

The resulting probability density is rendered as an interactive Leaflet heatmap highlighting the most probable microphone location.

### 4. Fitted Rest Frequency, Robust Loss & Shared Clock
The rest frequency is not trusted from a hand estimate. For each event and each clock hypothesis the matcher solves, in closed form, for $f_0$ and a bounded linear source drift evaluated at **emission** time:

$$f(t) = g(t)\,\big(b_0 + b_1\,u(\tau)\big), \qquad g = \frac{c}{c + u_r},\quad |b_1/b_0| \le 2\%$$

Residuals are scored with a pseudo-Huber loss (iteratively reweighted), every measured sample must fall inside the trajectory's arrival-time support (no extrapolation), and **all events share one recording clock**: a second flyover cannot be given its own time shift. Ridges in the same harmonic group must map to the same flight.

### 5. Why one flyover is not enough
A straight, constant-velocity pass is mirror-symmetric about the vertical plane containing the flight track, and with an unknown clock the observer can also slide along the track. A single flyover therefore yields a ridge of equally good locations. A second, differently oriented flyover on the same clock breaks the symmetry. The app reports this explicitly instead of presenting the best grid cell as an answer.

---

## 🖥️ System Features

### 1. Audio Analysis & Interactive Spectrogram
- **Audio Ingestion**: Drag-and-drop or file upload for MP3, WAV, FLAC, and OGG recordings.
- **Web Audio API STFT**: High-resolution Fourier Transform (2048-point Hanning window) rendered in real-time on HTML5 Canvas.
- **Color Palettes**: Turbo, Inferno, Magma, and Classic Jet palettes with adjustable dynamic range.
- **Synced Playback**: Scrubbable playhead cursor synced with audio playback.
- **Doppler UI Tracing Tool**:
  - Interactive click-and-drag tracing tool over the spectrogram.
  - **Snap-to-Peak**: Sub-bin quadratic peak locking that snaps drawn points to the highest local spectral energy.
  - Automatic extraction of fundamental frequency $f_0$ from the S-curve inflection point.

### 1b. Automatic Doppler Ridge Discovery
- **Whole-recording scan**: no manual time window or frequency band. Spectrogram → stationary-tone removal (per-bin median) → robust per-frame noise normalisation → **bounded-slope Viterbi** ridge tracking → threshold segmentation → mask and repeat.
- **Harmonic grouping**: ridges that overlap in time with a stable small-integer ratio (×2, ×3, 3/2 …) are grouped as one flyover.
- **Constant-velocity flyover fit** per ridge: exact retarded-time solution giving rest frequency, aircraft speed, CPA slant range and CPA time (plus uncertainties). Used to flag non-Doppler-like ridges, align clocks, and check kinematic consistency with ADS-B tracks.
- Runs on the server (`/api/audio/discover`, WAV natively, other formats via `ffmpeg`) or entirely in the browser.

### 2. ADS-B Flight Tracking & Physics Engine
- **OpenSky Network Integration**: Queries live state vectors and historical tracks within spatial bounding boxes.
- **ADSB.lol fallback**: Open, keyless community ADS-B feed used when OpenSky returns nothing (geometric altitude preferred, basis recorded).
- **Realistic Kinematic Interpolator**: Continuous position and velocity vector calculation using spherical-to-Cartesian geodesic mapping.
- **Curated Real-World Flyover Scenarios**:
  - **SFO Approach**: United Airlines UAL421 (Boeing 737-800) passing over Foster City / San Mateo.
  - **LHR Approach**: British Airways BAW282 (Airbus A350-900) crossing Richmond / Twickenham.
  - **JFK Departure**: Delta Air Lines DAL410 (Airbus A321) climbing past Queens / Long Island.
  - **Two flyovers, one clock** (default): UAL1714 and SWA2286 cross Foster City 34 s apart in a single 75 s recording, with an unrelated ASA331 decoy track. Resolves to the true microphone position.
- **Synthetic Flyover WAV Generator**: Synthesizes authentic flyover audio obeying Doppler shifts, engine harmonics, and $1/r$ acoustic attenuation.

### 2b. Joint Matching, Refinement & Evidence Checks
- **Joint multi-event search** over the candidate grid with one shared recording clock (free, or a trusted UTC window ± tolerance).
- **Continuous refinement** (Nelder-Mead) of every local optimum; competing optima are kept as **alternatives**.
- **Sensitivity analysis**: re-refines under sound speed ±5 m/s, aircraft altitude ±150 m and frequency offset/tilt; the maximum displacement is reported as an engineering radius (uncalibrated).
- **Decision layer** returns `supported` or `inconclusive` with a machine-readable reason: `ambiguous_locations`, `insufficient_independent_events`, `inconsistent_event_match`, `insufficient_track_support`, `source_drift_unresolved`, `refinement_boundary_limited`, `location_uncertainty_too_large`, … Thresholds (≤3 Hz RMSE, ≤25 % outliers, ≥95 % track support, ≥2 independent flyovers, 0.5 Hz near-tie at >1 km) are predeclared engineering choices, **not calibrated probabilities**.

### 3. Mapping, Heatmaps & Telemetry
- **CARTO Maps Integration**: Integrated CARTO basemaps with API key support, featuring CARTO Voyager, CARTO Dark Matter, and CARTO Positron tile layers with an interactive layer switcher.
- **Dynamic Acoustic Heatmap**: Renders heat map distribution with click-to-inspect candidate points.
- **Hottest Point Pinpoint**: Highlights the best-ranked position, labelled as *supported* or as a *hypothesis* depending on the decision, with the sensitivity radius and near-tied alternatives drawn around it.
- **Curve Comparison HUD**: Real-time canvas chart overlaying user-traced acoustic curve vs. modeled 3D theoretical Doppler curve.

---

## 🚀 Quickstart

### Prerequisites
- Python 3.8+
- Modern Web Browser (Chrome, Firefox, Safari, Edge)

### Installation
```bash
# Clone the repository
git clone https://github.com/awne8886/doppler.git
cd doppler

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
python3 run.py
```
Open your browser to `http://localhost:8000`.

---

## 🧪 Testing

Run the physics, API, discovery and matching test suites:
```bash
python3 -m pytest -q
```

The discovery/matching suite includes negative controls: noise-only audio must yield no ridges, a single flyover must be reported inconclusive, a 6 s clock error on one of two flyovers must not be absorbed, and samples outside trajectory support must not match.

Regenerate the curated sample audio (e.g. the two-flyover recording):
```bash
python3 -c "from app.audio.generator import generate_all_scenario_samples as g; g(only=['sfo_dual'])"
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/health` | GET | Health check status |
| `GET /api/scenarios` | GET | List curated flyover scenarios & sample audio |
| `POST /api/flights/search` | POST | Fetch or generate ADS-B trajectories in search sector |
| `POST /api/geolocate` | POST | Joint multi-event matching with shared clock, refinement, sensitivity and decision |
| `POST /api/audio/discover` | POST | Automatic ridge discovery (`file`, `sample_id` or `saved_as`) |
| `POST /api/flyover/fit` | POST | Constant-velocity flyover fit of one traced curve |
| `POST /api/audio/upload` | POST | Upload user audio recording (MP3/WAV) |

`/api/geolocate` accepts either the legacy `traced_curve` + `f0`, or `events: [{id, group_id, traced_curve, flight_id?}]` with options `fit_f0`, `recording_start_epoch` + `clock_tolerance_s`, `temperature_c`, `observer_alt_m`, `sensitivity`, `max_rmse_hz`, `min_independent_flyovers` and `precision_target_m`. The response adds `decision`, `alternatives`, `sensitivity`, per-event `events` diagnostics and `flyover_fits`.

---

## 🙏 Acknowledgements

The ridge discovery, fitted-source matching with a locked recording clock, multi-flyover evidence and inconclusive-outcome policy are inspired by the methodology documented in [DOPPLERGEIST](https://github.com/1337isnot1337/DOPPLERGEIST) by Cinna Davis, itself an independent reconstruction of an approach shown in a video by Colsto. DOPPLERGEIST is licensed under PolyForm Noncommercial 1.0.0; **no DOPPLERGEIST source code is included here**. The implementations in this repository were written independently from its published method descriptions and remain under this project's own license.
