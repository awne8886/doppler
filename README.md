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

### 2. ADS-B Flight Tracking & Physics Engine
- **OpenSky Network Integration**: Queries live state vectors and historical tracks within spatial bounding boxes.
- **Realistic Kinematic Interpolator**: Continuous position and velocity vector calculation using spherical-to-Cartesian geodesic mapping.
- **Curated Real-World Flyover Scenarios**:
  - **SFO Approach**: United Airlines UAL421 (Boeing 737-800) passing over Foster City / San Mateo.
  - **LHR Approach**: British Airways BAW282 (Airbus A350-900) crossing Richmond / Twickenham.
  - **JFK Departure**: Delta Air Lines DAL410 (Airbus A321) climbing past Queens / Long Island.
- **Synthetic Flyover WAV Generator**: Synthesizes authentic flyover audio obeying Doppler shifts, engine harmonics, and $1/r$ acoustic attenuation.

### 3. Mapping, Heatmaps & Telemetry
- **Leaflet.js Dark Mode Map**: Renders search sector boundary, aircraft flight trajectories, and candidate observer grid.
- **Dynamic Acoustic Heatmap**: Renders heat map distribution with click-to-inspect candidate points.
- **Hottest Point Pinpoint**: Highlights microphone ground truth with pulsing marker and coordinates popup.
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

Run physics engine and API test suites:
```bash
python3 tests/test_physics.py
python3 tests/test_api.py
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `GET /api/health` | GET | Health check status |
| `GET /api/scenarios` | GET | List curated flyover scenarios & sample audio |
| `POST /api/flights/search` | POST | Fetch or generate ADS-B trajectories in search sector |
| `POST /api/geolocate` | POST | Run Least Squares Doppler physics grid evaluation |
| `POST /api/audio/upload` | POST | Upload user audio recording (MP3/WAV) |
