"""
Acoustic flyover audio generator for Doppler geolocator testing.
Synthesizes WAV files with realistic Doppler-shifted engine tones, harmonics,
inverse-square amplitude decay, and atmospheric sound propagation.
"""

import wave
import os
from typing import Any, Dict, List, Tuple
import numpy as np

from app.physics.doppler_engine import DopplerEngine
from app.adsb.sample_flights import get_curated_scenarios


def synthesize_flyover_wav(
    trajectory: Dict[str, Any],
    observer_coords: Tuple[float, float],
    output_wav_path: str,
    duration_sec: float = 30.0,
    sample_rate: int = 44100,
    f0: float = 800.0,
) -> str:
    """
    Synthesize an audio WAV file of a flight passing over an observer,
    strictly obeying Doppler shift and 1/r acoustic attenuation.
    """
    engine = DopplerEngine()
    obs_lat, obs_lon = observer_coords

    # Extract trajectory
    t_emit = np.array(trajectory["timestamps"], dtype=np.float64)
    # Re-zero timestamps to 0..T
    t_emit = t_emit - t_emit[0]

    plane_lats = np.array(trajectory["latitudes"], dtype=np.float64)
    plane_lons = np.array(trajectory["longitudes"], dtype=np.float64)
    plane_alts = np.array(trajectory["altitudes"], dtype=np.float64)

    # Convert to Cartesian relative to observer
    plane_x, plane_y, plane_z = engine.geodetic_to_cartesian(
        plane_lats, plane_lons, plane_alts, obs_lat, obs_lon, 0.0
    )
    plane_pos = np.stack([plane_x, plane_y, plane_z], axis=1)

    # Calculate velocity vectors
    vx = np.gradient(plane_x, t_emit)
    vy = np.gradient(plane_y, t_emit)
    vz = np.gradient(plane_z, t_emit)
    plane_vel = np.stack([vx, vy, vz], axis=1)

    obs_pos = np.array([0.0, 0.0, 0.0])

    # Compute continuous observer times and frequencies
    t_obs, f_obs = engine.compute_doppler_curve(t_emit, plane_pos, plane_vel, obs_pos, f0)
    d_slant, _ = engine.compute_radial_velocity(plane_pos, plane_vel, obs_pos)

    # Find CPA time at observer
    idx_cpa = np.argmin(d_slant)
    cpa_t_obs = t_obs[idx_cpa]

    # Select audio time window centered on CPA (e.g. CPA +/- duration/2)
    t_start = max(0.0, cpa_t_obs - (duration_sec / 2.0))
    t_end = t_start + duration_sec

    # Audio synthesis time samples
    num_samples = int(duration_sec * sample_rate)
    t_audio = np.linspace(t_start, t_end, num_samples)

    # Interpolate observed frequency and slant distance at audio timestamps
    sort_idx = np.argsort(t_obs)
    t_obs_sorted = t_obs[sort_idx]
    f_obs_sorted = f_obs[sort_idx]
    d_slant_sorted = d_slant[sort_idx]

    f_instant = np.interp(t_audio, t_obs_sorted, f_obs_sorted)
    d_instant = np.interp(t_audio, t_obs_sorted, d_slant_sorted)

    # Compute phase as cumulative integral of instantaneous frequency
    # phi(t) = 2 * pi * sum(f * dt)
    dt_sample = 1.0 / sample_rate
    phase = 2.0 * np.pi * np.cumsum(f_instant) * dt_sample

    # Amplitude modulation: inverse square law with atmospheric absorption and ground decay
    # normalize amplitude so CPA is ~0.8 max
    min_dist = np.min(d_instant)
    amplitude = (min_dist / np.maximum(d_instant, 100.0)) ** 1.1

    # Sound tone synthesis: fundamental f0 + harmonics (blade pass / turbine tone)
    # Fundamental
    tone = np.sin(phase)
    # 2nd harmonic
    tone += 0.35 * np.sin(2.0 * phase)
    # 3rd harmonic
    tone += 0.15 * np.sin(3.0 * phase)
    # Add ambient low rumble / broad jet noise
    noise = np.random.normal(0, 0.08, num_samples)
    # Simple lowpass filter on noise
    noise = np.convolve(noise, np.ones(15) / 15.0, mode="same")

    signal = (tone * 0.7 + noise) * amplitude
    # Apply fade in/out envelope at boundaries to prevent clicks
    fade_len = int(sample_rate * 0.5)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    signal[:fade_len] *= fade_in
    signal[-fade_len:] *= fade_out

    # Normalize to 16-bit PCM
    signal = signal / (np.max(np.abs(signal)) + 1e-6)
    int_samples = (signal * 32760.0).astype(np.int16)

    # Write WAV file
    with wave.open(output_wav_path, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())

    return output_wav_path


def synthesize_multi_flyover_wav(
    trajectories: List[Dict[str, Any]],
    observer_coords: Tuple[float, float],
    output_wav_path: str,
    recording_start_epoch: float,
    duration_sec: float,
    sample_rate: int = 16000,
    reference_distance_m: float = 1500.0,
    noise_level: float = 0.05,
    seed: int = 7,
) -> str:
    """
    Synthesize one recording containing several flyovers on a shared absolute clock.

    Audio sample 0 corresponds to ``recording_start_epoch``.  Each aircraft's tone
    (fundamental ``f0_hz`` plus two harmonics) follows its exact emission-time
    Doppler curve and is silent outside its trajectory's arrival-time support.
    """
    engine = DopplerEngine()
    rng = np.random.default_rng(seed)
    obs_lat, obs_lon = observer_coords
    n = int(duration_sec * sample_rate)
    t_abs = recording_start_epoch + np.arange(n) / sample_rate
    mix = np.zeros(n, dtype=np.float64)

    for traj in trajectories:
        t_emit = np.asarray(traj["timestamps"], dtype=np.float64)
        px, py, pz = engine.geodetic_to_cartesian(
            np.asarray(traj["latitudes"]), np.asarray(traj["longitudes"]), np.asarray(traj["altitudes"]),
            obs_lat, obs_lon, 0.0,
        )
        pos = np.stack([px, py, pz], axis=1)
        vel = np.stack([np.gradient(px, t_emit), np.gradient(py, t_emit), np.gradient(pz, t_emit)], axis=1)
        f0 = float(traj.get("f0_hz", 800.0))
        t_obs, f_obs = engine.compute_doppler_curve(t_emit, pos, vel, np.zeros(3), f0)
        d, _ = engine.compute_radial_velocity(pos, vel, np.zeros(3))
        order = np.argsort(t_obs)
        t_obs, f_obs, d = t_obs[order], f_obs[order], d[order]
        live = (t_abs >= t_obs[0]) & (t_abs <= t_obs[-1])
        if not np.any(live):
            continue
        f_inst = np.interp(t_abs, t_obs, f_obs)
        d_inst = np.interp(t_abs, t_obs, d)
        phase = 2.0 * np.pi * np.cumsum(np.where(live, f_inst, 0.0)) / sample_rate
        amp = np.where(live, (reference_distance_m / np.maximum(d_inst, 100.0)) ** 1.1, 0.0)
        # short taper at support edges to avoid clicks
        edge = np.convolve(live.astype(float), np.ones(int(0.2 * sample_rate)) / (0.2 * sample_rate), mode="same")
        tone = np.sin(phase) + 0.35 * np.sin(2.0 * phase) + 0.15 * np.sin(3.0 * phase)
        mix += 0.7 * tone * amp * np.minimum(edge, 1.0)

    noise = rng.normal(0.0, noise_level, n)
    noise = np.convolve(noise, np.ones(9) / 9.0, mode="same")
    signal = mix + noise
    fade = int(0.3 * sample_rate)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    with wave.open(output_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((signal * 32760.0).astype(np.int16).tobytes())
    return output_wav_path


def emission_arrivals(trajectory: Dict[str, Any], observer_coords: Tuple[float, float],
                      wind_en: Tuple[float, float] = (0.0, 0.0), speed_of_sound: float = 343.2,
                      observer_alt_m: float = 0.0, dt: float = 0.02) -> Dict[str, np.ndarray]:
    """
    First-principles forward model, independent of the matcher's Doppler formula.

    The trajectory is densified (linear interpolation in local ENU, exact for
    straight constant-velocity tracks).  For every emission time tau the
    travel time T through air moving uniformly with ``wind_en`` (east, north
    toward) is found by fixed-point iteration of ``c T = |R - W T|`` (sound
    moves at c relative to the air; the observer drifts by -W T in the air
    frame).  The received frequency factor is the numerical derivative
    ``d tau / d t_arrival`` - no Doppler equation is used, so this checks the
    matcher's analytic model rather than repeating it.
    """
    engine = DopplerEngine()
    t = np.asarray(trajectory["timestamps"], dtype=np.float64)
    x, y, z = engine.geodetic_to_cartesian(np.asarray(trajectory["latitudes"]), np.asarray(trajectory["longitudes"]),
                                           np.asarray(trajectory["altitudes"]), observer_coords[0], observer_coords[1], 0.0)
    tau = t[0] + dt * np.arange(int(np.floor((t[-1] - t[0]) / dt + 1e-9)) + 1)
    S = np.stack([np.interp(tau, t, x), np.interp(tau, t, y), np.interp(tau, t, z)], axis=1)
    R = np.array([0.0, 0.0, observer_alt_m]) - S
    W = np.array([wind_en[0], wind_en[1], 0.0])
    T = np.linalg.norm(R, axis=1) / speed_of_sound
    for _ in range(12):
        T = np.linalg.norm(R - np.outer(T, W), axis=1) / speed_of_sound
    t_arr = tau + T
    factor = np.gradient(tau, t_arr)
    return {"tau": tau, "t": t_arr, "factor": factor, "d": np.linalg.norm(R, axis=1)}


def simulate_events(trajectories: List[Dict[str, Any]], observer_coords: Tuple[float, float],
                    recording_start_epoch: float, duration_sec: float, wind_en: Tuple[float, float] = (0.0, 0.0),
                    speed_of_sound: float = 343.2, observer_alt_m: float = 0.0, step: float = 0.25,
                    min_duration_s: float = 8.0, margin_s: float = 1.0) -> List[Dict[str, Any]]:
    """Noise-free traced events (recording-relative time) for every flight audible in the recording."""
    events = []
    for tr in trajectories:
        a = emission_arrivals(tr, observer_coords, wind_en, speed_of_sound, observer_alt_m)
        lo = max(a["t"][2], recording_start_epoch + margin_s)
        hi = min(a["t"][-3], recording_start_epoch + duration_sec - margin_s)
        if hi - lo < min_duration_s:
            continue
        t_abs = np.arange(lo, hi, step)
        f = np.interp(t_abs, a["t"], a["factor"]) * float(tr.get("f0_hz", 800.0))
        events.append({"id": tr["flight_id"], "traced_curve": [
            {"time": float(tt - recording_start_epoch), "freq": float(ff)} for tt, ff in zip(t_abs, f)]})
    return events


def synthesize_emission_phase_wav(trajectories: List[Dict[str, Any]], observer_coords: Tuple[float, float],
                                  output_wav_path: str, recording_start_epoch: float, duration_sec: float,
                                  wind_en: Tuple[float, float] = (0.0, 0.0), speed_of_sound: float = 343.2,
                                  observer_alt_m: float = 0.0, sample_rate: int = 16000,
                                  reference_distance_m: float = 1500.0, noise_level: float = 0.05, seed: int = 7,
                                  harmonics: Tuple[float, ...] = (1.0, 0.35, 0.15)) -> str:
    """
    Audio whose phase is the source phase at emission, ``2 pi f0 tau(t)``,
    with tau(t) from :func:`emission_arrivals` (wind and observer height
    included).  The Doppler shift emerges from the propagation geometry.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_sec * sample_rate)
    t_abs = recording_start_epoch + np.arange(n) / sample_rate
    mix = np.zeros(n)
    for tr in trajectories:
        a = emission_arrivals(tr, observer_coords, wind_en, speed_of_sound, observer_alt_m)
        live = (t_abs >= a["t"][0]) & (t_abs <= a["t"][-1])
        if not np.any(live):
            continue
        tau = np.interp(t_abs, a["t"], a["tau"])
        phase = 2.0 * np.pi * float(tr.get("f0_hz", 800.0)) * (tau - a["tau"][0])
        d = np.interp(t_abs, a["t"], a["d"])
        amp = np.where(live, (reference_distance_m / np.maximum(d, 100.0)) ** 1.1, 0.0)
        edge = np.convolve(live.astype(float), np.ones(int(0.2 * sample_rate)) / (0.2 * sample_rate), mode="same")
        tone = sum(h * np.sin((k + 1) * phase) for k, h in enumerate(harmonics))
        mix += 0.7 * tone * amp * np.minimum(edge, 1.0)
    noise = np.convolve(rng.normal(0.0, noise_level, n), np.ones(9) / 9.0, mode="same")
    signal = mix + noise
    fade = int(0.3 * sample_rate)
    signal[:fade] *= np.linspace(0, 1, fade)
    signal[-fade:] *= np.linspace(1, 0, fade)
    signal = signal / (np.max(np.abs(signal)) + 1e-9)
    with wave.open(output_wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((signal * 32760.0).astype(np.int16).tobytes())
    return output_wav_path


SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "samples")


def generate_all_scenario_samples(output_dir: str = SAMPLES_DIR, only: List[str] = None):
    """Generate sample WAV audio files for all curated scenarios."""
    scenarios = get_curated_scenarios()
    os.makedirs(output_dir, exist_ok=True)
    for s_id, s_data in scenarios.items():
        if only and s_id not in only:
            continue
        out_path = os.path.join(output_dir, f"{s_id}.wav")
        if len(s_data.get("flights", [])) > 1:
            gt = s_data["search_center"]
            synthesize_multi_flyover_wav(
                trajectories=s_data["flights"],
                observer_coords=(gt["lat"], gt["lon"]),
                output_wav_path=out_path,
                recording_start_epoch=s_data["recording_start_epoch"],
                duration_sec=s_data["audio_duration_sec"],
            )
            continue
        gt_lat = s_data["trajectory"]["ground_truth_mic"]["lat"]
        gt_lon = s_data["trajectory"]["ground_truth_mic"]["lon"]
        f0 = s_data["f0_target"]
        synthesize_flyover_wav(
            trajectory=s_data["trajectory"],
            observer_coords=(gt_lat, gt_lon),
            output_wav_path=out_path,
            duration_sec=25.0,
            f0=f0,
        )
