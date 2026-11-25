"""
Automatic Doppler ridge discovery.

Scans a whole recording for moving tonal ridges (the Doppler S-curves of
passing aircraft) without the user choosing a time window or frequency band:

1. Decode to mono and resample to a modest analysis rate.
2. Short-time spectrogram in dB.
3. Contrast normalisation: subtract each frequency bin's median over time
   (removes stationary hum / tonal baselines), then robust z-score each frame
   against its own median and MAD (normalises the broadband noise floor).
4. Bounded-slope Viterbi tracking: the best path through the contrast image
   whose frequency can change by at most ``max_slope_hz_s``.  A small per-bin
   jump penalty keeps paths smooth.
5. The path is segmented where its contrast clears a detection threshold;
   accepted segments are masked out and the search repeats to find further
   ridges (other aircraft or harmonics).
6. Ridges that overlap in time with a stable small-integer frequency ratio are
   grouped into one harmonic group (one physical flyover).

A detected ridge is acoustic evidence, not proof that the source is an
aircraft: maneuvering aircraft, sirens, birds or changing engine RPM can all
produce moving tones.  Each ridge therefore carries a constant-velocity
flyover fit so downstream steps can judge how Doppler-like it is.
"""

from __future__ import annotations

import io
import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile

from app.physics.flyover_fit import fit_flyover


@dataclass
class DiscoverySettings:
    fmin_hz: float = 60.0
    fmax_hz: float = 2000.0
    window_s: float = 0.25
    hop_s: float = 0.05
    max_slope_hz_s: float = 160.0
    jump_penalty: float = 0.35
    detection_z: float = 4.0
    min_duration_s: float = 3.0
    max_gap_s: float = 0.6
    max_ridges: int = 6
    mask_half_width_hz: float = 25.0
    output_step_s: float = 0.25
    max_duration_s: float = 1800.0
    harmonic_max_integer: int = 6
    harmonic_tolerance: float = 0.012


def load_audio(path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
    """Load an audio file as mono float32.  WAV is read natively; other formats need ffmpeg."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        sr, data = wavfile.read(path)
        x = _to_float_mono(data)
    else:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise ValueError(
                f"Decoding '{ext}' requires ffmpeg on PATH; upload a WAV file or install ffmpeg."
            )
        cmd = [ffmpeg, "-v", "error", "-i", path, "-ac", "1", "-f", "wav", "-acodec", "pcm_s16le", "-"]
        proc = subprocess.run(cmd, capture_output=True, check=False, timeout=600)
        if proc.returncode != 0:
            raise ValueError(f"ffmpeg failed to decode audio: {proc.stderr.decode(errors='ignore')[:200]}")
        sr, data = wavfile.read(io.BytesIO(proc.stdout))
        x = _to_float_mono(data)

    if target_sr and target_sr != sr:
        g = math.gcd(int(sr), int(target_sr))
        x = signal.resample_poly(x, int(target_sr) // g, int(sr) // g).astype(np.float32)
        sr = int(target_sr)
    return x, int(sr)


def _to_float_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        x = data.astype(np.float32) / float(max(abs(info.min), info.max))
    else:
        x = data.astype(np.float32)
    return x


def contrast_spectrogram(
    x: np.ndarray, sr: int, settings: DiscoverySettings
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, freqs, power_db, contrast) restricted to the analysis band."""
    nperseg = int(2 ** round(math.log2(max(64, settings.window_s * sr))))
    hop = max(1, int(round(settings.hop_s * sr)))
    noverlap = max(0, nperseg - hop)
    freqs, times, sxx = signal.spectrogram(
        x, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap, mode="psd", scaling="density"
    )
    band = (freqs >= settings.fmin_hz) & (freqs <= min(settings.fmax_hz, sr / 2.0))
    freqs = freqs[band]
    power_db = 10.0 * np.log10(np.maximum(sxx[band, :].T, 1e-20))  # (T, F)

    # Remove stationary tonal baselines (per-bin median across time).
    base = np.median(power_db, axis=0, keepdims=True)
    z = power_db - base
    # Robust per-frame noise-floor normalisation.
    med = np.median(z, axis=1, keepdims=True)
    mad = np.median(np.abs(z - med), axis=1, keepdims=True) * 1.4826
    contrast = (z - med) / np.maximum(mad, 1e-3)
    contrast = np.clip(contrast, -3.0, 12.0)
    return times, freqs, power_db, contrast


def viterbi_ridge(contrast: np.ndarray, max_jump: int, jump_penalty: float) -> np.ndarray:
    """Highest-scoring path through a (T, F) score image with bounded per-frame bin jumps."""
    T, F = contrast.shape
    score = contrast[0].astype(np.float64).copy()
    back = np.zeros((T, F), dtype=np.int32)
    idx = np.arange(F)
    for t in range(1, T):
        best = np.full(F, -np.inf)
        arg = idx.copy()
        for k in range(-max_jump, max_jump + 1):
            # predecessor bin for state j is j + k
            shifted = np.full(F, -np.inf)
            if k >= 0:
                shifted[: F - k] = score[k:]
            else:
                shifted[-k:] = score[: F + k]
            cand = shifted - jump_penalty * abs(k)
            better = cand > best
            best[better] = cand[better]
            arg[better] = idx[better] + k
        back[t] = arg
        score = best + contrast[t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _refine_peak(power_db: np.ndarray, t_idx: int, f_idx: int, df: float, f_base: float) -> float:
    F = power_db.shape[1]
    if 0 < f_idx < F - 1:
        a, b, c = power_db[t_idx, f_idx - 1], power_db[t_idx, f_idx], power_db[t_idx, f_idx + 1]
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
    else:
        delta = 0.0
    return f_base + (f_idx + delta) * df


def _rational_ratio(r: float, max_int: int, tol: float) -> Optional[Fraction]:
    best = None
    for q in range(1, max_int + 1):
        for p in range(q + 1, max_int * q + 1):
            if p > max_int:
                break
            frac = Fraction(p, q)
            if frac.denominator != q:
                continue
            err = abs(r - p / q) / r
            if err < tol and (best is None or err < best[0]):
                best = (err, frac)
    return best[1] if best else None


def discover_ridges(x: np.ndarray, sr: int, settings: Optional[DiscoverySettings] = None) -> Dict[str, Any]:
    """Run automatic ridge discovery on a mono signal."""
    s = settings or DiscoverySettings()
    max_samples = int(s.max_duration_s * sr)
    truncated = x.size > max_samples
    if truncated:
        x = x[:max_samples]

    times, freqs, power_db, contrast = contrast_spectrogram(x, sr, s)
    if times.size < 5 or freqs.size < 5:
        return _result([], [], s, sr, x.size / sr, truncated, "Recording too short for analysis.")

    df = float(freqs[1] - freqs[0])
    hop = float(times[1] - times[0]) if times.size > 1 else s.hop_s
    max_jump = max(1, int(math.ceil(s.max_slope_hz_s * hop / df)))
    min_frames = int(math.ceil(s.min_duration_s / hop))
    gap_frames = int(math.ceil(s.max_gap_s / hop))
    mask_w = max(1, int(round(s.mask_half_width_hz / df)))

    work = contrast.copy()
    ridges: List[Dict[str, Any]] = []
    attempts = 0
    while len(ridges) < s.max_ridges and attempts < s.max_ridges * 3:
        attempts += 1
        path = viterbi_ridge(work, max_jump, s.jump_penalty)
        along = work[np.arange(times.size), path]
        # Smooth the along-path contrast so single weak frames do not split a ridge.
        smooth = signal.medfilt(along, kernel_size=5) if along.size >= 5 else along
        detected = smooth >= s.detection_z
        # Close short gaps.
        runs = _runs(detected)
        merged: List[List[int]] = []
        for a, b in runs:
            if merged and a - merged[-1][1] <= gap_frames:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        accepted = [(a, b) for a, b in merged if b - a >= min_frames]
        if not accepted:
            break
        for a, b in accepted:
            seg_t = times[a:b]
            seg_bins = path[a:b]
            seg_f = np.array([
                _refine_peak(power_db, a + i, int(seg_bins[i]), df, float(freqs[0])) for i in range(b - a)
            ])
            seg_c = along[a:b]
            ridges.append(_build_ridge(seg_t, seg_f, seg_c, s))
            # Mask the accepted ridge so the next pass finds something else.
            for i in range(a, b):
                lo = max(0, int(seg_bins[i - a]) - mask_w)
                hi = min(freqs.size, int(seg_bins[i - a]) + mask_w + 1)
                work[i, lo:hi] = -3.0
            if len(ridges) >= s.max_ridges:
                break

    ridges.sort(key=lambda r: (r["t_start"], r["median_freq_hz"]))
    for i, r in enumerate(ridges):
        r["id"] = f"R{i + 1}"
    groups = _group_harmonics(ridges, s)
    msg = None if ridges else "No moving tonal ridge cleared the detection threshold."
    return _result(ridges, groups, s, sr, x.size / sr, truncated, msg)


def _build_ridge(seg_t: np.ndarray, seg_f: np.ndarray, seg_c: np.ndarray, s: DiscoverySettings) -> Dict[str, Any]:
    # Frequency measurement scatter relative to a smooth local fit.
    k = min(len(seg_f) // 2 * 2 - 1, 9)
    smooth = signal.savgol_filter(seg_f, k, 2) if k >= 5 else seg_f
    scatter = float(np.median(np.abs(seg_f - smooth)) * 1.4826)

    # Down-sample to the output step for matching (median within each bin).
    step = max(s.output_step_s, float(seg_t[1] - seg_t[0]) if seg_t.size > 1 else s.output_step_s)
    edges = np.arange(seg_t[0], seg_t[-1] + step, step)
    pts = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (seg_t >= lo) & (seg_t < hi)
        if np.any(sel):
            pts.append({"time": round(float(np.mean(seg_t[sel])), 3),
                        "freq": round(float(np.median(seg_f[sel])), 2)})

    slope = np.diff(smooth)
    falling_fraction = float(np.mean(slope < 0)) if slope.size else 0.0

    fit = None
    try:
        fit = fit_flyover([p["time"] for p in pts], [p["freq"] for p in pts], loss_scale_hz=max(1.0, scatter))
        fit.pop("model_curve", None)
    except Exception:
        fit = None

    doppler_like = bool(
        fit is not None
        and fit["rmse_hz"] <= max(4.0, 3.0 * scatter)
        and falling_fraction >= 0.6
        and (np.max(seg_f) - np.min(seg_f)) >= 5.0
    )
    return {
        "id": None,
        "harmonic_group_id": None,
        "t_start": round(float(seg_t[0]), 3),
        "t_end": round(float(seg_t[-1]), 3),
        "duration_s": round(float(seg_t[-1] - seg_t[0]), 3),
        "median_freq_hz": round(float(np.median(seg_f)), 2),
        "freq_range_hz": [round(float(np.min(seg_f)), 2), round(float(np.max(seg_f)), 2)],
        "mean_contrast": round(float(np.mean(seg_c)), 2),
        "frequency_uncertainty_hz": round(max(scatter, 0.1), 3),
        "falling_fraction": round(falling_fraction, 3),
        "doppler_like": doppler_like,
        "flyover_fit": fit,
        "points": pts,
    }


def _overlap_ratio(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Median frequency ratio (>= 1) over the overlapping time span, and its relative spread."""
    lo, hi = max(a["t_start"], b["t_start"]), min(a["t_end"], b["t_end"])
    shorter = min(a["duration_s"], b["duration_s"])
    if shorter <= 0 or hi - lo < 0.5 * shorter:
        return None
    ta = np.array([p["time"] for p in a["points"]]); fa = np.array([p["freq"] for p in a["points"]])
    tb = np.array([p["time"] for p in b["points"]]); fb = np.array([p["freq"] for p in b["points"]])
    grid = np.linspace(lo, hi, 25)
    ra, rb = np.interp(grid, ta, fa), np.interp(grid, tb, fb)
    ratio = np.maximum(ra, rb) / np.maximum(np.minimum(ra, rb), 1e-6)
    med = float(np.median(ratio))
    return med, float(np.median(np.abs(ratio - med)) / med)


def _group_harmonics(ridges: List[Dict[str, Any]], s: DiscoverySettings) -> List[Dict[str, Any]]:
    n = len(ridges)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            # Harmonics of a real flyover are themselves Doppler-shaped; linking
            # through non-Doppler ridges would chain unrelated sounds together.
            if not (ridges[i]["doppler_like"] and ridges[j]["doppler_like"]):
                continue
            ov = _overlap_ratio(ridges[i], ridges[j])
            if ov is None:
                continue
            med, spread = ov
            frac = _rational_ratio(med, s.harmonic_max_integer, s.harmonic_tolerance)
            if frac is not None and spread < s.harmonic_tolerance:
                parent[find(i)] = find(j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out = []
    for gi, members in enumerate(sorted(groups.values(), key=lambda m: min(ridges[k]["t_start"] for k in m))):
        gid = f"G{gi + 1}"
        members.sort(key=lambda k: ridges[k]["median_freq_hz"])
        base = ridges[members[0]]["median_freq_hz"]
        for k in members:
            ridges[k]["harmonic_group_id"] = gid
            ov = _overlap_ratio(ridges[members[0]], ridges[k]) if k != members[0] else (1.0, 0.0)
            ratio = ov[0] if ov else ridges[k]["median_freq_hz"] / base
            frac = _rational_ratio(ratio, s.harmonic_max_integer, s.harmonic_tolerance) if k != members[0] else Fraction(1)
            ridges[k]["ratio_to_lowest"] = round(float(ratio), 4)
            ridges[k]["harmonic_ratio"] = str(frac) if frac is not None else None
        primary = max(members, key=lambda k: (ridges[k]["doppler_like"], ridges[k]["duration_s"], ridges[k]["mean_contrast"]))
        out.append({
            "id": gid,
            "ridge_ids": [ridges[k]["id"] for k in members],
            "primary_ridge_id": ridges[primary]["id"],
            "t_start": min(ridges[k]["t_start"] for k in members),
            "t_end": max(ridges[k]["t_end"] for k in members),
            "doppler_like": any(ridges[k]["doppler_like"] for k in members),
        })
    return out


def _result(ridges, groups, s, sr, duration, truncated, message) -> Dict[str, Any]:
    return {
        "ridges": ridges,
        "harmonic_groups": groups,
        "n_ridges": len(ridges),
        "n_groups": len(groups),
        "analysis_sample_rate": sr,
        "duration_s": round(float(duration), 3),
        "truncated": bool(truncated),
        "settings": asdict(s),
        "message": message,
        "note": "Ridges are acoustic proposals, not verified aircraft detections.",
    }


def discover_file(path: str, settings: Optional[DiscoverySettings] = None) -> Dict[str, Any]:
    s = settings or DiscoverySettings()
    target = int(min(48000, max(4000, 4 * s.fmax_hz)))
    x, sr = load_audio(path, target_sr=target)
    return discover_ridges(x, sr, s)
