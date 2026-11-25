"""
Recording start time from file metadata.

A trusted recording clock collapses the single-flyover clock/position
ambiguity, so it is worth reading whatever the recorder wrote - but every
source has caveats and the result is only a **suggestion** for the user to
confirm:

* **BWF ``bext``** (Broadcast WAV): ``OriginationDate`` / ``OriginationTime``
  and ``TimeReference`` (samples since midnight).  These are *local* recorder
  times with no time-zone information.  When ``TimeReference`` is non-zero it
  gives the start to the sample: midnight of ``OriginationDate`` +
  TimeReference / sample rate.
* **iXML** (``BWF_ORIGINATION_DATE``/``TIME``, ``BWF_TIME_REFERENCE_LOW/HIGH``
  or ``TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_LO/HI``): same semantics.
* **ffprobe ``creation_time``** (MP4/M4A/MOV/…): UTC by specification, but
  many devices write the time the file was *closed*, i.e. the end of the
  recording.  Both readings are returned.

Without a time zone the epoch is computed *as if* the local time were UTC and
``timezone_known`` is false; callers must say so and let the user adjust.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Optional, Tuple


def _riff_chunks(path: str):
    """Yield (chunk_id, payload) for a RIFF/WAVE (or RF64 header) file; reads only small chunks fully."""
    with open(path, "rb") as fh:
        head = fh.read(12)
        if len(head) < 12 or head[:4] not in (b"RIFF", b"RF64") or head[8:12] != b"WAVE":
            return
        while True:
            hdr = fh.read(8)
            if len(hdr) < 8:
                return
            cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
            if cid in (b"fmt ", b"bext", b"iXML"):
                yield cid, fh.read(size)
            else:
                fh.seek(size, 1)
            if size % 2:
                fh.seek(1, 1)


def parse_bext(payload: bytes) -> Dict[str, Any]:
    """Decode the fields of a BWF bext chunk that matter for timing."""
    if len(payload) < 346:
        return {}
    date = payload[320:330].decode("ascii", errors="ignore").strip("\x00 ")
    tod = payload[330:338].decode("ascii", errors="ignore").strip("\x00 ")
    lo, hi = struct.unpack("<II", payload[338:346])
    return {"origination_date": date, "origination_time": tod, "time_reference": (hi << 32) | lo}


def parse_ixml(payload: bytes) -> Dict[str, Any]:
    try:
        root = ET.fromstring(payload.rstrip(b"\x00").decode("utf-8", errors="ignore"))
    except ET.ParseError:
        return {}
    vals = {el.tag.upper(): (el.text or "").strip() for el in root.iter()}

    def as_int(*names):
        for n in names:
            if vals.get(n, "").isdigit():
                return int(vals[n])
        return None

    lo = as_int("BWF_TIME_REFERENCE_LOW", "TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_LO")
    hi = as_int("BWF_TIME_REFERENCE_HIGH", "TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_HI") or 0
    return {
        "origination_date": vals.get("BWF_ORIGINATION_DATE", ""),
        "origination_time": vals.get("BWF_ORIGINATION_TIME", ""),
        "time_reference": (hi << 32) | lo if lo is not None else 0,
        "sample_rate": as_int("TIMESTAMP_SAMPLE_RATE", "FILE_SAMPLE_RATE"),
    }


def _local_start(date: str, tod: str, time_ref: int, sr: Optional[int]) -> Optional[Tuple[datetime, str]]:
    date = date.replace(":", "-").replace("/", "-").replace(".", "-")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return None
    tod = tod.replace("-", ":").replace(".", ":")
    if time_ref and sr:
        midnight = datetime.strptime(date, "%Y-%m-%d")
        return midnight + timedelta(seconds=time_ref / sr), "time_reference"
    if re.fullmatch(r"\d{2}:\d{2}:\d{2}", tod):
        return datetime.strptime(f"{date} {tod}", "%Y-%m-%d %H:%M:%S"), "origination_time"
    return None


def read_wav_start(path: str) -> Optional[Dict[str, Any]]:
    sr = None
    bext = ixml = None
    for cid, payload in _riff_chunks(path):
        if cid == b"fmt " and len(payload) >= 8:
            sr = struct.unpack("<I", payload[4:8])[0]
        elif cid == b"bext":
            bext = parse_bext(payload)
        elif cid == b"iXML":
            ixml = parse_ixml(payload)
    for source, info in (("bwf_bext", bext), ("ixml", ixml)):
        if not info:
            continue
        local = _local_start(info.get("origination_date", ""), info.get("origination_time", ""),
                             info.get("time_reference", 0), info.get("sample_rate") or sr)
        if local is None:
            continue
        dt, field = local
        return {
            "start_epoch": dt.replace(tzinfo=UTC).timestamp(),
            "start_local_iso": dt.isoformat(timespec="milliseconds"),
            "source": source,
            "field": field,
            "timezone_known": False,
            "notes": [
                "Broadcast-WAV times are the recorder's local clock with no time zone; the epoch assumes UTC. "
                "Adjust for the recorder's time zone and clock error before trusting it.",
            ],
        }
    return None


def parse_ffprobe_json(info: Dict[str, Any], duration_s: Optional[float] = None) -> Optional[Dict[str, Any]]:
    tags = {}
    for scope in [info.get("format") or {}] + list(info.get("streams") or []):
        for k, v in (scope.get("tags") or {}).items():
            tags.setdefault(k.lower(), v)
    ct = tags.get("creation_time")
    if not ct:
        return None
    iso = ct.strip().replace(" ", "T", 1)
    if iso.endswith(("Z", "z")):
        iso = iso[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return None
    tz_known = dt.tzinfo is not None
    if not tz_known:
        dt = dt.replace(tzinfo=UTC)
    epoch = dt.timestamp()
    if duration_s is None:
        try:
            duration_s = float((info.get("format") or {}).get("duration"))
        except (TypeError, ValueError):
            duration_s = None
    out = {
        "start_epoch": epoch,
        "start_local_iso": None,
        "source": "ffprobe_creation_time",
        "field": "creation_time",
        "timezone_known": tz_known,
        "notes": ["creation_time is often written when the file is closed (end of recording); "
                  "confirm whether it marks the start or the end."],
        "alternatives": [],
    }
    if duration_s:
        out["alternatives"].append({"label": "creation_time marks the end of the recording",
                                    "start_epoch": epoch - duration_s})
    return out


def read_recording_start(path: str) -> Dict[str, Any]:
    """
    Best-effort recording start.  Always returns a dict with ``start_epoch``
    (or None), ``source``, ``timezone_known``, ``requires_confirmation`` and
    ``notes``.
    """
    result: Optional[Dict[str, Any]] = None
    if os.path.splitext(path)[1].lower() == ".wav":
        try:
            result = read_wav_start(path)
        except (OSError, struct.error):
            result = None
    if result is None:
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            try:
                proc = subprocess.run([ffprobe, "-v", "quiet", "-print_format", "json", "-show_format",
                                       "-show_streams", path], capture_output=True, timeout=30, check=False)
                if proc.returncode == 0:
                    result = parse_ffprobe_json(json.loads(proc.stdout or b"{}"))
            except (OSError, ValueError, subprocess.SubprocessError):
                result = None
    if result is None:
        return {"start_epoch": None, "source": None, "timezone_known": False, "requires_confirmation": True,
                "notes": ["No recording start time found in the file metadata."]}
    result["requires_confirmation"] = True
    return result
