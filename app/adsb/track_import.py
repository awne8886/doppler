"""
Import aircraft tracks from files so recordings can be matched against real,
historical trajectories.

Supported formats (``fmt``):

``csv``       Generic CSV with a header row.  Columns (case-insensitive,
              aliases accepted): time, lat, lon, altitude, optional speed,
              track, vertical rate, flight id / icao24, callsign.  Units are
              read from the header (``alt_ft``, ``altitude (m)``, ``speed_kt``
              ...) or from the ``altitude_unit`` / ``speed_unit`` options;
              an altitude column with no known unit is rejected.
``fr24_csv``  FlightRadar24 CSV export
              (``Timestamp,UTC,Callsign,Position,Altitude,Speed,Direction``;
              feet, knots, barometric altitude).
``kml``       KML ``gx:Track`` (``when`` + ``gx:coord``) or time-stamped
              Placemark points (FR24 KML export).  Altitude in metres (KML).
``geojson``   LineString features with ``properties.times`` /
              ``coordTimes``, or Point features with a ``time`` property
              grouped by ``flight_id``/``icao24``/``callsign``.  Metres.
``readsb``    readsb / tar1090 trace JSON (``trace_full_<hex>.json``, gzip
              accepted).  Geometric altitude is used when present.

Times may be Unix seconds, Unix milliseconds or ISO 8601.  ISO times without
an offset are ambiguous and rejected unless ``timezone`` (``UTC`` or
``+HH:MM``) is given.  Every track is validated (coordinates, monotonic time,
plausible altitude and implied speed) and tagged ``data_source="uploaded"``.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

FT_TO_M = 0.3048
KT_TO_MPS = 0.514444
KMH_TO_MPS = 1.0 / 3.6
FPM_TO_MPS = 0.00508
MAX_ALTITUDE_M = 20000.0
MIN_ALTITUDE_M = -500.0
MAX_IMPLIED_SPEED_MPS = 450.0
FORMATS = ("csv", "fr24_csv", "kml", "geojson", "readsb")


class TrackImportError(ValueError):
    """Raised for files that cannot be imported; the message is safe to show to users."""


@dataclass
class ImportOptions:
    altitude_unit: str = "auto"      # auto | ft | m
    speed_unit: str = "auto"         # auto | kt | mps | kmh
    vertical_rate_unit: str = "auto"  # auto | fpm | mps
    timezone: Optional[str] = None   # applied to ISO times without an offset: "UTC" or "+02:00"
    altitude_basis: Optional[str] = None  # override: geometric | barometric | unknown


# --------------------------------------------------------------------- time

def _tz_from_option(tz: Optional[str]) -> Optional[timezone]:
    if tz is None or str(tz).strip() == "":
        return None
    s = str(tz).strip().upper()
    if s in ("UTC", "Z", "GMT"):
        return UTC
    m = re.fullmatch(r"(?:UTC)?([+-])(\d{1,2}):?(\d{2})?", s)
    if not m:
        raise TrackImportError(f"Unsupported timezone '{tz}'. Use 'UTC' or an offset such as '+02:00'.")
    sign = 1 if m.group(1) == "+" else -1
    return timezone(sign * timedelta(hours=int(m.group(2)), minutes=int(m.group(3) or 0)))


def parse_time(value: Any, tz: Optional[timezone]) -> float:
    """Parse a timestamp into Unix seconds (UTC)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _epoch_from_number(float(value))
    s = str(value).strip()
    if not s:
        raise TrackImportError("Empty timestamp.")
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", s):
        return _epoch_from_number(float(s))
    iso = s.replace(" ", "T", 1) if re.match(r"\d{4}-\d{2}-\d{2} \d", s) else s
    if iso.endswith(("Z", "z")):
        iso = iso[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        raise TrackImportError(f"Unrecognised timestamp '{s}'. Use Unix seconds/milliseconds or ISO 8601.") from None
    if dt.tzinfo is None:
        if tz is None:
            raise TrackImportError(
                f"Timestamp '{s}' has no time-zone offset. Pass timezone='UTC' (or an offset like '+02:00') "
                "to state how it should be read.")
        dt = dt.replace(tzinfo=tz)
    return _check_epoch(dt.timestamp())


def _epoch_from_number(x: float) -> float:
    if x > 1e11:  # milliseconds
        x /= 1000.0
    if x < 1e8:
        raise TrackImportError(
            f"Timestamp {x:g} looks relative, not absolute. Tracks need Unix epoch or ISO 8601 times.")
    return _check_epoch(x)


def _check_epoch(x: float) -> float:
    if not (631152000.0 <= x <= 4102444800.0):  # 1990-01-01 .. 2100-01-01
        raise TrackImportError(f"Timestamp {x:.0f} is outside 1990-2100.")
    return x


# -------------------------------------------------------------------- units

def _unit_from_name(name: str, kind: str) -> Optional[str]:
    n = name.lower().replace(" ", "")
    if kind == "alt":
        if re.search(r"(_|\()(ft|feet)\)?$|feet", n):
            return "ft"
        if re.search(r"(_|\()(m|meters|metres)\)?$", n):
            return "m"
    elif kind == "speed":
        if re.search(r"(kt|kts|knots)\)?$", n):
            return "kt"
        if re.search(r"(mps|m/s|ms)\)?$", n):
            return "mps"
        if re.search(r"(kmh|km/h|kph)\)?$", n):
            return "kmh"
    elif kind == "vrate":
        if re.search(r"(fpm|ft/min)\)?$", n):
            return "fpm"
        if re.search(r"(mps|m/s)\)?$", n):
            return "mps"
    return None


def _alt_factor(unit: str) -> float:
    return {"ft": FT_TO_M, "m": 1.0}[unit]


def _speed_factor(unit: str) -> float:
    return {"kt": KT_TO_MPS, "mps": 1.0, "kmh": KMH_TO_MPS}[unit]


def _vrate_factor(unit: str) -> float:
    return {"fpm": FPM_TO_MPS, "mps": 1.0}[unit]


def _num(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        x = float(v)
    else:
        s = str(v).strip()
        if s == "" or s.lower() in ("nan", "null", "none", "-", "ground"):
            return None
        try:
            x = float(s)
        except ValueError:
            return None
    return x if math.isfinite(x) else None


# ----------------------------------------------------------------- building

class _TrackBuilder:
    def __init__(self, key: str):
        self.key = key
        self.callsign: Optional[str] = None
        self.icao24: Optional[str] = None
        self.aircraft_type: Optional[str] = None
        self.rows: List[Tuple[float, float, float, float, Optional[float], Optional[float], Optional[float]]] = []

    def add(self, t, lat, lon, alt, speed=None, track=None, vrate=None):
        self.rows.append((t, lat, lon, alt, speed, track, vrate))


def _finish(builders: Iterable[_TrackBuilder], basis: str, fmt: str, source_name: str,
            warnings: List[str]) -> List[Dict[str, Any]]:
    flights = []
    for b in builders:
        traj = _validate(b, basis, fmt, source_name, warnings)
        if traj is not None:
            flights.append(traj)
    if not flights:
        raise TrackImportError("No usable track (each track needs at least 3 valid, time-stamped airborne points).")
    return flights


def _validate(b: _TrackBuilder, basis: str, fmt: str, source_name: str, warnings: List[str]) -> Optional[Dict[str, Any]]:
    rows = sorted(b.rows, key=lambda r: r[0])
    dedup = []
    for r in rows:
        if not (-90.0 <= r[1] <= 90.0 and -180.0 <= r[2] <= 180.0):
            raise TrackImportError(f"Track {b.key}: coordinate out of range ({r[1]}, {r[2]}).")
        if dedup and abs(r[0] - dedup[-1][0]) < 1e-6:
            continue  # duplicate timestamp: keep the first report
        dedup.append(r)
    if len(dedup) < 3:
        warnings.append(f"Track {b.key}: fewer than 3 distinct points; skipped.")
        return None
    alts = [r[3] for r in dedup]
    if max(alts) > MAX_ALTITUDE_M or min(alts) < MIN_ALTITUDE_M:
        raise TrackImportError(
            f"Track {b.key}: altitude range {min(alts):.0f}..{max(alts):.0f} m is implausible; check the altitude unit "
            "(feet vs metres).")
    # Drop isolated position jumps that imply impossible speeds.
    kept = [dedup[0]]
    dropped = 0
    for r in dedup[1:]:
        p = kept[-1]
        dt = r[0] - p[0]
        d = _haversine_m(p[1], p[2], r[1], r[2])
        if d / dt > MAX_IMPLIED_SPEED_MPS:
            dropped += 1
            continue
        kept.append(r)
    if dropped:
        warnings.append(f"Track {b.key}: {dropped} point(s) dropped for implying > {MAX_IMPLIED_SPEED_MPS:.0f} m/s.")
    if len(kept) < 3:
        warnings.append(f"Track {b.key}: fewer than 3 consistent points; skipped.")
        return None
    speeds = [r[4] for r in kept]
    if all(s is not None for s in speeds) and len(kept) >= 5:
        implied = [_haversine_m(a[1], a[2], c[1], c[2]) / (c[0] - a[0]) for a, c in zip(kept[:-2], kept[2:])]
        rep = sorted(speeds)[len(speeds) // 2]
        imp = sorted(implied)[len(implied) // 2]
        if imp > 20.0 and rep > 0:
            ratio = rep / imp
            if abs(ratio - 1.0) > 0.35:
                warnings.append(
                    f"Track {b.key}: reported speed ({rep:.0f} m/s) differs from position-implied speed ({imp:.0f} m/s) "
                    f"by a factor {ratio:.2f}; check the speed unit.")
    fid = (b.icao24 or b.callsign or b.key or "TRACK").strip()
    return {
        "flight_id": fid.upper(),
        "icao24": (b.icao24 or "").lower() or None,
        "callsign": (b.callsign or "").strip() or fid.upper(),
        "aircraft_type": b.aircraft_type or "Unknown (uploaded track)",
        "timestamps": [r[0] for r in kept],
        "latitudes": [r[1] for r in kept],
        "longitudes": [r[2] for r in kept],
        "altitudes": [r[3] for r in kept],
        "speeds": [r[4] for r in kept] if all(r[4] is not None for r in kept) else [],
        "headings": [r[5] for r in kept] if all(r[5] is not None for r in kept) else [],
        "vertical_rates": [r[6] for r in kept] if all(r[6] is not None for r in kept) else [],
        "data_source": "uploaded",
        "altitude_basis": basis,
        "source_format": fmt,
        "source_name": source_name,
    }


def _haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000.0 * math.asin(min(1.0, math.sqrt(a)))


# ------------------------------------------------------------------ detection

def detect_format(data: bytes, filename: str = "") -> str:
    name = (filename or "").lower()
    head = data[:4096].lstrip()
    if head[:1] in (b"{", b"["):
        try:
            obj = json.loads(data.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise TrackImportError("File looks like JSON but could not be parsed.") from None
        if isinstance(obj, dict) and "trace" in obj:
            return "readsb"
        if isinstance(obj, dict) and obj.get("type") in ("FeatureCollection", "Feature"):
            return "geojson"
        raise TrackImportError("JSON file is neither GeoJSON nor a readsb/tar1090 trace.")
    if head.startswith(b"<?xml") or b"<kml" in head[:2048] or name.endswith(".kml"):
        return "kml"
    first = data.decode("utf-8-sig", errors="replace").splitlines()[0] if data else ""
    cols = [c.strip().lower() for c in first.split(",")]
    if {"timestamp", "utc", "callsign", "position"} <= set(cols):
        return "fr24_csv"
    return "csv"


def parse_tracks(data: bytes, filename: str = "", fmt: Optional[str] = None,
                 options: Optional[ImportOptions] = None) -> Dict[str, Any]:
    """Parse and validate a track file.  Returns ``{"format", "flights", "warnings"}``."""
    opts = options or ImportOptions()
    if data[:2] == b"\x1f\x8b":
        try:
            data = gzip.decompress(data)
        except OSError:
            raise TrackImportError("Corrupt gzip file.") from None
    if not data.strip():
        raise TrackImportError("Empty file.")
    fmt = (fmt or "").strip().lower() or detect_format(data, filename)
    if fmt not in FORMATS:
        raise TrackImportError(f"Unknown format '{fmt}'. Supported: {', '.join(FORMATS)}.")
    for unit, allowed in ((opts.altitude_unit, ("auto", "ft", "m")), (opts.speed_unit, ("auto", "kt", "mps", "kmh")),
                          (opts.vertical_rate_unit, ("auto", "fpm", "mps"))):
        if unit not in allowed:
            raise TrackImportError(f"Unsupported unit '{unit}'; expected one of {allowed}.")
    if opts.altitude_basis not in (None, "", "geometric", "barometric", "unknown"):
        raise TrackImportError("altitude_basis must be geometric, barometric or unknown.")
    tz = _tz_from_option(opts.timezone)
    warnings: List[str] = []
    parser = {"csv": _parse_csv, "fr24_csv": _parse_fr24, "kml": _parse_kml, "geojson": _parse_geojson,
              "readsb": _parse_readsb}[fmt]
    flights = parser(data, filename, opts, tz, warnings)
    return {"format": fmt, "flights": flights, "warnings": warnings}


# ----------------------------------------------------------------------- CSV

_ALIASES = {
    "time": ("timestamp", "time", "utc", "datetime", "date_time", "epoch", "unix_time", "unixtime", "t", "time_utc"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "lng", "long", "longitude"),
    "alt": ("alt", "altitude", "alt_geom", "geo_altitude", "geoaltitude", "alt_baro", "baro_altitude",
            "baroaltitude", "height"),
    "speed": ("speed", "gs", "ground_speed", "groundspeed", "velocity"),
    "track": ("track", "heading", "true_track", "course", "direction"),
    "vrate": ("vertical_rate", "vrate", "baro_rate", "geom_rate", "vert_rate", "climb_rate"),
    "id": ("flight_id", "icao24", "hex", "icao", "icao_hex"),
    "callsign": ("callsign", "flight", "call_sign"),
    "type": ("aircraft_type", "type", "actype"),
}


def _strip_unit(col: str) -> str:
    c = col.strip().lower()
    c = re.sub(r"\s*\(.*\)\s*$", "", c)
    c = re.sub(r"_(ft|feet|m|meters|metres|kt|kts|knots|mps|ms|kmh|kph|fpm|deg|degrees|s|utc)$", "", c)
    return c.replace(" ", "_")


def _match_columns(header: List[str]) -> Dict[str, int]:
    found: Dict[str, int] = {}
    for idx, col in enumerate(header):
        base = _strip_unit(col)
        for key, aliases in _ALIASES.items():
            if key not in found and (base in aliases or col.strip().lower() in aliases):
                found[key] = idx
    return found


def _parse_csv(data: bytes, filename: str, opts: ImportOptions, tz, warnings) -> List[Dict[str, Any]]:
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise TrackImportError("CSV file has no header row.") from None
    cols = _match_columns(header)
    missing = [k for k in ("time", "lat", "lon", "alt") if k not in cols]
    if missing:
        raise TrackImportError(f"CSV is missing required column(s): {', '.join(missing)}. Header: {header}")
    alt_name = header[cols["alt"]]
    alt_unit = opts.altitude_unit if opts.altitude_unit != "auto" else _unit_from_name(alt_name, "alt")
    if alt_unit is None:
        raise TrackImportError(
            f"Altitude column '{alt_name}' does not state its unit. Rename it (e.g. alt_ft / alt_m) or pass altitude_unit.")
    speed_unit = None
    if "speed" in cols:
        speed_unit = opts.speed_unit if opts.speed_unit != "auto" else _unit_from_name(header[cols["speed"]], "speed")
        if speed_unit is None:
            warnings.append(f"Speed column '{header[cols['speed']]}' has no unit; speeds ignored (pass speed_unit to use them).")
    vr_unit = None
    if "vrate" in cols:
        vr_unit = opts.vertical_rate_unit if opts.vertical_rate_unit != "auto" else _unit_from_name(header[cols["vrate"]], "vrate")
        if vr_unit is None:
            warnings.append(f"Vertical-rate column '{header[cols['vrate']]}' has no unit; ignored.")
    basis = opts.altitude_basis or _basis_from_name(alt_name)
    builders: Dict[str, _TrackBuilder] = {}
    skipped = 0
    for row in reader:
        if not row or all(not c.strip() for c in row):
            continue
        try:
            t = parse_time(row[cols["time"]], tz)
        except IndexError:
            skipped += 1
            continue
        lat, lon, alt = _num(_get(row, cols, "lat")), _num(_get(row, cols, "lon")), _num(_get(row, cols, "alt"))
        if lat is None or lon is None or alt is None:
            skipped += 1
            continue
        key = (_get(row, cols, "id") or _get(row, cols, "callsign") or "TRACK1").strip()
        b = builders.setdefault(key, _TrackBuilder(key))
        if "id" in cols:
            b.icao24 = b.icao24 or (_get(row, cols, "id") or "").strip()
        if "callsign" in cols:
            b.callsign = b.callsign or (_get(row, cols, "callsign") or "").strip()
        if "type" in cols:
            b.aircraft_type = b.aircraft_type or (_get(row, cols, "type") or "").strip() or None
        spd = _num(_get(row, cols, "speed"))
        vr = _num(_get(row, cols, "vrate"))
        b.add(t, lat, lon, alt * _alt_factor(alt_unit),
              spd * _speed_factor(speed_unit) if spd is not None and speed_unit else None,
              _num(_get(row, cols, "track")),
              vr * _vrate_factor(vr_unit) if vr is not None and vr_unit else None)
    if skipped:
        warnings.append(f"{skipped} CSV row(s) without time, position or altitude were skipped.")
    return _finish(builders.values(), basis, "csv", filename, warnings)


def _get(row: List[str], cols: Dict[str, int], key: str) -> Optional[str]:
    i = cols.get(key)
    return row[i] if i is not None and i < len(row) else None


def _basis_from_name(name: str) -> str:
    n = name.lower()
    if "geom" in n or "geo_" in n or "geoalt" in n or "gnss" in n or "gps" in n:
        return "geometric"
    if "baro" in n:
        return "barometric"
    return "unknown"


def _parse_fr24(data: bytes, filename: str, opts: ImportOptions, tz, warnings) -> List[Dict[str, Any]]:
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    alt_unit = opts.altitude_unit if opts.altitude_unit != "auto" else "ft"
    speed_unit = opts.speed_unit if opts.speed_unit != "auto" else "kt"
    b = _TrackBuilder(filename or "FR24")
    skipped = 0
    for row in reader:
        r = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
        try:
            t = parse_time(r.get("timestamp") or r.get("utc"), tz or UTC)
            lat_s, lon_s = (r.get("position") or "").split(",")
            lat, lon = float(lat_s), float(lon_s)
        except (ValueError, TrackImportError):
            skipped += 1
            continue
        alt = _num(r.get("altitude"))
        if alt is None:
            skipped += 1
            continue
        if alt <= 0 and _num(r.get("speed")) is not None and _num(r.get("speed")) < 50:
            continue  # on the ground
        b.callsign = b.callsign or r.get("callsign") or None
        spd = _num(r.get("speed"))
        b.add(t, lat, lon, alt * _alt_factor(alt_unit), spd * _speed_factor(speed_unit) if spd is not None else None,
              _num(r.get("direction")), None)
    if skipped:
        warnings.append(f"{skipped} FR24 row(s) could not be parsed and were skipped.")
    b.key = b.callsign or b.key
    return _finish([b], opts.altitude_basis or "barometric", "fr24_csv", filename, warnings)


# ----------------------------------------------------------------------- KML

def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_kml(data: bytes, filename: str, opts: ImportOptions, tz, warnings) -> List[Dict[str, Any]]:
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise TrackImportError(f"Invalid KML: {e}") from e
    alt_unit = opts.altitude_unit if opts.altitude_unit != "auto" else "m"
    builders: List[_TrackBuilder] = []
    doc_name = next((el.text for el in root.iter() if _local(el.tag) == "name" and el.text), None) or filename or "KML"
    # gx:Track elements
    for i, track in enumerate(el for el in root.iter() if _local(el.tag) == "Track"):
        whens = [el.text for el in track if _local(el.tag) == "when"]
        coords = [el.text for el in track if _local(el.tag) == "coord"]
        if len(whens) != len(coords):
            raise TrackImportError("KML gx:Track has a different number of <when> and <gx:coord> elements.")
        b = _TrackBuilder(f"{doc_name}-{i + 1}" if i else doc_name)
        b.callsign = doc_name
        for w, c in zip(whens, coords):
            parts = (c or "").split()
            if len(parts) < 3:
                continue
            b.add(parse_time(w, tz), float(parts[1]), float(parts[0]), float(parts[2]) * _alt_factor(alt_unit))
        builders.append(b)
    if not builders:
        b = _TrackBuilder(doc_name)
        b.callsign = doc_name
        for pm in (el for el in root.iter() if _local(el.tag) == "Placemark"):
            when = next((el.text for el in pm.iter() if _local(el.tag) == "when"), None)
            point = next((el for el in pm.iter() if _local(el.tag) == "Point"), None)
            if when is None or point is None:
                continue
            ctext = next((el.text for el in point.iter() if _local(el.tag) == "coordinates"), "") or ""
            parts = ctext.strip().split(",")
            if len(parts) < 3:
                continue
            b.add(parse_time(when, tz), float(parts[1]), float(parts[0]), float(parts[2]) * _alt_factor(alt_unit))
        builders.append(b)
    return _finish(builders, opts.altitude_basis or "unknown", "kml", filename, warnings)


# ------------------------------------------------------------------- GeoJSON

def _parse_geojson(data: bytes, filename: str, opts: ImportOptions, tz, warnings) -> List[Dict[str, Any]]:
    obj = json.loads(data.decode("utf-8-sig"))
    feats = obj.get("features") if obj.get("type") == "FeatureCollection" else [obj]
    alt_unit = opts.altitude_unit if opts.altitude_unit != "auto" else "m"
    builders: Dict[str, _TrackBuilder] = {}
    for i, f in enumerate(feats or []):
        geom = (f or {}).get("geometry") or {}
        props = (f or {}).get("properties") or {}
        gtype = geom.get("type")
        key = str(props.get("flight_id") or props.get("icao24") or props.get("callsign") or props.get("name")
                  or f"TRACK{i + 1}")
        if gtype in ("LineString", "MultiLineString"):
            lines = geom["coordinates"] if gtype == "MultiLineString" else [geom["coordinates"]]
            coords = [c for line in lines for c in line]
            times = props.get("times") or props.get("coordTimes") or props.get("timestamps")
            if isinstance(times, list) and times and isinstance(times[0], list):
                times = [t for seg in times for t in seg]
            if not isinstance(times, list) or len(times) != len(coords):
                raise TrackImportError(f"GeoJSON feature '{key}' needs a 'times' (or 'coordTimes') array matching its coordinates.")
            b = builders.setdefault(key + f"#{i}", _TrackBuilder(key))
            for c, t in zip(coords, times):
                if len(c) < 3:
                    raise TrackImportError(f"GeoJSON feature '{key}' coordinates need [lon, lat, altitude].")
                b.add(parse_time(t, tz), float(c[1]), float(c[0]), float(c[2]) * _alt_factor(alt_unit),
                      *_geojson_kinematics(props, None, opts, warnings))
        elif gtype == "Point":
            c = geom.get("coordinates") or []
            t = props.get("time", props.get("timestamp"))
            alt = c[2] if len(c) >= 3 else _num(props.get("altitude_m"))
            if t is None or len(c) < 2 or alt is None:
                continue
            b = builders.setdefault(key, _TrackBuilder(key))
            b.add(parse_time(t, tz), float(c[1]), float(c[0]), float(alt) * _alt_factor(alt_unit),
                  *_geojson_kinematics(props, 0, opts, warnings))
        else:
            continue
        b.callsign = b.callsign or (str(props["callsign"]) if props.get("callsign") else None)
        b.icao24 = b.icao24 or (str(props["icao24"]) if props.get("icao24") else None)
        b.aircraft_type = b.aircraft_type or props.get("aircraft_type")
    return _finish(builders.values(), opts.altitude_basis or str(_first_prop(feats, "altitude_basis") or "unknown"),
                   "geojson", filename, warnings)


def _geojson_kinematics(props, idx, opts, warnings):
    if idx is None:
        return (None, None, None)
    spd = _num(props.get("speed_mps"))
    return (spd, _num(props.get("track")), _num(props.get("vertical_rate_mps")))


def _first_prop(feats, key):
    for f in feats or []:
        v = ((f or {}).get("properties") or {}).get(key)
        if v:
            return v
    return None


# -------------------------------------------------------------------- readsb

def _parse_readsb(data: bytes, filename: str, opts: ImportOptions, tz, warnings) -> List[Dict[str, Any]]:
    """
    readsb/tar1090 trace rows: [dt, lat, lon, alt_baro_ft|"ground"|null, gs_kt, track, flags,
    vrate_fpm, aircraft{}|null, source, alt_geom_ft, geom_rate_fpm, ias, roll].
    flags: 1 stale, 2 new leg, 4 vertical rate geometric, 8 altitude field geometric.

    One altitude basis is used for the whole track: geometric when every point
    has it, else barometric when every point has that, else geometric points
    only (if they are at least 80 %), else the file is rejected.
    """
    obj = json.loads(data.decode("utf-8-sig"))
    base = _num(obj.get("timestamp"))
    if base is None:
        raise TrackImportError("readsb trace has no 'timestamp'.")
    icao = str(obj.get("icao") or "").lower()
    b = _TrackBuilder(icao or "TRACE")
    b.icao24 = icao or None
    b.aircraft_type = obj.get("t") or obj.get("desc")
    rows = []
    n_ground = 0
    for row in obj.get("trace") or []:
        if not isinstance(row, list) or len(row) < 6:
            continue
        if row[3] == "ground":
            n_ground += 1
            continue
        flags = int(row[6]) if len(row) > 6 and isinstance(row[6], (int, float)) else 0
        alt_field = _num(row[3])
        alt_geom = _num(row[10]) if len(row) > 10 else None
        alt_baro = alt_field
        if flags & 8:  # the altitude field itself is geometric
            alt_geom, alt_baro = (alt_geom if alt_geom is not None else alt_field), None
        lat, lon = _num(row[1]), _num(row[2])
        if lat is None or lon is None or (alt_geom is None and alt_baro is None):
            continue
        ac = row[8] if len(row) > 8 and isinstance(row[8], dict) else None
        if ac and ac.get("flight") and not b.callsign:
            b.callsign = str(ac["flight"]).strip()
        gs = _num(row[4])
        geom_rate = _num(row[11]) if len(row) > 11 else None
        vr = geom_rate if geom_rate is not None else _num(row[7])
        rows.append((_check_epoch(base + float(row[0])), lat, lon, alt_geom, alt_baro,
                     gs * KT_TO_MPS if gs is not None else None, _num(row[5]),
                     vr * FPM_TO_MPS if vr is not None else None))
    if n_ground:
        warnings.append(f"{n_ground} on-ground trace point(s) skipped.")
    n_geom = sum(r[3] is not None for r in rows)
    n_baro = sum(r[4] is not None for r in rows)
    if rows and n_geom == len(rows):
        basis, idx = "geometric", 3
    elif rows and n_baro == len(rows):
        basis, idx = "barometric", 4
    elif rows and n_geom >= 0.8 * len(rows):
        basis, idx = "geometric", 3
        warnings.append(f"{len(rows) - n_geom} point(s) without geometric altitude dropped (mixed altitude bases).")
    elif rows:
        raise TrackImportError("Trace mixes geometric and barometric altitudes; re-export with a single altitude basis.")
    else:
        basis, idx = "unknown", 3
    for r in rows:
        if r[idx] is not None:
            b.add(r[0], r[1], r[2], r[idx] * FT_TO_M, r[5], r[6], r[7])
    return _finish([b], opts.altitude_basis or basis, "readsb", filename, warnings)


def validate_uploaded_trajectory(traj: Dict[str, Any], index: int = 0) -> Dict[str, Any]:
    """
    Validate a trajectory dict supplied by a client (e.g. the result of a
    previous import) with the same rules as file imports, and force
    ``data_source="uploaded"`` so provenance cannot be spoofed.
    """
    if not isinstance(traj, dict):
        raise TrackImportError(f"Trajectory {index + 1} must be an object.")
    keys = ("timestamps", "latitudes", "longitudes", "altitudes")
    try:
        cols = [list(traj[k]) for k in keys]
    except (KeyError, TypeError):
        raise TrackImportError(f"Trajectory {index + 1} needs timestamps, latitudes, longitudes and altitudes arrays.") from None
    n = len(cols[0])
    if any(len(c) != n for c in cols):
        raise TrackImportError(f"Trajectory {index + 1}: arrays have different lengths.")
    opt = {}
    for k in ("speeds", "headings", "vertical_rates"):
        arr = traj.get(k) or []
        opt[k] = list(arr) if len(arr) == n else [None] * n
    key = str(traj.get("flight_id") or traj.get("icao24") or traj.get("callsign") or f"TRACK{index + 1}")
    b = _TrackBuilder(key)
    b.icao24 = str(traj["icao24"]) if traj.get("icao24") else None
    b.callsign = str(traj["callsign"]) if traj.get("callsign") else None
    b.aircraft_type = str(traj["aircraft_type"]) if traj.get("aircraft_type") else None
    for i in range(n):
        vals = [_num(cols[j][i]) for j in range(4)]
        if any(v is None for v in vals):
            raise TrackImportError(f"Trajectory {index + 1}: non-numeric value at point {i + 1}.")
        b.add(_check_epoch(vals[0]), vals[1], vals[2], vals[3], _num(opt["speeds"][i]), _num(opt["headings"][i]),
              _num(opt["vertical_rates"][i]))
    basis = traj.get("altitude_basis") if traj.get("altitude_basis") in ("geometric", "barometric", "unknown") else "unknown"
    warnings: List[str] = []
    out = _validate(b, basis, str(traj.get("source_format") or "client"), str(traj.get("source_name") or ""), warnings)
    if out is None:
        raise TrackImportError(f"Trajectory {index + 1}: " + "; ".join(warnings))
    out["flight_id"] = str(traj.get("flight_id") or out["flight_id"])
    return out
