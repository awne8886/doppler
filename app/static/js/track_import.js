/**
 * Browser track import (mirrors app/adsb/track_import.py) so the static
 * GitHub Pages build can match recordings against real, uploaded tracks.
 *
 * Formats: generic CSV, FlightRadar24 CSV, KML (gx:Track or time-stamped
 * Placemarks), GeoJSON (LineString + times, or time-stamped Points) and
 * readsb/tar1090 trace JSON (gzip accepted where DecompressionStream exists).
 * Same validation rules: explicit units, ISO times need an offset or a
 * timezone option, plausible coordinates / altitude / implied speed.
 */
(function (root) {
    const FT = 0.3048, KT = 0.514444, KMH = 1 / 3.6, FPM = 0.00508;
    const MAX_ALT = 20000, MIN_ALT = -500, MAX_SPEED = 450;

    class TrackImportError extends Error {}

    function tzOffsetMinutes(tz) {
        if (tz == null || String(tz).trim() === '') return null;
        const s = String(tz).trim().toUpperCase();
        if (['UTC', 'Z', 'GMT'].includes(s)) return 0;
        const m = /^(?:UTC)?([+-])(\d{1,2}):?(\d{2})?$/.exec(s);
        if (!m) throw new TrackImportError(`Unsupported timezone '${tz}'. Use 'UTC' or an offset such as '+02:00'.`);
        return (m[1] === '+' ? 1 : -1) * (parseInt(m[2], 10) * 60 + parseInt(m[3] || '0', 10));
    }

    function checkEpoch(x) {
        if (!(x >= 631152000 && x <= 4102444800)) throw new TrackImportError(`Timestamp ${Math.round(x)} is outside 1990-2100.`);
        return x;
    }

    function epochFromNumber(x) {
        if (x > 1e11) x /= 1000;
        if (x < 1e8) throw new TrackImportError(`Timestamp ${x} looks relative, not absolute. Tracks need Unix epoch or ISO 8601 times.`);
        return checkEpoch(x);
    }

    function parseTime(value, tzMin) {
        if (typeof value === 'number') return epochFromNumber(value);
        const s = String(value == null ? '' : value).trim();
        if (!s) throw new TrackImportError('Empty timestamp.');
        if (/^[+-]?\d+(\.\d+)?$/.test(s)) return epochFromNumber(parseFloat(s));
        let iso = /^\d{4}-\d{2}-\d{2} \d/.test(s) ? s.replace(' ', 'T') : s;
        const hasZone = /([zZ]|[+-]\d{2}:?\d{2})$/.test(iso);
        if (!/^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?/.test(iso)) {
            throw new TrackImportError(`Unrecognised timestamp '${s}'. Use Unix seconds/milliseconds or ISO 8601.`);
        }
        if (!hasZone) {
            if (tzMin == null) {
                throw new TrackImportError(`Timestamp '${s}' has no time-zone offset. Pass timezone='UTC' (or an offset like '+02:00') to state how it should be read.`);
            }
            const ms = Date.parse(iso + 'Z');
            if (isNaN(ms)) throw new TrackImportError(`Unrecognised timestamp '${s}'.`);
            return checkEpoch(ms / 1000 - tzMin * 60);
        }
        const ms = Date.parse(iso);
        if (isNaN(ms)) throw new TrackImportError(`Unrecognised timestamp '${s}'.`);
        return checkEpoch(ms / 1000);
    }

    function unitFromName(name, kind) {
        const n = name.toLowerCase().replace(/ /g, '');
        if (kind === 'alt') {
            if (/(_|\()(ft|feet)\)?$|feet/.test(n)) return 'ft';
            if (/(_|\()(m|meters|metres)\)?$/.test(n)) return 'm';
        } else if (kind === 'speed') {
            if (/(kt|kts|knots)\)?$/.test(n)) return 'kt';
            if (/(mps|m\/s|ms)\)?$/.test(n)) return 'mps';
            if (/(kmh|km\/h|kph)\)?$/.test(n)) return 'kmh';
        } else if (kind === 'vrate') {
            if (/(fpm|ft\/min)\)?$/.test(n)) return 'fpm';
            if (/(mps|m\/s)\)?$/.test(n)) return 'mps';
        }
        return null;
    }
    const ALT_F = { ft: FT, m: 1 }, SPD_F = { kt: KT, mps: 1, kmh: KMH }, VR_F = { fpm: FPM, mps: 1 };

    function num(v) {
        if (v == null) return null;
        if (typeof v === 'number') return Number.isFinite(v) ? v : null;
        const s = String(v).trim();
        if (!s || ['nan', 'null', 'none', '-', 'ground'].includes(s.toLowerCase())) return null;
        const x = Number(s);
        return Number.isFinite(x) ? x : null;
    }

    function haversine(lat1, lon1, lat2, lon2) {
        const r = Math.PI / 180, p1 = lat1 * r, p2 = lat2 * r, dp = p2 - p1, dl = (lon2 - lon1) * r;
        const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
        return 2 * 6371000 * Math.asin(Math.min(1, Math.sqrt(a)));
    }

    function builder(key) { return { key, callsign: null, icao24: null, aircraft_type: null, rows: [] }; }

    function validate(b, basis, fmt, sourceName, warnings) {
        const rows = b.rows.slice().sort((p, q) => p[0] - q[0]);
        const dedup = [];
        for (const r of rows) {
            if (!(r[1] >= -90 && r[1] <= 90 && r[2] >= -180 && r[2] <= 180)) {
                throw new TrackImportError(`Track ${b.key}: coordinate out of range (${r[1]}, ${r[2]}).`);
            }
            if (dedup.length && Math.abs(r[0] - dedup[dedup.length - 1][0]) < 1e-6) continue;
            dedup.push(r);
        }
        if (dedup.length < 3) { warnings.push(`Track ${b.key}: fewer than 3 distinct points; skipped.`); return null; }
        const alts = dedup.map(r => r[3]);
        if (Math.max(...alts) > MAX_ALT || Math.min(...alts) < MIN_ALT) {
            throw new TrackImportError(`Track ${b.key}: altitude range ${Math.round(Math.min(...alts))}..${Math.round(Math.max(...alts))} m is implausible; check the altitude unit (feet vs metres).`);
        }
        const kept = [dedup[0]];
        let dropped = 0;
        for (const r of dedup.slice(1)) {
            const p = kept[kept.length - 1];
            if (haversine(p[1], p[2], r[1], r[2]) / (r[0] - p[0]) > MAX_SPEED) { dropped++; continue; }
            kept.push(r);
        }
        if (dropped) warnings.push(`Track ${b.key}: ${dropped} point(s) dropped for implying > ${MAX_SPEED} m/s.`);
        if (kept.length < 3) { warnings.push(`Track ${b.key}: fewer than 3 consistent points; skipped.`); return null; }
        const all = (k) => kept.every(r => r[k] != null);
        if (all(4) && kept.length >= 5) {
            const implied = [];
            for (let i = 0; i + 2 < kept.length; i++) {
                implied.push(haversine(kept[i][1], kept[i][2], kept[i + 2][1], kept[i + 2][2]) / (kept[i + 2][0] - kept[i][0]));
            }
            const med = (a) => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)];
            const rep = med(kept.map(r => r[4])), imp = med(implied);
            if (imp > 20 && rep > 0 && Math.abs(rep / imp - 1) > 0.35) {
                warnings.push(`Track ${b.key}: reported speed (${Math.round(rep)} m/s) differs from position-implied speed (${Math.round(imp)} m/s) by a factor ${(rep / imp).toFixed(2)}; check the speed unit.`);
            }
        }
        const fid = String(b.icao24 || b.callsign || b.key || 'TRACK').trim();
        return {
            flight_id: fid.toUpperCase(), icao24: b.icao24 ? b.icao24.toLowerCase() : null,
            callsign: (b.callsign || '').trim() || fid.toUpperCase(),
            aircraft_type: b.aircraft_type || 'Unknown (uploaded track)',
            timestamps: kept.map(r => r[0]), latitudes: kept.map(r => r[1]), longitudes: kept.map(r => r[2]),
            altitudes: kept.map(r => r[3]),
            speeds: all(4) ? kept.map(r => r[4]) : [], headings: all(5) ? kept.map(r => r[5]) : [],
            vertical_rates: all(6) ? kept.map(r => r[6]) : [],
            data_source: 'uploaded', altitude_basis: basis, source_format: fmt, source_name: sourceName,
        };
    }

    function finish(builders, basis, fmt, name, warnings) {
        const flights = builders.map(b => validate(b, basis, fmt, name, warnings)).filter(Boolean);
        if (!flights.length) throw new TrackImportError('No usable track (each track needs at least 3 valid, time-stamped airborne points).');
        return flights;
    }

    // ---------------------------------------------------------------- CSV
    function splitCsvLine(line) {
        const out = [];
        let cur = '', q = false;
        for (let i = 0; i < line.length; i++) {
            const ch = line[i];
            if (q) {
                if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; } else if (ch === '"') q = false; else cur += ch;
            } else if (ch === '"') q = true;
            else if (ch === ',') { out.push(cur); cur = ''; } else cur += ch;
        }
        out.push(cur);
        return out;
    }
    const ALIASES = {
        time: ['timestamp', 'time', 'utc', 'datetime', 'date_time', 'epoch', 'unix_time', 'unixtime', 't', 'time_utc'],
        lat: ['lat', 'latitude'], lon: ['lon', 'lng', 'long', 'longitude'],
        alt: ['alt', 'altitude', 'alt_geom', 'geo_altitude', 'geoaltitude', 'alt_baro', 'baro_altitude', 'baroaltitude', 'height'],
        speed: ['speed', 'gs', 'ground_speed', 'groundspeed', 'velocity'],
        track: ['track', 'heading', 'true_track', 'course', 'direction'],
        vrate: ['vertical_rate', 'vrate', 'baro_rate', 'geom_rate', 'vert_rate', 'climb_rate'],
        id: ['flight_id', 'icao24', 'hex', 'icao', 'icao_hex'], callsign: ['callsign', 'flight', 'call_sign'],
        type: ['aircraft_type', 'type', 'actype'],
    };
    function stripUnit(col) {
        return col.trim().toLowerCase().replace(/\s*\(.*\)\s*$/, '')
            .replace(/_(ft|feet|m|meters|metres|kt|kts|knots|mps|ms|kmh|kph|fpm|deg|degrees|s|utc)$/, '').replace(/ /g, '_');
    }
    function basisFromName(name) {
        const n = name.toLowerCase();
        if (/geom|geo_|geoalt|gnss|gps/.test(n)) return 'geometric';
        if (/baro/.test(n)) return 'barometric';
        return 'unknown';
    }

    function parseCsv(text, name, o, tz, warnings) {
        const lines = text.replace(/^﻿/, '').split(/\r?\n/);
        const header = splitCsvLine(lines[0] || '');
        const cols = {};
        header.forEach((c, i) => {
            const base = stripUnit(c);
            for (const [k, al] of Object.entries(ALIASES)) if (!(k in cols) && (al.includes(base) || al.includes(c.trim().toLowerCase()))) cols[k] = i;
        });
        const missing = ['time', 'lat', 'lon', 'alt'].filter(k => !(k in cols));
        if (missing.length) throw new TrackImportError(`CSV is missing required column(s): ${missing.join(', ')}.`);
        const altName = header[cols.alt];
        const altUnit = o.altitude_unit !== 'auto' ? o.altitude_unit : unitFromName(altName, 'alt');
        if (!altUnit) throw new TrackImportError(`Altitude column '${altName}' does not state its unit. Rename it (e.g. alt_ft / alt_m) or pass altitude_unit.`);
        let spdUnit = null, vrUnit = null;
        if ('speed' in cols) {
            spdUnit = o.speed_unit !== 'auto' ? o.speed_unit : unitFromName(header[cols.speed], 'speed');
            if (!spdUnit) warnings.push(`Speed column '${header[cols.speed]}' has no unit; speeds ignored (pass speed_unit to use them).`);
        }
        if ('vrate' in cols) {
            vrUnit = o.vertical_rate_unit !== 'auto' ? o.vertical_rate_unit : unitFromName(header[cols.vrate], 'vrate');
            if (!vrUnit) warnings.push(`Vertical-rate column '${header[cols.vrate]}' has no unit; ignored.`);
        }
        const basis = o.altitude_basis || basisFromName(altName);
        const builders = new Map();
        let skipped = 0;
        for (const line of lines.slice(1)) {
            if (!line.trim()) continue;
            const row = splitCsvLine(line);
            const get = (k) => (k in cols && cols[k] < row.length ? row[cols[k]] : null);
            const t = parseTime(get('time'), tz);
            const lat = num(get('lat')), lon = num(get('lon')), alt = num(get('alt'));
            if (lat == null || lon == null || alt == null) { skipped++; continue; }
            const key = String(get('id') || get('callsign') || 'TRACK1').trim();
            if (!builders.has(key)) builders.set(key, builder(key));
            const b = builders.get(key);
            if ('id' in cols) b.icao24 = b.icao24 || String(get('id') || '').trim();
            if ('callsign' in cols) b.callsign = b.callsign || String(get('callsign') || '').trim();
            if ('type' in cols) b.aircraft_type = b.aircraft_type || String(get('type') || '').trim() || null;
            const spd = num(get('speed')), vr = num(get('vrate'));
            b.rows.push([t, lat, lon, alt * ALT_F[altUnit], spd != null && spdUnit ? spd * SPD_F[spdUnit] : null,
                num(get('track')), vr != null && vrUnit ? vr * VR_F[vrUnit] : null]);
        }
        if (skipped) warnings.push(`${skipped} CSV row(s) without time, position or altitude were skipped.`);
        return finish([...builders.values()], basis, 'csv', name, warnings);
    }

    function parseFr24(text, name, o, tz, warnings) {
        const lines = text.replace(/^﻿/, '').split(/\r?\n/);
        const header = splitCsvLine(lines[0]).map(h => h.trim().toLowerCase());
        const altUnit = o.altitude_unit !== 'auto' ? o.altitude_unit : 'ft';
        const spdUnit = o.speed_unit !== 'auto' ? o.speed_unit : 'kt';
        const b = builder(name || 'FR24');
        let skipped = 0;
        for (const line of lines.slice(1)) {
            if (!line.trim()) continue;
            const v = splitCsvLine(line);
            const r = {};
            header.forEach((h, i) => { r[h] = (v[i] || '').trim(); });
            let t, lat, lon;
            try {
                t = parseTime(r.timestamp || r.utc, tz == null ? 0 : tz);
                [lat, lon] = (r.position || '').split(',').map(Number);
                if (!Number.isFinite(lat) || !Number.isFinite(lon)) throw new Error('bad position');
            } catch (e) { skipped++; continue; }
            const alt = num(r.altitude);
            if (alt == null) { skipped++; continue; }
            const spd = num(r.speed);
            if (alt <= 0 && spd != null && spd < 50) continue;
            b.callsign = b.callsign || r.callsign || null;
            b.rows.push([t, lat, lon, alt * ALT_F[altUnit], spd != null ? spd * SPD_F[spdUnit] : null, num(r.direction), null]);
        }
        if (skipped) warnings.push(`${skipped} FR24 row(s) could not be parsed and were skipped.`);
        b.key = b.callsign || b.key;
        return finish([b], o.altitude_basis || 'barometric', 'fr24_csv', name, warnings);
    }

    // ---------------------------------------------------------------- KML
    function parseKml(text, name, o, tz, warnings) {
        if (typeof DOMParser === 'undefined') throw new TrackImportError('KML parsing needs a browser DOMParser.');
        const doc = new DOMParser().parseFromString(text, 'application/xml');
        if (doc.getElementsByTagName('parsererror').length) throw new TrackImportError('Invalid KML.');
        const altUnit = o.altitude_unit !== 'auto' ? o.altitude_unit : 'm';
        const local = (el) => el.localName || el.nodeName.replace(/^.*:/, '');
        const all = (tag, rootEl = doc) => [...rootEl.getElementsByTagName('*')].filter(el => local(el) === tag);
        const nameEl = all('name').find(el => el.textContent.trim());
        const docName = nameEl ? nameEl.textContent.trim() : (name || 'KML');
        const builders = [];
        all('Track').forEach((track, i) => {
            const whens = [...track.children].filter(el => local(el) === 'when').map(el => el.textContent);
            const coords = [...track.children].filter(el => local(el) === 'coord').map(el => el.textContent);
            if (whens.length !== coords.length) throw new TrackImportError('KML gx:Track has a different number of <when> and <gx:coord> elements.');
            const b = builder(i ? `${docName}-${i + 1}` : docName);
            b.callsign = docName;
            whens.forEach((w, k) => {
                const p = coords[k].trim().split(/\s+/).map(Number);
                if (p.length >= 3) b.rows.push([parseTime(w, tz), p[1], p[0], p[2] * ALT_F[altUnit], null, null, null]);
            });
            builders.push(b);
        });
        if (!builders.length) {
            const b = builder(docName);
            b.callsign = docName;
            for (const pm of all('Placemark')) {
                const when = all('when', pm)[0], point = all('Point', pm)[0];
                if (!when || !point) continue;
                const c = (all('coordinates', point)[0] || { textContent: '' }).textContent.trim().split(',').map(Number);
                if (c.length >= 3) b.rows.push([parseTime(when.textContent, tz), c[1], c[0], c[2] * ALT_F[altUnit], null, null, null]);
            }
            builders.push(b);
        }
        return finish(builders, o.altitude_basis || 'unknown', 'kml', name, warnings);
    }

    // ------------------------------------------------------------ GeoJSON
    function parseGeojson(obj, name, o, tz, warnings) {
        const feats = obj.type === 'FeatureCollection' ? obj.features : [obj];
        const altUnit = o.altitude_unit !== 'auto' ? o.altitude_unit : 'm';
        const builders = new Map();
        let basisProp = null;
        (feats || []).forEach((f, i) => {
            const g = (f && f.geometry) || {}, p = (f && f.properties) || {};
            basisProp = basisProp || p.altitude_basis;
            const key = String(p.flight_id || p.icao24 || p.callsign || p.name || `TRACK${i + 1}`);
            let b;
            if (g.type === 'LineString' || g.type === 'MultiLineString') {
                const coords = g.type === 'MultiLineString' ? g.coordinates.flat() : g.coordinates;
                let times = p.times || p.coordTimes || p.timestamps;
                if (Array.isArray(times) && Array.isArray(times[0])) times = times.flat();
                if (!Array.isArray(times) || times.length !== coords.length) {
                    throw new TrackImportError(`GeoJSON feature '${key}' needs a 'times' (or 'coordTimes') array matching its coordinates.`);
                }
                b = builder(key);
                builders.set(`${key}#${i}`, b);
                coords.forEach((c, k) => {
                    if (c.length < 3) throw new TrackImportError(`GeoJSON feature '${key}' coordinates need [lon, lat, altitude].`);
                    b.rows.push([parseTime(times[k], tz), c[1], c[0], c[2] * ALT_F[altUnit], null, null, null]);
                });
            } else if (g.type === 'Point') {
                const c = g.coordinates || [];
                const t = p.time != null ? p.time : p.timestamp;
                const alt = c.length >= 3 ? c[2] : num(p.altitude_m);
                if (t == null || c.length < 2 || alt == null) return;
                if (!builders.has(key)) builders.set(key, builder(key));
                b = builders.get(key);
                b.rows.push([parseTime(t, tz), c[1], c[0], alt * ALT_F[altUnit], num(p.speed_mps), num(p.track), num(p.vertical_rate_mps)]);
            } else return;
            b.callsign = b.callsign || (p.callsign ? String(p.callsign) : null);
            b.icao24 = b.icao24 || (p.icao24 ? String(p.icao24) : null);
            b.aircraft_type = b.aircraft_type || p.aircraft_type || null;
        });
        return finish([...builders.values()], o.altitude_basis || basisProp || 'unknown', 'geojson', name, warnings);
    }

    // ------------------------------------------------------------- readsb
    function parseReadsb(obj, name, o, tz, warnings) {
        const base = num(obj.timestamp);
        if (base == null) throw new TrackImportError("readsb trace has no 'timestamp'.");
        const icao = String(obj.icao || '').toLowerCase();
        const b = builder(icao || 'TRACE');
        b.icao24 = icao || null;
        b.aircraft_type = obj.t || obj.desc || null;
        const rows = [];
        let ground = 0;
        for (const row of obj.trace || []) {
            if (!Array.isArray(row) || row.length < 6) continue;
            if (row[3] === 'ground') { ground++; continue; }
            const flags = typeof row[6] === 'number' ? row[6] : 0;
            const field = num(row[3]);
            let geom = row.length > 10 ? num(row[10]) : null, baro = field;
            if (flags & 8) { geom = geom != null ? geom : field; baro = null; }
            const lat = num(row[1]), lon = num(row[2]);
            if (lat == null || lon == null || (geom == null && baro == null)) continue;
            const ac = row.length > 8 && row[8] && typeof row[8] === 'object' ? row[8] : null;
            if (ac && ac.flight && !b.callsign) b.callsign = String(ac.flight).trim();
            const gs = num(row[4]);
            const gr = row.length > 11 ? num(row[11]) : null;
            const vr = gr != null ? gr : num(row[7]);
            rows.push([checkEpoch(base + Number(row[0])), lat, lon, geom, baro, gs != null ? gs * KT : null, num(row[5]), vr != null ? vr * FPM : null]);
        }
        if (ground) warnings.push(`${ground} on-ground trace point(s) skipped.`);
        const nGeom = rows.filter(r => r[3] != null).length, nBaro = rows.filter(r => r[4] != null).length;
        let basis = 'unknown', idx = 3;
        if (rows.length && nGeom === rows.length) basis = 'geometric';
        else if (rows.length && nBaro === rows.length) { basis = 'barometric'; idx = 4; }
        else if (rows.length && nGeom >= 0.8 * rows.length) {
            basis = 'geometric';
            warnings.push(`${rows.length - nGeom} point(s) without geometric altitude dropped (mixed altitude bases).`);
        } else if (rows.length) throw new TrackImportError('Trace mixes geometric and barometric altitudes; re-export with a single altitude basis.');
        for (const r of rows) if (r[idx] != null) b.rows.push([r[0], r[1], r[2], r[idx] * FT, r[5], r[6], r[7]]);
        return finish([b], o.altitude_basis || basis, 'readsb', name, warnings);
    }

    // ------------------------------------------------------------ entry
    function detectFormat(text, name) {
        const head = text.slice(0, 4096).trimStart();
        if (head[0] === '{' || head[0] === '[') {
            let obj;
            try { obj = JSON.parse(text); } catch (e) { throw new TrackImportError('File looks like JSON but could not be parsed.'); }
            if (obj && obj.trace) return ['readsb', obj];
            if (obj && (obj.type === 'FeatureCollection' || obj.type === 'Feature')) return ['geojson', obj];
            throw new TrackImportError('JSON file is neither GeoJSON nor a readsb/tar1090 trace.');
        }
        if (head.startsWith('<?xml') || head.slice(0, 2048).includes('<kml') || /\.kml$/i.test(name || '')) return ['kml', null];
        const cols = (text.split(/\r?\n/)[0] || '').split(',').map(c => c.trim().toLowerCase());
        if (['timestamp', 'utc', 'callsign', 'position'].every(c => cols.includes(c))) return ['fr24_csv', null];
        return ['csv', null];
    }

    async function bytesToText(buf) {
        const u8 = new Uint8Array(buf);
        if (u8[0] === 0x1f && u8[1] === 0x8b) {
            if (typeof DecompressionStream === 'undefined') throw new TrackImportError('This browser cannot decompress gzip files; upload the uncompressed file.');
            const stream = new Blob([u8]).stream().pipeThrough(new DecompressionStream('gzip'));
            return await new Response(stream).text();
        }
        return new TextDecoder('utf-8').decode(u8);
    }

    /** Parse track text (already decompressed). Returns {format, flights, warnings}. */
    function parseTracksText(text, name = '', fmt = null, options = {}) {
        const o = Object.assign({ altitude_unit: 'auto', speed_unit: 'auto', vertical_rate_unit: 'auto', timezone: null, altitude_basis: null }, options);
        if (!text.trim()) throw new TrackImportError('Empty file.');
        const tz = tzOffsetMinutes(o.timezone);
        const warnings = [];
        let obj = null;
        if (!fmt) [fmt, obj] = detectFormat(text, name);
        else if (fmt === 'readsb' || fmt === 'geojson') obj = JSON.parse(text);
        const parsers = { csv: parseCsv, fr24_csv: parseFr24, kml: parseKml };
        let flights;
        if (fmt === 'geojson') flights = parseGeojson(obj, name, o, tz, warnings);
        else if (fmt === 'readsb') flights = parseReadsb(obj, name, o, tz, warnings);
        else if (parsers[fmt]) flights = parsers[fmt](text, name, o, tz, warnings);
        else throw new TrackImportError(`Unknown format '${fmt}'.`);
        return { format: fmt, flights, warnings, data_source: 'uploaded', count: flights.length };
    }

    async function parseTrackFile(file, fmt = null, options = {}) {
        const text = await bytesToText(await file.arrayBuffer());
        return parseTracksText(text, file.name, fmt, options);
    }

    root.TrackImport = { parseTracksText, parseTrackFile, parseTime, TrackImportError };
})(typeof window !== 'undefined' ? window : globalThis);
