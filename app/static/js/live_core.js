/**
 * Live listening core (DOM-free, unit-tested in tests/js/live_core.test.mjs).
 *
 * The microphone cannot be located from sound alone. What a live session can
 * do is what the rest of the app does with a recording: detect aircraft
 * flyovers (Doppler ridges) and match them against real ADS-B trajectories on
 * a shared, trusted clock. Live mode has two advantages over an old file:
 *   - the clock is known (each audio chunk is time-stamped as it arrives), and
 *   - live ADS-B can be polled for the whole session, so the browser assembles
 *     real multi-point tracks instead of single projected snapshots.
 * Every additional flyover from a different direction narrows the estimate.
 *
 * Pieces:
 *   Resampler        streaming linear-interpolation resampler (input is low-passed in the audio graph)
 *   RingBuffer       last N seconds of audio at the analysis rate, addressed by absolute sample index
 *   SampleClock      absolute sample index -> UTC epoch, re-anchored on gaps and slow clock drift
 *   FlyoverTracker   turns repeated ridge discoveries over a sliding window into committed, de-duplicated flyovers
 *   AdsbTrackStore   accumulates live ADS-B state vectors per aircraft into tracks
 *   refineSearchArea where to search next, given the latest result
 */
(function (root) {
    'use strict';

    // ------------------------------------------------------------- resampler
    class Resampler {
        constructor(inRate, outRate) {
            this.inRate = inRate;
            this.outRate = outRate;
            this.step = inRate / outRate; // input samples per output sample
            this.pos = 0; // position of the next output sample, relative to the start of the pending input
            this.prev = null; // last input sample of the previous chunk (for interpolation across chunks)
        }

        /** Resample one chunk; returns a Float32Array at outRate. */
        process(input) {
            const n = input.length;
            if (!n) return new Float32Array(0);
            // Index -1 refers to the previous chunk's last sample.
            const at = (i) => (i < 0 ? (this.prev == null ? input[0] : this.prev) : input[i]);
            const out = [];
            let p = this.pos;
            while (p <= n - 1) {
                const i0 = Math.floor(p);
                const frac = p - i0;
                const a = at(i0), b = i0 + 1 <= n - 1 ? input[i0 + 1] : a;
                out.push(a + (b - a) * frac);
                p += this.step;
            }
            this.pos = p - n;
            this.prev = input[n - 1];
            return Float32Array.from(out);
        }
    }

    // ----------------------------------------------------------- ring buffer
    class RingBuffer {
        constructor(capacity) {
            this.capacity = Math.max(1, Math.floor(capacity));
            this.data = new Float32Array(this.capacity);
            this.total = 0; // samples ever written; absolute index of the next sample
        }

        get oldest() { return Math.max(0, this.total - this.capacity); }

        push(chunk) {
            for (let i = 0; i < chunk.length; i++) this.data[(this.total + i) % this.capacity] = chunk[i];
            this.total += chunk.length;
        }

        /** Copy absolute samples [from, to) (clamped to what is still held). */
        read(from, to) {
            const a = Math.max(this.oldest, Math.floor(from)), b = Math.min(this.total, Math.floor(to));
            const out = new Float32Array(Math.max(0, b - a));
            for (let k = a; k < b; k++) out[k - a] = this.data[k % this.capacity];
            return { start: a, samples: out };
        }
    }

    // ----------------------------------------------------------------- clock
    /**
     * Maps absolute sample indices to UTC epoch seconds. Each audio chunk
     * reports the epoch of its first sample; a new anchor is started when the
     * reported time disagrees with the running prediction by more than
     * `jumpS` (dropped audio, suspended context), and small disagreements are
     * folded in slowly (audio vs system clock drift, timestamp jitter).
     */
    class SampleClock {
        constructor(rate, { jumpS = 0.25, gain = 0.05, minAnchorSpacingS = 10 } = {}) {
            this.rate = rate;
            this.jumpS = jumpS;
            this.gain = gain;
            this.minSpacing = minAnchorSpacingS * rate;
            this.anchors = []; // [{k, epoch}] increasing k
            this.offset = 0; // smoothed correction relative to the latest anchor's prediction
            this.discontinuities = 0;
        }

        predict(k) {
            const a = this.anchors[this.anchors.length - 1];
            return a.epoch + (k - a.k) / this.rate;
        }

        addChunk(k, epoch) {
            if (!Number.isFinite(epoch)) return;
            if (!this.anchors.length) { this.anchors.push({ k, epoch }); this.offset = 0; return; }
            const err = epoch - this.predict(k);
            if (Math.abs(err) > this.jumpS) {
                this.anchors.push({ k, epoch });
                this.offset = 0;
                this.discontinuities++;
                return;
            }
            this.offset += this.gain * (err - this.offset);
            const last = this.anchors[this.anchors.length - 1];
            if (k - last.k >= this.minSpacing && Math.abs(this.offset) > 1e-3) {
                this.anchors.push({ k, epoch: this.predict(k) + this.offset });
                this.offset = 0;
            }
        }

        toEpoch(k) {
            if (!this.anchors.length) return NaN;
            let lo = 0, hi = this.anchors.length - 1;
            if (k < this.anchors[0].k) lo = hi = 0;
            while (lo < hi) {
                const mid = (lo + hi + 1) >> 1;
                if (this.anchors[mid].k <= k) lo = mid; else hi = mid - 1;
            }
            const a = this.anchors[lo];
            return a.epoch + (k - a.k) / this.rate;
        }

        prune(kOldest) {
            while (this.anchors.length > 1 && this.anchors[1].k <= kOldest) this.anchors.shift();
        }
    }

    // -------------------------------------------------------- flyover tracker
    const TRACKER_DEFAULTS = {
        edge_guard_s: 2.0, // a group starting this close to the window start may be clipped
        settle_s: 8.0, // a group must have ended this long before the window end (flyover finished)
        min_duration_s: 6.0,
        same_freq_tol: 0.03, // relative frequency tolerance when matching a re-detection
        same_overlap: 0.5, // fraction of the shorter time span that must overlap
        max_flyovers: 40,
    };

    /**
     * Commits complete Doppler-like harmonic groups from successive analyses
     * of a sliding window. Times are stored as UTC epochs so flyovers keep
     * their identity as the window moves.
     */
    class FlyoverTracker {
        constructor(opts = {}) {
            this.opts = Object.assign({}, TRACKER_DEFAULTS, opts);
            this.flyovers = []; // [{id, epoch_start, epoch_end, ridges:[{epoch_times, freqs, median_freq_hz, ...}], updates}]
            this.counter = 0;
        }

        /**
         * @param discovery RidgeDiscovery result for the window
         * @param windowStartK absolute sample index of window time 0
         * @param rate analysis sample rate
         * @param toEpoch absolute sample index -> epoch
         * @param windowDurS window length in seconds
         * @returns {{added: Array, updated: Array}}
         */
        ingest(discovery, windowStartK, rate, toEpoch, windowDurS) {
            const o = this.opts;
            const ridges = (discovery && discovery.ridges) || [];
            const groups = new Map();
            for (const r of ridges) {
                if (!r.doppler_like || !r.points || r.points.length < 3) continue;
                const gid = r.harmonic_group_id || r.id;
                if (!groups.has(gid)) groups.set(gid, []);
                groups.get(gid).push(r);
            }
            const added = [], updated = [];
            for (const members of groups.values()) {
                const ts = Math.min(...members.map(r => r.t_start)), te = Math.max(...members.map(r => r.t_end));
                if (ts < o.edge_guard_s || te > windowDurS - o.settle_s || te - ts < o.min_duration_s) continue;
                const conv = (t) => toEpoch(windowStartK + t * rate);
                const cand = {
                    epoch_start: conv(ts),
                    epoch_end: conv(te),
                    ridges: members.map(r => ({
                        epoch_times: r.points.map(p => conv(p.time)),
                        freqs: r.points.map(p => p.freq),
                        median_freq_hz: r.median_freq_hz,
                        harmonic_ratio: r.harmonic_ratio || null,
                        flyover_fit: r.flyover_fit || null,
                    })),
                };
                if (!Number.isFinite(cand.epoch_start) || !Number.isFinite(cand.epoch_end)) continue;
                const same = this.flyovers.find(f => this.isSame(f, cand));
                if (same) {
                    const nNew = cand.ridges.reduce((s, r) => s + r.freqs.length, 0);
                    const nOld = same.ridges.reduce((s, r) => s + r.freqs.length, 0);
                    if (nNew > nOld) {
                        Object.assign(same, cand);
                        same.updates++;
                        updated.push(same);
                    }
                    continue;
                }
                const fly = Object.assign({ id: `F${++this.counter}`, updates: 0 }, cand);
                this.flyovers.push(fly);
                added.push(fly);
            }
            this.flyovers.sort((a, b) => a.epoch_start - b.epoch_start);
            while (this.flyovers.length > o.max_flyovers) this.flyovers.shift();
            return { added, updated };
        }

        isSame(f, c) {
            const lo = Math.max(f.epoch_start, c.epoch_start), hi = Math.min(f.epoch_end, c.epoch_end);
            const shorter = Math.min(f.epoch_end - f.epoch_start, c.epoch_end - c.epoch_start);
            if (shorter <= 0 || hi - lo < this.opts.same_overlap * shorter) return false;
            return f.ridges.some(a => c.ridges.some(b =>
                Math.abs(a.median_freq_hz - b.median_freq_hz) <= this.opts.same_freq_tol * Math.max(a.median_freq_hz, b.median_freq_hz)));
        }

        /**
         * Geolocation events for the most recent `maxFlyovers` flyovers, on a
         * clock starting at `origin` (UTC epoch, returned): one event per
         * ridge, ridges of one flyover share a group id.
         */
        events(maxFlyovers = 8) {
            const use = this.flyovers.slice(-maxFlyovers);
            if (!use.length) return { origin: null, events: [], flyovers: [] };
            const origin = Math.floor(Math.min(...use.map(f => f.epoch_start)) - 1);
            const events = [];
            for (const f of use) {
                f.ridges.forEach((r, j) => {
                    events.push({
                        id: `${f.id}${f.ridges.length > 1 ? `h${j + 1}` : ''}`,
                        group_id: f.id,
                        traced_curve: r.epoch_times.map((t, i) => ({
                            time: Math.round((t - origin) * 1000) / 1000,
                            freq: Math.round(r.freqs[i] * 100) / 100,
                        })),
                    });
                });
            }
            return { origin, events, flyovers: use };
        }
    }

    // ------------------------------------------------------- ADS-B track store
    const LIVE_SOURCES = new Set(['opensky_live', 'adsblol_live']);

    /**
     * Accumulates the single reported state of each live-snapshot trajectory
     * (see FlightManager._project_state: `state_time` marks the real report;
     * the rest of the arrays is a linear projection and is ignored).
     */
    class AdsbTrackStore {
        constructor({ maxAgeS = 3600 } = {}) {
            this.maxAgeS = maxAgeS;
            this.tracks = new Map(); // icao24 -> {icao24, callsign, aircraft_type, altitude_basis, data_source, points: []}
            this.polls = 0;
            this.lastSource = null;
        }

        static reportedState(f) {
            const st = Number(f.state_time);
            const ts = f.timestamps || [];
            if (!Number.isFinite(st) || !ts.length) return null;
            let best = -1, bestD = Infinity;
            for (let i = 0; i < ts.length; i++) {
                const d = Math.abs(ts[i] - st);
                if (d < bestD) { bestD = d; best = i; }
            }
            if (best < 0 || bestD > 0.6) return null;
            const num = (arr) => (arr && arr[best] != null && Number.isFinite(Number(arr[best])) ? Number(arr[best]) : null);
            const p = { t: st, lat: num(f.latitudes), lon: num(f.longitudes), alt: num(f.altitudes),
                speed: num(f.speeds), heading: num(f.headings), vr: num(f.vertical_rates) };
            return p.lat == null || p.lon == null || p.alt == null ? null : p;
        }

        /** Ingest one /api/flights/search response; returns the number of new state reports. */
        ingest(resp) {
            this.polls++;
            let added = 0;
            for (const f of (resp && resp.flights) || []) {
                if (!LIVE_SOURCES.has(f.data_source) || f.projection !== 'linear_from_single_state') continue;
                const p = AdsbTrackStore.reportedState(f);
                const key = String(f.icao24 || f.flight_id || '').toLowerCase();
                if (!p || !key) continue;
                let tr = this.tracks.get(key);
                if (!tr) {
                    tr = { icao24: key, callsign: f.callsign || key.toUpperCase(), aircraft_type: f.aircraft_type || null,
                        altitude_basis: f.altitude_basis || 'unknown', data_source: f.data_source, points: [] };
                    this.tracks.set(key, tr);
                }
                if (f.callsign) tr.callsign = f.callsign;
                if (tr.points.some(q => Math.abs(q.t - p.t) < 1e-3)) continue;
                tr.points.push(p);
                tr.points.sort((a, b) => a.t - b.t);
                tr.data_source = f.data_source;
                added++;
                this.lastSource = f.data_source;
            }
            return added;
        }

        prune(nowEpoch) {
            for (const [k, tr] of this.tracks) {
                tr.points = tr.points.filter(p => p.t >= nowEpoch - this.maxAgeS);
                if (!tr.points.length) this.tracks.delete(k);
            }
        }

        get size() { return this.tracks.size; }

        /** Tracks with >= 3 reports overlapping [t0 - pad, t1 + pad], in the trajectory format the API accepts. */
        trajectories(t0, t1, pad = 120) {
            const out = [];
            for (const tr of this.tracks.values()) {
                const pts = tr.points.filter(p => p.t >= t0 - pad && p.t <= t1 + pad);
                if (pts.length < 3) continue;
                out.push({
                    flight_id: tr.icao24.toUpperCase(), icao24: tr.icao24, callsign: tr.callsign,
                    aircraft_type: tr.aircraft_type || 'Unknown (ADS-B live)', altitude_basis: tr.altitude_basis,
                    source_format: 'live_poll', source_name: `${tr.data_source} polled by the browser`,
                    timestamps: pts.map(p => p.t), latitudes: pts.map(p => p.lat), longitudes: pts.map(p => p.lon),
                    altitudes: pts.map(p => p.alt), speeds: pts.map(p => p.speed), headings: pts.map(p => p.heading),
                    vertical_rates: pts.map(p => p.vr),
                });
            }
            return out;
        }

        toJSON() { return [...this.tracks.values()]; }
    }

    // ------------------------------------------------------------ refinement
    /**
     * Next matching area. After a supported result the search re-centres on
     * the estimate and shrinks to a few sensitivity radii (finer grid cells,
     * so more precision from the same grid size). After an inconclusive
     * result it widens back towards the user's area so a wrong lock-in can
     * recover.
     */
    function refineSearchArea(result, current, base, { minRadiusKm = 2, factor = 4 } = {}) {
        const decision = result && result.decision;
        const best = result && result.best_location;
        if (result && result.status === 'success' && decision && decision.status === 'supported' && best
            && Number.isFinite(best.uncertainty_radius_m)) {
            const r = Math.min(base.radius_km, Math.max(minRadiusKm, (factor * best.uncertainty_radius_m) / 1000));
            return { lat: best.lat, lon: best.lon, radius_km: Math.round(r * 10) / 10, refined: true };
        }
        const r = Math.min(base.radius_km, current.radius_km * 2);
        if (r >= base.radius_km) return { lat: base.lat, lon: base.lon, radius_km: base.radius_km, refined: false };
        return { lat: current.lat, lon: current.lon, radius_km: Math.round(r * 10) / 10, refined: current.refined };
    }

    function areaChanged(a, b) {
        if (!a || !b) return true;
        const dKm = Math.hypot((a.lat - b.lat) * 111.0, (a.lon - b.lon) * 111.0 * Math.cos((a.lat * Math.PI) / 180));
        return dKm > 0.25 * Math.min(a.radius_km, b.radius_km) || Math.abs(a.radius_km - b.radius_km) > 0.2 * Math.min(a.radius_km, b.radius_km);
    }

    root.LiveCore = { Resampler, RingBuffer, SampleClock, FlyoverTracker, AdsbTrackStore, refineSearchArea, areaChanged,
        TRACKER_DEFAULTS };
})(typeof window !== 'undefined' ? window : globalThis);
