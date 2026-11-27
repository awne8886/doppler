/**
 * Client-side Doppler analysis (mirrors the Python backend so the static
 * GitHub Pages build works without a server):
 *
 *  - FlyoverFit:      constant-velocity flyover model (exact retarded time) + robust fit
 *  - RidgeDiscovery:  automatic bounded-slope Viterbi ridge tracking + harmonic grouping
 *  - JointMatcher:    fitted f0 / bounded drift, pseudo-Huber loss, shared recording clock,
 *                     harmonic groups, refinement, alternatives and sensitivity
 *  - decideLocation:  supported vs inconclusive evidence checks
 *
 * Scores are ranking statistics, not calibrated probabilities.
 */
(function (root) {
    const window = root;
    const C_DEFAULT = 343.2;

    // ------------------------------------------------------------- geodesy
    /** Local tangent-plane conversions (same equirectangular mid-latitude form as the Python engine). */
    class GeoEngine {
        constructor(speedOfSound = C_DEFAULT, earthRadius = 6371000.0) {
            this.c = speedOfSound;
            this.R = earthRadius;
        }
        geodeticToCartesian(lat, lon, alt, refLat, refLon, refAlt = 0.0) {
            const toRad = Math.PI / 180.0;
            const latRad = lat * toRad, refLatRad = refLat * toRad;
            const latMid = (latRad + refLatRad) / 2.0;
            return {
                x: this.R * (lon * toRad - refLon * toRad) * Math.cos(latMid),
                y: this.R * (latRad - refLatRad),
                z: alt - refAlt,
            };
        }
        cartesianToGeodetic(x, y, z, refLat, refLon, refAlt = 0.0) {
            const toDeg = 180.0 / Math.PI, toRad = Math.PI / 180.0;
            const refLatRad = refLat * toRad;
            const latRad = refLatRad + y / this.R;
            const latMid = (latRad + refLatRad) / 2.0;
            const lonRad = refLon * toRad + x / (this.R * Math.cos(latMid));
            return { lat: latRad * toDeg, lon: lonRad * toDeg, alt: z + refAlt };
        }
    }

    /** Speed of sound in humid air, Cramer (1993); mirrors DopplerEngine.calculate_speed_of_sound. */
    function speedOfSound(tempC = 20.0, relativeHumidity = 0.5, pressurePa = 101325.0, co2 = 0.0004) {
        let h = relativeHumidity > 1 ? relativeHumidity / 100 : relativeHumidity;
        h = Math.min(Math.max(h, 0), 1);
        const t = tempC, T = t + 273.15, p = pressurePa;
        const psv = Math.exp(1.2811805e-5 * T * T - 1.9509874e-2 * T + 34.04926034 - 6.3536311e3 / T);
        const fEnh = 1.00062 + 3.14e-8 * p + 5.6e-7 * t * t;
        const xw = h * fEnh * psv / p, xc = co2;
        const a = [331.5024, 0.603055, -0.000528, 51.471935, 0.1495874, -0.000782, -1.82e-7, 3.73e-8, -2.93e-10,
            -85.20931, -0.228525, 5.91e-5, -2.835149, -2.15e-13, 29.179762, 0.000486];
        return a[0] + a[1] * t + a[2] * t * t
            + (a[3] + a[4] * t + a[5] * t * t) * xw
            + (a[6] + a[7] * t + a[8] * t * t) * p
            + (a[9] + a[10] * t + a[11] * t * t) * xc
            + a[12] * xw * xw + a[13] * p * p + a[14] * xc * xc + a[15] * xw * p * xc;
    }

    /** API semantics: 343.2 m/s unless weather is given; then Cramer with RH defaulting to 50 %. */
    function speedOfSoundFor(temperatureC, humidityPct) {
        if (temperatureC == null && humidityPct == null) return C_DEFAULT;
        return speedOfSound(temperatureC == null ? 20 : temperatureC, (humidityPct == null ? 50 : humidityPct) / 100);
    }

    // ------------------------------------------------------------ utilities
    function median(arr) {
        // Typed-array sort is numeric and several times faster than a comparator sort (same result for finite values).
        const a = Float64Array.from(arr).sort();
        const n = a.length;
        if (n === 0) return NaN;
        return n % 2 ? a[(n - 1) >> 1] : 0.5 * (a[n / 2 - 1] + a[n / 2]);
    }
    function percentile(arr, p) {
        const a = Array.from(arr).sort((x, y) => x - y);
        const idx = (p / 100) * (a.length - 1);
        const lo = Math.floor(idx), hi = Math.ceil(idx);
        return a[lo] + (a[hi] - a[lo]) * (idx - lo);
    }
    function interpSorted(x, xs, ys) {
        const n = xs.length;
        if (x <= xs[0]) return ys[0];
        if (x >= xs[n - 1]) return ys[n - 1];
        let lo = 0, hi = n - 1;
        while (hi - lo > 1) {
            const mid = (lo + hi) >> 1;
            if (xs[mid] <= x) lo = mid; else hi = mid;
        }
        const f = (x - xs[lo]) / (xs[hi] - xs[lo] || 1e-12);
        return ys[lo] + f * (ys[hi] - ys[lo]);
    }
    /** numpy.round semantics (round half to even). */
    function roundHalfEven(x) {
        const f = Math.floor(x), d = x - f;
        if (d > 0.5) return f + 1;
        if (d < 0.5) return f;
        return f % 2 === 0 ? f : f + 1;
    }
    /** numpy.arange semantics: start + i * step for i < ceil((stop - start) / step). */
    function arange(start, stop, step) {
        const n = Math.max(0, Math.ceil((stop - start) / step));
        const out = new Array(n);
        for (let i = 0; i < n; i++) out[i] = start + i * step;
        return out;
    }
    /** numpy.linspace semantics (endpoint included). */
    function linspace(start, stop, num) {
        if (num === 1) return [start];
        const step = (stop - start) / (num - 1);
        const out = new Array(num);
        for (let i = 0; i < num; i++) out[i] = i * step + start;
        out[num - 1] = stop;
        return out;
    }
    function uniqueSorted(arr) {
        const a = arr.slice().sort((x, y) => x - y);
        return a.filter((v, i) => i === 0 || v !== a[i - 1]);
    }

    function nelderMead(f, x0, steps, maxIter = 400, tol = 1e-8) {
        const n = x0.length;
        let simplex = [x0.slice()];
        for (let i = 0; i < n; i++) {
            const p = x0.slice();
            p[i] += steps[i];
            simplex.push(p);
        }
        let vals = simplex.map(f);
        for (let it = 0; it < maxIter; it++) {
            const order = vals.map((v, i) => i).sort((a, b) => vals[a] - vals[b]);
            simplex = order.map(i => simplex[i]);
            vals = order.map(i => vals[i]);
            if (Math.abs(vals[n] - vals[0]) <= tol * (Math.abs(vals[0]) + 1e-12)) break;
            const cen = new Array(n).fill(0);
            for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cen[j] += simplex[i][j] / n;
            const xr = cen.map((c, j) => c + (c - simplex[n][j]));
            const fr = f(xr);
            if (fr < vals[0]) {
                const xe = cen.map((c, j) => c + 2 * (c - simplex[n][j]));
                const fe = f(xe);
                if (fe < fr) { simplex[n] = xe; vals[n] = fe; } else { simplex[n] = xr; vals[n] = fr; }
            } else if (fr < vals[n - 1]) {
                simplex[n] = xr; vals[n] = fr;
            } else {
                const xc = cen.map((c, j) => c + 0.5 * (simplex[n][j] - c));
                const fc = f(xc);
                if (fc < vals[n]) { simplex[n] = xc; vals[n] = fc; }
                else {
                    for (let i = 1; i <= n; i++) {
                        simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j]));
                        vals[i] = f(simplex[i]);
                    }
                }
            }
        }
        let bi = 0;
        for (let i = 1; i <= n; i++) if (vals[i] < vals[bi]) bi = i;
        return { x: simplex[bi], fx: vals[bi] };
    }

    // ---------------------------------------------------------- flyover fit
    const FlyoverFit = {
        frequency(t, f0, v, d, tau0, c = C_DEFAULT) {
            const T = t - tau0;
            const a = c * c - v * v;
            const disc = Math.max(0, c * c * v * v * T * T + a * d * d);
            const s = (c * c * T - Math.sqrt(disc)) / a;
            const r = Math.sqrt(d * d + v * v * s * s);
            const ur = (v * v * s) / Math.max(r, 1e-9);
            return (f0 * c) / (c + ur);
        },

        fit(times, freqs, c = C_DEFAULT, lossScale = 2.0) {
            const pts = times.map((t, i) => [t, freqs[i]]).filter(p => isFinite(p[0]) && isFinite(p[1]))
                .sort((a, b) => a[0] - b[0]);
            if (pts.length < 5) throw new Error('At least 5 ridge samples are required for a flyover fit.');
            const t = pts.map(p => p[0]), f = pts.map(p => p[1]);
            const fHi = percentile(f, 95), fLo = percentile(f, 5), fMid = 0.5 * (fHi + fLo);
            const span = t[t.length - 1] - t[0];
            const ratio = Math.max(1.0001, fHi / Math.max(fLo, 1e-6));
            const vInit = Math.min(329, Math.max(16, (c * (ratio - 1)) / (ratio + 1)));
            let iMid = 0;
            for (let i = 1; i < f.length; i++) if (Math.abs(f[i] - fMid) < Math.abs(f[iMid] - fMid)) iMid = i;
            const tMid = t[iMid];
            const lo = [fLo * 0.5, 15, 30, t[0] - Math.max(30, span)];
            const hi = [fHi * 1.5, 330, 40000, t[t.length - 1] + Math.max(30, span)];
            const clamp = (p) => p.map((v, i) => Math.min(hi[i], Math.max(lo[i], v)));
            const d2 = lossScale * lossScale;
            const obj = (p) => {
                const q = clamp(p);
                let pen = 0;
                for (let i = 0; i < 4; i++) pen += Math.abs(p[i] - q[i]);
                let s = 0;
                for (let i = 0; i < t.length; i++) {
                    const r = this.frequency(t[i], q[0], q[1], q[2], q[3], c) - f[i];
                    s += 2 * d2 * (Math.sqrt(1 + (r * r) / d2) - 1);
                }
                return s + 1e3 * pen;
            };
            let best = null;
            for (const dInit of [300, 800, 1500, 3000, 6000, 12000]) {
                let x = clamp([fMid, vInit, dInit, tMid - dInit / c]);
                let r = nelderMead(obj, x, [Math.max(1, fMid * 0.02), 15, dInit * 0.3, 1.0], 700);
                r = nelderMead(obj, r.x, [Math.max(0.5, fMid * 0.005), 4, Math.max(20, r.x[2] * 0.08), 0.3], 500);
                if (!best || r.fx < best.fx) best = r;
            }
            const [f0, v, d, tau0] = clamp(best.x);
            let ss = 0, out = 0;
            for (let i = 0; i < t.length; i++) {
                const r = this.frequency(t[i], f0, v, d, tau0, c) - f[i];
                ss += r * r;
                if (Math.abs(r) > 3 * lossScale) out++;
            }
            return {
                f0_hz: +f0.toFixed(3), speed_mps: +v.toFixed(2), cpa_slant_m: +d.toFixed(1),
                cpa_emission_time_s: +tau0.toFixed(3), cpa_arrival_time_s: +(tau0 + d / c).toFixed(3),
                rmse_hz: +Math.sqrt(ss / t.length).toFixed(3), outlier_fraction: +(out / t.length).toFixed(4),
                n_samples: t.length, speed_of_sound_mps: c,
            };
        },
    };

    // ------------------------------------------------------- ridge discovery
    const DEFAULT_DISCOVERY = {
        fmin_hz: 60, fmax_hz: 2000, window_s: 0.25, hop_s: 0.05, max_slope_hz_s: 160,
        jump_penalty: 0.35, detection_z: 4.0, min_duration_s: 3.0, max_gap_s: 0.6, max_ridges: 6,
        mask_half_width_hz: 25, output_step_s: 0.25, max_duration_s: 1800,
        harmonic_max_integer: 6, harmonic_tolerance: 0.012,
    };

    function fft(real, imag) {
        const n = real.length;
        for (let i = 1, j = 0; i < n; i++) {
            let bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                let t = real[i]; real[i] = real[j]; real[j] = t;
                t = imag[i]; imag[i] = imag[j]; imag[j] = t;
            }
        }
        for (let len = 2; len <= n; len <<= 1) {
            const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang);
            for (let i = 0; i < n; i += len) {
                let cr = 1, ci = 0;
                for (let k = 0; k < len / 2; k++) {
                    const a = i + k, b = a + len / 2;
                    const vr = real[b] * cr - imag[b] * ci, vi = real[b] * ci + imag[b] * cr;
                    real[b] = real[a] - vr; imag[b] = imag[a] - vi;
                    real[a] += vr; imag[a] += vi;
                    const nr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = nr;
                }
            }
        }
    }

    function rationalRatio(r, maxInt, tol) {
        let best = null;
        for (let q = 1; q <= maxInt; q++) {
            for (let p = q + 1; p <= maxInt; p++) {
                if (gcd(p, q) !== 1) continue;
                const err = Math.abs(r - p / q) / r;
                if (err < tol && (!best || err < best.err)) best = { err, p, q };
            }
        }
        return best ? (best.q === 1 ? `${best.p}` : `${best.p}/${best.q}`) : null;
    }
    function gcd(a, b) { return b ? gcd(b, a % b) : a; }

    const RidgeDiscovery = {
        async resample(audioBuffer, targetSr) {
            const dur = Math.min(audioBuffer.duration, DEFAULT_DISCOVERY.max_duration_s);
            const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
            const ctx = new Offline(1, Math.max(1, Math.ceil(dur * targetSr)), targetSr);
            const src = ctx.createBufferSource();
            src.buffer = audioBuffer;
            src.connect(ctx.destination);
            src.start(0);
            const rendered = await ctx.startRendering();
            return rendered.getChannelData(0);
        },

        /** Resampling needs OfflineAudioContext (main thread); the ridge search can run in a worker. */
        async discover(audioBuffer, settings = {}, worker = null) {
            const s = Object.assign({}, DEFAULT_DISCOVERY, settings);
            const sr = Math.min(48000, Math.max(4000, 4 * s.fmax_hz));
            const x = await this.resample(audioBuffer, sr);
            if (worker) {
                try {
                    return await worker('discover', { x, sr, settings: s }, null, [x.buffer]);
                } catch (e) {
                    if (!e || !e.workerUnavailable) throw e;
                    return this.discoverSignal(await this.resample(audioBuffer, sr), sr, s);
                }
            }
            return this.discoverSignal(x, sr, s);
        },

        discoverSignal(x, sr, s) {
            s = Object.assign({}, DEFAULT_DISCOVERY, s || {});
            const N = Math.pow(2, Math.round(Math.log2(Math.max(64, s.window_s * sr))));
            const hop = Math.max(1, Math.round(s.hop_s * sr));
            const T = Math.floor((x.length - N) / hop) + 1;
            if (T < 5) return this.result([], [], s, sr, x.length / sr, 'Recording too short for analysis.');
            const df0 = sr / N;
            const b0 = Math.ceil(s.fmin_hz / df0), b1 = Math.floor(Math.min(s.fmax_hz, sr / 2) / df0);
            const F = b1 - b0 + 1;
            const win = new Float32Array(N);
            for (let i = 0; i < N; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N);
            const power = new Float32Array(T * F);
            const re = new Float64Array(N), im = new Float64Array(N);
            const times = new Float64Array(T);
            for (let t = 0; t < T; t++) {
                const off = t * hop;
                for (let i = 0; i < N; i++) { re[i] = x[off + i] * win[i]; im[i] = 0; }
                fft(re, im);
                for (let j = 0; j < F; j++) {
                    const k = b0 + j;
                    power[t * F + j] = 10 * Math.log10(Math.max(re[k] * re[k] + im[k] * im[k], 1e-20));
                }
                times[t] = (off + N / 2) / sr;
            }
            // Contrast: remove per-bin median over time, then robust per-frame z-score.
            const contrast = new Float32Array(T * F);
            const col = new Float32Array(T);
            for (let j = 0; j < F; j++) {
                for (let t = 0; t < T; t++) col[t] = power[t * F + j];
                const m = median(col);
                for (let t = 0; t < T; t++) contrast[t * F + j] = power[t * F + j] - m;
            }
            const row = new Float32Array(F), dev = new Float32Array(F);
            for (let t = 0; t < T; t++) {
                for (let j = 0; j < F; j++) row[j] = contrast[t * F + j];
                const m = median(row);
                for (let j = 0; j < F; j++) dev[j] = Math.abs(row[j] - m);
                const mad = Math.max(median(dev) * 1.4826, 1e-3);
                for (let j = 0; j < F; j++) contrast[t * F + j] = Math.min(12, Math.max(-3, (row[j] - m) / mad));
            }

            const hopS = hop / sr;
            const maxJump = Math.max(1, Math.ceil((s.max_slope_hz_s * hopS) / df0));
            const minFrames = Math.ceil(s.min_duration_s / hopS);
            const gapFrames = Math.ceil(s.max_gap_s / hopS);
            const maskW = Math.max(1, Math.round(s.mask_half_width_hz / df0));
            const fBase = b0 * df0;
            const ridges = [];
            let attempts = 0;
            while (ridges.length < s.max_ridges && attempts < s.max_ridges * 3) {
                attempts++;
                const path = this.viterbi(contrast, T, F, maxJump, s.jump_penalty);
                const along = new Float32Array(T);
                for (let t = 0; t < T; t++) along[t] = contrast[t * F + path[t]];
                const smooth = new Float32Array(T);
                for (let t = 0; t < T; t++) {
                    const w = [];
                    for (let k = Math.max(0, t - 2); k <= Math.min(T - 1, t + 2); k++) w.push(along[k]);
                    smooth[t] = t >= 2 && t < T - 2 ? median(w) : along[t];
                }
                const runs = [];
                let st = -1;
                for (let t = 0; t <= T; t++) {
                    const on = t < T && smooth[t] >= s.detection_z;
                    if (on && st < 0) st = t;
                    if (!on && st >= 0) { runs.push([st, t]); st = -1; }
                }
                const merged = [];
                for (const r of runs) {
                    if (merged.length && r[0] - merged[merged.length - 1][1] <= gapFrames) merged[merged.length - 1][1] = r[1];
                    else merged.push(r.slice());
                }
                const accepted = merged.filter(([a, b]) => b - a >= minFrames);
                if (!accepted.length) break;
                for (const [a, b] of accepted) {
                    const segT = [], segF = [], segC = [];
                    for (let t = a; t < b; t++) {
                        const j = path[t];
                        let delta = 0;
                        if (j > 0 && j < F - 1) {
                            const p0 = power[t * F + j - 1], p1 = power[t * F + j], p2 = power[t * F + j + 1];
                            const den = p0 - 2 * p1 + p2;
                            delta = Math.abs(den) > 1e-12 ? Math.max(-0.5, Math.min(0.5, (0.5 * (p0 - p2)) / den)) : 0;
                        }
                        segT.push(times[t]); segF.push(fBase + (j + delta) * df0); segC.push(along[t]);
                    }
                    ridges.push(this.buildRidge(segT, segF, segC, s));
                    for (let t = a; t < b; t++) {
                        const lo = Math.max(0, path[t] - maskW), hi = Math.min(F, path[t] + maskW + 1);
                        for (let j = lo; j < hi; j++) contrast[t * F + j] = -3;
                    }
                    if (ridges.length >= s.max_ridges) break;
                }
            }
            ridges.sort((p, q) => p.t_start - q.t_start || p.median_freq_hz - q.median_freq_hz);
            ridges.forEach((r, i) => { r.id = `R${i + 1}`; });
            const groups = this.groupHarmonics(ridges, s);
            return this.result(ridges, groups, s, sr, x.length / sr,
                ridges.length ? null : 'No moving tonal ridge cleared the detection threshold.');
        },

        viterbi(score, T, F, maxJump, penalty) {
            let prev = new Float64Array(F);
            for (let j = 0; j < F; j++) prev[j] = score[j];
            const back = new Int32Array(T * F);
            let cur = new Float64Array(F);
            for (let t = 1; t < T; t++) {
                for (let j = 0; j < F; j++) {
                    let best = -Infinity, arg = j;
                    const lo = Math.max(0, j - maxJump), hi = Math.min(F - 1, j + maxJump);
                    for (let k = lo; k <= hi; k++) {
                        const v = prev[k] - penalty * Math.abs(k - j);
                        if (v > best) { best = v; arg = k; }
                    }
                    cur[j] = best + score[t * F + j];
                    back[t * F + j] = arg;
                }
                const tmp = prev; prev = cur; cur = tmp;
            }
            const path = new Int32Array(T);
            let bj = 0;
            for (let j = 1; j < F; j++) if (prev[j] > prev[bj]) bj = j;
            path[T - 1] = bj;
            for (let t = T - 1; t > 0; t--) path[t - 1] = back[t * F + path[t]];
            return path;
        },

        buildRidge(segT, segF, segC, s) {
            const n = segF.length;
            const smooth = segF.map((v, i) => {
                let sum = 0, cnt = 0;
                for (let k = Math.max(0, i - 4); k <= Math.min(n - 1, i + 4); k++) { sum += segF[k]; cnt++; }
                return sum / cnt;
            });
            const scatter = median(segF.map((v, i) => Math.abs(v - smooth[i]))) * 1.4826;
            const step = Math.max(s.output_step_s, n > 1 ? segT[1] - segT[0] : s.output_step_s);
            const pts = [];
            for (let lo = segT[0]; lo <= segT[n - 1]; lo += step) {
                const ts = [], fs = [];
                for (let i = 0; i < n; i++) if (segT[i] >= lo && segT[i] < lo + step) { ts.push(segT[i]); fs.push(segF[i]); }
                if (ts.length) pts.push({ time: +(ts.reduce((a, b) => a + b, 0) / ts.length).toFixed(3), freq: +median(fs).toFixed(2) });
            }
            let falling = 0;
            for (let i = 1; i < n; i++) if (smooth[i] - smooth[i - 1] < 0) falling++;
            const fallingFrac = n > 1 ? falling / (n - 1) : 0;
            let fit = null;
            try { fit = FlyoverFit.fit(pts.map(p => p.time), pts.map(p => p.freq), C_DEFAULT, Math.max(1, scatter)); } catch (e) { fit = null; }
            const fMin = Math.min(...segF), fMax = Math.max(...segF);
            return {
                id: null, harmonic_group_id: null,
                t_start: +segT[0].toFixed(3), t_end: +segT[n - 1].toFixed(3),
                duration_s: +(segT[n - 1] - segT[0]).toFixed(3),
                median_freq_hz: +median(segF).toFixed(2), freq_range_hz: [+fMin.toFixed(2), +fMax.toFixed(2)],
                mean_contrast: +(segC.reduce((a, b) => a + b, 0) / n).toFixed(2),
                frequency_uncertainty_hz: +Math.max(scatter, 0.1).toFixed(3),
                falling_fraction: +fallingFrac.toFixed(3),
                doppler_like: !!(fit && fit.rmse_hz <= Math.max(4, 3 * scatter) && fallingFrac >= 0.6 && fMax - fMin >= 5),
                flyover_fit: fit, points: pts,
            };
        },

        overlapRatio(a, b) {
            const lo = Math.max(a.t_start, b.t_start), hi = Math.min(a.t_end, b.t_end);
            const shorter = Math.min(a.duration_s, b.duration_s);
            if (shorter <= 0 || hi - lo < 0.5 * shorter) return null;
            const ta = a.points.map(p => p.time), fa = a.points.map(p => p.freq);
            const tb = b.points.map(p => p.time), fb = b.points.map(p => p.freq);
            const ratios = [];
            for (let i = 0; i < 25; i++) {
                const t = lo + ((hi - lo) * i) / 24;
                const ra = interpSorted(t, ta, fa), rb = interpSorted(t, tb, fb);
                ratios.push(Math.max(ra, rb) / Math.max(Math.min(ra, rb), 1e-6));
            }
            const med = median(ratios);
            return { ratio: med, spread: median(ratios.map(r => Math.abs(r - med))) / med };
        },

        groupHarmonics(ridges, s) {
            const n = ridges.length;
            const parent = ridges.map((_, i) => i);
            const find = (i) => { while (parent[i] !== i) { parent[i] = parent[parent[i]]; i = parent[i]; } return i; };
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    if (!(ridges[i].doppler_like && ridges[j].doppler_like)) continue;
                    const ov = this.overlapRatio(ridges[i], ridges[j]);
                    if (!ov) continue;
                    if (rationalRatio(ov.ratio, s.harmonic_max_integer, s.harmonic_tolerance) && ov.spread < s.harmonic_tolerance) {
                        parent[find(i)] = find(j);
                    }
                }
            }
            const map = new Map();
            for (let i = 0; i < n; i++) {
                const r = find(i);
                if (!map.has(r)) map.set(r, []);
                map.get(r).push(i);
            }
            const lists = [...map.values()].sort((a, b) => Math.min(...a.map(k => ridges[k].t_start)) - Math.min(...b.map(k => ridges[k].t_start)));
            return lists.map((members, gi) => {
                const gid = `G${gi + 1}`;
                members.sort((a, b) => ridges[a].median_freq_hz - ridges[b].median_freq_hz);
                const base = ridges[members[0]];
                for (const k of members) {
                    const r = ridges[k];
                    r.harmonic_group_id = gid;
                    if (k === members[0]) { r.ratio_to_lowest = 1; r.harmonic_ratio = '1'; continue; }
                    const ov = this.overlapRatio(base, r);
                    const ratio = ov ? ov.ratio : r.median_freq_hz / base.median_freq_hz;
                    r.ratio_to_lowest = +ratio.toFixed(4);
                    r.harmonic_ratio = rationalRatio(ratio, s.harmonic_max_integer, s.harmonic_tolerance);
                }
                const primary = members.reduce((p, k) => {
                    const a = ridges[p], b = ridges[k];
                    const key = (r) => [r.doppler_like ? 1 : 0, r.duration_s, r.mean_contrast];
                    const ka = key(a), kb = key(b);
                    for (let i = 0; i < 3; i++) if (kb[i] !== ka[i]) return kb[i] > ka[i] ? k : p;
                    return p;
                }, members[0]);
                return {
                    id: gid, ridge_ids: members.map(k => ridges[k].id), primary_ridge_id: ridges[primary].id,
                    t_start: Math.min(...members.map(k => ridges[k].t_start)),
                    t_end: Math.max(...members.map(k => ridges[k].t_end)),
                    doppler_like: members.some(k => ridges[k].doppler_like),
                };
            });
        },

        result(ridges, groups, s, sr, duration, message) {
            return {
                ridges, harmonic_groups: groups, n_ridges: ridges.length, n_groups: groups.length,
                analysis_sample_rate: sr, duration_s: +duration.toFixed(3), settings: s, message,
                note: 'Ridges are acoustic proposals, not verified aircraft detections.', mode: 'client',
            };
        },
    };

    // ---------------------------------------------------------- joint matcher
    const DEFAULT_MATCH = {
        speed_of_sound: C_DEFAULT, fit_f0: true, drift_fraction: 0.02, loss_scale_hz: 2.0, min_support: 0.95,
        irls_iterations: 2, coarse_halfwidth_s: 6.0, coarse_step_s: 0.25, fine_halfwidth_s: 0.4, fine_step_s: 0.02,
        refine_clock_halfwidth_s: 1.0, max_gap_s: 20.0, observer_alt_m: 0.0, altitude_bias_m: 0.0,
        max_alternatives: 6, grid_max_points: 40,
        wind_e: 0.0, wind_n: 0.0, wind_uncertainty_mps: 5.0, elevation: null, refine_clock_step_s: 0.08,
    };

    /**
     * Travel time and Doppler factor (mirrors matching.propagation): exact
     * uniform-wind solution c T = |R - W T|, f/f0 = (c + W.k) / (c + W.k - v.k)
     * with k the air-frame wave normal. R: aircraft -> observer.
     */
    function propagation(rx, ry, rz, vx, vy, vz, c, we, wn) {
        const d = Math.max(1, Math.sqrt(rx * rx + ry * ry + rz * rz));
        if (!we && !wn) {
            let u = -(rx * vx + ry * vy + rz * vz) / d;
            u = Math.max(-0.95 * c, Math.min(0.95 * c, u));
            return [d / c, c / (c + u)];
        }
        const w2 = we * we + wn * wn, rw = rx * we + ry * wn, a = c * c - w2;
        const T = (-rw + Math.sqrt(rw * rw + a * d * d)) / a;
        const kx = (rx - we * T) / (c * T), ky = (ry - wn * T) / (c * T), kz = rz / (c * T);
        const ck = c + kx * we + ky * wn;
        let u = -(vx * kx + vy * ky + vz * kz);
        u = Math.max(-0.95 * ck, Math.min(0.95 * ck, u));
        return [T, ck / (ck + u)];
    }

    /** Meteorological wind (speed, FROM direction) -> [east, north] toward. */
    function windVector(speed, fromDeg) {
        if (!speed) return [0, 0];
        const th = (fromDeg || 0) * Math.PI / 180;
        return [-speed * Math.sin(th), -speed * Math.cos(th)];
    }

    // ISA pressure altitude -> QNH altitude (mirrors app/physics/atmosphere.py).
    const ISA = { T0: 288.15, L: 0.0065, P0: 1013.25, EXP: 9.80665 / (287.05287 * 0.0065) };
    function qnhAltitude(pa, qnh) {
        const p = ISA.P0 * Math.pow(1 - ISA.L * pa / ISA.T0, ISA.EXP);
        return (ISA.T0 / ISA.L) * (1 - Math.pow(p / qnh, 1 / ISA.EXP));
    }
    function applyQnh(traj, qnh) {
        if (traj.altitude_basis !== 'barometric' || traj.altitude_correction) return traj;
        return Object.assign({}, traj, { altitudes: traj.altitudes.map(a => +qnhAltitude(a, qnh).toFixed(2)),
            altitude_correction: { type: 'qnh', qnh_hpa: qnh } });
    }

    /** Bilinear lat/lon elevation grid as a local (x, y) -> metres callable (mirrors elevation.LatLonElevation). */
    function latLonElevation(grid, centerLat, centerLon, datumOffset = 0) {
        const v = grid.values, ny = v.length, nx = v[0].length;
        if (ny < 2 || nx < 2 || !(grid.lat_max > grid.lat_min && grid.lon_max > grid.lon_min)) throw new Error('Invalid elevation grid.');
        const geo = new GeoEngine();
        return (x, y) => {
            const g = geo.cartesianToGeodetic(x, y, 0, centerLat, centerLon, 0);
            const fy = Math.min(Math.max((g.lat - grid.lat_min) / (grid.lat_max - grid.lat_min) * (ny - 1), 0), ny - 1);
            const fx = Math.min(Math.max((g.lon - grid.lon_min) / (grid.lon_max - grid.lon_min) * (nx - 1), 0), nx - 1);
            const iy = Math.min(Math.floor(fy), ny - 2), ix = Math.min(Math.floor(fx), nx - 2), ty = fy - iy, tx = fx - ix;
            return (1 - ty) * ((1 - tx) * v[iy][ix] + tx * v[iy][ix + 1]) + ty * ((1 - tx) * v[iy + 1][ix] + tx * v[iy + 1][ix + 1]) + datumOffset;
        };
    }

    class TrajectoryModel {
        constructor(traj, centerLat, centerLon, engine, altBias = 0, maxGap = 20) {
            this.raw = traj;
            this.flight_id = String(traj.flight_id || 'UNKNOWN');
            this.altitude_basis = String(traj.altitude_basis || 'unknown');
            this.altitude_corrected = !!traj.altitude_correction;
            const n = traj.timestamps.length;
            const idx = [...Array(n).keys()].sort((a, b) => traj.timestamps[a] - traj.timestamps[b]);
            this.t = idx.map(i => traj.timestamps[i]);
            this.pos = idx.map(i => engine.geodeticToCartesian(traj.latitudes[i], traj.longitudes[i], traj.altitudes[i] + altBias, centerLat, centerLon, 0));
            const ve = traj.velocity_east, vn = traj.velocity_north, vu = traj.velocity_up;
            if (ve && vn && vu && ve.length === n && vn.length === n && vu.length === n) {
                // Conditioned tracks carry fused (reported + smoothed) velocities; use them as-is.
                this.vel = idx.map(i => ({ vx: ve[i], vy: vn[i], vz: vu[i] }));
            } else {
                // np.gradient semantics: one-sided at the ends, non-uniform central differences inside.
                this.vel = this.pos.map((p, i) => {
                    if (n < 2) return { vx: 0, vy: 0, vz: 0 };
                    if (i === 0 || i === n - 1) {
                        const a = i === 0 ? 0 : n - 2, b = a + 1, dt = Math.max(1e-9, this.t[b] - this.t[a]);
                        return { vx: (this.pos[b].x - this.pos[a].x) / dt, vy: (this.pos[b].y - this.pos[a].y) / dt, vz: (this.pos[b].z - this.pos[a].z) / dt };
                    }
                    const h1 = this.t[i] - this.t[i - 1], h2 = this.t[i + 1] - this.t[i];
                    const g = (k) => (h1 * h1 * this.pos[i + 1][k] + (h2 * h2 - h1 * h1) * this.pos[i][k] - h2 * h2 * this.pos[i - 1][k]) / (h1 * h2 * (h1 + h2));
                    return { vx: g('x'), vy: g('y'), vz: g('z') };
                });
            }
            this.gapAfter = [];
            for (let i = 0; i < n - 1; i++) if (this.t[i + 1] - this.t[i] > maxGap) this.gapAfter.push(i);
        }
        arrival(obs, c, wind = null) {
            const n = this.t.length;
            const rec = new Array(n);
            const we = wind ? wind[0] : 0, wn = wind ? wind[1] : 0;
            let dmin = Infinity, tcpa = 0;
            for (let i = 0; i < n; i++) {
                const rx = obs.x - this.pos[i].x, ry = obs.y - this.pos[i].y, rz = obs.z - this.pos[i].z;
                const d = Math.max(1, Math.sqrt(rx * rx + ry * ry + rz * rz));
                const [T, g] = propagation(rx, ry, rz, this.vel[i].vx, this.vel[i].vy, this.vel[i].vz, c, we, wn);
                const to = this.t[i] + T;
                rec[i] = [to, g, this.t[i]];
                if (d < dmin) { dmin = d; tcpa = to; }
            }
            const gaps = this.gapAfter.map(i => [rec[i][0], rec[i + 1][0]]);
            rec.sort((a, b) => a[0] - b[0]);
            return { t: rec.map(r => r[0]), g: rec.map(r => r[1]), tau: rec.map(r => r[2]), t_cpa: tcpa, gaps };
        }
    }

    function fitEventAt(curve, ev, start, o) {
        const M = ev.t.length;
        const t0 = curve.t[0], t1 = curve.t[curve.t.length - 1];
        const G = new Float64Array(M), U = new Float64Array(M), inside = new Uint8Array(M);
        let nIn = 0, tauSum = 0, tauMin = Infinity, tauMax = -Infinity;
        const tau = new Float64Array(M);
        for (let i = 0; i < M; i++) {
            const q = ev.t[i] + start;
            let ok = q >= t0 && q <= t1;
            for (const [a, b] of curve.gaps) if (q > a && q < b) ok = false;
            if (!ok) continue;
            inside[i] = 1; nIn++;
            G[i] = interpSorted(q, curve.t, curve.g);
            tau[i] = interpSorted(q, curve.t, curve.tau);
            tauSum += tau[i]; tauMin = Math.min(tauMin, tau[i]); tauMax = Math.max(tauMax, tau[i]);
        }
        const support = nIn / M;
        const fail = { loss: Infinity, rmse: Infinity, outlier: 1, support, f0: 0, drift: 0, drift_at_bound: false, model: null };
        if (support < o.min_support || nIn < 3) return fail;
        const tc = tauSum / nIn, span = Math.max(tauMax - tauMin, 1e-3);
        for (let i = 0; i < M; i++) if (inside[i]) U[i] = (tau[i] - tc) / span;
        const delta = o.loss_scale_hz, df = o.drift_fraction;
        const w = new Float64Array(M);
        for (let i = 0; i < M; i++) w[i] = inside[i];
        let b0 = 0, b1 = 0, atBound = false;
        const resid = new Float64Array(M);
        for (let it = 0; it < 1 + o.irls_iterations; it++) {
            if (ev.f0_fixed) { b0 = ev.f0_fixed; b1 = 0; }
            else {
                let a11 = 0, a12 = 0, a22 = 0, y1 = 0, y2 = 0;
                for (let i = 0; i < M; i++) {
                    if (!inside[i]) continue;
                    const g = G[i], h = g * U[i], wi = w[i], fi = ev.f[i];
                    a11 += wi * g * g; a12 += wi * g * h; a22 += wi * h * h; y1 += wi * fi * g; y2 += wi * fi * h;
                }
                const det = a11 * a22 - a12 * a12;
                if (Math.abs(det) > 1e-12 * Math.max(a11 * a22, 1e-30)) { b0 = (y1 * a22 - y2 * a12) / det; b1 = (a11 * y2 - a12 * y1) / det; }
                else { b0 = y1 / Math.max(a11, 1e-12); b1 = 0; }
                const ratio = b1 / Math.max(Math.abs(b0), 1e-9);
                atBound = Math.abs(ratio) > df;
                if (atBound) {
                    const sg = Math.sign(ratio);
                    let num = 0, den = 0;
                    for (let i = 0; i < M; i++) {
                        if (!inside[i]) continue;
                        const h = G[i] * (1 + sg * df * U[i]);
                        num += w[i] * ev.f[i] * h; den += w[i] * h * h;
                    }
                    b0 = num / Math.max(den, 1e-12); b1 = sg * df * b0;
                }
            }
            for (let i = 0; i < M; i++) {
                if (!inside[i]) continue;
                resid[i] = ev.f[i] - G[i] * (b0 + b1 * U[i]);
                w[i] = 1 / Math.sqrt(1 + (resid[i] / delta) ** 2);
            }
        }
        if (b0 <= 0) return fail;
        let loss = 0, ss = 0, out = 0;
        const model = new Array(M).fill(null);
        for (let i = 0; i < M; i++) {
            if (!inside[i]) continue;
            const r = resid[i];
            loss += delta * delta * (Math.sqrt(1 + (r / delta) ** 2) - 1);
            ss += r * r;
            if (Math.abs(r) > 3 * delta) out++;
            model[i] = G[i] * (b0 + b1 * U[i]);
        }
        return {
            loss: loss / nIn, rmse: Math.sqrt(ss / nIn), outlier: out / nIn, support,
            f0: b0, drift: b1 / Math.max(Math.abs(b0), 1e-9), drift_at_bound: atBound, model,
        };
    }

    const robustRmse = (loss) => (isFinite(loss) ? Math.sqrt(Math.max(0, 2 * loss)) : Infinity);

    class JointMatcher {
        constructor(trajectories, rawEvents, centerLat, centerLon, opts = {}, clockWindow = null) {
            this.o = Object.assign({}, DEFAULT_MATCH, opts);
            this.center = { lat: centerLat, lon: centerLon };
            this.rawTrajectories = trajectories;
            this.engine = new GeoEngine(this.o.speed_of_sound);
            this.models = trajectories.map(t => new TrajectoryModel(t, centerLat, centerLon, this.engine, this.o.altitude_bias_m, this.o.max_gap_s));
            if (!this.models.length) throw new Error('No flight trajectories supplied.');
            this.events = rawEvents.map((ev, i) => {
                const pts = (ev.traced_curve || ev.points || []).slice().sort((a, b) => a.time - b.time);
                if (pts.length < 3) throw new Error(`Event ${i + 1} needs at least 3 points.`);
                const t = pts.map(p => p.time), f = pts.map(p => p.freq);
                const id = String(ev.id || `E${i + 1}`);
                return {
                    id, group_id: String(ev.group_id || id), t, f,
                    f0_fixed: this.o.fit_f0 ? null : ev.f0, flight_ids: ev.flight_id ? [ev.flight_id] : null,
                    cpa_time_s: estimateCpa(t, f, this.o.speed_of_sound),
                };
            });
            this.eventsCoarse = this.events.map(e => decimate(e, this.o.grid_max_points));
            this.groups = {};
            this.events.forEach((e, i) => { (this.groups[e.group_id] = this.groups[e.group_id] || []).push(i); });
            this.groupIds = Object.keys(this.groups);
            this.clockWindow = clockWindow;
        }

        observer(x, y, o) {
            return { x, y, z: o.observer_alt_m + (o.elevation ? o.elevation(x, y) : 0) };
        }

        allowed(gid) {
            let restrict = null;
            for (const ei of this.groups[gid]) {
                const ids = this.events[ei].flight_ids;
                if (ids) restrict = restrict ? restrict.filter(x => ids.includes(x)) : ids.slice();
            }
            const idx = this.models.map((m, k) => k).filter(k => !restrict || restrict.includes(this.models[k].flight_id));
            return idx.length ? idx : this.models.map((m, k) => k);
        }

        candidateStarts(curves) {
            const o = this.o;
            if (this.clockWindow) {
                const [lo, hi] = this.clockWindow;
                if (hi - lo < 1e-9) return [lo];
                const n = Math.min(4000, Math.max(2, Math.ceil((hi - lo) / o.coarse_step_s) + 1));
                return linspace(lo, hi, n);
            }
            // Mirrors JointMatcher._candidate_starts exactly (numpy rounding and arange semantics).
            const noms = [];
            for (const gid of this.groupIds) {
                const ev = this.events[this.groups[gid][0]];
                for (const k of this.allowed(gid)) noms.push(roundHalfEven((curves[k].t_cpa - ev.cpa_time_s) / o.coarse_step_s) * o.coarse_step_s);
            }
            const offs = arange(-o.coarse_halfwidth_s, o.coarse_halfwidth_s + 1e-9, o.coarse_step_s);
            const all = [];
            for (const nom of uniqueSorted(noms)) for (const d of offs) all.push(nom + d);
            return uniqueSorted(all);
        }

        clipClock(starts) {
            if (!this.clockWindow) return starts;
            const [lo, hi] = this.clockWindow;
            const s = starts.filter(x => x >= lo - 1e-9 && x <= hi + 1e-9);
            return s.length ? s : [Math.min(Math.max(starts.reduce((a, b) => a + b, 0) / starts.length, lo), hi)];
        }

        joint(curves, starts, o, events) {
            let bestTotal = Infinity, bestStart = starts[0], bestChoice = null;
            const choiceAt = {};
            for (const s of starts) {
                let total = 0;
                const choice = {};
                for (const gid of this.groupIds) {
                    let gBest = Infinity, gArg = -1;
                    for (const k of this.allowed(gid)) {
                        let acc = 0;
                        for (const ei of this.groups[gid]) {
                            const ev = events[ei];
                            // cheap support bound (ignores gaps) before the full fit
                            const tt = curves[k].t;
                            if (countInRange(ev.t, tt[0] - s, tt[tt.length - 1] - s) / ev.t.length < o.min_support) { acc = Infinity; break; }
                            acc += fitEventAt(curves[k], ev, s, o).loss;
                            if (!isFinite(acc)) break;
                        }
                        acc /= this.groups[gid].length;
                        if (acc < gBest) { gBest = acc; gArg = k; }
                    }
                    total += gBest;
                    choice[gid] = gArg;
                    if (!isFinite(total)) break;
                }
                if (total < bestTotal) { bestTotal = total; bestStart = s; bestChoice = choice; }
            }
            return { total: bestTotal, start: bestStart, choice: bestChoice };
        }

        evaluatePoint(obs, starts = null, o = null, models = null, events = null) {
            o = o || this.o; models = models || this.models; events = events || this.events;
            const curves = models.map(m => m.arrival(obs, o.speed_of_sound, [o.wind_e, o.wind_n]));
            starts = starts || this.candidateStarts(curves);
            const coarse = this.joint(curves, starts, o, events);
            if (!isFinite(coarse.total)) return coarse;
            const fine = [];
            for (const d of arange(-o.fine_halfwidth_s, o.fine_halfwidth_s + 1e-9, o.fine_step_s)) fine.push(coarse.start + d);
            const f = this.joint(curves, this.clipClock(fine), o, events);
            return f.total <= coarse.total ? f : coarse;
        }

        async gridSearch(radiusKm, res, onProgress) {
            const R = radiusKm * 1000, step = (2 * R) / (res - 1);
            this.grid = { R, step, res, cells: [] };
            for (let iy = 0; iy < res; iy++) {
                for (let ix = 0; ix < res; ix++) {
                    const x = -R + ix * step, y = -R + iy * step;
                    const inside = Math.hypot(x, y) <= R * 1.05;
                    const cell = { ix, iy, x, y, inside, loss: Infinity, start: 0 };
                    if (inside) {
                        const r = this.evaluatePoint(this.observer(x, y, this.o), null, null, null, this.eventsCoarse);
                        cell.loss = r.total; cell.start = r.start;
                    }
                    this.grid.cells.push(cell);
                }
                if (onProgress) onProgress((iy + 1) / res);
                await new Promise(r => setTimeout(r, 0));
            }
            return this.grid;
        }

        localMinima() {
            const { res, cells } = this.grid;
            const at = (ix, iy) => cells[iy * res + ix];
            const mins = [];
            for (const c of cells) {
                if (!isFinite(c.loss)) continue;
                let ok = true;
                for (let dy = -1; dy <= 1 && ok; dy++) for (let dx = -1; dx <= 1; dx++) {
                    const ix = c.ix + dx, iy = c.iy + dy;
                    if (ix < 0 || iy < 0 || ix >= res || iy >= res) continue;
                    const n = at(ix, iy);
                    if (isFinite(n.loss) && n.loss < c.loss) { ok = false; break; }
                }
                if (ok) mins.push(c);
            }
            return mins.sort((a, b) => a.loss - b.loss);
        }

        refine(x0, y0, s0, o = null, models = null, events = null, maxIter = 120) {
            o = o || this.o;
            const hw = o.refine_clock_halfwidth_s, starts = [];
            for (const v of arange(s0 - hw, s0 + hw + 1e-9, o.refine_clock_step_s)) starts.push(v);
            const st = this.clipClock(starts);
            const step = this.grid ? this.grid.step : 400;
            const obj = (p) => {
                const r = this.evaluatePoint(this.observer(p[0], p[1], o), st, o, models, events);
                return isFinite(r.total) ? r.total : 1e12;
            };
            const nm = nelderMead(obj, [x0, y0], [0.5 * step, 0.5 * step], maxIter, 1e-6);
            const r = this.evaluatePoint(this.observer(nm.x[0], nm.x[1], o), st, o, models, events);
            const clockEdge = !this.clockWindow && (Math.abs(r.start - st[0]) < 1e-6 || Math.abs(r.start - st[st.length - 1]) < 1e-6);
            return { x: nm.x[0], y: nm.x[1], loss: r.total, start: r.start, choice: r.choice, clock_at_refine_edge: clockEdge };
        }

        toGeo(x, y) {
            const g = this.engine.cartesianToGeodetic(x, y, 0, this.center.lat, this.center.lon, 0);
            return { lat: g.lat, lon: g.lon };
        }

        diagnostics(x, y, start, choice) {
            const obs = this.observer(x, y, this.o);
            const curves = this.models.map(m => m.arrival(obs, this.o.speed_of_sound, [this.o.wind_e, this.o.wind_n]));
            const out = [];
            for (const gid of this.groupIds) {
                const k = choice ? choice[gid] : -1;
                for (const ei of this.groups[gid]) {
                    const ev = this.events[ei];
                    if (k === undefined || k < 0) { out.push({ event_id: ev.id, group_id: gid, matched: false, flight_id: null }); continue; }
                    const r = fitEventAt(curves[k], ev, start, this.o);
                    const m = this.models[k];
                    out.push({
                        event_id: ev.id, group_id: gid, matched: isFinite(r.loss), flight_id: m.flight_id,
                        callsign: m.raw.callsign, aircraft_type: m.raw.aircraft_type,
                        f0_hz: +r.f0.toFixed(2), drift_fraction: +r.drift.toFixed(5), drift_at_bound: r.drift_at_bound,
                        rmse_hz: +r.rmse.toFixed(3), robust_rmse_hz: +robustRmse(r.loss).toFixed(3),
                        outlier_fraction: +r.outlier.toFixed(4), support_fraction: +r.support.toFixed(4),
                        n_samples: ev.t.length, times: ev.t, traced: ev.f, modeled: r.model || [],
                    });
                }
            }
            return out;
        }

        async run(radiusKm, res = 25, { sensitivity = true, onProgress = null } = {}) {
            await this.gridSearch(radiusKm, res, p => onProgress && onProgress('grid', p));
            const mins = this.localMinima();
            if (!mins.length) return { status: 'no_match', message: 'No candidate position explained all events with full trajectory support.' };
            const refined = [];
            for (const c of mins.slice(0, this.o.max_alternatives)) {
                const r = this.refine(c.x, c.y, c.start);
                if (!isFinite(r.loss)) continue;
                if (refined.some(q => Math.hypot(r.x - q.x, r.y - q.y) < 0.5 * this.grid.step)) continue;
                refined.push(r);
                if (onProgress) onProgress('refine', refined.length / this.o.max_alternatives);
                await new Promise(r2 => setTimeout(r2, 0));
            }
            if (!refined.length) return { status: 'no_match', message: 'Refinement failed for all candidate optima.' };
            refined.sort((a, b) => a.loss - b.loss);
            const best = refined[0];
            const nGroups = Math.max(1, this.groupIds.length);
            for (const r of refined) {
                Object.assign(r, this.toGeo(r.x, r.y));
                r.robust_rmse_hz = robustRmse(r.loss / nGroups);
                r.distance_from_best_m = Math.hypot(r.x - best.x, r.y - best.y);
            }
            const boundary = Math.hypot(best.x, best.y) > this.grid.R - 0.5 * this.grid.step;
            const diag = this.diagnostics(best.x, best.y, best.start, best.choice);
            const sens = sensitivity ? await this.sensitivity(best, onProgress) : null;
            return {
                status: 'ok', best, alternatives: refined.slice(1), diagnostics: diag,
                boundary_limited: boundary || best.clock_at_refine_edge, sensitivity: sens,
                robust_rmse_hz: robustRmse(best.loss / nGroups), n_groups: nGroups,
            };
        }

        /** Mirrors JointMatcher.altitude_perturbation_m: 150 m geometric, 200 m baro+QNH, 300 m otherwise. */
        altitudePerturbation(choice) {
            const idx = choice ? Object.values(choice).filter(k => k != null && k >= 0) : [];
            const list = idx.length ? idx : this.models.map((m, k) => k);
            let delta = 150;
            for (const k of list) {
                const m = this.models[k];
                if (m.altitude_basis === 'geometric') continue;
                delta = Math.max(delta, m.altitude_basis === 'barometric' && m.altitude_corrected ? 200 : 300);
            }
            return delta;
        }

        /** Same declared perturbations as the Python backend (sound speed, altitude, wind, frequency). */
        async sensitivity(best, onProgress) {
            const scenarios = [];
            const base = this.o;
            const altDelta = this.altitudePerturbation(best.choice);
            const dw = base.wind_uncertainty_mps;
            const d = base.loss_scale_hz;
            const tilt = (t) => { const span = Math.max(t[t.length - 1] - t[0], 1e-6), mid = 0.5 * (t[0] + t[t.length - 1]); return t.map(x => ((x - mid) / span) * 2); };
            const fmt = (v) => `${v > 0 ? '+' : ''}${Math.round(v)}`;
            const jobs = [];
            for (const dc of [-5, 5]) jobs.push([`sound_speed_${fmt(dc)}mps`, Object.assign({}, base, { speed_of_sound: base.speed_of_sound + dc }), null, null]);
            for (const da of [-altDelta, altDelta]) {
                jobs.push([`altitude_${fmt(da)}m`, base, this.rawTrajectories.map(t => new TrajectoryModel(t, this.center.lat, this.center.lon, this.engine, base.altitude_bias_m + da, base.max_gap_s)), null]);
            }
            if (dw > 0) {
                for (const [name, de, dn] of [['wind_east', dw, 0], ['wind_west', -dw, 0], ['wind_north', 0, dw], ['wind_south', 0, -dw]]) {
                    jobs.push([`${name}_${Math.round(dw)}mps`, Object.assign({}, base, { wind_e: base.wind_e + de, wind_n: base.wind_n + dn }), null, null]);
                }
            }
            const shift = (fn) => this.events.map(e => Object.assign({}, e, { f: fn(e) }));
            jobs.push(['freq_offset_+', base, null, shift(e => e.f.map(v => v + d))]);
            jobs.push(['freq_offset_-', base, null, shift(e => e.f.map(v => v - d))]);
            jobs.push(['freq_tilt_+', base, null, shift(e => { const w = tilt(e.t); return e.f.map((v, i) => v + d * w[i]); })]);
            jobs.push(['freq_tilt_-', base, null, shift(e => { const w = tilt(e.t); return e.f.map((v, i) => v - d * w[i]); })]);
            for (const [name, o, models, events] of jobs) {
                const r = this.refine(best.x, best.y, best.start, o, models, events);
                const ok = isFinite(r.loss);
                scenarios.push({ scenario: name, ok, displacement_m: ok ? +Math.hypot(r.x - best.x, r.y - best.y).toFixed(1) : null,
                    clock_shift_s: ok ? +(r.start - best.start).toFixed(3) : null });
                if (onProgress) onProgress('sensitivity', scenarios.length / jobs.length);
                await new Promise(r2 => setTimeout(r2, 0));
            }
            const oks = scenarios.filter(s => s.ok);
            return {
                complete: oks.length === scenarios.length,
                radius_m: oks.length ? Math.max(...oks.map(s => s.displacement_m)) : null,
                scenarios, sources: dw > 0 ? ['sound_speed', 'altitude', 'wind', 'frequency'] : ['sound_speed', 'altitude', 'frequency'],
                altitude_perturbation_m: altDelta, wind_perturbation_mps: dw,
                note: 'Declared engineering perturbations; not a calibrated confidence interval.',
            };
        }
    }

    function estimateCpa(t, f, c) {
        if (t.length >= 5) {
            try { return FlyoverFit.fit(t, f, c).cpa_arrival_time_s; } catch (e) { /* fall through */ }
        }
        let best = 0, minSlope = Infinity;
        for (let i = 1; i < t.length; i++) {
            const s = (f[i] - f[i - 1]) / Math.max(1e-6, t[i] - t[i - 1]);
            if (s < minSlope) { minSlope = s; best = i; }
        }
        return t[best];
    }

    function lowerBound(arr, x) {
        let lo = 0, hi = arr.length;
        while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < x) lo = m + 1; else hi = m; }
        return lo;
    }
    function countInRange(sorted, lo, hi) {
        let a = lowerBound(sorted, lo), b = lowerBound(sorted, hi);
        while (b < sorted.length && sorted[b] <= hi) b++;
        return b - a;
    }

    function decimate(ev, n) {
        if (ev.t.length <= n) return ev;
        const idx = uniqueSorted(linspace(0, ev.t.length - 1, n).map(roundHalfEven));
        return Object.assign({}, ev, { t: idx.map(i => ev.t[i]), f: idx.map(i => ev.f[i]) });
    }

    // ---------------------------------------------- trajectory conditioning
    // Mirrors app/adsb/conditioning.py: Gaussian-kernel local quadratic regression
    // onto a 1 s grid per gap-free segment, fused with reported velocities.
    function localQuadratic(t, y, tq, h) {
        const reach = 4 * h, n = t.length;
        const val = new Array(tq.length), der = new Array(tq.length);
        for (let q = 0; q < tq.length; q++) {
            const tc = tq[q];
            let lo = lowerBound(t, tc - reach);
            let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, r0 = 0, r1 = 0, r2 = 0;
            for (let j = lo; j < n && t[j] <= tc + reach; j++) {
                const d = t[j] - tc, w = Math.exp(-0.5 * (d / h) ** 2), d2 = d * d;
                s0 += w; s1 += w * d; s2 += w * d2; s3 += w * d2 * d; s4 += w * d2 * d2;
                r0 += w * y[j]; r1 += w * d * y[j]; r2 += w * d2 * y[j];
            }
            const det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s3 * s2) + s2 * (s1 * s3 - s2 * s2);
            const det2 = s0 * s2 - s1 * s1;
            if (Math.abs(det) > 1e-9 * Math.max(Math.abs(s0 * s2 * s4), 1e-300)) {
                val[q] = (r0 * (s2 * s4 - s3 * s3) - s1 * (r1 * s4 - s3 * r2) + s2 * (r1 * s3 - s2 * r2)) / det;
                der[q] = (s0 * (r1 * s4 - s3 * r2) - r0 * (s1 * s4 - s3 * s2) + s2 * (s1 * r2 - r1 * s2)) / det;
            } else if (Math.abs(det2) > 1e-9 * Math.max(Math.abs(s0 * s2), 1e-300)) {
                val[q] = (s2 * r0 - s1 * r1) / det2;
                der[q] = (s0 * r1 - s1 * r0) / det2;
            } else {
                val[q] = r0 / Math.max(s0, 1e-300);
                der[q] = 0;
            }
        }
        return { val, der };
    }

    function conditionTrajectory(traj, { maxGapS = 20, dt = 1, bandwidthS = null, reportedWeight = 0.7 } = {}) {
        const R = 6371000, rad = Math.PI / 180;
        const nRaw = traj.timestamps.length;
        const order = [...Array(nRaw).keys()].sort((a, b) => traj.timestamps[a] - traj.timestamps[b] || a - b);
        const idx = [];
        for (const i of order) if (!idx.length || traj.timestamps[i] - traj.timestamps[idx[idx.length - 1]] > 1e-6) idx.push(i);
        if (idx.length < 2) return Object.assign({}, traj);
        const t = idx.map(i => traj.timestamps[i]);
        const lat = idx.map(i => traj.latitudes[i]), lon = idx.map(i => traj.longitudes[i]), alt = idx.map(i => traj.altitudes[i]);
        const lat0 = lat[0], lon0 = lon[0];
        const x = lat.map((la, k) => R * (lon[k] - lon0) * rad * Math.cos(0.5 * (la + lat0) * rad));
        const y = lat.map(la => R * (la - lat0) * rad);
        const rep = (key) => {
            const a = traj[key] || [];
            return a.length === nRaw && a.every(v => v != null) ? idx.map(i => a[i]) : null;
        };
        const gs = rep('speeds'), trk = rep('headings'), vr = rep('vertical_rates');
        const hasH = !!(gs && trk);
        const veRep = hasH ? gs.map((g, k) => g * Math.sin(trk[k] * rad)) : null;
        const vnRep = hasH ? gs.map((g, k) => g * Math.cos(trk[k] * rad)) : null;
        const segs = [];
        let a = 0;
        for (let k = 1; k <= t.length; k++) if (k === t.length || t[k] - t[k - 1] > maxGapS) { segs.push([a, k]); a = k; }
        const T = [], X = [], Y = [], Z = [], VE = [], VN = [], VU = [], hs = [];
        let nSeg = 0;
        for (const [s0, s1] of segs) {
            if (s1 - s0 < 2) continue;
            nSeg++;
            const ts = t.slice(s0, s1), sl = (arr) => arr.slice(s0, s1);
            const nOut = Math.floor((ts[ts.length - 1] - ts[0]) / dt + 1e-9) + 1;
            const tg = [...Array(nOut).keys()].map(k => ts[0] + dt * k);
            if (tg[tg.length - 1] < ts[ts.length - 1] - 1e-6) tg.push(ts[ts.length - 1]);
            const diffs = ts.slice(1).map((v, k) => v - ts[k]);
            const h = bandwidthS != null ? bandwidthS : Math.min(30, Math.max(2, 2 * median(diffs)));
            hs.push(h);
            const px = localQuadratic(ts, sl(x), tg, h), py = localQuadratic(ts, sl(y), tg, h), pz = localQuadratic(ts, sl(alt), tg, h);
            let dx = px.der, dy = py.der, dz = pz.der;
            if (hasH) {
                const re = localQuadratic(ts, sl(veRep), tg, h).val, rn = localQuadratic(ts, sl(vnRep), tg, h).val;
                dx = dx.map((v, k) => reportedWeight * re[k] + (1 - reportedWeight) * v);
                dy = dy.map((v, k) => reportedWeight * rn[k] + (1 - reportedWeight) * v);
            }
            if (vr) {
                const rz = localQuadratic(ts, sl(vr), tg, h).val;
                dz = dz.map((v, k) => reportedWeight * rz[k] + (1 - reportedWeight) * v);
            }
            T.push(...tg); X.push(...px.val); Y.push(...py.val); Z.push(...pz.val); VE.push(...dx); VN.push(...dy); VU.push(...dz);
        }
        if (!T.length) return Object.assign({}, traj);
        const LAT = Y.map(v => lat0 + v / R / rad);
        const LON = X.map((v, k) => lon0 + v / (R * Math.cos(0.5 * (LAT[k] + lat0) * rad)) / rad);
        const out = Object.assign({}, traj);
        Object.assign(out, {
            timestamps: T, latitudes: LAT, longitudes: LON, altitudes: Z,
            velocity_east: VE, velocity_north: VN, velocity_up: VU,
            speeds: VE.map((v, k) => Math.hypot(v, VN[k])),
            headings: VE.map((v, k) => ((Math.atan2(v, VN[k]) / rad) % 360 + 360) % 360),
            vertical_rates: VU.slice(),
            conditioning: {
                method: 'gaussian_local_quadratic', bandwidth_s: hs.length ? median(hs) : null, resample_dt_s: dt,
                segments: nSeg, max_gap_s: maxGapS, fused_reported_velocity: hasH, fused_reported_vertical_rate: !!vr,
                reported_weight: reportedWeight, input_points: nRaw, output_points: T.length,
            },
        });
        return out;
    }

    // ------------------------------------------------------------- prefilter
    // Mirrors app/physics/prefilter.py (fixed tolerances, identical decisions).
    const PREFILTER = { speedTolMin: 30, speedTolFrac: 0.25, slantTolMin: 800, slantTolFrac: 0.6, maxDelayS: 200, fitMaxRmse: 4, fitMinSamples: 8 };

    function fitIsReliable(fit) {
        if (!fit) return false;
        const at = fit.parameter_at_bound || {};
        const v = +fit.speed_mps || 0, d = +fit.cpa_slant_m || 0;
        const atBound = !!(at.speed || at.distance) || !(v > 15.5 && v < 329.5) || !(d > 31 && d < 39990);
        return (fit.n_samples || 0) >= PREFILTER.fitMinSamples && (fit.rmse_hz != null ? fit.rmse_hz : 1e9) <= PREFILTER.fitMaxRmse && !atBound;
    }

    function flightGroundSpeed(traj) {
        const sp = (traj.speeds || []).filter(v => v != null);
        if (sp.length) return median(sp);
        const t = traj.timestamps || [];
        if (t.length < 2) return null;
        const rad = Math.PI / 180, v = [];
        for (let i = 1; i < t.length; i++) {
            const dt = t[i] - t[i - 1];
            if (dt <= 0) continue;
            const la1 = traj.latitudes[i - 1] * rad, la2 = traj.latitudes[i] * rad;
            const dx = 6371000 * (traj.longitudes[i] - traj.longitudes[i - 1]) * rad * Math.cos(0.5 * (la1 + la2));
            const dy = 6371000 * (la2 - la1);
            v.push(Math.hypot(dx, dy) / dt);
        }
        return v.length ? median(v) : null;
    }

    /** Diagnostic kinematic comparison of a flyover fit with a trajectory (mirrors flyover_fit.kinematic_consistency). */
    function kinematicConsistency(fit, traj, observerAltM = 0) {
        const speeds = (traj.speeds || []).filter(v => v != null);
        const alts = (traj.altitudes || []).filter(v => v != null);
        const vFit = +fit.speed_mps, dFit = +fit.cpa_slant_m;
        const unc = fit.uncertainty || {};
        const vTol = Math.max(25, 3 * (unc.speed_mps || 0), 0.2 * vFit);
        const dTol = Math.max(500, 3 * (unc.cpa_slant_m || 0), 0.5 * dFit);
        const vFlight = speeds.length ? median(speeds) : NaN;
        const minH = alts.length ? Math.min(...alts) - observerAltM : NaN;
        const speedOk = Number.isFinite(vFlight) && Math.abs(vFlight - vFit) <= vTol;
        const heightOk = Number.isFinite(minH) && minH <= dFit + dTol;
        return {
            flight_speed_mps: Number.isFinite(vFlight) ? +vFlight.toFixed(1) : null, fitted_speed_mps: +vFit.toFixed(1),
            speed_tolerance_mps: +vTol.toFixed(1), min_height_above_observer_m: Number.isFinite(minH) ? +minH.toFixed(1) : null,
            fitted_cpa_slant_m: +dFit.toFixed(1), slant_tolerance_m: +dTol.toFixed(1),
            speed_compatible: speedOk, height_compatible: heightOk, compatible: speedOk && heightOk,
        };
    }

    function prefilterFlights(flights, rawEvents, flyoverFits, clockWindow = null, observerAltM = 0) {
        const spans = new Map();
        rawEvents.forEach((ev, i) => {
            const pts = ev.traced_curve || ev.points || [];
            if (!pts.length) return;
            const eid = String(ev.id || `E${i + 1}`), gid = String(ev.group_id || eid);
            const lo = Math.min(...pts.map(p => p.time)), hi = Math.max(...pts.map(p => p.time));
            const cur = spans.get(gid);
            spans.set(gid, cur ? [Math.min(cur[0], lo), Math.max(cur[1], hi), cur[2].concat([eid])] : [lo, hi, [eid]]);
        });
        const groupFit = new Map();
        for (const [gid, [, , ids]] of spans) {
            const fits = ids.map(e => flyoverFits[e]).filter(fitIsReliable);
            if (fits.length) groupFit.set(gid, fits.reduce((a, b) => ((b.n_samples || 0) > (a.n_samples || 0) ? b : a)));
        }
        const kept = [], discarded = [];
        for (const f of flights) {
            const t = f.timestamps || [];
            const vg = flightGroundSpeed(f);
            const alts = (f.altitudes || []).filter(a => a != null);
            const minH = alts.length ? Math.min(...alts) - observerAltM : null;
            let excludedAll = spans.size > 0;
            const reasons = [];
            for (const [gid, [lo, hi]] of spans) {
                const why = [];
                if (clockWindow && t.length) {
                    const aLo = t[0], aHi = t[t.length - 1] + PREFILTER.maxDelayS;
                    if (clockWindow[1] + hi < aLo || clockWindow[0] + lo > aHi) {
                        why.push({ reason: 'no_time_overlap', event_window: [+(clockWindow[0] + lo).toFixed(1), +(clockWindow[1] + hi).toFixed(1)], flight_arrival_window: [+aLo.toFixed(1), +aHi.toFixed(1)] });
                    }
                }
                const fit = groupFit.get(gid);
                if (fit) {
                    const vFit = +fit.speed_mps, dFit = +fit.cpa_slant_m;
                    const vTol = Math.max(PREFILTER.speedTolMin, PREFILTER.speedTolFrac * vFit);
                    const dTol = Math.max(PREFILTER.slantTolMin, PREFILTER.slantTolFrac * dFit);
                    if (vg != null && Math.abs(vg - vFit) > vTol) why.push({ reason: 'speed_incompatible', flight_speed_mps: +vg.toFixed(1), fitted_speed_mps: +vFit.toFixed(1), tolerance_mps: +vTol.toFixed(1) });
                    if (minH != null && minH > dFit + dTol) why.push({ reason: 'height_incompatible', min_height_above_observer_m: +minH.toFixed(1), fitted_cpa_slant_m: +dFit.toFixed(1), tolerance_m: +dTol.toFixed(1) });
                }
                if (why.length) reasons.push({ group_id: gid, reasons: why }); else excludedAll = false;
            }
            if (excludedAll) discarded.push({ flight_id: f.flight_id, callsign: f.callsign, data_source: f.data_source, groups: reasons });
            else kept.push(f);
        }
        return { kept, discarded };
    }

    // ------------------------------------------------------------- decision
    const DEFAULT_POLICY = {
        name: 'engineering-v2', max_rmse_hz: 3.0, max_outlier_fraction: 0.25, min_support_fraction: 0.95,
        min_independent_flyovers: 2, near_tie_hz: 0.5, precision_target_m: 1000, require_sensitivity: false,
        residual_metric: 'robust',
    };
    const REASONS = {
        no_doppler_events: 'No usable Doppler event was supplied.',
        no_adsb_data: 'No real ADS-B trajectories were available for the search area and time window (demo data is opt-in).',
        no_matching_flights: "No candidate flight explained every event with full trajectory support.",
        insufficient_track_support: "Some measured samples fall outside the matched trajectory's arrival-time support.",
        inconsistent_event_match: 'Event residuals or outlier fraction exceed the declared thresholds.',
        source_drift_unresolved: 'Fitted source-frequency drift reached its bound; the tone was not stable enough.',
        refinement_boundary_limited: 'The best position (or clock) sits on the search boundary; enlarge the search.',
        insufficient_independent_events: 'Fewer independent flyovers than required (and no trusted recording clock).',
        ambiguous_locations: 'A materially different location fits almost equally well.',
        uncertainty_not_established: 'Sensitivity analysis was not available or did not complete.',
        location_uncertainty_too_large: 'The sensitivity radius exceeds the precision target.',
        supported_by_evidence_checks: 'Passed all declared evidence checks (uncalibrated; not a probability).',
    };

    function decideLocation(result, policy = {}, nEvents = null) {
        const p = Object.assign({}, DEFAULT_POLICY, policy);
        const checks = [];
        const add = (check, passed, detail = {}) => checks.push(Object.assign({ check, passed: !!passed }, detail));
        const outcome = (code) => ({
            status: code === 'supported_by_evidence_checks' ? 'supported' : 'inconclusive',
            reason_code: code, message: REASONS[code], policy: p, checks, calibrated: false,
        });
        if (nEvents === 0) return outcome('no_doppler_events');
        if (result && result.status === 'no_adsb_data') return outcome('no_adsb_data');
        if (!result || result.status !== 'ok') return outcome('no_matching_flights');
        const diag = result.diagnostics || [];
        const matched = diag.filter(d => d.matched);
        if (!matched.length || matched.length < diag.length) {
            add('all_events_matched', false, { unmatched: diag.filter(d => !d.matched).map(d => d.event_id) });
            return outcome('no_matching_flights');
        }
        add('all_events_matched', true);
        let first = null;
        const minSupport = Math.min(...matched.map(d => d.support_fraction));
        add('track_support', minSupport >= p.min_support_fraction, { min_support_fraction: minSupport });
        if (minSupport < p.min_support_fraction) first = first || 'insufficient_track_support';
        const metricKey = p.residual_metric === 'robust' ? 'robust_rmse_hz' : 'rmse_hz';
        const worstRmse = Math.max(...matched.map(d => (d[metricKey] != null ? d[metricKey] : d.rmse_hz)));
        const worstOut = Math.max(...matched.map(d => d.outlier_fraction));
        const residOk = worstRmse <= p.max_rmse_hz && worstOut <= p.max_outlier_fraction;
        add('residuals', residOk, { worst_rmse_hz: worstRmse, worst_outlier_fraction: worstOut, metric: p.residual_metric });
        if (!residOk) first = first || 'inconsistent_event_match';
        const driftBound = matched.filter(d => d.drift_at_bound).map(d => d.event_id);
        add('source_drift', !driftBound.length, { events_at_bound: driftBound });
        if (driftBound.length) first = first || 'source_drift_unresolved';
        add('boundary', !result.boundary_limited);
        if (result.boundary_limited) first = first || 'refinement_boundary_limited';
        const independent = Math.min(new Set(matched.map(d => d.flight_id)).size, new Set(matched.map(d => d.group_id)).size);
        add('independent_flyovers', independent >= p.min_independent_flyovers, { independent_flyovers: independent, required: p.min_independent_flyovers });
        const ties = (result.alternatives || []).filter(a => a.robust_rmse_hz - result.robust_rmse_hz <= p.near_tie_hz && a.distance_from_best_m > p.precision_target_m)
            .map(a => ({ lat: a.lat, lon: a.lon, distance_m: +a.distance_from_best_m.toFixed(1), rmse_gap_hz: +(a.robust_rmse_hz - result.robust_rmse_hz).toFixed(3) }));
        add('ambiguity', !ties.length, { near_tied_alternatives: ties });
        const sens = result.sensitivity;
        let sensOk = true;
        if (!sens || !sens.complete) {
            add('sensitivity', !p.require_sensitivity, { available: !!sens, complete: !!(sens && sens.complete) });
            if (p.require_sensitivity) { sensOk = false; first = first || 'uncertainty_not_established'; }
        } else {
            sensOk = (sens.radius_m || 0) <= p.precision_target_m;
            add('sensitivity', sensOk, { radius_m: sens.radius_m, precision_target_m: p.precision_target_m });
        }
        if (first) return outcome(first);
        if (independent < p.min_independent_flyovers) return outcome(ties.length ? 'ambiguous_locations' : 'insufficient_independent_events');
        if (ties.length) return outcome('ambiguous_locations');
        if (!sensOk) return outcome('location_uncertainty_too_large');
        return outcome('supported_by_evidence_checks');
    }


    /**
     * Browser geolocation pipeline (mirrors app.main.run_geolocation):
     * QNH, conditioning, flyover fits, pre-filter, joint matching, decision
     * and result formatting.  `found` is the flight-resolution result
     * ({flights, data_source, notes, limitations, message}).  Runs on the
     * main thread or inside analysis_worker.js.
     */
    async function runPipeline(payload, found, onProgress = null) {
        const DA = window.DopplerAnalysis;
        const events = payload.events;
        let flights = found.flights;
        const provenance = { data_source: found.data_source, data_notes: found.notes, data_limitations: found.limitations };
        if (!flights || flights.length === 0) {
            return { status: 'no_adsb_data', message: found.message, mode: 'client_physics',
                decision: DA.decideLocation({ status: 'no_adsb_data' }, {}, events.length), ...provenance };
        }
        if (payload.flight_id) {
            const target = flights.filter(f => f.flight_id === payload.flight_id);
            if (target.length) flights = target;
        }
        if (payload.qnh_hpa != null) flights = flights.map(f => DA.applyQnh(f, payload.qnh_hpa));
        if (payload.condition_tracks !== false) {
            flights = flights.map(f => (['historical', 'uploaded'].includes(f.data_source) ? DA.conditionTrajectory(f) : f));
        }
        const c = DA.speedOfSoundFor(payload.temperature_c, payload.relative_humidity_pct);
        const [windE, windN] = DA.windVector(payload.wind_speed_mps, payload.wind_from_deg);
        let elevation = null, elevationNote = 'flat_ground';
        if (payload.observer_elevation_grid) {
            elevation = DA.latLonElevation(payload.observer_elevation_grid, payload.center_lat, payload.center_lon, payload.elevation_datum_offset_m || 0);
            elevationNote = 'user_grid';
        } else if (payload.dem_lookup) {
            elevationNote = 'dem_unavailable_flat_ground';
        }
        const opts = {
            speed_of_sound: c,
            fit_f0: payload.fit_f0 !== false,
            loss_scale_hz: payload.loss_scale_hz || 2.0,
            drift_fraction: payload.drift_fraction != null ? payload.drift_fraction : 0.02,
            observer_alt_m: payload.observer_alt_m || 0,
            wind_e: windE, wind_n: windN,
            wind_uncertainty_mps: payload.wind_uncertainty_mps != null ? payload.wind_uncertainty_mps : 5.0,
            elevation,
        };
        provenance.atmosphere = {
            speed_of_sound_mps: +c.toFixed(3), wind_east_mps: +windE.toFixed(3), wind_north_mps: +windN.toFixed(3),
            wind_speed_mps: payload.wind_speed_mps || 0, wind_from_deg: payload.wind_from_deg ?? null,
            wind_uncertainty_mps: opts.wind_uncertainty_mps, qnh_hpa: payload.qnh_hpa ?? null, elevation: elevationNote,
        };
        if (!opts.fit_f0) events.forEach(e => { e.f0 = e.f0 || payload.f0; });
        const clock = payload.recording_start_epoch != null
            ? [payload.recording_start_epoch - (payload.clock_tolerance_s ?? 2), payload.recording_start_epoch + (payload.clock_tolerance_s ?? 2)]
            : null;

        const flyoverFits = {};
        for (const ev of events) {
            const pts = ev.traced_curve || [];
            if (pts.length < 5) continue;
            try { flyoverFits[ev.id] = DA.FlyoverFit.fit(pts.map(p => p.time), pts.map(p => p.freq), c); } catch (e) { /* skip */ }
        }
        let candidates = flights, discarded = [];
        if (payload.prefilter !== false) {
            ({ kept: candidates, discarded } = DA.prefilterFlights(flights, events, flyoverFits, clock, opts.observer_alt_m));
        }
        provenance.discarded_flights = discarded;
        const summaries = (diags) => flights.map(f => {
            const d = discarded.find(x => x.flight_id === f.flight_id);
            return {
                flight_id: f.flight_id, callsign: f.callsign, aircraft_type: f.aircraft_type,
                data_source: f.data_source, altitude_basis: f.altitude_basis,
                matched_events: diags.filter(x => x.flight_id === f.flight_id && x.matched).map(x => x.event_id),
                discarded: !!d, discard_reasons: d ? d.groups : [], conditioning: f.conditioning || null,
                kinematic_consistency: Object.fromEntries(Object.entries(flyoverFits).map(([eid, fit]) => [eid, kinematicConsistency(fit, f, opts.observer_alt_m)])),
            };
        });
        if (!candidates.length) {
            return { status: 'no_match', message: 'Every candidate flight was excluded by the kinematic / time pre-filter.',
                decision: DA.decideLocation({ status: 'no_match' }, {}, events.length), flyover_fits: flyoverFits,
                evaluated_flights: summaries([]), mode: 'client_physics', ...provenance };
        }
        const matcher = new DA.JointMatcher(candidates, events, payload.center_lat, payload.center_lon, opts, clock);
        const res = await matcher.run(payload.radius_km, payload.grid_resolution || 25, {
            sensitivity: payload.sensitivity !== false, onProgress,
        });
        const decision = DA.decideLocation(res, {
            max_rmse_hz: payload.max_rmse_hz || 3.0,
            residual_metric: payload.residual_metric || 'robust',
            min_independent_flyovers: payload.min_independent_flyovers || 2,
            precision_target_m: payload.precision_target_m || 1000,
        }, events.length);

        if (res.status !== 'ok') {
            return { status: 'no_match', message: res.message, decision, flyover_fits: flyoverFits, evaluated_flights: summaries([]), mode: 'client_physics', ...provenance };
        }

        const sigma = 3 * opts.loss_scale_hz;
        const scoreOf = (rr) => (isFinite(rr) ? 100 * Math.exp(-0.5 * (rr / sigma) ** 2) : 0);
        const nG = res.n_groups;
        const heatmap = matcher.grid.cells.filter(cell => cell.inside).map(cell => {
            const g = matcher.toGeo(cell.x, cell.y);
            const rr = DA.robustRmse(cell.loss / nG);
            return { lat: +g.lat.toFixed(6), lon: +g.lon.toFixed(6), score: +scoreOf(rr).toFixed(2), rmse: isFinite(rr) ? +rr.toFixed(2) : null };
        });
        const best = res.best;
        const bg = matcher.toGeo(best.x, best.y);
        const diags = res.diagnostics;
        const primary = diags.find(d => d.matched) || diags[0];
        const pk = matcher.models.find(m => m.flight_id === primary.flight_id) || matcher.models[0];
        const comparison = [];
        const eventsOut = diags.map(d => {
            const { times = [], traced = [], modeled = [], ...rest } = d;
            times.forEach((t, i) => comparison.push({
                event_id: d.event_id, time: +t.toFixed(3), traced_freq: +traced[i].toFixed(2),
                modeled_freq: modeled[i] != null ? +modeled[i].toFixed(2) : null,
            }));
            return rest;
        });
        const matchedIds = new Set(diags.filter(d => d.matched).map(d => d.flight_id));
        return {
            status: 'success',
            mode: 'client_physics',
            ...provenance,
            best_flight: { flight_id: pk.flight_id, callsign: pk.raw.callsign, aircraft_type: pk.raw.aircraft_type, data_source: pk.raw.data_source },
            best_location: {
                lat: +bg.lat.toFixed(6), lon: +bg.lon.toFixed(6),
                score: +scoreOf(res.robust_rmse_hz).toFixed(2),
                rmse_hz: +Math.max(...diags.filter(d => d.matched).map(d => d.rmse_hz)).toFixed(3),
                robust_rmse_hz: +res.robust_rmse_hz.toFixed(3),
                time_offset_sec: +(best.start - pk.t[0]).toFixed(3),
                recording_start_epoch: +best.start.toFixed(3),
                uncertainty_radius_m: res.sensitivity ? res.sensitivity.radius_m : null,
            },
            decision,
            alternatives: res.alternatives.map(a => ({
                lat: +a.lat.toFixed(6), lon: +a.lon.toFixed(6), robust_rmse_hz: +a.robust_rmse_hz.toFixed(3),
                distance_from_best_m: +a.distance_from_best_m.toFixed(1), recording_start_epoch: +a.start.toFixed(3),
            })),
            sensitivity: res.sensitivity,
            events: eventsOut,
            flyover_fits: flyoverFits,
            heatmap,
            curve_comparison: comparison,
            flight_trajectory: {
                flight_id: pk.flight_id, callsign: pk.raw.callsign, aircraft_type: pk.raw.aircraft_type,
                data_source: pk.raw.data_source, latitudes: pk.raw.latitudes, longitudes: pk.raw.longitudes, altitudes: pk.raw.altitudes, speeds: pk.raw.speeds,
            },
            matched_flights: matcher.models.filter(m => matchedIds.has(m.flight_id)).map(m => ({
                flight_id: m.flight_id, callsign: m.raw.callsign, aircraft_type: m.raw.aircraft_type,
                data_source: m.raw.data_source, latitudes: m.raw.latitudes, longitudes: m.raw.longitudes,
            })),
            f0_hz: primary.f0_hz,
            evaluated_flights: summaries(diags),
            grid_resolution: payload.grid_resolution || 25,
            speed_of_sound_mps: c,
            clock_mode: clock ? 'trusted_window' : 'free',
        };
    }

    window.DopplerAnalysis = {
        FlyoverFit, RidgeDiscovery, JointMatcher, TrajectoryModel, GeoEngine, decideLocation, robustRmse, nelderMead,
        fitEventAt, speedOfSound, speedOfSoundFor, DEFAULT_MATCH, DEFAULT_POLICY, REASONS,
        conditionTrajectory, localQuadratic, prefilterFlights, fitIsReliable, flightGroundSpeed, PREFILTER,
        propagation, windVector, qnhAltitude, applyQnh, latLonElevation, runPipeline, DEFAULT_DISCOVERY, kinematicConsistency,
    };
})(typeof window !== 'undefined' ? window : globalThis);
