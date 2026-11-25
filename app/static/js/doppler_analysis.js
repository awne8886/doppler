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
        const a = Array.from(arr).sort((x, y) => x - y);
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

        async discover(audioBuffer, settings = {}) {
            const s = Object.assign({}, DEFAULT_DISCOVERY, settings);
            const sr = Math.min(48000, Math.max(4000, 4 * s.fmax_hz));
            const x = await this.resample(audioBuffer, sr);
            return this.discoverSignal(x, sr, s);
        },

        discoverSignal(x, sr, s) {
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
    };

    class TrajectoryModel {
        constructor(traj, centerLat, centerLon, engine, altBias = 0, maxGap = 20) {
            this.raw = traj;
            this.flight_id = String(traj.flight_id || 'UNKNOWN');
            const n = traj.timestamps.length;
            const idx = [...Array(n).keys()].sort((a, b) => traj.timestamps[a] - traj.timestamps[b]);
            this.t = idx.map(i => traj.timestamps[i]);
            this.pos = idx.map(i => engine.geodeticToCartesian(traj.latitudes[i], traj.longitudes[i], traj.altitudes[i] + altBias, centerLat, centerLon, 0));
            this.vel = this.pos.map((p, i) => {
                const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1);
                const dt = Math.max(1e-3, this.t[b] - this.t[a]);
                return { vx: (this.pos[b].x - this.pos[a].x) / dt, vy: (this.pos[b].y - this.pos[a].y) / dt, vz: (this.pos[b].z - this.pos[a].z) / dt };
            });
            this.gapAfter = [];
            for (let i = 0; i < n - 1; i++) if (this.t[i + 1] - this.t[i] > maxGap) this.gapAfter.push(i);
        }
        arrival(obs, c) {
            const n = this.t.length;
            const rec = new Array(n);
            let dmin = Infinity, tcpa = 0;
            for (let i = 0; i < n; i++) {
                const rx = this.pos[i].x - obs.x, ry = this.pos[i].y - obs.y, rz = this.pos[i].z - obs.z;
                const d = Math.max(1, Math.sqrt(rx * rx + ry * ry + rz * rz));
                let ur = (rx * this.vel[i].vx + ry * this.vel[i].vy + rz * this.vel[i].vz) / d;
                ur = Math.max(-0.95 * c, Math.min(0.95 * c, ur));
                const to = this.t[i] + d / c;
                rec[i] = [to, c / (c + ur), this.t[i]];
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
                return [...Array(n).keys()].map(i => lo + ((hi - lo) * i) / (n - 1));
            }
            const set = new Set();
            for (const gid of this.groupIds) {
                const ev = this.events[this.groups[gid][0]];
                for (const k of this.allowed(gid)) {
                    const nom = Math.round((curves[k].t_cpa - ev.cpa_time_s) / o.coarse_step_s) * o.coarse_step_s;
                    for (let d = -o.coarse_halfwidth_s; d <= o.coarse_halfwidth_s + 1e-9; d += o.coarse_step_s) set.add(+(nom + d).toFixed(4));
                }
            }
            return [...set].sort((a, b) => a - b);
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
            const curves = models.map(m => m.arrival(obs, o.speed_of_sound));
            starts = starts || this.candidateStarts(curves);
            const coarse = this.joint(curves, starts, o, events);
            if (!isFinite(coarse.total)) return coarse;
            const fine = [];
            for (let d = -o.fine_halfwidth_s; d <= o.fine_halfwidth_s + 1e-9; d += o.fine_step_s) fine.push(coarse.start + d);
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
                        const r = this.evaluatePoint({ x, y, z: this.o.observer_alt_m }, null, null, null, this.eventsCoarse);
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
            for (let d = -hw; d <= hw + 1e-9; d += o.fine_step_s * 2) starts.push(s0 + d);
            const st = this.clipClock(starts);
            const step = this.grid ? this.grid.step : 400;
            const obj = (p) => {
                const r = this.evaluatePoint({ x: p[0], y: p[1], z: o.observer_alt_m }, st, o, models, events);
                return isFinite(r.total) ? r.total : 1e12;
            };
            const nm = nelderMead(obj, [x0, y0], [0.5 * step, 0.5 * step], maxIter, 1e-6);
            const r = this.evaluatePoint({ x: nm.x[0], y: nm.x[1], z: o.observer_alt_m }, st, o, models, events);
            const clockEdge = !this.clockWindow && (Math.abs(r.start - st[0]) < 1e-6 || Math.abs(r.start - st[st.length - 1]) < 1e-6);
            return { x: nm.x[0], y: nm.x[1], loss: r.total, start: r.start, choice: r.choice, clock_at_refine_edge: clockEdge };
        }

        toGeo(x, y) {
            const g = this.engine.cartesianToGeodetic(x, y, 0, this.center.lat, this.center.lon, 0);
            return { lat: g.lat, lon: g.lon };
        }

        diagnostics(x, y, start, choice) {
            const obs = { x, y, z: this.o.observer_alt_m };
            const curves = this.models.map(m => m.arrival(obs, this.o.speed_of_sound));
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

        async run(radiusKm, res = 21, { sensitivity = true, onProgress = null } = {}) {
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

        async sensitivity(best, onProgress) {
            const scenarios = [];
            const base = this.o;
            const runOne = async (name, o, models) => {
                const r = this.refine(best.x, best.y, best.start, o, models, null, 60);
                const ok = isFinite(r.loss);
                scenarios.push({ scenario: name, ok, displacement_m: ok ? +Math.hypot(r.x - best.x, r.y - best.y).toFixed(1) : null });
                if (onProgress) onProgress('sensitivity', scenarios.length / 4);
                await new Promise(r2 => setTimeout(r2, 0));
            };
            for (const dc of [-5, 5]) await runOne(`sound_speed_${dc > 0 ? '+' : ''}${dc}mps`, Object.assign({}, base, { speed_of_sound: base.speed_of_sound + dc }), null);
            for (const da of [-150, 150]) {
                const models = this.rawTrajectories.map(t => new TrajectoryModel(t, this.center.lat, this.center.lon, this.engine, base.altitude_bias_m + da, base.max_gap_s));
                await runOne(`altitude_${da > 0 ? '+' : ''}${da}m`, base, models);
            }
            const oks = scenarios.filter(s => s.ok);
            return {
                complete: oks.length === scenarios.length,
                radius_m: oks.length ? Math.max(...oks.map(s => s.displacement_m)) : null,
                scenarios, sources: ['sound_speed', 'altitude'],
                note: 'Declared engineering perturbations (reduced browser set); not a calibrated confidence interval.',
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
        const idx = [...new Set([...Array(n).keys()].map(i => Math.round((i * (ev.t.length - 1)) / (n - 1))))];
        return Object.assign({}, ev, { t: idx.map(i => ev.t[i]), f: idx.map(i => ev.f[i]) });
    }

    // ------------------------------------------------------------- decision
    const DEFAULT_POLICY = {
        name: 'engineering-v1', max_rmse_hz: 3.0, max_outlier_fraction: 0.25, min_support_fraction: 0.95,
        min_independent_flyovers: 2, near_tie_hz: 0.5, precision_target_m: 1000, require_sensitivity: false,
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
        const worstRmse = Math.max(...matched.map(d => d.rmse_hz));
        const worstOut = Math.max(...matched.map(d => d.outlier_fraction));
        const residOk = worstRmse <= p.max_rmse_hz && worstOut <= p.max_outlier_fraction;
        add('residuals', residOk, { worst_rmse_hz: worstRmse, worst_outlier_fraction: worstOut });
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

    window.DopplerAnalysis = {
        FlyoverFit, RidgeDiscovery, JointMatcher, TrajectoryModel, GeoEngine, decideLocation, robustRmse, nelderMead,
        fitEventAt, speedOfSound, speedOfSoundFor, DEFAULT_MATCH, DEFAULT_POLICY, REASONS,
    };
})(typeof window !== 'undefined' ? window : globalThis);
