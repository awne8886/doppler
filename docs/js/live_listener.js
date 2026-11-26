/**
 * Live microphone listening.
 *
 * Asks for microphone access, then listens continuously:
 *   1. audio is low-passed, resampled to 8 kHz, time-stamped and kept for the last few minutes;
 *   2. every ~15 s the latest window is scanned for Doppler ridges (Web Worker); each finished
 *      flyover is committed once, with UTC times;
 *   3. live ADS-B is polled every ~10 s (server mode) and accumulated into per-aircraft tracks;
 *   4. after every new flyover all recent flyovers are matched jointly on the shared, trusted clock;
 *      a supported result re-centres and narrows the search, so later flyovers refine it further.
 *
 * Analysis and polling are driven by incoming audio blocks rather than timers, so they keep
 * running when the tab is in the background (browsers throttle timers in hidden tabs).
 */
(function () {
    'use strict';

    const RATE = 8000; // analysis sample rate (Doppler tones of interest are < 2 kHz)
    const FMAX_HZ = 2000;
    const LOWPASS_HZ = 2400;
    const WINDOW_S = 240; // analysis window
    const HOLD_S = 300; // audio kept in memory
    const ANALYZE_EVERY_S = 15;
    const FIRST_ANALYSIS_S = 30;
    const POLL_EVERY_S = 10;
    const MAX_FLYOVERS_PER_MATCH = 8;

    const $ = (id) => document.getElementById(id);

    class LiveListener {
        constructor() {
            this.active = false;
            this.history = [];
            this.tracker = null;
            this.adsb = null;
            this.lastResult = null;
        }

        init() {
            this.btn = $('liveToggleBtn');
            if (!this.btn) return;
            this.statusEl = $('liveStatus');
            this.historyEl = $('liveHistory');
            this.btn.addEventListener('click', () => (this.active ? this.stop('Live listening stopped.') : this.start()));
            $('liveUseMapBtn').addEventListener('click', () => {
                const c = window.mapView.map.getCenter();
                window.DopplerApp.setSearchCenter(c.lat, c.lng);
                window.DopplerUtil.toast('Search centre set to the map centre.', 'ok', 3000);
            });
            $('liveExportBtn').addEventListener('click', () => this.exportData());
            setInterval(() => this.updateUi(), 1000);
            this.updateUi();
        }

        setStatus(html, warn = false) {
            if (!this.statusEl) return;
            this.statusEl.innerHTML = html;
            this.statusEl.classList.toggle('warn', !!warn);
        }

        // ------------------------------------------------------------ start / stop
        async start() {
            const { toast } = window.DopplerUtil;
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                toast('Microphone access needs a secure page (HTTPS or http://localhost) and a browser that supports it.', 'error', 10000);
                return;
            }
            const area = window.DopplerApp.getSearchArea();
            if (![area.lat, area.lon, area.radius_km].every(Number.isFinite)) {
                toast('Set a search centre and radius first (roughly where you are, e.g. your city).', 'warn');
                return;
            }
            this.btn.disabled = true;
            this.setStatus('Requesting microphone access...');
            let stream;
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
                });
            } catch (err) {
                this.btn.disabled = false;
                const denied = err && (err.name === 'NotAllowedError' || err.name === 'SecurityError');
                this.setStatus(denied ? 'Microphone access was denied. Allow it in the browser\'s site settings to listen live.'
                    : `Could not open the microphone: ${window.DopplerUtil.escapeHtml(err.message || err.name)}`, true);
                return;
            }
            try {
                await this.setupAudio(stream);
            } catch (err) {
                stream.getTracks().forEach(t => t.stop());
                this.btn.disabled = false;
                this.setStatus(`Could not start audio capture: ${window.DopplerUtil.escapeHtml(err.message || String(err))}`, true);
                return;
            }

            const LC = window.LiveCore;
            this.stream = stream;
            this.ring = new LC.RingBuffer(HOLD_S * RATE);
            this.clock = new LC.SampleClock(RATE);
            this.resampler = new LC.Resampler(this.ctx.sampleRate, RATE);
            this.tracker = new LC.FlyoverTracker();
            this.adsb = new LC.AdsbTrackStore();
            this.history = [];
            this.lastResult = null;
            this.baseArea = { lat: area.lat, lon: area.lon, radius_km: area.radius_km, refined: false };
            this.area = Object.assign({}, this.baseArea);
            this.startedAt = Date.now() / 1000;
            this.lastAnalysis = this.startedAt;
            this.lastPoll = 0;
            this.levelDb = -120;
            this.analyzing = false;
            this.polling = false;
            this.geoRunning = false;
            this.pendingGeo = false;
            this.pollFailures = 0;
            this.active = true;

            stream.getAudioTracks().forEach(t => t.addEventListener('ended', () => this.stop('The microphone was disconnected.')));
            try { this.wakeLock = navigator.wakeLock ? await navigator.wakeLock.request('screen') : null; } catch (e) { this.wakeLock = null; }

            window.DopplerApp.enterLiveMode();
            window.DopplerApp.resetResults();
            window.mapView.clearLiveHistory();
            window.mapView.displayFlightTrajectories([]);
            window.mapView.clearAmbiguity();
            if (window.audioAnalyzer.playheadLine) window.audioAnalyzer.playheadLine.style.display = 'none';
            window.mapView.updateSearchRadiusCircle(area.lat, area.lon, area.radius_km);
            const maxSel = $('maxFreqSelect');
            if (maxSel && maxSel.value !== String(FMAX_HZ)) { maxSel.value = String(FMAX_HZ); maxSel.dispatchEvent(new Event('change')); }

            this.backend = await window.apiClient.checkBackend();
            this.btn.disabled = false;
            this.renderHistory();
            this.setStatus(this.backend
                ? 'Listening. Waiting for an aircraft to fly past; each complete flyover is matched against live ADS-B tracks collected during this session.'
                : 'Listening. <strong>Static (GitHub Pages) build: no live ADS-B access.</strong> Flyovers are still detected and time-stamped; '
                  + 'import ADS-B tracks covering this session, or run the server, to turn them into a location.', !this.backend);
            if (this.backend) this.pollAdsb();
            this.updateUi();
        }

        async setupAudio(stream) {
            const AC = window.AudioContext || window.webkitAudioContext;
            this.ctx = new AC();
            if (this.ctx.state === 'suspended') await this.ctx.resume();
            const src = this.ctx.createMediaStreamSource(stream);
            // Anti-alias before resampling to 8 kHz: 3 cascaded 2nd-order low-passes (~-48 dB at 6 kHz).
            let node = src;
            for (let i = 0; i < 3; i++) {
                const f = this.ctx.createBiquadFilter();
                f.type = 'lowpass';
                f.frequency.value = LOWPASS_HZ;
                f.Q.value = Math.SQRT1_2;
                node.connect(f);
                node = f;
            }
            const mute = this.ctx.createGain();
            mute.gain.value = 0; // the graph must reach the destination to be processed; keep it silent
            mute.connect(this.ctx.destination);
            this.usingWorklet = false;
            if (this.ctx.audioWorklet && typeof AudioWorkletNode !== 'undefined') {
                try {
                    await this.ctx.audioWorklet.addModule(new URL('js/live_capture_worklet.js', document.baseURI).href);
                    this.capture = new AudioWorkletNode(this.ctx, 'live-capture', {
                        numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1, channelCountMode: 'explicit',
                    });
                    this.capture.port.onmessage = (e) => this.onChunk(e.data.frame, e.data.data);
                    this.usingWorklet = true;
                } catch (e) {
                    console.warn('AudioWorklet unavailable, using ScriptProcessor:', e);
                }
            }
            if (!this.usingWorklet) {
                const sp = this.ctx.createScriptProcessor(4096, 1, 1);
                let frame = 0;
                sp.onaudioprocess = (e) => {
                    const d = e.inputBuffer.getChannelData(0).slice();
                    this.onChunk(frame, d);
                    frame += d.length;
                };
                this.capture = sp;
            }
            node.connect(this.capture);
            this.capture.connect(mute);
            this.sources = { src, mute };
        }

        stop(message = 'Live listening stopped.') {
            if (!this.active) return;
            this.active = false;
            try { if (this.capture) this.capture.disconnect(); } catch (e) { /* ignore */ }
            if (this.capture && this.capture.port) this.capture.port.onmessage = null;
            if (this.stream) this.stream.getTracks().forEach(t => t.stop());
            if (this.ctx) this.ctx.close().catch(() => {});
            if (this.wakeLock) this.wakeLock.release().catch(() => {});
            this.capture = null;
            this.ctx = null;
            this.stream = null;
            this.wakeLock = null;
            this.stoppedAt = Date.now() / 1000;
            this.setStatus(`${window.DopplerUtil.escapeHtml(message)} ${this.tracker ? `${this.tracker.flyovers.length} flyover(s) were detected; ` : ''}`
                + 'use <strong>Export live data</strong> to keep them.');
            this.updateUi();
        }

        // ------------------------------------------------------------- audio path
        chunkEpoch(frame, n) {
            // AudioContext clock -> wall clock via getOutputTimestamp (tracks audio/system clock drift).
            if (this.usingWorklet && this.ctx && this.ctx.getOutputTimestamp) {
                const ts = this.ctx.getOutputTimestamp();
                if (ts && ts.performanceTime > 0) {
                    return (performance.timeOrigin + ts.performanceTime) / 1000 + (frame / this.ctx.sampleRate - ts.contextTime);
                }
            }
            return Date.now() / 1000 - n / (this.ctx ? this.ctx.sampleRate : 48000);
        }

        onChunk(frame, data) {
            if (!this.active || !data || !data.length) return;
            const epoch = this.chunkEpoch(frame, data.length);
            const out = this.resampler.process(data);
            this.clock.addChunk(this.ring.total, epoch);
            this.ring.push(out);
            this.clock.prune(this.ring.oldest);
            let ss = 0;
            for (let i = 0; i < data.length; i++) ss += data[i] * data[i];
            const db = 10 * Math.log10(Math.max(ss / data.length, 1e-12));
            this.levelDb = Math.max(db, this.levelDb - 1.5);

            const now = Date.now() / 1000;
            if (this.backend && !this.polling && now - this.lastPoll >= POLL_EVERY_S) this.pollAdsb();
            const heard = this.ring.total / RATE;
            if (!this.analyzing && heard >= FIRST_ANALYSIS_S && now - this.lastAnalysis >= ANALYZE_EVERY_S) this.analyze();
        }

        // --------------------------------------------------------------- analysis
        async analyze() {
            this.analyzing = true;
            this.lastAnalysis = Date.now() / 1000;
            try {
                const k1 = this.ring.total, k0 = Math.max(this.ring.oldest, k1 - WINDOW_S * RATE);
                const { start, samples } = this.ring.read(k0, k1);
                const dur = samples.length / RATE;
                const keep = samples.slice();
                const settings = { fmax_hz: FMAX_HZ };
                let res;
                try {
                    res = await window.apiClient.runInWorker('discover', { x: samples, sr: RATE, settings }, null, [samples.buffer]);
                } catch (e) {
                    if (!e || !e.workerUnavailable) throw e;
                    res = window.DopplerAnalysis.RidgeDiscovery.discoverSignal(keep.slice(), RATE, settings);
                }
                if (!this.active) return;
                if ($('liveSpectroCheck').checked) this.renderWindow(keep, res);
                const { added, updated } = this.tracker.ingest(res, start, RATE, (k) => this.clock.toEpoch(k), dur);
                if (added.length) {
                    const names = added.map(f => `${f.id} (${new Date(f.epoch_start * 1000).toISOString().slice(11, 19)} UTC)`).join(', ');
                    window.DopplerUtil.toast(`Flyover detected: ${names}`, 'ok', 5000);
                }
                if (added.length || updated.length) this.scheduleGeolocate();
            } catch (err) {
                console.warn('Live analysis failed:', err);
                this.setStatus(`Ridge analysis failed: ${window.DopplerUtil.escapeHtml(err.message || String(err))}`, true);
            } finally {
                this.analyzing = false;
            }
        }

        renderWindow(samples, res) {
            const aa = window.audioAnalyzer;
            let buf = null;
            try {
                aa.initContext();
                buf = aa.audioCtx.createBuffer(1, samples.length, RATE);
                buf.copyToChannel(samples, 0);
            } catch (e) {
                // Some browsers reject 8 kHz buffers; the spectrogram only needs these fields.
                buf = { sampleRate: RATE, length: samples.length, duration: samples.length / RATE, numberOfChannels: 1,
                    getChannelData: () => samples };
            }
            if (aa.audioElement && aa.audioElement.getAttribute('src')) {
                aa.audioElement.pause();
                aa.audioElement.removeAttribute('src'); // no playback of the live window
                aa.audioElement.load();
            }
            aa.audioBuffer = buf;
            aa.processAudioBuffer();
            $('audioFileName').textContent = `Live microphone (last ${Math.round(samples.length / RATE)} s)`;
            if (res && res.ridges && res.ridges.length) window.tracer.loadDiscoveredRidges(res);
            const dop = ((res && res.ridges) || []).filter(r => r.doppler_like).length;
            const ds = $('discoveryStatus');
            ds.classList.remove('warn');
            ds.innerHTML = `Live window: ${dop} Doppler-like ridge(s) in view. A flyover is committed once it has fully passed `
                + '(about 10 s after the tone fades). <em>Ridges are acoustic proposals, not verified aircraft.</em>';
        }

        // ------------------------------------------------------------------ ADS-B
        async pollAdsb() {
            this.polling = true;
            this.lastPoll = Date.now() / 1000;
            try {
                const resp = await window.apiClient.searchFlights({
                    center_lat: this.area.lat, center_lon: this.area.lon,
                    radius_km: Math.max(this.baseArea.radius_km, 10), demo: false,
                });
                if (!this.active) return;
                this.adsb.ingest(resp);
                const now = Date.now() / 1000;
                this.adsb.prune(now);
                // Until there is a result to show, draw what is being collected (last 10 minutes).
                if (!this.history.length && !this.geoRunning) window.mapView.displayFlightTrajectories(this.adsb.trajectories(now - 600, now, 0));
                this.pollFailures = resp && resp.status === 'ok' ? 0 : this.pollFailures + 1;
                this.lastPollNote = resp && resp.status !== 'ok' ? ((resp.notes || []).slice(-1)[0] || 'no live ADS-B data') : null;
            } catch (err) {
                this.pollFailures++;
                this.lastPollNote = err.message || String(err);
            } finally {
                this.polling = false;
            }
        }

        // ------------------------------------------------------------ geolocation
        scheduleGeolocate() {
            this.pendingGeo = true;
            if (!this.geoRunning) this.runGeolocate();
        }

        async runGeolocate() {
            this.geoRunning = true;
            try {
                while (this.pendingGeo) {
                    this.pendingGeo = false;
                    await this.geolocateOnce();
                }
            } finally {
                this.geoRunning = false;
            }
        }

        async geolocateOnce() {
            const { escapeHtml } = window.DopplerUtil;
            const { origin, events, flyovers } = this.tracker.events(MAX_FLYOVERS_PER_MATCH);
            if (!events.length) return;
            const imported = window.DopplerApp.getImportedTracks();
            const tEnd = Math.max(...flyovers.map(f => f.epoch_end));
            const trajectories = imported.length ? imported : this.adsb.trajectories(origin, tEnd);
            if (!trajectories.length) {
                this.setStatus(`${flyovers.length} flyover(s) detected, but no ADS-B track covers ${flyovers.length > 1 ? 'them' : 'it'} yet. `
                    + (this.backend ? 'Live ADS-B is being collected; the match runs again with the next flyover. '
                        + 'If this persists, check that the search centre is roughly right (within ~10 km).'
                        : 'Import tracks for this session to locate it.'), true);
                return;
            }
            const area = this.area;
            const payload = Object.assign(window.DopplerApp.basePayload(), {
                center_lat: area.lat, center_lon: area.lon, radius_km: area.radius_km,
                events, trajectories, recording_start_epoch: origin,
                clock_tolerance_s: window.DopplerApp.clockToleranceS(), scenario_id: null, demo: false,
            });
            this.setStatus(`Matching ${flyovers.length} flyover(s) against ${trajectories.length} ADS-B track(s) `
                + `within ${area.radius_km} km of ${area.lat.toFixed(4)}, ${area.lon.toFixed(4)}...`);
            let result;
            try {
                result = await window.apiClient.geolocateObserver(payload, (stage, frac) => {
                    this.setStatus(`Matching ${flyovers.length} flyover(s) (${escapeHtml(stage)}): ${Math.round(frac * 100)}%`);
                });
            } catch (err) {
                this.setStatus(`Matching failed: ${escapeHtml(err.message || String(err))}`, true);
                return;
            }
            this.lastResult = result;
            window.DopplerApp.displayGeolocationResults(result, { quiet: true });
            const decision = result.decision || {};
            if (result.status === 'success' && result.best_location) {
                const b = result.best_location;
                this.history.push({
                    at: Date.now() / 1000, lat: b.lat, lon: b.lon, supported: decision.status === 'supported',
                    reason: decision.reason_code || null, uncertainty_m: Number.isFinite(b.uncertainty_radius_m) ? b.uncertainty_radius_m : null,
                    n_flyovers: flyovers.length, radius_km: area.radius_km,
                    matched: (result.matched_flights || []).map(f => f.callsign || f.flight_id),
                });
                window.mapView.displayLiveHistory(this.history);
                this.renderHistory();
            }
            const next = window.LiveCore.refineSearchArea(result, this.area, this.baseArea);
            const changed = window.LiveCore.areaChanged(next, this.area);
            this.area = next;
            window.mapView.updateSearchRadiusCircle(next.lat, next.lon, next.radius_km);
            const supported = decision.status === 'supported';
            let msg = supported
                ? `<strong>Supported</strong> from ${flyovers.length} flyover(s). Still listening: each new flyover refines it.`
                : `<strong>Inconclusive</strong> (${escapeHtml(decision.reason_code || result.status)}) from ${flyovers.length} flyover(s). `
                  + (flyovers.length < 2 ? 'One flyover cannot fix a position (mirror ambiguity); waiting for another from a different direction.'
                      : 'Still listening for more flyovers.');
            if (changed && next.refined) {
                msg += ` Narrowing the search to ${next.radius_km} km around the estimate and re-matching...`;
                this.pendingGeo = true;
            }
            this.setStatus(msg, !supported);
        }

        // ------------------------------------------------------------------- UI
        renderHistory() {
            if (!this.historyEl) return;
            const { escapeHtml, fmtNum } = window.DopplerUtil;
            this.historyEl.innerHTML = this.history.slice().reverse().map((h, i) => `
                <li class="${h.supported ? 'ok' : 'tentative'}">
                    <span class="lh-n">#${this.history.length - i}</span>
                    <span class="lh-pos">${fmtNum(h.lat, 5)}, ${fmtNum(h.lon, 5)}</span>
                    <span>${h.supported ? 'supported' : 'inconclusive'}${h.uncertainty_m != null ? ` &plusmn;${fmtNum(h.uncertainty_m)} m` : ''}</span>
                    <span>${h.n_flyovers} flyover(s) &middot; ${escapeHtml(h.matched.join(' + ') || '--')}</span>
                </li>`).join('');
        }

        updateUi() {
            if (!this.btn) return;
            const { fmtNum } = window.DopplerUtil;
            this.btn.classList.toggle('listening', this.active);
            this.btn.innerHTML = this.active ? '&#9632; Stop listening' : '&#127908; Start live listening';
            $('liveCard').classList.toggle('active', this.active);
            const el = this.active && this.startedAt ? Date.now() / 1000 - this.startedAt : 0;
            $('liveElapsed').textContent = this.active
                ? `${String(Math.floor(el / 60)).padStart(2, '0')}:${String(Math.floor(el % 60)).padStart(2, '0')}` : 'off';
            const lvl = this.active ? Math.max(0, Math.min(1, (this.levelDb + 90) / 80)) : 0;
            $('liveLevelBar').style.width = `${Math.round(lvl * 100)}%`;
            $('liveFlyovers').textContent = this.tracker ? String(this.tracker.flyovers.length) : '0';
            let aircraft = '--';
            if (this.adsb && this.active) {
                aircraft = this.backend ? `${this.adsb.size}${this.pollFailures > 2 ? ' (ADS-B unavailable)' : ''}` : 'no access';
                if (this.pollFailures > 2 && this.lastPollNote) $('liveAircraft').title = this.lastPollNote;
            } else if (this.adsb) aircraft = String(this.adsb.size);
            $('liveAircraft').textContent = aircraft;
            const h = this.history[this.history.length - 1];
            $('liveEstimate').textContent = h
                ? `${fmtNum(h.lat, 5)}, ${fmtNum(h.lon, 5)}${h.supported && h.uncertainty_m != null ? ` ±${fmtNum(h.uncertainty_m)} m` : ' (tentative)'}`
                : '--';
        }

        exportData() {
            if (!this.tracker) { window.DopplerUtil.toast('Start live listening first.', 'warn'); return; }
            const data = {
                kind: 'doppler-live-session', version: 1,
                started_epoch: this.startedAt, stopped_epoch: this.active ? null : this.stoppedAt,
                search_area: { base: this.baseArea, current: this.area },
                clock: { analysis_rate_hz: RATE, discontinuities: this.clock ? this.clock.discontinuities : 0,
                    note: 'Flyover times are UTC epochs from the browser clock.' },
                flyovers: this.tracker.flyovers,
                adsb_tracks: this.adsb ? this.adsb.toJSON() : [],
                estimates: this.history,
                last_result: this.lastResult,
            };
            window.DopplerUtil.downloadFile(`doppler-live-${new Date().toISOString().replace(/[:.]/g, '-')}.json`, data);
        }
    }

    window.liveListener = new LiveListener();
})();
