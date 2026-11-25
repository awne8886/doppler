/**
 * Spectrogram UI Tracing Tool
 * Allows clicking, dragging, and snap-to-peak tracing of Doppler curves.
 * Extracts frequency drop, fundamental frequency f0, and exact time offsets.
 */
class DopplerTracer {
    constructor() {
        this.traceCanvas = document.getElementById('traceCanvas');
        this.ctx = this.traceCanvas.getContext('2d');
        this.crosshairHUD = document.getElementById('crosshairHUD');
        this.tracePointCount = document.getElementById('tracePointCount');
        this.f0Input = document.getElementById('f0Input');
        this.deltaFDisplay = document.getElementById('deltaFDisplay');

        // Multiple Doppler events share one recording clock. Each event:
        // { id, group_id, points: [{time, freq, x, y}], color, include, meta }
        this.events = [];
        this.active = -1;
        this.eventCounter = 0;
        this.eventList = document.getElementById('eventList');
        this.palette = ['#22d3ee', '#a3e635', '#f472b6', '#fbbf24', '#c084fc', '#fb7185', '#34d399', '#60a5fa'];
        this.duration = 0.0;
        this.maxFreq = 2000.0;
        this.spectralData = null;
        this.sampleRate = 44100;
        this.fftSize = 2048;
        this.hopSize = 512;

        this.isTracing = false;
        this.isMouseDown = false;
        this.snapToPeak = true;

        this.initEventListeners();
        this.newEvent();
    }

    get points() {
        if (this.active < 0 || !this.events[this.active]) this.newEvent();
        return this.events[this.active].points;
    }

    set points(arr) {
        if (this.active < 0 || !this.events[this.active]) this.newEvent();
        this.events[this.active].points = arr;
    }

    newEvent(opts = {}) {
        this.eventCounter++;
        const id = opts.id || `T${this.eventCounter}`;
        const ev = {
            id,
            group_id: opts.group_id || null,
            points: [],
            color: opts.color || this.palette[(this.events.length) % this.palette.length],
            include: opts.include !== false,
            meta: opts.meta || null,
        };
        this.events.push(ev);
        this.active = this.events.length - 1;
        this.renderEventList();
        return ev;
    }

    setActive(i) {
        if (i >= 0 && i < this.events.length) {
            this.active = i;
            this.renderTrace();
            this.updateStats();
            this.renderEventList();
        }
    }

    removeActive() {
        if (this.active < 0) return;
        this.events.splice(this.active, 1);
        this.active = this.events.length ? Math.min(this.active, this.events.length - 1) : -1;
        if (this.active < 0) this.newEvent();
        this.renderTrace();
        this.updateStats();
        this.renderEventList();
    }

    /** Replace all events with automatically discovered ridges. */
    loadDiscoveredRidges(discovery) {
        this.events = [];
        this.active = -1;
        const ridges = discovery.ridges || [];
        ridges.forEach((r, i) => {
            const groupIdx = Math.max(0, (discovery.harmonic_groups || []).findIndex(g => g.id === r.harmonic_group_id));
            const ev = this.newEvent({
                id: r.id,
                group_id: r.harmonic_group_id,
                color: this.palette[groupIdx % this.palette.length],
                include: !!r.doppler_like,
                meta: r,
            });
            ev.points = r.points.map(p => {
                const pos = this.timeFreqToCanvas(p.time, p.freq);
                return { time: p.time, freq: p.freq, x: pos.x, y: pos.y };
            });
        });
        if (!this.events.length) this.newEvent();
        // Select the longest included ridge for editing.
        let best = 0;
        this.events.forEach((e, i) => {
            const b = this.events[best];
            if (e.include && (!b.include || e.points.length > b.points.length)) best = i;
        });
        this.active = best;
        const fit = this.events[best].meta && this.events[best].meta.flyover_fit;
        if (fit && fit.f0_hz) this.f0Input.value = Number(fit.f0_hz).toFixed(1);
        this.renderTrace();
        this.updateStats();
        this.renderEventList();
    }

    /** Events (with >= 3 points) selected for geolocation. */
    getEvents() {
        return this.events
            .filter(e => e.include && e.points.length >= 3)
            .map(e => ({
                id: e.id,
                group_id: e.group_id || e.id,
                traced_curve: e.points.map(p => ({
                    time: Math.round(p.time * 1000) / 1000,
                    freq: Math.round(p.freq * 100) / 100,
                })),
            }));
    }

    renderEventList() {
        if (!this.eventList) return;
        this.eventList.innerHTML = '';
        this.events.forEach((e, i) => {
            const chip = document.createElement('div');
            chip.className = 'event-chip' + (i === this.active ? ' active' : '') + (e.include ? '' : ' excluded');
            chip.style.setProperty('--chip-color', e.color);
            const fit = e.meta && e.meta.flyover_fit;
            const parts = [`<strong>${e.id}</strong>`];
            if (e.group_id) parts.push(`${e.group_id}${e.meta && e.meta.harmonic_ratio ? ' &times;' + e.meta.harmonic_ratio : ''}`);
            if (fit) parts.push(`f&#8320; ${Math.round(fit.f0_hz)} Hz`, `${Math.round(fit.speed_mps)} m/s`, `CPA ${(fit.cpa_slant_m / 1000).toFixed(2)} km`);
            else parts.push(`${e.points.length} pts`);
            if (e.meta && !e.meta.doppler_like) parts.push('<em>not Doppler-like</em>');
            chip.innerHTML = `
                <input type="checkbox" ${e.include ? 'checked' : ''} title="Include in geolocation">
                <span class="chip-dot"></span>
                <span class="chip-text">${parts.join(' &middot; ')}</span>`;
            chip.querySelector('input').addEventListener('click', (ev) => {
                ev.stopPropagation();
                e.include = ev.target.checked;
                this.renderTrace();
                this.renderEventList();
            });
            chip.addEventListener('click', () => this.setActive(i));
            this.eventList.appendChild(chip);
        });
    }

    initEventListeners() {
        const canvas = this.traceCanvas;

        canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        window.addEventListener('mouseup', () => this.onMouseUp());
        canvas.addEventListener('mouseleave', () => this.onMouseLeave());

        // Buttons
        document.getElementById('snapPeakBtn').addEventListener('click', (e) => {
            this.snapToPeak = !this.snapToPeak;
            e.currentTarget.classList.toggle('active', this.snapToPeak);
        });

        document.getElementById('undoTraceBtn').addEventListener('click', () => this.undoPoint());
        document.getElementById('clearTraceBtn').addEventListener('click', () => this.clearTrace());
        document.getElementById('autoDetectF0Btn').addEventListener('click', () => this.autoDetectF0());
        const newEvBtn = document.getElementById('newEventBtn');
        if (newEvBtn) newEvBtn.addEventListener('click', () => { this.newEvent(); this.renderTrace(); this.updateStats(); });
        const delEvBtn = document.getElementById('deleteEventBtn');
        if (delEvBtn) delEvBtn.addEventListener('click', () => this.removeActive());
    }

    onAudioLoaded(duration, maxFreq, spectralData, sampleRate, fftSize, hopSize) {
        this.duration = duration;
        this.maxFreq = maxFreq;
        this.spectralData = spectralData;
        this.sampleRate = sampleRate;
        this.fftSize = fftSize;
        this.hopSize = hopSize;
        this.clearAll();
    }

    getCanvasCoordinates(e) {
        const rect = this.traceCanvas.getBoundingClientRect();
        const scaleX = this.traceCanvas.width / rect.width;
        const scaleY = this.traceCanvas.height / rect.height;

        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;
        return { x, y };
    }

    canvasToTimeFreq(x, y) {
        const width = this.traceCanvas.width;
        const height = this.traceCanvas.height;

        const time = Math.max(0.0, Math.min(this.duration, (x / width) * this.duration));
        const freq = Math.max(0.0, Math.min(this.maxFreq, (1.0 - (y / height)) * this.maxFreq));
        return { time, freq };
    }

    timeFreqToCanvas(time, freq) {
        const width = this.traceCanvas.width;
        const height = this.traceCanvas.height;

        const x = (time / this.duration) * width;
        const y = (1.0 - (freq / this.maxFreq)) * height;
        return { x, y };
    }

    onMouseMove(e) {
        if (!this.duration) return;
        const { x, y } = this.getCanvasCoordinates(e);
        const { time, freq } = this.canvasToTimeFreq(x, y);

        this.crosshairHUD.textContent = `t: ${time.toFixed(2)}s | f: ${Math.round(freq)} Hz`;
        this.crosshairHUD.style.display = 'block';

        if (this.isMouseDown) {
            this.addTracePoint(x, y);
        }
    }

    onMouseDown(e) {
        if (!this.duration) return;
        this.isMouseDown = true;
        const { x, y } = this.getCanvasCoordinates(e);
        this.addTracePoint(x, y);
    }

    onMouseUp() {
        this.isMouseDown = false;
        if (this.points.length >= 3) {
            this.autoDetectF0();
        }
    }

    onMouseLeave() {
        this.crosshairHUD.style.display = 'none';
        this.isMouseDown = false;
    }

    addTracePoint(canvasX, canvasY) {
        let { time, freq } = this.canvasToTimeFreq(canvasX, canvasY);

        // Snap to peak if enabled
        if (this.snapToPeak && this.spectralData) {
            freq = this.findLocalSpectralPeak(time, freq);
        }

        const pointPos = this.timeFreqToCanvas(time, freq);

        // Avoid adding points too close in time (< 0.2s)
        const minTimeDist = 0.15;
        const existingIdx = this.points.findIndex(p => Math.abs(p.time - time) < minTimeDist);

        if (existingIdx !== -1) {
            this.points[existingIdx] = { time, freq, x: pointPos.x, y: pointPos.y };
        } else {
            this.points.push({ time, freq, x: pointPos.x, y: pointPos.y });
        }

        // Sort by time
        this.points.sort((a, b) => a.time - b.time);
        this.renderTrace();
        this.updateStats();
    }

    findLocalSpectralPeak(time, targetFreq) {
        if (!this.spectralData || this.spectralData.length === 0) return targetFreq;

        const timeFrac = Math.max(0.0, Math.min(1.0, time / this.duration));
        const sliceIdx = Math.floor(timeFrac * (this.spectralData.length - 1));
        const slice = this.spectralData[sliceIdx];
        if (!slice) return targetFreq;

        const binHz = (this.sampleRate / 2) / slice.length;
        const targetBin = Math.round(targetFreq / binHz);
        const searchRadiusBins = Math.round(120.0 / binHz); // search within +/- 120 Hz

        let bestBin = targetBin;
        let maxVal = -Infinity;

        const minBin = Math.max(0, targetBin - searchRadiusBins);
        const maxBin = Math.min(slice.length - 1, targetBin + searchRadiusBins);

        for (let b = minBin; b <= maxBin; b++) {
            if (slice[b] > maxVal) {
                maxVal = slice[b];
                bestBin = b;
            }
        }

        // Quadratic peak interpolation around bestBin for sub-bin precision
        if (bestBin > 0 && bestBin < slice.length - 1) {
            const alpha = slice[bestBin - 1];
            const beta = slice[bestBin];
            const gamma = slice[bestBin + 1];
            const delta = (0.5 * (alpha - gamma)) / (alpha - 2.0 * beta + gamma + 1e-9);
            return (bestBin + delta) * binHz;
        }

        return bestBin * binHz;
    }

    renderTrace() {
        this.ctx.clearRect(0, 0, this.traceCanvas.width, this.traceCanvas.height);
        this.events.forEach((ev, i) => {
            if (i !== this.active) this.drawEvent(ev, false);
        });
        if (this.events[this.active]) this.drawEvent(this.events[this.active], true);
        this.drawF0Line();
    }

    drawEvent(ev, isActive) {
        const pts = ev.points;
        if (!pts.length) return;
        const ctx = this.ctx;
        ctx.save();
        ctx.globalAlpha = ev.include ? (isActive ? 1.0 : 0.75) : 0.3;
        ctx.lineWidth = isActive ? 3 : 2;
        ctx.strokeStyle = ev.color;
        ctx.shadowColor = ev.color;
        ctx.shadowBlur = isActive ? 10 : 0;
        if (!ev.include) ctx.setLineDash([4, 4]);
        ctx.beginPath();
        if (pts.length === 1) {
            ctx.arc(pts[0].x, pts[0].y, 2, 0, Math.PI * 2);
        } else {
            ctx.moveTo(pts[0].x, pts[0].y);
            for (let i = 0; i < pts.length - 1; i++) {
                const xc = (pts[i].x + pts[i + 1].x) / 2;
                const yc = (pts[i].y + pts[i + 1].y) / 2;
                ctx.quadraticCurveTo(pts[i].x, pts[i].y, xc, yc);
            }
            ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
        if (isActive) {
            // Only draw handles when sparse enough to be useful.
            const stride = Math.max(1, Math.ceil(pts.length / 40));
            for (let i = 0; i < pts.length; i += stride) {
                const p = pts[i];
                ctx.beginPath();
                ctx.fillStyle = '#fff';
                ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.beginPath();
                ctx.strokeStyle = ev.color;
                ctx.lineWidth = 1.5;
                ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2);
                ctx.stroke();
            }
        }
        ctx.fillStyle = ev.color;
        ctx.shadowBlur = 0;
        ctx.font = '11px "JetBrains Mono", monospace';
        ctx.fillText(ev.id, pts[0].x + 4, pts[0].y - 6);
        ctx.restore();
    }

    drawF0Line() {
        const currentF0 = parseFloat(this.f0Input.value);
        if (currentF0 > 0 && currentF0 <= this.maxFreq) {
            const yF0 = (1.0 - (currentF0 / this.maxFreq)) * this.traceCanvas.height;
            this.ctx.save();
            this.ctx.beginPath();
            this.ctx.setLineDash([6, 6]);
            this.ctx.lineWidth = 1.5;
            this.ctx.strokeStyle = 'rgba(245, 158, 11, 0.7)';
            this.ctx.moveTo(0, yF0);
            this.ctx.lineTo(this.traceCanvas.width, yF0);
            this.ctx.stroke();
            this.ctx.fillStyle = '#f59e0b';
            this.ctx.font = '11px "JetBrains Mono", monospace';
            this.ctx.fillText(`f0: ${Math.round(currentF0)}Hz`, 8, yF0 - 5);
            this.ctx.restore();
        }
    }

    /** Re-project all events after the frequency axis changes. */
    reproject(maxFreq) {
        this.maxFreq = maxFreq;
        for (const ev of this.events) {
            for (const p of ev.points) {
                const pos = this.timeFreqToCanvas(p.time, p.freq);
                p.x = pos.x; p.y = pos.y;
            }
        }
        this.renderTrace();
    }

    updateStats() {
        const nInc = this.events.filter(e => e.include && e.points.length >= 3).length;
        this.tracePointCount.textContent = `${this.points.length} (${nInc} event${nInc === 1 ? '' : 's'})`;

        if (this.points.length >= 2) {
            const freqs = this.points.map(p => p.freq);
            const maxF = Math.max(...freqs);
            const minF = Math.min(...freqs);
            this.deltaFDisplay.textContent = `${Math.round(maxF - minF)} Hz`;
        } else {
            this.deltaFDisplay.textContent = '-- Hz';
        }
    }

    autoDetectF0() {
        if (this.points.length < 3) return;

        // Prefer a constant-velocity flyover fit (estimates f0, speed and CPA range).
        if (this.points.length >= 5 && window.DopplerAnalysis) {
            try {
                const fit = window.DopplerAnalysis.FlyoverFit.fit(this.points.map(p => p.time), this.points.map(p => p.freq));
                if (fit.f0_hz > 0 && fit.rmse_hz < 25) {
                    this.f0Input.value = fit.f0_hz.toFixed(1);
                    const ev = this.events[this.active];
                    if (ev) {
                        ev.meta = Object.assign({}, ev.meta || { doppler_like: true }, { flyover_fit: fit });
                        this.renderEventList();
                    }
                    this.renderTrace();
                    return;
                }
            } catch (e) { /* fall back to inflection heuristic */ }
        }

        // In a Doppler S-curve, f0 is located at the inflection point
        // where the slope (df/dt) is steepest (most negative)
        let maxSlope = 0;
        let bestF0 = 0;

        for (let i = 0; i < this.points.length - 1; i++) {
            const dt = this.points[i + 1].time - this.points[i].time;
            const df = this.points[i + 1].freq - this.points[i].freq;
            if (dt > 0.05) {
                const slope = Math.abs(df / dt);
                if (slope > maxSlope) {
                    maxSlope = slope;
                    bestF0 = (this.points[i].freq + this.points[i + 1].freq) / 2.0;
                }
            }
        }

        if (bestF0 === 0) {
            // Fallback to midpoint between max and min
            const freqs = this.points.map(p => p.freq);
            bestF0 = (Math.max(...freqs) + Math.min(...freqs)) / 2.0;
        }

        this.f0Input.value = Math.round(bestF0).toFixed(1);
        this.renderTrace();
    }

    undoPoint() {
        if (this.points.length > 0) {
            this.points.pop();
            this.renderTrace();
            this.updateStats();
        }
    }

    clearTrace() {
        this.clearAll();
    }

    clearAll() {
        this.events = [];
        this.active = -1;
        this.eventCounter = 0;
        this.newEvent();
        this.renderTrace();
        this.updateStats();
    }

    getTracedCurve() {
        return this.points.map(p => ({
            time: Math.round(p.time * 1000) / 1000,
            freq: Math.round(p.freq * 100) / 100,
        }));
    }

    loadSynthesizedPoints(pointsArray, f0) {
        this.clearAll();
        this.points = [];
        for (const pt of pointsArray) {
            const pos = this.timeFreqToCanvas(pt.time, pt.freq);
            this.points.push({
                time: pt.time,
                freq: pt.freq,
                x: pos.x,
                y: pos.y,
            });
        }
        if (f0) {
            this.f0Input.value = Number(f0).toFixed(1);
        }
        this.renderTrace();
        this.updateStats();
    }
}

window.tracer = new DopplerTracer();
