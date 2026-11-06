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

        this.points = []; // [{ time: float, freq: float, x: float, y: float }]
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
    }

    onAudioLoaded(duration, maxFreq, spectralData, sampleRate, fftSize, hopSize) {
        this.duration = duration;
        this.maxFreq = maxFreq;
        this.spectralData = spectralData;
        this.sampleRate = sampleRate;
        this.fftSize = fftSize;
        this.hopSize = hopSize;
        this.clearTrace();
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

        if (this.points.length === 0) return;

        // Draw connecting spline curve
        this.ctx.save();
        this.ctx.lineWidth = 3;
        this.ctx.strokeStyle = '#22d3ee';
        this.ctx.shadowColor = '#06b6d4';
        this.ctx.shadowBlur = 10;
        this.ctx.beginPath();

        if (this.points.length === 1) {
            this.ctx.arc(this.points[0].x, this.points[0].y, 2, 0, Math.PI * 2);
        } else {
            this.ctx.moveTo(this.points[0].x, this.points[0].y);
            for (let i = 0; i < this.points.length - 1; i++) {
                const xc = (this.points[i].x + this.points[i + 1].x) / 2;
                const yc = (this.points[i].y + this.points[i + 1].y) / 2;
                this.ctx.quadraticCurveTo(this.points[i].x, this.points[i].y, xc, yc);
            }
            this.ctx.lineTo(this.points[this.points.length - 1].x, this.points[this.points.length - 1].y);
        }
        this.ctx.stroke();

        // Draw points
        for (let i = 0; i < this.points.length; i++) {
            const p = this.points[i];
            this.ctx.beginPath();
            this.ctx.fillStyle = '#fff';
            this.ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
            this.ctx.fill();

            this.ctx.beginPath();
            this.ctx.strokeStyle = '#06b6d4';
            this.ctx.lineWidth = 1.5;
            this.ctx.arc(p.x, p.y, 6, 0, Math.PI * 2);
            this.ctx.stroke();
        }

        // Draw horizontal line at current f0
        const currentF0 = parseFloat(this.f0Input.value);
        if (currentF0 > 0 && currentF0 <= this.maxFreq) {
            const yF0 = (1.0 - (currentF0 / this.maxFreq)) * this.traceCanvas.height;
            this.ctx.beginPath();
            this.ctx.setLineDash([6, 6]);
            this.ctx.lineWidth = 1.5;
            this.ctx.strokeStyle = 'rgba(245, 158, 11, 0.7)'; // Amber
            this.ctx.shadowBlur = 0;
            this.ctx.moveTo(0, yF0);
            this.ctx.lineTo(this.traceCanvas.width, yF0);
            this.ctx.stroke();

            // Label f0
            this.ctx.fillStyle = '#f59e0b';
            this.ctx.font = '11px "JetBrains Mono", monospace';
            this.ctx.fillText(`f0: ${Math.round(currentF0)}Hz`, 8, yF0 - 5);
        }

        this.ctx.restore();
    }

    updateStats() {
        this.tracePointCount.textContent = this.points.length;

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
        this.points = [];
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
