/**
 * High-Resolution Audio Analyzer & Spectrogram Engine
 * Uses Web Audio API and windowed STFT to render interactive spectrograms.
 */
class AudioAnalyzer {
    constructor() {
        this.audioCtx = null;
        this.audioBuffer = null;
        this.spectrogramCanvas = document.getElementById('spectrogramCanvas');
        this.ctx = this.spectrogramCanvas.getContext('2d');
        this.yAxisRuler = document.getElementById('yAxisRuler');
        this.xAxisRuler = document.getElementById('xAxisRuler');
        this.audioElement = document.getElementById('audioElement');
        this.playheadLine = document.getElementById('playheadLine');
        this.audioTimeDisplay = document.getElementById('audioTimeDisplay');

        this.fftSize = 2048;
        this.hopSize = 512;
        this.maxFreq = 2000; // Hz
        this.colormap = 'turbo';

        this.spectralData = null; // 2D array: [timeSliceIndex][freqBinIndex]
        this.sampleRate = 44100;
        this.duration = 0.0;
        this.timeSlices = 0;
        this.freqBins = 0;

        this.colorMaps = this.generateColormaps();
        this.initEventListeners();
    }

    initContext() {
        if (!this.audioCtx) {
            const AudioContext = window.AudioContext || window.webkitAudioContext;
            this.audioCtx = new AudioContext();
        }
        if (this.audioCtx.state === 'suspended') {
            this.audioCtx.resume();
        }
    }

    initEventListeners() {
        this.audioElement.addEventListener('timeupdate', () => this.updatePlayhead());
        this.audioElement.addEventListener('ended', () => {
            this.playheadLine.style.display = 'none';
            document.getElementById('playIcon').innerHTML = '&#9658;';
        });

        document.getElementById('colormapSelect').addEventListener('change', (e) => {
            this.colormap = e.target.value;
            if (this.spectralData) this.renderSpectrogram();
        });

        document.getElementById('maxFreqSelect').addEventListener('change', (e) => {
            this.maxFreq = parseFloat(e.target.value);
            if (this.spectralData) {
                this.renderSpectrogram();
                this.renderRulers();
            }
        });
    }

    async loadAudioFromUrl(url, filename = 'sample.wav') {
        this.initContext();
        document.getElementById('audioFileName').textContent = filename;
        this.audioElement.src = url;

        const resp = await fetch(url);
        const arrayBuffer = await resp.arrayBuffer();
        this.audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);
        this.processAudioBuffer();
    }

    async loadAudioFromFile(file) {
        this.initContext();
        document.getElementById('audioFileName').textContent = file.name;
        this.audioElement.src = URL.createObjectURL(file);

        const arrayBuffer = await file.arrayBuffer();
        this.audioBuffer = await this.audioCtx.decodeAudioData(arrayBuffer);
        this.processAudioBuffer();
    }

    processAudioBuffer() {
        const pcmData = this.audioBuffer.getChannelData(0); // Mono or left channel
        this.sampleRate = this.audioBuffer.sampleRate;
        this.duration = this.audioBuffer.duration;

        this.audioTimeDisplay.textContent = `00:00 / ${this.formatTime(this.duration)}`;

        // Compute STFT
        this.computeSTFT(pcmData);
        this.renderSpectrogram();
        this.renderRulers();

        // Notify tracer that new audio is ready
        if (window.tracer) {
            window.tracer.onAudioLoaded(this.duration, this.maxFreq, this.spectralData, this.sampleRate, this.fftSize, this.hopSize);
        }
    }

    computeSTFT(pcmData) {
        const N = this.fftSize;
        const hop = this.hopSize;
        const totalSamples = pcmData.length;
        this.timeSlices = Math.floor((totalSamples - N) / hop);
        this.freqBins = N / 2;

        this.spectralData = [];

        // Precompute Hanning window
        const window = new Float32Array(N);
        for (let i = 0; i < N; i++) {
            window[i] = 0.5 * (1.0 - Math.cos((2.0 * Math.PI * i) / (N - 1)));
        }

        const real = new Float32Array(N);
        const imag = new Float32Array(N);

        let maxDb = -Infinity;
        let minDb = Infinity;

        // Process frames
        for (let slice = 0; slice < this.timeSlices; slice++) {
            const offset = slice * hop;
            for (let i = 0; i < N; i++) {
                real[i] = pcmData[offset + i] * window[i];
                imag[i] = 0.0;
            }

            this.fftRadix2(real, imag);

            const sliceMagnitudes = new Float32Array(this.freqBins);
            for (let k = 0; k < this.freqBins; k++) {
                const mag = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
                const db = 20.0 * Math.log10(Math.max(mag, 1e-6));
                sliceMagnitudes[k] = db;
                if (db > maxDb) maxDb = db;
                if (db < minDb) minDb = db;
            }
            this.spectralData.push(sliceMagnitudes);
        }

        // Normalize dB values to [0.0, 1.0] with dynamic range floor
        const dynamicRange = 65.0; // dB
        const floorDb = maxDb - dynamicRange;

        for (let s = 0; s < this.timeSlices; s++) {
            const slice = this.spectralData[s];
            for (let k = 0; k < this.freqBins; k++) {
                slice[k] = Math.max(0.0, Math.min(1.0, (slice[k] - floorDb) / dynamicRange));
            }
        }
    }

    // In-place Radix-2 Cooley-Tukey FFT
    fftRadix2(real, imag) {
        const n = real.length;
        // Bit reversal
        let j = 0;
        for (let i = 0; i < n - 1; i++) {
            if (i < j) {
                const tempR = real[i]; real[i] = real[j]; real[j] = tempR;
                const tempI = imag[i]; imag[i] = imag[j]; imag[j] = tempI;
            }
            let k = n >> 1;
            while (k <= j) {
                j -= k;
                k >>= 1;
            }
            j += k;
        }

        // Butterfly calculations
        for (let len = 2; len <= n; len <<= 1) {
            const halfLen = len >> 1;
            const angle = (-2.0 * Math.PI) / len;
            const wStepR = Math.cos(angle);
            const wStepI = Math.sin(angle);

            for (let i = 0; i < n; i += len) {
                let wR = 1.0;
                let wI = 0.0;
                for (let k = 0; k < halfLen; k++) {
                    const uR = real[i + k];
                    const uI = imag[i + k];
                    const vR = real[i + k + halfLen] * wR - imag[i + k + halfLen] * wI;
                    const vI = real[i + k + halfLen] * wI + imag[i + k + halfLen] * wR;

                    real[i + k] = uR + vR;
                    imag[i + k] = uI + vI;
                    real[i + k + halfLen] = uR - vR;
                    imag[i + k + halfLen] = uI - vI;

                    const nextWR = wR * wStepR - wI * wStepI;
                    wI = wR * wStepI + wI * wStepR;
                    wR = nextWR;
                }
            }
        }
    }

    renderSpectrogram() {
        if (!this.spectralData || this.spectralData.length === 0) return;

        const width = this.spectrogramCanvas.width;
        const height = this.spectrogramCanvas.height;
        const imgData = this.ctx.createImageData(width, height);
        const data = imgData.data;

        const lut = this.colorMaps[this.colormap] || this.colorMaps.turbo;
        const maxBin = Math.min(
            this.freqBins,
            Math.floor((this.maxFreq / (this.sampleRate / 2)) * this.freqBins)
        );

        for (let x = 0; x < width; x++) {
            const sliceIdx = Math.floor((x / width) * this.timeSlices);
            const slice = this.spectralData[sliceIdx] || this.spectralData[0];

            for (let y = 0; y < height; y++) {
                // High frequency at top (y=0), 0 Hz at bottom (y=height-1)
                const binFrac = (height - 1 - y) / height;
                const binIdx = Math.min(maxBin - 1, Math.floor(binFrac * maxBin));
                const val = slice[binIdx];
                const lutIdx = Math.floor(val * 255.0) * 3;

                const pixelIdx = (y * width + x) * 4;
                data[pixelIdx] = lut[lutIdx];         // R
                data[pixelIdx + 1] = lut[lutIdx + 1]; // G
                data[pixelIdx + 2] = lut[lutIdx + 2]; // B
                data[pixelIdx + 3] = 255;             // Alpha
            }
        }

        this.ctx.putImageData(imgData, 0, 0);
    }

    renderRulers() {
        // Y-Axis (Frequency)
        this.yAxisRuler.innerHTML = '';
        const freqSteps = 5;
        for (let i = 0; i <= freqSteps; i++) {
            const frac = i / freqSteps;
            const freqVal = Math.round((1.0 - frac) * this.maxFreq);
            const tick = document.createElement('div');
            tick.className = 'y-tick';
            tick.style.top = `${frac * 100}%`;
            tick.textContent = `${freqVal}Hz`;
            this.yAxisRuler.appendChild(tick);
        }

        // X-Axis (Time)
        this.xAxisRuler.innerHTML = '';
        const timeSteps = 6;
        for (let i = 0; i <= timeSteps; i++) {
            const frac = i / timeSteps;
            const timeVal = ((frac * this.duration)).toFixed(1);
            const tick = document.createElement('div');
            tick.className = 'x-tick';
            tick.style.left = `${frac * 100}%`;
            tick.textContent = `${timeVal}s`;
            this.xAxisRuler.appendChild(tick);
        }
    }

    updatePlayhead() {
        if (!this.duration || this.duration === 0) return;
        const curTime = this.audioElement.currentTime;
        const frac = Math.min(1.0, curTime / this.duration);
        const width = this.spectrogramCanvas.parentElement.clientWidth;
        const xPos = frac * width;

        this.playheadLine.style.display = 'block';
        this.playheadLine.style.left = `${xPos}px`;
        this.audioTimeDisplay.textContent = `${this.formatTime(curTime)} / ${this.formatTime(this.duration)}`;
    }

    formatTime(sec) {
        const m = Math.floor(sec / 60);
        const s = Math.floor(sec % 60);
        return `${m < 10 ? '0' : ''}${m}:${s < 10 ? '0' : ''}${s}`;
    }

    generateColormaps() {
        const makeLut = (colors) => {
            const lut = new Uint8Array(256 * 3);
            for (let i = 0; i < 256; i++) {
                const t = i / 255.0;
                let c1 = colors[0], c2 = colors[colors.length - 1];
                let segmentT = t;

                for (let k = 0; k < colors.length - 1; k++) {
                    if (t >= colors[k].pos && t <= colors[k + 1].pos) {
                        c1 = colors[k];
                        c2 = colors[k + 1];
                        segmentT = (t - c1.pos) / (c2.pos - c1.pos);
                        break;
                    }
                }
                lut[i * 3] = Math.round(c1.r + (c2.r - c1.r) * segmentT);
                lut[i * 3 + 1] = Math.round(c1.g + (c2.g - c1.g) * segmentT);
                lut[i * 3 + 2] = Math.round(c1.b + (c2.b - c1.b) * segmentT);
            }
            return lut;
        };

        return {
            turbo: makeLut([
                { pos: 0.0, r: 48, g: 18, b: 59 },
                { pos: 0.2, r: 70, g: 134, b: 251 },
                { pos: 0.4, r: 27, g: 229, b: 181 },
                { pos: 0.6, r: 164, g: 252, b: 60 },
                { pos: 0.8, r: 251, g: 153, b: 44 },
                { pos: 1.0, r: 122, g: 4, b: 2 }
            ]),
            inferno: makeLut([
                { pos: 0.0, r: 0, g: 0, b: 4 },
                { pos: 0.25, r: 87, g: 16, b: 110 },
                { pos: 0.5, r: 187, g: 55, b: 84 },
                { pos: 0.75, r: 249, g: 142, b: 9 },
                { pos: 1.0, r: 252, g: 255, b: 164 }
            ]),
            magma: makeLut([
                { pos: 0.0, r: 0, g: 0, b: 4 },
                { pos: 0.25, r: 60, g: 15, b: 110 },
                { pos: 0.5, r: 150, g: 40, b: 120 },
                { pos: 0.75, r: 230, g: 100, b: 110 },
                { pos: 1.0, r: 250, g: 250, b: 210 }
            ]),
            jet: makeLut([
                { pos: 0.0, r: 0, g: 0, b: 130 },
                { pos: 0.25, r: 0, g: 255, b: 255 },
                { pos: 0.5, r: 0, g: 255, b: 0 },
                { pos: 0.75, r: 255, g: 255, b: 0 },
                { pos: 1.0, r: 180, g: 0, b: 0 }
            ]),
        };
    }
}

window.audioAnalyzer = new AudioAnalyzer();
