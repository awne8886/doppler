/**
 * AudioWorklet that forwards microphone samples to the main thread in blocks,
 * tagged with the context frame of the block's first sample (so the main
 * thread can time-stamp them and notice dropped audio).
 */
/* global AudioWorkletProcessor, registerProcessor, currentFrame */
class LiveCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.block = 2048;
        this.buf = new Float32Array(this.block);
        this.fill = 0;
        this.startFrame = 0;
    }

    process(inputs) {
        const ch = inputs[0] && inputs[0][0];
        if (!ch) return true;
        for (let i = 0; i < ch.length; i++) {
            if (this.fill === 0) this.startFrame = currentFrame + i;
            this.buf[this.fill++] = ch[i];
            if (this.fill === this.block) {
                const out = this.buf;
                this.port.postMessage({ frame: this.startFrame, data: out }, [out.buffer]);
                this.buf = new Float32Array(this.block);
                this.fill = 0;
            }
        }
        return true;
    }
}

registerProcessor('live-capture', LiveCaptureProcessor);
