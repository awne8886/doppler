/**
 * Small shared helpers (loaded first).
 *
 * escapeHtml must wrap every value that originates outside this app
 * (ADS-B callsigns, aircraft types, uploaded track names, server messages)
 * before it is interpolated into innerHTML or a Leaflet popup.
 */
(function (root) {
    const HTML_ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;', '`': '&#96;' };

    function escapeHtml(value) {
        if (value === null || value === undefined) return '';
        return String(value).replace(/[&<>"'`]/g, ch => HTML_ESCAPES[ch]);
    }

    /** Format a number for HTML output; non-finite / non-numeric values become a placeholder. */
    function fmtNum(value, digits = 0, fallback = '--') {
        const n = typeof value === 'number' ? value : parseFloat(value);
        return Number.isFinite(n) ? n.toFixed(digits) : fallback;
    }

    /**
     * Recording start from Broadcast-WAV metadata (bext / iXML), mirroring
     * app/audio/metadata.py. BWF times are the recorder's local clock with no
     * time zone, so the epoch assumes UTC and must be confirmed by the user.
     */
    function readWavStart(arrayBuffer) {
        const dv = new DataView(arrayBuffer);
        const tag = (o) => String.fromCharCode(dv.getUint8(o), dv.getUint8(o + 1), dv.getUint8(o + 2), dv.getUint8(o + 3));
        if (dv.byteLength < 12 || !['RIFF', 'RF64'].includes(tag(0)) || tag(8) !== 'WAVE') return null;
        let off = 12, sr = null, bext = null, ixml = null;
        const ascii = (o, n) => { let s = ''; for (let i = 0; i < n; i++) { const c = dv.getUint8(o + i); if (c) s += String.fromCharCode(c); } return s.trim(); };
        while (off + 8 <= dv.byteLength) {
            const id = tag(off), size = dv.getUint32(off + 4, true), body = off + 8;
            if (body + size > dv.byteLength && id !== 'data') break;
            if (id === 'fmt ' && size >= 8) sr = dv.getUint32(body + 4, true);
            else if (id === 'bext' && size >= 346) {
                bext = { date: ascii(body + 320, 10), time: ascii(body + 330, 8),
                    ref: dv.getUint32(body + 338, true) + dv.getUint32(body + 342, true) * 4294967296 };
            } else if (id === 'iXML') {
                const txt = new TextDecoder().decode(new Uint8Array(arrayBuffer, body, size));
                const get = (k) => { const m = new RegExp(`<${k}>\\s*([^<]*)\\s*</${k}>`, 'i').exec(txt); return m ? m[1].trim() : ''; };
                const lo = get('BWF_TIME_REFERENCE_LOW') || get('TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_LO');
                const hi = get('BWF_TIME_REFERENCE_HIGH') || get('TIMESTAMP_SAMPLES_SINCE_MIDNIGHT_HI') || '0';
                ixml = { date: get('BWF_ORIGINATION_DATE'), time: get('BWF_ORIGINATION_TIME'),
                    ref: /^\d+$/.test(lo) ? Number(lo) + Number(hi) * 4294967296 : 0,
                    sr: Number(get('TIMESTAMP_SAMPLE_RATE') || get('FILE_SAMPLE_RATE')) || null };
            }
            off = body + size + (size % 2);
        }
        for (const [source, info] of [['bwf_bext', bext], ['ixml', ixml]]) {
            if (!info) continue;
            const date = info.date.replace(/[:/.]/g, '-');
            if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) continue;
            const rate = info.sr || sr;
            let epoch = null, field = null;
            if (info.ref && rate) { epoch = Date.parse(`${date}T00:00:00Z`) / 1000 + info.ref / rate; field = 'time_reference'; }
            else if (/^\d{2}[:.-]\d{2}[:.-]\d{2}$/.test(info.time)) { epoch = Date.parse(`${date}T${info.time.replace(/[.-]/g, ':')}Z`) / 1000; field = 'origination_time'; }
            if (epoch == null || isNaN(epoch)) continue;
            return {
                start_epoch: epoch, start_local_iso: new Date(epoch * 1000).toISOString().replace('Z', ''), source, field,
                timezone_known: false, requires_confirmation: true,
                notes: ['Broadcast-WAV times are the recorder\'s local clock with no time zone; the epoch assumes UTC. Adjust for the recorder\'s time zone and clock error before trusting it.'],
            };
        }
        return null;
    }

    /**
     * Non-blocking notification (replaces alert()). kind: info | ok | warn | error.
     * The message is set as text, never as HTML.
     */
    function toast(message, kind = 'info', timeoutMs = 7000) {
        if (typeof document === 'undefined') return null;
        let host = document.getElementById('toastHost');
        if (!host) {
            host = document.createElement('div');
            host.id = 'toastHost';
            host.className = 'toast-host';
            host.setAttribute('aria-live', 'polite');
            document.body.appendChild(host);
        }
        const el = document.createElement('div');
        el.className = `toast ${kind}`;
        el.setAttribute('role', kind === 'error' || kind === 'warn' ? 'alert' : 'status');
        const msg = document.createElement('span');
        msg.className = 'toast-msg';
        msg.textContent = String(message);
        const close = document.createElement('button');
        close.type = 'button';
        close.setAttribute('aria-label', 'Dismiss');
        close.textContent = '\u00d7';
        close.addEventListener('click', () => el.remove());
        el.append(msg, close);
        host.appendChild(el);
        if (timeoutMs > 0) setTimeout(() => el.remove(), timeoutMs);
        return el;
    }

    /** Offer a JS object / string as a file download. */
    function downloadFile(filename, content, mime = 'application/json') {
        const blob = new Blob([typeof content === 'string' ? content : JSON.stringify(content, null, 2)], { type: mime });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }

    root.DopplerUtil = Object.assign(root.DopplerUtil || {}, { escapeHtml, fmtNum, readWavStart, toast, downloadFile });
})(typeof window !== 'undefined' ? window : globalThis);
