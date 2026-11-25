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

    root.DopplerUtil = Object.assign(root.DopplerUtil || {}, { escapeHtml, fmtNum });
})(typeof window !== 'undefined' ? window : globalThis);
