/**
 * Frontend runtime configuration.
 *
 * On GitHub Pages this static file is used as-is. The FastAPI server serves
 * its own /js/config.js instead, built from the DOPPLER_CARTO_KEY environment
 * variable, so deployments can use their own key without editing code.
 * CARTO keys are visible to every visitor: restrict them by HTTP referrer in
 * the CARTO dashboard. Without a key the public CARTO basemaps are used.
 */
window.DOPPLER_CONFIG = Object.assign({
    cartoKey: 'cb1_53g1_1_fd641ece496f7c5c49a5afbe',
}, window.DOPPLER_CONFIG || {});
