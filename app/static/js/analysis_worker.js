/**
 * Web Worker for the browser analysis (ridge search and joint matching), so
 * long computations do not freeze the page. Messages: {id, type, args};
 * replies: {id, kind: 'progress'|'result'|'error', ...}.
 */
/* global importScripts, DopplerAnalysis */
self.window = self;
importScripts('util.js', 'doppler_analysis.js');

self.onmessage = async (e) => {
    const { id, type, args } = e.data;
    const progress = (stage, frac) => self.postMessage({ id, kind: 'progress', stage, frac });
    try {
        let result;
        if (type === 'geolocate') {
            result = await DopplerAnalysis.runPipeline(args.payload, args.found, progress);
        } else if (type === 'discover') {
            result = DopplerAnalysis.RidgeDiscovery.discoverSignal(args.x, args.sr, args.settings);
        } else {
            throw new Error(`Unknown worker task '${type}'.`);
        }
        self.postMessage({ id, kind: 'result', result });
    } catch (err) {
        self.postMessage({ id, kind: 'error', message: err && err.message ? err.message : String(err) });
    }
};
