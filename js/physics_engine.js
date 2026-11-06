/**
 * Client-Side Acoustic Doppler Physics Engine & Grid Scorer.
 * Enables 100% standalone, serverless operation on GitHub Pages.
 */

class ClientDopplerEngine {
    constructor(speedOfSound = 343.2, earthRadius = 6371000.0) {
        this.c = speedOfSound;
        this.R = earthRadius;
    }

    geodeticToCartesian(lat, lon, alt, refLat, refLon, refAlt = 0.0) {
        const toRad = Math.PI / 180.0;
        const latRad = lat * toRad;
        const lonRad = lon * toRad;
        const refLatRad = refLat * toRad;
        const refLonRad = refLon * toRad;

        const latMid = (latRad + refLatRad) / 2.0;
        const x = this.R * (lonRad - refLonRad) * Math.cos(latMid);
        const y = this.R * (latRad - refLatRad);
        const z = alt - refAlt;
        return { x, y, z };
    }

    cartesianToGeodetic(x, y, z, refLat, refLon, refAlt = 0.0) {
        const toDeg = 180.0 / Math.PI;
        const toRad = Math.PI / 180.0;
        const refLatRad = refLat * toRad;

        const dLat = y / this.R;
        const latRad = refLatRad + dLat;
        const latMid = (latRad + refLatRad) / 2.0;
        const dLon = x / (this.R * Math.cos(latMid));
        const lonRad = (refLon * toRad) + dLon;

        return {
            lat: latRad * toDeg,
            lon: lonRad * toDeg,
            alt: z + refAlt,
        };
    }

    computeRadialVelocity(planePos, planeVel, obsPos) {
        // r_vec = planePos - obsPos
        const rx = planePos.x - obsPos.x;
        const ry = planePos.y - obsPos.y;
        const rz = planePos.z - obsPos.z;

        const d = Math.max(1.0, Math.sqrt(rx * rx + ry * ry + rz * rz));

        // Radial velocity towards observer: - (r_vec . planeVel) / d
        const dot = rx * planeVel.vx + ry * planeVel.vy + rz * planeVel.vz;
        const vr = -dot / d;

        return { d, vr };
    }

    computeDopplerCurve(tEmit, planePositions, planeVelocities, obsPos, f0) {
        const n = tEmit.length;
        const tObs = new Float64Array(n);
        const fObs = new Float64Array(n);

        for (let i = 0; i < n; i++) {
            const { d, vr } = this.computeRadialVelocity(planePositions[i], planeVelocities[i], obsPos);
            // Clamp Mach limit for numerical safety
            const vrSafe = Math.max(-0.95 * this.c, Math.min(0.95 * this.c, vr));

            // Doppler formula: f = f0 * (c / (c - vr))
            fObs[i] = f0 * (this.c / (this.c - vrSafe));

            // Acoustic delay: t_obs = t_emit + d / c
            tObs[i] = tEmit[i] + (d / this.c);
        }

        return { tObs, fObs };
    }
}

class ClientGridScorer {
    constructor(engine = null) {
        this.engine = engine || new ClientDopplerEngine();
    }

    evaluateGrid(centerLat, centerLon, radiusKm, trajectory, tracedCurve, f0, gridRes = 21) {
        const radiusM = radiusKm * 1000.0;
        const tEmit = new Float64Array(trajectory.timestamps);
        const lats = trajectory.latitudes;
        const lons = trajectory.longitudes;
        const alts = trajectory.altitudes;
        const nTraj = tEmit.length;

        // Convert plane trajectory to Cartesian meters relative to center
        const planePos = new Array(nTraj);
        for (let i = 0; i < nTraj; i++) {
            planePos[i] = this.engine.geodeticToCartesian(lats[i], lons[i], alts[i], centerLat, centerLon, 0.0);
        }

        // Numerical gradient velocities
        const planeVel = new Array(nTraj);
        for (let i = 0; i < nTraj; i++) {
            let dt, vx, vy, vz;
            if (i === 0) {
                dt = Math.max(0.01, tEmit[1] - tEmit[0]);
                vx = (planePos[1].x - planePos[0].x) / dt;
                vy = (planePos[1].y - planePos[0].y) / dt;
                vz = (planePos[1].z - planePos[0].z) / dt;
            } else if (i === nTraj - 1) {
                dt = Math.max(0.01, tEmit[nTraj - 1] - tEmit[nTraj - 2]);
                vx = (planePos[nTraj - 1].x - planePos[nTraj - 2].x) / dt;
                vy = (planePos[nTraj - 1].y - planePos[nTraj - 2].y) / dt;
                vz = (planePos[nTraj - 1].z - planePos[nTraj - 2].z) / dt;
            } else {
                dt = Math.max(0.01, tEmit[i + 1] - tEmit[i - 1]);
                vx = (planePos[i + 1].x - planePos[i - 1].x) / dt;
                vy = (planePos[i + 1].y - planePos[i - 1].y) / dt;
                vz = (planePos[i + 1].z - planePos[i - 1].z) / dt;
            }
            planeVel[i] = { vx, vy, vz };
        }

        // User traced curve points
        const m = tracedCurve.length;
        const tracedTimes = new Float64Array(m);
        const tracedFreqs = new Float64Array(m);
        const tBase = tracedCurve[0].time;
        for (let i = 0; i < m; i++) {
            tracedTimes[i] = tracedCurve[i].time - tBase;
            tracedFreqs[i] = tracedCurve[i].freq;
        }

        // Generate 2D candidate grid
        const step = (2.0 * radiusM) / (gridRes - 1);
        const heatmap = [];
        let bestRmse = Infinity;
        let bestScore = 0.0;
        let bestLocation = null;
        let bestModeled = null;
        let bestTimeOffset = 0.0;

        for (let ix = 0; ix < gridRes; ix++) {
            const x = -radiusM + ix * step;
            for (let iy = 0; iy < gridRes; iy++) {
                const y = -radiusM + iy * step;
                const distCenter = Math.sqrt(x * x + y * y);
                if (distCenter > radiusM * 1.05) continue; // Inside search circle

                const geo = this.engine.cartesianToGeodetic(x, y, 0.0, centerLat, centerLon, 0.0);
                const obsPos = { x, y, z: 0.0 };

                // Compute theoretical Doppler curve at this candidate observer
                const { tObs, fObs } = this.engine.computeDopplerCurve(tEmit, planePos, planeVel, obsPos, f0);

                // Find CPA index in modeled curve (closest to f0)
                let minDiff = Infinity, cpaIdx = 0;
                for (let k = 0; k < nTraj; k++) {
                    const diff = Math.abs(fObs[k] - f0);
                    if (diff < minDiff) { minDiff = diff; cpaIdx = k; }
                }

                // Find CPA in user trace
                let minTraceDiff = Infinity, traceCpaIdx = 0;
                for (let k = 0; k < m; k++) {
                    const diff = Math.abs(tracedFreqs[k] - f0);
                    if (diff < minTraceDiff) { minTraceDiff = diff; traceCpaIdx = k; }
                }

                const nominalOffset = tObs[cpaIdx] - tracedTimes[traceCpaIdx];
                let candBestMse = Infinity;
                let candBestOffset = nominalOffset;
                let candBestModeled = new Float64Array(m);

                // Coarse search around nominal offset (+/- 4s)
                const minTObs = tObs[0];
                const maxTObs = tObs[nTraj - 1];

                for (let stepOff = -8; stepOff <= 8; stepOff++) {
                    const offset = nominalOffset + stepOff * 0.5;
                    let mse = 0.0;
                    let valid = true;

                    for (let k = 0; k < m; k++) {
                        const tQuery = tracedTimes[k] + offset;
                        if (tQuery < minTObs || tQuery > maxTObs) {
                            valid = false;
                            break;
                        }
                        // Linear interpolation
                        const fInterp = this.interpolate(tQuery, tObs, fObs);
                        const err = tracedFreqs[k] - fInterp;
                        mse += err * err;
                    }

                    if (valid && (mse / m) < candBestMse) {
                        candBestMse = mse / m;
                        candBestOffset = offset;
                    }
                }

                // Fine refinement (+/- 0.5s in 0.05s steps)
                if (candBestMse !== Infinity) {
                    const coarseOffset = candBestOffset;
                    for (let stepFine = -5; stepFine <= 5; stepFine++) {
                        const fineOff = coarseOffset + stepFine * 0.1;
                        let mse = 0.0;
                        let valid = true;

                        for (let k = 0; k < m; k++) {
                            const tQuery = tracedTimes[k] + fineOff;
                            if (tQuery < minTObs || tQuery > maxTObs) {
                                valid = false;
                                break;
                            }
                            const fInterp = this.interpolate(tQuery, tObs, fObs);
                            const err = tracedFreqs[k] - fInterp;
                            mse += err * err;
                        }

                        if (valid && (mse / m) < candBestMse) {
                            candBestMse = mse / m;
                            candBestOffset = fineOff;
                        }
                    }
                }

                if (candBestMse === Infinity) {
                    candBestMse = 2500.0; // High penalty
                }

                const rmse = Math.sqrt(candBestMse);
                const sigma = 18.0;
                const score = 100.0 * Math.exp(-0.5 * Math.pow(rmse / sigma, 2));

                heatmap.push({
                    lat: Math.round(geo.lat * 1e6) / 1e6,
                    lon: Math.round(geo.lon * 1e6) / 1e6,
                    score: Math.round(score * 100) / 100,
                    rmse: Math.round(rmse * 100) / 100,
                });

                if (rmse < bestRmse) {
                    bestRmse = rmse;
                    bestScore = score;
                    bestTimeOffset = candBestOffset;
                    bestLocation = {
                        lat: Math.round(geo.lat * 1e6) / 1e6,
                        lon: Math.round(geo.lon * 1e6) / 1e6,
                        score: Math.round(score * 100) / 100,
                        rmse_hz: Math.round(rmse * 100) / 100,
                        time_offset_sec: Math.round(candBestOffset * 100) / 100,
                    };

                    // Compute final best modeled curve points
                    bestModeled = new Float64Array(m);
                    for (let k = 0; k < m; k++) {
                        bestModeled[k] = this.interpolate(tracedTimes[k] + candBestOffset, tObs, fObs);
                    }
                }
            }
        }

        const curveComparison = [];
        for (let k = 0; k < m; k++) {
            curveComparison.push({
                time: Math.round(tracedCurve[k].time * 1000) / 1000,
                traced_freq: Math.round(tracedCurve[k].freq * 100) / 100,
                modeled_freq: bestModeled ? Math.round(bestModeled[k] * 100) / 100 : 0.0,
            });
        }

        return {
            flight_id: trajectory.flight_id,
            callsign: trajectory.callsign,
            aircraft_type: trajectory.aircraft_type,
            best_location: bestLocation,
            heatmap,
            curve_comparison: curveComparison,
            f0_hz: f0,
            total_candidates_evaluated: heatmap.length,
        };
    }

    interpolate(x, xArr, yArr) {
        const n = xArr.length;
        if (x <= xArr[0]) return yArr[0];
        if (x >= xArr[n - 1]) return yArr[n - 1];

        // Binary search
        let low = 0, high = n - 1;
        while (low <= high) {
            const mid = (low + high) >> 1;
            if (xArr[mid] < x) low = mid + 1;
            else high = mid - 1;
        }

        const i = Math.max(1, low);
        const x0 = xArr[i - 1], x1 = xArr[i];
        const y0 = yArr[i - 1], y1 = yArr[i];
        const frac = (x - x0) / (x1 - x0 + 1e-9);
        return y0 + frac * (y1 - y0);
    }
}

window.clientPhysics = new ClientGridScorer();
