/**
 * Gaussian + RGB similarity against the reference spec (Splat-Simplify.md
 * §2.1, §4).
 *
 * The closed forms used as expectations: for two isotropic Gaussians of equal
 * scale sigma separated by delta the log-determinant terms cancel exactly and
 * log B = -delta^2 / (8 sigma^2); for equal geometry the whole similarity
 * reduces to the negative squared RGB distance.
 *
 * Tolerances allow for the decode's EPS_COV diagonal floor, which perturbs those
 * forms by roughly EPS_COV/sigma^2 relative.
 */
import assert from 'node:assert';
import { describe, it } from 'node:test';

import {
    createSimTile, fillSimTile, logSimilarity, dissimilarity, blendedRadius
} from '../src/lib/decimate-voxel/similarity.js';
import { makeView, tileOf } from './helpers/voxel-splats.mjs';

describe('voxel similarity', () => {
    it('a splat is perfectly similar to itself', () => {
        const tile = tileOf([{ p: [1, 2, 3], s: 0.05, a: 0.8, rgb: [0.3, 0.6, 0.9] }]);
        const l = logSimilarity(tile, 0, 0);
        assert.ok(Math.abs(l) < 1e-12, `expected ~0, got ${l}`);
        assert.ok(dissimilarity(l) < 1e-12);
    });

    it('separation is measured relative to scale, not in world units', () => {
        // Same 0.05 gap; the tight pair should be far more dissimilar.
        const tight = tileOf([
            { p: [0, 0, 0], s: 0.05, a: 0.8 },
            { p: [0.05, 0, 0], s: 0.05, a: 0.8 }
        ]);
        const loose = tileOf([
            { p: [0, 0, 0], s: 0.5, a: 0.8 },
            { p: [0.05, 0, 0], s: 0.5, a: 0.8 }
        ]);

        // Equal isotropic scales: log B = -delta^2 / (8 sigma^2), exactly.
        assert.ok(Math.abs(logSimilarity(tight, 0, 1) - -0.125) < 1e-6);
        assert.ok(Math.abs(logSimilarity(loose, 0, 1) - -0.00125) < 1e-6);
        assert.ok(dissimilarity(logSimilarity(tight, 0, 1)) >
                  dissimilarity(logSimilarity(loose, 0, 1)));
    });

    it('covariance mismatch alone is penalized, at identical positions', () => {
        const tile = tileOf([
            { p: [0, 0, 0], s: 0.05, a: 0.8 },
            { p: [0, 0, 0], s: 0.5, a: 0.8 }
        ]);
        // 1.5 * ln(sa sb / mean) with mean = (sa^2 + sb^2) / 2.
        const expected = 1.5 * Math.log((0.05 * 0.5) / ((0.0025 + 0.25) / 2));
        assert.ok(Math.abs(logSimilarity(tile, 0, 1) - expected) < 1e-5,
            `expected ${expected}, got ${logSimilarity(tile, 0, 1)}`);
        assert.ok(logSimilarity(tile, 0, 1) < -1);
    });

    it('the colour term is the squared display-RGB distance', () => {
        const tile = tileOf([
            { p: [0, 0, 0], s: 0.05, a: 0.8, rgb: [0.8, 0.5, 0.5] },
            { p: [0, 0, 0], s: 0.05, a: 0.8, rgb: [0.2, 0.5, 0.5] }
        ]);
        assert.ok(Math.abs(logSimilarity(tile, 0, 1) - -0.36) < 1e-6);
    });

    it('colour saturates: out-of-range DC clamps to the display cube', () => {
        const tile = tileOf([
            { p: [0, 0, 0], s: 0.05, a: 0.8, rgb: [3, 0.5, 0.5] },
            { p: [0, 0, 0], s: 0.05, a: 0.8, rgb: [0.8, 0.5, 0.5] }
        ]);
        // 3.0 clamps to 1.0, so the distance is 0.2 rather than 2.2.
        assert.ok(Math.abs(logSimilarity(tile, 0, 1) - -0.04) < 1e-6);
    });

    it('blended radius is the 3-sigma radius for an isotropic splat', () => {
        const tile = tileOf([{ p: [0, 0, 0], s: 0.05, a: 0.8 }]);
        assert.ok(Math.abs(blendedRadius(tile, 0) - 0.15) < 1e-6);
        // Elongating at fixed volume raises the RMS radius, so rho grows.
        const long = tileOf([{ p: [0, 0, 0], s: [0.2, 0.05, 0.0125], a: 0.8 }]);
        assert.ok(blendedRadius(long, 0) > blendedRadius(tile, 0));
    });

    it('degenerate covariance yields infinite dissimilarity', () => {
        const tile = tileOf([
            { p: [0, 0, 0], s: 0.05, a: 0.8 },
            { p: [0.05, 0, 0], s: 0.05, a: 0.8 }
        ]);
        tile.logDet[1] = -Infinity;
        assert.strictEqual(logSimilarity(tile, 0, 1), -Infinity);
        assert.strictEqual(dissimilarity(logSimilarity(tile, 0, 1)), Infinity);
        assert.strictEqual(dissimilarity(-Infinity), Infinity);
    });

    it('tiles grow rather than overflow when filled beyond capacity', () => {
        const view = makeView([
            { p: [0, 0, 0], s: 0.05, a: 0.8 },
            { p: [0.05, 0, 0], s: 0.05, a: 0.8 },
            { p: [0.1, 0, 0], s: 0.05, a: 0.8 }
        ]);
        const grown = fillSimTile(view, [0, 1, 2], 3, createSimTile(1));
        assert.strictEqual(grown.count, 3);
        // 2 sigma apart, so log B = -0.5 up to the decode's EPS_COV floor, which
        // shifts quad by EPS_COV/sigma^2 — here about 2e-6.
        assert.ok(Math.abs(logSimilarity(grown, 0, 2) - -0.5) < 1e-5);
    });
});
