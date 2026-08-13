/**
 * Voxel-grid arithmetic against the reference spec (Splat-Simplify.md §3):
 * budget V = clamp(floor(T/24), 1, N), dims proportional to extent with product
 * near V, shrink by 0.85 until occupied cells fit, subdivision threshold
 * H = max(2, ceil(8N/V)).
 */
import assert from 'node:assert';
import { describe, it } from 'node:test';

import {
    voxelBudget, gridDims, fitDims, subdivideThreshold
} from '../src/lib/decimate-voxel/voxel-grid.js';

describe('voxel grid', () => {
    it('budget targets 24 representatives per cell, clamped to [1, n]', () => {
        assert.strictEqual(voxelBudget(862238, 55183186), 35926);
        assert.strictEqual(voxelBudget(24, 1000), 1);
        assert.strictEqual(voxelBudget(1, 1000), 1);          // floor(1/24)=0 -> clamped up
        assert.strictEqual(voxelBudget(1000000, 10), 10);      // clamped to n
    });

    it('dims follow extent proportions with product near the budget', () => {
        const d = gridDims([100, 100, 100], 1000);
        assert.deepStrictEqual(d, { dx: 10, dy: 10, dz: 10 });
        // A flat scene should stay flat: the short axis gets fewer cells.
        const flat = gridDims([100, 10, 100], 1000);
        assert.ok(flat.dy < flat.dx, `expected dy < dx, got ${JSON.stringify(flat)}`);
        assert.strictEqual(flat.dx, flat.dz);
    });

    it('degenerate axes never produce a zero dimension', () => {
        const d = gridDims([100, 0, 100], 1000);
        assert.strictEqual(d.dy, 1);
        assert.ok(d.dx >= 1 && d.dz >= 1);
    });

    it('fitDims shrinks until occupancy fits, and terminates at 1,1,1', () => {
        // Probe reports the full cell product as occupied: forces shrinking.
        const count = d => d.dx * d.dy * d.dz;
        const fitted = fitDims({ dx: 10, dy: 10, dz: 10 }, 27, count);
        assert.ok(count(fitted) <= 27, `got ${JSON.stringify(fitted)}`);
        // Unsatisfiable budget must still terminate rather than spin.
        const floored = fitDims({ dx: 4, dy: 4, dz: 4 }, 0, count);
        assert.deepStrictEqual(floored, { dx: 1, dy: 1, dz: 1 });
    });

    it('fitDims leaves dims alone when already within budget', () => {
        const d = { dx: 3, dy: 3, dz: 3 };
        assert.deepStrictEqual(fitDims(d, 1000, () => 27), d);
    });

    it('subdivision threshold is 8x the mean occupancy, floored at 2', () => {
        assert.strictEqual(subdivideThreshold(55183186, 35926), 12289);
        assert.strictEqual(subdivideThreshold(10, 1000), 2);   // floored
    });
});
