/**
 * Budget allocation against the reference spec (Splat-Simplify.md §6):
 * B_local = min(T, N, 24G), one representative per occupied voxel, and the
 * surplus shared proportional to per-voxel opacity with largest-remainder
 * rounding under per-voxel capacity limits.
 */
import assert from 'node:assert';
import { describe, it } from 'node:test';

import { localFillBudget, allocateExtras } from '../src/lib/decimate-voxel/budget.js';

const sum = arr => arr.reduce((a, b) => a + b, 0);

describe('voxel budget', () => {
    it('local-fill budget is the tightest of target, input and 24 per voxel', () => {
        assert.strictEqual(localFillBudget(1000, 5000, 100), 1000);   // target binds
        assert.strictEqual(localFillBudget(1000, 500, 100), 500);     // input binds
        assert.strictEqual(localFillBudget(1000, 5000, 10), 240);     // 24G binds
    });

    it('a target equal to the voxel count leaves no extras', () => {
        const quota = allocateExtras([10, 10, 10], [1, 1, 1], 3, 3, 30);
        assert.deepStrictEqual(Array.from(quota), [0, 0, 0]);
    });

    it('extras follow opacity mass', () => {
        // remaining = 6 - 2 = 4, split 3:1 by opacity.
        const quota = allocateExtras([100, 100], [3, 1], 2, 6, 200);
        assert.deepStrictEqual(Array.from(quota), [3, 1]);
    });

    it('capacity clamping spills onto the voxels that can hold it', () => {
        // Voxel 0 has the opacity but holds only one extra; the rest must go to
        // voxel 1 rather than being dropped.
        const quota = allocateExtras([2, 100], [10, 1], 2, 12, 102);
        assert.deepStrictEqual(Array.from(quota), [1, 9]);
        assert.strictEqual(sum(Array.from(quota)), 10);
    });

    it('singleton voxels are ineligible', () => {
        const quota = allocateExtras([1, 1, 5], [5, 5, 1], 3, 7, 7);
        assert.deepStrictEqual(Array.from(quota), [0, 0, 4]);
    });

    it('an all-transparent scene still spends the budget, round-robin', () => {
        const quota = allocateExtras([10, 10, 10], [0, 0, 0], 3, 8, 30);
        assert.strictEqual(sum(Array.from(quota)), 5);
        // Even split, remainder to the lower indices.
        assert.deepStrictEqual(Array.from(quota), [2, 2, 1]);
    });

    it('spends exactly the local-fill budget and never exceeds capacity', () => {
        const counts = [3, 17, 1, 42, 8, 2, 60, 5];
        const opacity = [1.5, 9, 0.2, 4, 0.5, 3, 12, 0.1];
        const n = sum(counts);

        for (const target of [10, 25, 60, 138, 500]) {
            const quota = Array.from(allocateExtras(counts, opacity, counts.length, target, n));
            const expected = Math.max(0, localFillBudget(target, n, counts.length) - counts.length);
            assert.strictEqual(sum(quota), expected, `target ${target}: got ${quota}`);
            quota.forEach((q, g) => {
                assert.ok(q <= counts[g] - 1, `voxel ${g} over capacity at target ${target}`);
                assert.ok(q >= 0);
            });
        }
    });

    it('terminates when every voxel is full before the budget is spent', () => {
        // B_local = min(100, 4, 48) = 4, so two extras against two slots of room.
        const quota = allocateExtras([2, 2], [1, 1], 2, 100, 4);
        assert.deepStrictEqual(Array.from(quota), [1, 1]);
    });
});
