/**
 * Global budget fill (Splat-Simplify.md §7 stage 2) as a threshold over
 * per-voxel pick sequences.
 *
 * The property that makes the threshold valid — each voxel's picks are
 * non-increasing in priority, so its survivors are always a prefix — is what
 * most of these check, along with an exhaustive comparison against a plain
 * sequential greedy on random inputs.
 */
import assert from 'node:assert';
import { describe, it } from 'node:test';

import { ALWAYS_TAKE, globalFill } from '../src/lib/decimate-voxel/global-fill.js';
import { mulberry32 } from './helpers/voxel-splats.mjs';

/**
 * Build the input from per-voxel priority lists. The first entry of each voxel
 * is the §5 pick and carries ALWAYS_TAKE, mirroring the kernel.
 */
const build = (voxels, localTake, target) => {
    const voxelCount = voxels.length;
    const runStart = new Int32Array(voxelCount + 1);
    for (let g = 0; g < voxelCount; g++) runStart[g + 1] = runStart[g] + voxels[g].length;
    const pickPriority = new Float32Array(runStart[voxelCount]);
    const pickCount = new Int32Array(voxelCount);
    for (let g = 0; g < voxelCount; g++) {
        pickCount[g] = voxels[g].length;
        for (let k = 0; k < voxels[g].length; k++) {
            pickPriority[runStart[g] + k] = k === 0 ? ALWAYS_TAKE : voxels[g][k];
        }
    }
    return {
        voxelCount,
        runStart,
        pickCount,
        pickPriority,
        localTake: Int32Array.from(localTake),
        target
    };
};

/** Sequential greedy: repeatedly take the best remaining pick anywhere. */
const greedy = (input) => {
    const { voxelCount, runStart, pickCount, pickPriority, localTake, target } = input;
    const take = new Int32Array(voxelCount);
    let total = 0;
    for (let g = 0; g < voxelCount; g++) {
        take[g] = Math.min(pickCount[g], Math.max(0, localTake[g]));
        total += take[g];
    }
    while (total < target) {
        let best = -1;
        let bestPriority = -Infinity;
        for (let g = 0; g < voxelCount; g++) {
            if (take[g] >= pickCount[g]) continue;
            const p = pickPriority[runStart[g] + take[g]];
            if (p > bestPriority) {
                best = g;
                bestPriority = p;
            }
        }
        if (best < 0) break;
        take[best]++;
        total++;
    }
    return { take, total };
};

describe('voxel global fill', () => {
    it('keeps the local entitlement when the budget is already spent', () => {
        const input = build([[ALWAYS_TAKE, 5, 3], [ALWAYS_TAKE, 4, 2]], [2, 2], 4);
        const { take, total } = globalFill(input);
        assert.deepStrictEqual(Array.from(take), [2, 2]);
        assert.strictEqual(total, 4);
    });

    it('never drops a voxel below one representative', () => {
        // Even with a target under the voxel count, the coverage floor holds:
        // localTake is 1 everywhere and is honoured before any thresholding.
        const input = build([[ALWAYS_TAKE, 9], [ALWAYS_TAKE, 8], [ALWAYS_TAKE, 7]], [1, 1, 1], 2);
        const { take, total } = globalFill(input);
        assert.deepStrictEqual(Array.from(take), [1, 1, 1]);
        assert.strictEqual(total, 3);
    });

    it('spends the leftover on the highest priorities anywhere', () => {
        // Voxel 0 holds the three best tail picks; it should get all of them
        // rather than the budget being spread evenly.
        const input = build([
            [ALWAYS_TAKE, 9, 8, 7],
            [ALWAYS_TAKE, 2, 1, 0.5]
        ], [1, 1], 5);
        const { take, total } = globalFill(input);
        assert.deepStrictEqual(Array.from(take), [4, 1]);
        assert.strictEqual(total, 5);
    });

    it('interleaves voxels by priority', () => {
        const input = build([
            [ALWAYS_TAKE, 9, 4],
            [ALWAYS_TAKE, 8, 3],
            [ALWAYS_TAKE, 7, 2]
        ], [1, 1, 1], 6);
        const { take } = globalFill(input);
        // Budget 6 = 3 floors + 3 tails; the three 9/8/7 picks win.
        assert.deepStrictEqual(Array.from(take), [2, 2, 2]);
    });

    it('cannot exceed what the voxels actually hold', () => {
        const input = build([[ALWAYS_TAKE, 5], [ALWAYS_TAKE]], [1, 1], 100);
        const { take, total } = globalFill(input);
        assert.deepStrictEqual(Array.from(take), [2, 1]);
        assert.strictEqual(total, 3);
    });

    it('resolves ties deterministically by voxel then pick', () => {
        const input = build([
            [ALWAYS_TAKE, 5, 5],
            [ALWAYS_TAKE, 5, 5]
        ], [1, 1], 4);
        const { take, total } = globalFill(input);
        assert.strictEqual(total, 4);
        // All four tails tie, so the lower voxel takes its prefix first.
        assert.deepStrictEqual(Array.from(take), [3, 1]);
    });

    it('matches a sequential greedy on random sequences', () => {
        const rand = mulberry32(23);
        for (let trial = 0; trial < 200; trial++) {
            const voxelCount = 1 + Math.floor(rand() * 12);
            const voxels = [];
            for (let g = 0; g < voxelCount; g++) {
                const len = 1 + Math.floor(rand() * 8);
                // Non-increasing, as the greedy sequence always is.
                const seq = [ALWAYS_TAKE];
                let p = rand() * 10;
                for (let k = 1; k < len; k++) {
                    seq.push(p);
                    p *= rand();
                }
                voxels.push(seq);
            }
            const localTake = voxels.map(() => 1 + Math.floor(rand() * 3));
            const target = Math.floor(rand() * 40);
            const input = build(voxels, localTake, target);

            const got = globalFill(input);
            const want = greedy(input);
            assert.strictEqual(got.total, want.total, `trial ${trial}: totals differ`);
            assert.deepStrictEqual(Array.from(got.take), Array.from(want.take),
                `trial ${trial}: takes differ`);
        }
    });
});
