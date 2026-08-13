/**
 * Per-voxel survivor selection against the reference spec (Splat-Simplify.md
 * §5, §7): the coverage-biased weighted medoid, then saturating-novelty detail
 * representatives, over a slot range of a shared tile.
 */
import assert from 'node:assert';
import { describe, it } from 'node:test';

import { dissimilarity, logSimilarity } from '../src/lib/decimate-voxel/similarity.js';
import {
    NOVELTY_SCALE, createSelection, novelty, priority, firstRepresentative, selectVoxel
} from '../src/lib/decimate-voxel/representatives.js';
import { tileOf } from './helpers/voxel-splats.mjs';

/** Plain weighted medoid cost M_i (spec §5, before the /w_i coverage bias). */
const medoidCost = (tile, i, from, count) => {
    let wSum = 0;
    for (let j = from; j < from + count; j++) wSum += tile.w[j];
    let acc = 0;
    for (let j = from; j < from + count; j++) {
        if (j !== i) acc += tile.w[j] * dissimilarity(logSimilarity(tile, i, j));
    }
    return acc / wSum;
};

const repsOf = state => Array.from(state.reps.slice(0, state.repCount));

/** Three collinear splats one sigma apart, identical but for opacity. */
const collinear = [
    { p: [0, 0, 0], s: 0.05, a: 0.9 },
    { p: [0.05, 0, 0], s: 0.05, a: 0.02 },
    { p: [0.1, 0, 0], s: 0.05, a: 0.9 }
];

describe('voxel representatives', () => {
    it('the first representative is coverage-biased, not the plain medoid', () => {
        // The middle splat is the most central; it is also nearly invisible.
        const tile = tileOf(collinear);

        const costs = [0, 1, 2].map(i => medoidCost(tile, i, 0, 3));
        assert.ok(costs[1] < costs[0] && costs[1] < costs[2],
            `plain medoid should be the middle splat, got ${costs}`);

        // Dividing by w_i moves the choice to a visible splat; slots 0 and 2 are
        // symmetric, so the lower slot wins the tie.
        assert.strictEqual(firstRepresentative(tile, 0, 3), 0);
    });

    it('works on a range that does not start at slot 0', () => {
        // Two decoy splats ahead of the real voxel must not be considered.
        const tile = tileOf([
            { p: [9, 9, 9], s: 0.05, a: 0.99 },
            { p: [9, 9, 9.1], s: 0.05, a: 0.99 },
            ...collinear
        ]);
        assert.strictEqual(firstRepresentative(tile, 2, 3), 2);

        const state = selectVoxel(tile, 2, 3, 1, createSelection(5));
        const reps = repsOf(state);
        assert.strictEqual(reps.length, 2);
        assert.ok(reps.every(r => r >= 2), `range leaked, got ${reps}`);
    });

    it('handles empty and singleton voxels', () => {
        const tile = tileOf([{ p: [0, 0, 0], s: 0.05, a: 0.8 }]);
        assert.strictEqual(firstRepresentative(tile, 0, 1), 0);
        assert.strictEqual(firstRepresentative(tile, 0, 0), -1);

        const state = selectVoxel(tile, 0, 1, 5, createSelection(1));
        assert.deepStrictEqual(repsOf(state), [0]);
    });

    it('novelty saturates between 0 and 1', () => {
        assert.strictEqual(novelty(0), 0);
        assert.strictEqual(novelty(Infinity), 1);
        assert.ok(Math.abs(novelty(NOVELTY_SCALE) - (1 - Math.exp(-1))) < 1e-12);
        assert.ok(novelty(1) < novelty(2) && novelty(2) < novelty(100));
        assert.ok(novelty(1e6) <= 1);
    });

    it('spends a slot on a distant splat over a heavier near-duplicate', () => {
        const tile = tileOf([
            { p: [0, 0, 0], s: 0.05, a: 0.9 },        // heavy
            { p: [0.001, 0, 0], s: 0.05, a: 0.9 },    // heavy, ~duplicate of slot 0
            { p: [0.5, 0, 0], s: 0.05, a: 0.3 }       // 10 sigma away, much lighter
        ]);

        const state = selectVoxel(tile, 0, 3, 1, createSelection(3));
        const reps = repsOf(state);
        assert.strictEqual(reps.length, 2);
        assert.ok(reps.includes(2), `expected the distant splat, got ${reps}`);

        // The near-duplicate's novelty is ~1e-5, so its weight cannot rescue it.
        assert.ok(priority(tile, state, reps.includes(0) ? 1 : 0) < 1e-5);
    });

    it('exact duplicates are dropped as fully covered', () => {
        const dup = { p: [0.2, 0, 0], s: 0.05, a: 0.7, rgb: [0.4, 0.4, 0.4] };
        const tile = tileOf([dup, dup, dup]);

        // Quota is 2, but nothing new is left to keep.
        const state = selectVoxel(tile, 0, 3, 2, createSelection(3));
        assert.strictEqual(state.repCount, 1);
    });

    it('fills the quota with distinct slots and stops there', () => {
        const tile = tileOf([
            { p: [0, 0, 0], s: 0.05, a: 0.8 },
            { p: [0.4, 0, 0], s: 0.05, a: 0.8 },
            { p: [0, 0.4, 0], s: 0.05, a: 0.8 },
            { p: [0, 0, 0.4], s: 0.05, a: 0.8 }
        ]);

        const state = createSelection(4);
        assert.strictEqual(repsOf(selectVoxel(tile, 0, 4, 1, state)).length, 2);

        const all = repsOf(selectVoxel(tile, 0, 4, 3, state));
        assert.strictEqual(all.length, 4);
        assert.strictEqual(new Set(all).size, 4);

        // A quota beyond the member count is capped by availability.
        assert.strictEqual(repsOf(selectVoxel(tile, 0, 4, 99, state)).length, 4);
    });

    it('rejects a range that overruns the selection state', () => {
        const tile = tileOf([{ p: [0, 0, 0], s: 0.05, a: 0.8 }]);
        assert.throws(() => selectVoxel(tile, 0, 4, 1, createSelection(2)),
            /selection state too small/);
    });
});
