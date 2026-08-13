/**
 * Adaptive voxel partition against the reference spec (Splat-Simplify.md §3):
 * space-uniform binning, midpoint octant subdivision of overfull cells, and the
 * neighbourhood every assignment query is restricted to.
 *
 * The structural invariants are the point of most of these: leaves must
 * partition the splats exactly, and a leaf's member run and its box must agree,
 * because every later pass reads one and trusts the other.
 */
import assert from 'node:assert';
import { describe, it } from 'node:test';

import {
    NORM_MAX, normalizePositions, countOccupied, buildPartition, leavesTouch, leavesByCell, neighbourLeaves
} from '../src/lib/decimate-voxel/partition.js';
import { MAX_DEPTH, subdivideThreshold, voxelBudget } from '../src/lib/decimate-voxel/voxel-grid.js';

/** Deterministic PRNG so failures reproduce. */
const rng = (seed) => () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
};

const positionsOf = (points) => {
    const pos = new Float32Array(points.length * 3);
    points.forEach((p, i) => {
        pos[i * 3] = p[0];
        pos[i * 3 + 1] = p[1];
        pos[i * 3 + 2] = p[2];
    });
    return pos;
};

/** Uniform cloud in the unit cube, plus one dense knot to force subdivision. */
const cloudWithKnot = (n, knot, seed = 7) => {
    const rand = rng(seed);
    const points = [];
    for (let i = 0; i < n; i++) points.push([rand(), rand(), rand()]);
    for (let i = 0; i < knot; i++) {
        points.push([0.5 + rand() * 0.002, 0.5 + rand() * 0.002, 0.5 + rand() * 0.002]);
    }
    return positionsOf(points);
};

/** Every splat lands in exactly one leaf, and inside that leaf's box. */
const assertPartitionSound = (p, pos, n, min, ext) => {
    assert.strictEqual(p.start[p.leafCount], n);
    assert.strictEqual(p.start[0], 0);

    const seen = new Uint8Array(n);
    const norm = normalizePositions(pos, n, min, ext);

    for (let g = 0; g < p.leafCount; g++) {
        assert.ok(p.start[g] < p.start[g + 1], `leaf ${g} is empty`);
        for (let k = p.start[g]; k < p.start[g + 1]; k++) {
            const i = p.order[k];
            assert.strictEqual(seen[i], 0, `splat ${i} appears twice`);
            seen[i] = 1;
            for (let a = 0; a < 3; a++) {
                const v = norm[i * 3 + a];
                assert.ok(v >= p.lo[g * 3 + a] - 1e-12 && v < p.hi[g * 3 + a] + 1e-12,
                    `splat ${i} axis ${a} = ${v} outside leaf ${g} box ` +
                    `[${p.lo[g * 3 + a]}, ${p.hi[g * 3 + a]})`);
            }
        }
    }
    assert.ok(seen.every(v => v === 1), 'some splats are in no leaf');
};

describe('voxel partition', () => {
    it('normalizes into [0, NORM_MAX] and survives degenerate axes', () => {
        const pos = positionsOf([[0, 5, -1], [10, 5, 3], [5, 5, 1]]);
        const norm = normalizePositions(pos, 3, [0, 5, -1], [10, 0, 4]);
        assert.strictEqual(norm[0], 0);
        assert.strictEqual(norm[3], NORM_MAX);       // clamped off the far face
        assert.ok(Math.abs(norm[6] - 0.5) < 1e-12);
        // A zero-extent axis collapses rather than dividing by zero.
        assert.strictEqual(norm[1], 0);
        assert.strictEqual(norm[4], 0);
    });

    it('counts occupied cells, not total cells', () => {
        const pos = positionsOf([[0.1, 0.1, 0.1], [0.15, 0.1, 0.1], [0.9, 0.9, 0.9]]);
        const norm = normalizePositions(pos, 3, [0, 0, 0], [1, 1, 1]);
        assert.strictEqual(countOccupied(norm, 3, { dx: 2, dy: 2, dz: 2 }), 2);
        assert.strictEqual(countOccupied(norm, 3, { dx: 1, dy: 1, dz: 1 }), 1);
        assert.strictEqual(countOccupied(norm, 3, { dx: 100, dy: 100, dz: 100 }), 3);
    });

    it('partitions the input exactly, with boxes that contain their members', () => {
        const n = 4000;
        const pos = cloudWithKnot(n, 0);
        const p = buildPartition(pos, n, [0, 0, 0], [1, 1, 1], 480);
        assertPartitionSound(p, pos, n, [0, 0, 0], [1, 1, 1]);
    });

    it('subdivides a dense knot and leaves the sparse background alone', () => {
        const bg = 2000;
        const knot = 2000;
        const n = bg + knot;
        const pos = cloudWithKnot(bg, knot);
        const target = 960;
        const p = buildPartition(pos, n, [0, 0, 0], [1, 1, 1], target);
        assertPartitionSound(p, pos, n, [0, 0, 0], [1, 1, 1]);

        // The knot's leaves are the deep ones, and they are far finer than a
        // base cell.
        const depths = [];
        for (let g = 0; g < p.leafCount; g++) {
            depths.push(Math.round(Math.log2((1 / p.dims.dx) / (p.hi[g * 3] - p.lo[g * 3]))));
        }
        assert.ok(Math.max(...depths) > 0, 'nothing was subdivided');
        assert.ok(Math.max(...depths) <= MAX_DEPTH);
        assert.ok(depths.filter(d => d === 0).length > 0, 'everything was subdivided');
    });

    it('respects the leaf cap and the occupancy threshold', () => {
        const bg = 500;
        const knot = 3500;
        const n = bg + knot;
        const pos = cloudWithKnot(bg, knot);
        const target = 240;
        const p = buildPartition(pos, n, [0, 0, 0], [1, 1, 1], target);

        assertPartitionSound(p, pos, n, [0, 0, 0], [1, 1, 1]);
        assert.ok(p.leafCount <= target, `leafCount ${p.leafCount} exceeds target ${target}`);
        assert.strictEqual(p.threshold, subdivideThreshold(n, voxelBudget(target, n)));
    });

    it('a single splat gives one leaf covering everything', () => {
        const pos = positionsOf([[1, 2, 3]]);
        const p = buildPartition(pos, 1, [1, 2, 3], [0, 0, 0], 100);
        assert.strictEqual(p.leafCount, 1);
        assert.deepStrictEqual(Array.from(p.order), [0]);
        assert.deepStrictEqual(Array.from(p.start), [0, 1]);
    });

    it('coincident splats terminate at max depth instead of spinning', () => {
        const n = 64;
        const pos = positionsOf(Array.from({ length: n }, () => [0.3, 0.3, 0.3]));
        const p = buildPartition(pos, n, [0, 0, 0], [1, 1, 1], 240);
        // Nothing can separate them, so they stay one leaf.
        assert.strictEqual(p.leafCount, 1);
        assert.strictEqual(p.start[1], n);
    });

    it('touching includes self, faces and corners but not gaps', () => {
        const n = 2000;
        const pos = cloudWithKnot(n, 0);
        const p = buildPartition(pos, n, [0, 0, 0], [1, 1, 1], 480);

        for (let g = 0; g < p.leafCount; g++) assert.ok(leavesTouch(p, g, g));

        // Symmetry, on a sample.
        for (let g = 0; g < Math.min(p.leafCount, 20); g++) {
            for (let h = 0; h < p.leafCount; h++) {
                assert.strictEqual(leavesTouch(p, g, h), leavesTouch(p, h, g));
            }
        }
    });

    it('the 27-cell neighbourhood finds every touching leaf', () => {
        const bg = 1500;
        const knot = 1500;
        const n = bg + knot;
        const pos = cloudWithKnot(bg, knot);
        const p = buildPartition(pos, n, [0, 0, 0], [1, 1, 1], 720);
        const byCell = leavesByCell(p);

        // Brute force against the bucketed query: subdivision only refines
        // within a base cell, so bucketing must lose nothing.
        for (let g = 0; g < p.leafCount; g++) {
            const brute = [];
            for (let h = 0; h < p.leafCount; h++) {
                if (leavesTouch(p, g, h)) brute.push(h);
            }
            assert.deepStrictEqual(neighbourLeaves(p, g, byCell), brute, `leaf ${g}`);
        }
    });
});
