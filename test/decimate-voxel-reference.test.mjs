/**
 * End-to-end voxel decimation against the reference spec (Splat-Simplify.md
 * §3-§9): the coverage floor, assignment locality, and whether the novelty pass
 * actually spends the budget on distinct content.
 *
 * The behavioural test that matters here is the cluster one: given a budget of
 * exactly one splat per cluster, the selection must find one per cluster rather
 * than spending several on whichever cluster it started in.
 */
import assert from 'node:assert';
import { describe, it } from 'node:test';

import { groupByRepresentative } from '../src/lib/decimate-voxel/assign.js';
import { leavesByCell, leavesTouch, neighbourLeaves } from '../src/lib/decimate-voxel/partition.js';
import { decimateReference, leafAggregates } from '../src/lib/decimate-voxel/reference.js';
import { makeView, mulberry32 as rng } from './helpers/voxel-splats.mjs';

const boundsOf = (view, n) => {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < 3; a++) {
            const v = view.pos[i * 3 + a];
            if (v < min[a]) min[a] = v;
            if (v > max[a]) max[a] = v;
        }
    }
    return { min, ext: [max[0] - min[0], max[1] - min[1], max[2] - min[2]] };
};

/** A diffuse cloud, deterministic. */
const cloud = (n, seed = 3) => {
    const rand = rng(seed);
    return Array.from({ length: n }, () => ({
        p: [rand(), rand(), rand()],
        s: 0.01 + rand() * 0.02,
        a: 0.2 + rand() * 0.7,
        rgb: [rand(), rand(), rand()]
    }));
};

/** `k` tight, well-separated clusters of `per` splats each. */
const clusters = (k, per, seed = 11) => {
    const rand = rng(seed);
    const out = [];
    for (let c = 0; c < k; c++) {
        const centre = [(c % 2) * 4, (Math.floor(c / 2) % 2) * 4, Math.floor(c / 4) * 4];
        for (let i = 0; i < per; i++) {
            out.push({
                p: [
                    centre[0] + rand() * 0.05,
                    centre[1] + rand() * 0.05,
                    centre[2] + rand() * 0.05
                ],
                s: 0.02,
                a: 0.6,
                rgb: [0.5, 0.5, 0.5]
            });
        }
    }
    return out;
};

const run = (splats, target) => {
    const view = makeView(splats);
    const { min, ext } = boundsOf(view, splats.length);
    return { view, result: decimateReference(view, splats.length, min, ext, target) };
};

describe('voxel decimation reference', () => {
    it('keeps at most the target and at least one splat per occupied voxel', () => {
        const splats = cloud(600);
        const { result } = run(splats, 240);

        assert.ok(result.count > 0);
        assert.ok(result.count <= 240, `count ${result.count} exceeds target`);

        // The coverage floor: no occupied region of space can be emptied.
        for (let g = 0; g < result.partition.leafCount; g++) {
            assert.ok(result.reps.start[g + 1] > result.reps.start[g],
                `leaf ${g} kept nothing`);
        }
    });

    it('assigns every splat to a representative in a touching voxel', () => {
        const splats = cloud(600);
        const { result } = run(splats, 240);
        const { partition, reps, assignment } = result;

        // Which leaf owns each slot, and which leaf each representative is in.
        const leafOfSlot = new Int32Array(splats.length);
        for (let g = 0; g < partition.leafCount; g++) {
            leafOfSlot.fill(g, partition.start[g], partition.start[g + 1]);
        }

        const byCell = leavesByCell(partition);
        for (let slot = 0; slot < splats.length; slot++) {
            const rep = assignment[slot];
            assert.ok(rep >= 0, `slot ${slot} unassigned`);
            const g = leafOfSlot[slot];
            const h = leafOfSlot[rep];
            assert.ok(leavesTouch(partition, g, h),
                `slot ${slot} in leaf ${g} assigned across a gap to leaf ${h}`);
            assert.ok(neighbourLeaves(partition, g, byCell).includes(h));
        }

        // Representatives keep themselves.
        for (let k = 0; k < reps.count; k++) {
            assert.strictEqual(assignment[reps.slots[k]], reps.slots[k]);
        }
    });

    it('groups partition the input, with the representative first', () => {
        const splats = cloud(400, 5);
        const { result } = run(splats, 120);
        const groups = groupByRepresentative(result.assignment, result.reps);

        assert.strictEqual(groups.start[result.reps.count], splats.length);
        const seen = new Uint8Array(splats.length);
        for (let k = 0; k < result.reps.count; k++) {
            const from = groups.start[k];
            assert.ok(groups.start[k + 1] > from, `group ${k} is empty`);
            assert.strictEqual(groups.members[from], result.reps.slots[k],
                `group ${k} does not start with its representative`);
            for (let m = from; m < groups.start[k + 1]; m++) {
                const slot = groups.members[m];
                assert.strictEqual(seen[slot], 0, `slot ${slot} in two groups`);
                seen[slot] = 1;
            }
        }
        assert.ok(seen.every(v => v === 1));
    });

    it('spends a one-per-cluster budget on one splat per cluster', () => {
        const k = 8;
        const per = 20;
        const { result } = run(clusters(k, per), k);

        assert.strictEqual(result.count, k);

        // Which cluster each surviving splat came from.
        const clusterOf = slot => Math.floor(result.partition.order[slot] / per);
        const covered = new Set();
        for (let i = 0; i < result.reps.count; i++) {
            covered.add(clusterOf(result.reps.slots[i]));
        }
        assert.strictEqual(covered.size, k,
            `expected all ${k} clusters represented, got ${[...covered].sort()}`);

        // And every splat merged into its own cluster's survivor.
        for (let slot = 0; slot < k * per; slot++) {
            assert.strictEqual(clusterOf(result.assignment[slot]), clusterOf(slot));
        }
    });

    it('produces finite merged output', () => {
        const splats = cloud(400, 9);
        const { result } = run(splats, 120);
        assert.ok(Array.from(result.pos).every(Number.isFinite));
        assert.ok(Array.from(result.geo).every(Number.isFinite));
        assert.ok(Array.from(result.color).every(Number.isFinite));
    });

    it('is deterministic', () => {
        const splats = cloud(400, 13);
        const a = run(splats, 120).result;
        const b = run(splats, 120).result;
        assert.strictEqual(a.count, b.count);
        assert.deepStrictEqual(Array.from(a.pos), Array.from(b.pos));
        assert.deepStrictEqual(Array.from(a.geo), Array.from(b.geo));
        assert.deepStrictEqual(Array.from(a.color), Array.from(b.color));
    });

    it('aggregates per-leaf counts and opacity mass', () => {
        const splats = cloud(200, 17);
        const { view, result } = run(splats, 96);
        const { counts, opacity } = leafAggregates(view, result.partition);

        let total = 0;
        for (let g = 0; g < result.partition.leafCount; g++) {
            assert.strictEqual(counts[g], result.partition.start[g + 1] - result.partition.start[g]);
            assert.ok(opacity[g] > 0);
            total += counts[g];
        }
        assert.strictEqual(total, splats.length);
    });

    it('a target below the voxel count still returns something usable', () => {
        const splats = cloud(300, 21);
        const { result } = run(splats, 4);
        assert.ok(result.count > 0 && result.count <= 4, `got ${result.count}`);
        assert.ok(Array.from(result.geo).every(Number.isFinite));
    });
});
