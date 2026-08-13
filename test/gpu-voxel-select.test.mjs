/**
 * GPU voxel selection parity (GPU required; suites skip without a WebGPU
 * adapter).
 *
 * The comparison is deliberately against the whole CPU reference, which reaches
 * the same answer by a different route: it spends the global budget with a
 * sequential lazy heap in f64, while the kernels record each voxel's full pick
 * sequence and threshold the priorities in f32. Two unrelated algorithms
 * agreeing is a far stronger check than one implementation compared to a copy of
 * itself, and it is the only way to catch an error in the threshold argument.
 *
 * Structural results — per-voxel counts, the coverage floor, assignment locality
 * — must match exactly. Picks carry a small tolerance for near-ties, the one
 * place single precision can legitimately disagree.
 */
import assert from 'node:assert';
import { after, before, describe, it } from 'node:test';

import { allocateExtras } from '../src/lib/decimate-voxel/budget.js';
import { leavesTouch } from '../src/lib/decimate-voxel/partition.js';
import { decimateReference, leafAggregates } from '../src/lib/decimate-voxel/reference.js';
import { logSimilarity } from '../src/lib/decimate-voxel/similarity.js';
import { GpuVoxelSelect } from '../src/lib/gpu/gpu-voxel-select.js';
import { makeView, mulberry32 as rng } from './helpers/voxel-splats.mjs';

let device = null;

before(async () => {
    try {
        const { createDevice } = await import('../src/cli/node-device.js');
        device = await createDevice();
    } catch {
        device = null;
    }
});

after(() => {
    device?.destroy?.();
});

/** A diffuse cloud filling the unit cube. */
const cloud = (n, seed = 3) => {
    const rand = rng(seed);
    return Array.from({ length: n }, () => ({
        p: [rand(), rand(), rand()],
        s: 0.01 + rand() * 0.02,
        a: 0.2 + rand() * 0.7,
        rgb: [rand(), rand(), rand()]
    }));
};

/**
 * A cloud on a thin shell, so most base cells are empty. That drives `24G` well
 * below the target and pushes the majority of the budget through the global
 * pass — the case a dense cube never exercises.
 */
const shell = (n, seed = 3) => {
    const rand = rng(seed);
    return Array.from({ length: n }, () => {
        const theta = rand() * Math.PI * 2;
        const z = rand() * 2 - 1;
        const r = Math.sqrt(1 - z * z) * (0.98 + rand() * 0.04);
        return {
            p: [0.5 + 0.45 * r * Math.cos(theta), 0.5 + 0.45 * r * Math.sin(theta), 0.5 + 0.45 * z],
            s: 0.01 + rand() * 0.01,
            a: 0.2 + rand() * 0.7,
            rgb: [rand(), rand(), rand()]
        };
    });
};

/** Run both paths over identical inputs. */
const both = async (splats, target) => {
    const n = splats.length;
    const view = makeView(splats);
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < 3; a++) {
            const v = view.pos[i * 3 + a];
            if (v < min[a]) min[a] = v;
            if (v > max[a]) max[a] = v;
        }
    }
    const ext = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];

    const cpu = decimateReference(view, n, min, ext, target);
    const { partition } = cpu;
    const { counts, opacity } = leafAggregates(view, partition);
    const quotas = allocateExtras(counts, opacity, partition.leafCount, target, n);

    const gpu = new GpuVoxelSelect(device, n, partition.leafCount);
    try {
        const result = await gpu.execute(view, n, partition, quotas, target);
        return { view, n, target, partition, cpu, gpu: result };
    } finally {
        gpu.destroy();
    }
};

const repsOf = (reps, g) =>
    Array.from(reps.slots.slice(reps.start[g], reps.start[g + 1])).sort((a, b) => a - b);

const gpuRepsOf = (result, g) =>
    Array.from(result.repSlots.slice(result.repStart[g], result.repStart[g + 1]));

describe('gpu voxel select', () => {
    it('keeps the same count per voxel as the reference, on a dense cloud', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const { partition, cpu, gpu } = await both(cloud(2000), 480);
        assert.strictEqual(gpu.count, cpu.reps.count,
            `kept ${gpu.count}, reference kept ${cpu.reps.count}`);

        for (let g = 0; g < partition.leafCount; g++) {
            const want = cpu.reps.start[g + 1] - cpu.reps.start[g];
            assert.strictEqual(gpu.repCount[g], want,
                `leaf ${g}: kept ${gpu.repCount[g]}, reference kept ${want}`);
            assert.ok(gpu.repCount[g] > 0, `leaf ${g} kept nothing`);
        }
    });

    it('agrees with the reference where the global pass dominates', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        // Sparse occupancy: the local fill can only place 24 per voxel, so the
        // threshold pass has to reproduce the heap's choices for the rest.
        const { partition, cpu, gpu, target } = await both(shell(6000, 5), 1440);
        const local = 24 * partition.leafCount;
        assert.ok(local < target,
            `fixture does not exercise the global pass: 24G=${local} >= T=${target}`);

        assert.strictEqual(gpu.count, cpu.reps.count);

        let matched = 0;
        let total = 0;
        for (let g = 0; g < partition.leafCount; g++) {
            const want = new Set(repsOf(cpu.reps, g));
            for (const slot of gpuRepsOf(gpu, g)) {
                if (want.has(slot)) matched++;
                total++;
            }
        }
        assert.ok(matched / total > 0.99,
            `only ${matched}/${total} representatives agreed`);
    });

    it('picks the same representatives as the reference', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const { partition, cpu, gpu } = await both(cloud(2000, 5), 480);

        let matched = 0;
        let total = 0;
        for (let g = 0; g < partition.leafCount; g++) {
            const want = new Set(repsOf(cpu.reps, g));
            for (const slot of gpuRepsOf(gpu, g)) {
                if (want.has(slot)) matched++;
                total++;
            }
        }
        // Measured at 100%; the margin is for a stray exact tie, not drift.
        assert.ok(matched / total > 0.995,
            `only ${matched}/${total} representatives agreed`);
    });

    it('emits representatives ascending within each leaf', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const { partition, gpu } = await both(cloud(1500, 9), 360);
        for (let g = 0; g < partition.leafCount; g++) {
            const reps = gpuRepsOf(gpu, g);
            for (let k = 1; k < reps.length; k++) {
                assert.ok(reps[k] > reps[k - 1], `leaf ${g} not ascending: ${reps}`);
            }
        }
    });

    it('assigns every splat inside its own neighbourhood', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const { n, partition, gpu } = await both(cloud(2000, 7), 480);

        const leafOfSlot = new Int32Array(n);
        for (let g = 0; g < partition.leafCount; g++) {
            leafOfSlot.fill(g, partition.start[g], partition.start[g + 1]);
        }
        const chosen = new Set();
        for (let g = 0; g < partition.leafCount; g++) {
            for (const slot of gpuRepsOf(gpu, g)) chosen.add(slot);
        }

        for (let slot = 0; slot < n; slot++) {
            const rep = gpu.assignment[slot];
            assert.ok(chosen.has(rep), `slot ${slot} assigned to non-representative ${rep}`);
            assert.ok(leavesTouch(partition, leafOfSlot[slot], leafOfSlot[rep]),
                `slot ${slot} assigned across a gap`);
        }
        for (const rep of chosen) {
            assert.strictEqual(gpu.assignment[rep], rep);
        }
    });

    it('assignment matches the reference', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        const { n, cpu, gpu } = await both(cloud(1500, 11), 360);

        let agree = 0;
        let tolerated = 0;
        for (let slot = 0; slot < n; slot++) {
            if (gpu.assignment[slot] === cpu.assignment[slot]) {
                agree++;
                continue;
            }
            // Acceptable only if the two targets are indistinguishable under the
            // metric — which also covers a divergence inherited from selection.
            const a = logSimilarity(cpu.tile, slot, gpu.assignment[slot]);
            const b = logSimilarity(cpu.tile, slot, cpu.assignment[slot]);
            if (Math.abs(a - b) < 1e-4) tolerated++;
        }
        assert.strictEqual(agree + tolerated, n,
            `${n - agree - tolerated} assignments differ beyond tie tolerance`);
        assert.ok(agree / n > 0.999, `only ${agree}/${n} assignments matched exactly`);
    });

    it('drops duplicate splats as covered, exactly as the reference does', async (t) => {
        if (!device) return t.skip('no WebGPU adapter available');

        // Voxels stuffed with exact duplicates: the coverage rule has to fire on
        // both paths, and it cannot rely on an f32 log similarity landing on 0.
        // (An earlier COVERED_EPS of 1e-12 was below the f32 noise floor, so the
        // kernels kept spending budget on duplicates the reference discarded.)
        const rand = rng(19);
        const splats = [];
        for (let c = 0; c < 8; c++) {
            const proto = {
                p: [rand(), rand(), rand()],
                s: 0.02,
                a: 0.6,
                rgb: [rand(), rand(), rand()]
            };
            for (let i = 0; i < 40; i++) splats.push(proto);
        }

        const { partition, cpu, gpu } = await both(splats, 240);
        for (let g = 0; g < partition.leafCount; g++) {
            const want = cpu.reps.start[g + 1] - cpu.reps.start[g];
            assert.strictEqual(gpu.repCount[g], want,
                `leaf ${g}: kept ${gpu.repCount[g]}, reference kept ${want}`);
        }
        // Duplicates collapse hard: one survivor per distinct splat.
        assert.ok(gpu.count <= 8, `expected at most one survivor per cluster, got ${gpu.count}`);
    });
});
