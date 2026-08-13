/**
 * Whole-scene reference implementation of voxel decimation (spec §3 through §9),
 * single-threaded and resident.
 *
 * This is the oracle, not the shipping path. It exists so the GPU
 * implementation has something exact to be compared against stage by stage, and
 * so decimation quality can be measured before any WGSL is written. It holds
 * ~136 bytes of decoded tile per splat and runs O(occupancy²) per voxel, which
 * is fine for the scenes worth eyeballing and hopeless at 200M — that is what
 * the GPU path is for.
 *
 * The merge itself (§9) is not reimplemented: it is `mergeGroup`, the same
 * moment match every other decimator uses, so merged-mass compensation applies
 * here identically.
 */

import { assignToRepresentatives, groupByRepresentative, type RepresentativeSet } from './assign';
import { allocateExtras } from './budget';
import { buildPartition, leavesByCell, type VoxelPartition } from './partition';
import { addRepresentative, createSelection, priority, selectVoxel } from './representatives';
import { createSimTile, fillSimTile, type SimTile } from './similarity';
import { alphaDecode, createMergeScratch, mergeGroup, type MergedOut, type SplatView } from '../decimate/moment-match';

/**
 * Lazy max-heap over tile slots, keyed by §7 priority.
 *
 * The global completion pass needs the best candidate anywhere, repeatedly, and
 * a linear scan makes that quadratic in the budget — which matters because the
 * pass is not a rare tail: a surface-like scene leaves most base cells empty, so
 * `24G` undershoots the target badly and most of the budget is spent here.
 *
 * Adding a representative can only raise a candidate's nearest-similarity, so
 * priorities only ever fall. That makes lazy revalidation exact: pop the top,
 * recompute its priority, and if it dropped, push it back and continue.
 */
type MaxHeap = { keys: Float64Array; items: Int32Array; size: number; topKey: number };

const heapCreate = (capacity: number): MaxHeap => ({
    keys: new Float64Array(Math.max(capacity, 1)),
    items: new Int32Array(Math.max(capacity, 1)),
    size: 0,
    topKey: 0
});

const heapPush = (h: MaxHeap, key: number, item: number): void => {
    if (h.size === h.keys.length) {
        const keys = new Float64Array(h.size * 2);
        const items = new Int32Array(h.size * 2);
        keys.set(h.keys);
        items.set(h.items);
        h.keys = keys;
        h.items = items;
    }

    let i = h.size++;
    h.keys[i] = key;
    h.items[i] = item;
    while (i > 0) {
        const parent = (i - 1) >> 1;
        if (h.keys[parent] >= h.keys[i]) break;
        [h.keys[parent], h.keys[i]] = [h.keys[i], h.keys[parent]];
        [h.items[parent], h.items[i]] = [h.items[i], h.items[parent]];
        i = parent;
    }
};

/**
 * Pop the highest-keyed item, leaving its key in `topKey`.
 *
 * @param h - The heap.
 * @returns The item, or -1 when empty.
 */
const heapPop = (h: MaxHeap): number => {
    if (h.size === 0) return -1;
    const item = h.items[0];
    h.topKey = h.keys[0];

    h.size--;
    if (h.size > 0) {
        h.keys[0] = h.keys[h.size];
        h.items[0] = h.items[h.size];
        let i = 0;
        for (;;) {
            const l = i * 2 + 1;
            const r = l + 1;
            let largest = i;
            if (l < h.size && h.keys[l] > h.keys[largest]) largest = l;
            if (r < h.size && h.keys[r] > h.keys[largest]) largest = r;
            if (largest === i) break;
            [h.keys[largest], h.keys[i]] = [h.keys[i], h.keys[largest]];
            [h.items[largest], h.items[i]] = [h.items[i], h.items[largest]];
            i = largest;
        }
    }

    return item;
};

type ReferenceResult = {
    /** Output splat count, at most `target`. */
    count: number;
    /** Merged output columns, matching the input view's layout. */
    pos: Float32Array;
    geo: Float32Array;
    color: Float32Array;
    /** Stages kept for assertions against the GPU path. */
    partition: VoxelPartition;
    tile: SimTile;
    reps: RepresentativeSet;
    assignment: Int32Array;
};

/**
 * Per-leaf member counts and opacity mass (the spec §6 allocation weight).
 *
 * @param view - Splat columns.
 * @param partition - The partition.
 * @returns Counts and Σα per leaf.
 */
const leafAggregates = (
    view: SplatView,
    partition: VoxelPartition
): { counts: Int32Array; opacity: Float64Array } => {
    const counts = new Int32Array(partition.leafCount);
    const opacity = new Float64Array(partition.leafCount);

    for (let g = 0; g < partition.leafCount; g++) {
        const end = partition.start[g + 1];
        counts[g] = end - partition.start[g];
        let sum = 0;
        for (let k = partition.start[g]; k < end; k++) {
            sum += alphaDecode(view.geo[partition.order[k] * 8 + 7]);
        }
        opacity[g] = sum;
    }

    return { counts, opacity };
};

/**
 * Decimate a whole scene to at most `target` splats.
 *
 * @param view - Splat columns.
 * @param n - Input splat count.
 * @param min - Bounds minimum per axis.
 * @param ext - Bounds extent per axis.
 * @param target - Output budget T.
 * @returns The merged output plus every intermediate stage.
 */
const decimateReference = (
    view: SplatView,
    n: number,
    min: [number, number, number],
    ext: [number, number, number],
    target: number
): ReferenceResult => {
    const partition = buildPartition(view.pos, n, min, ext, target);
    const tile = fillSimTile(view, partition.order, n, createSimTile(n));

    const { counts, opacity } = leafAggregates(view, partition);
    const quotas = allocateExtras(counts, opacity, partition.leafCount, target, n);

    // Local fill (spec §7 stage 1). Selection state spans the scene, so the
    // global pass below can rank across voxels without recomputing anything.
    const state = createSelection(n);
    const perLeaf: number[][] = [];
    const leafOfSlot = new Int32Array(n);
    let total = 0;
    for (let g = 0; g < partition.leafCount; g++) {
        const from = partition.start[g];
        const count = counts[g];
        leafOfSlot.fill(g, from, from + count);
        selectVoxel(tile, from, count, quotas[g], state);
        const chosen = Array.from(state.reps.slice(0, state.repCount));
        perLeaf.push(chosen);
        total += chosen.length;
    }

    // Global completion (spec §7 stage 2): spend whatever the local budget left
    // over on the highest-priority candidate anywhere.
    if (total < target) {
        const heap = heapCreate(n);
        for (let slot = 0; slot < n; slot++) {
            const p = priority(tile, state, slot);
            if (p > 0) heapPush(heap, p, slot);
        }

        while (total < target) {
            const slot = heapPop(heap);
            if (slot < 0) break;

            const fresh = priority(tile, state, slot);
            if (fresh <= 0) continue;            // covered or promoted meanwhile
            if (fresh < heap.topKey) {
                heapPush(heap, fresh, slot);     // stale key, revalidate later
                continue;
            }

            const g = leafOfSlot[slot];
            addRepresentative(tile, state, partition.start[g], counts[g], slot);
            perLeaf[g].push(slot);
            total++;
        }
    }

    // Flatten to per-leaf runs, each ascending so the output is deterministic.
    const repStart = new Int32Array(partition.leafCount + 1);
    for (let g = 0; g < partition.leafCount; g++) {
        repStart[g + 1] = repStart[g] + perLeaf[g].length;
    }
    const repSlots = new Int32Array(total);
    for (let g = 0; g < partition.leafCount; g++) {
        perLeaf[g].sort((a, b) => a - b);
        repSlots.set(perLeaf[g], repStart[g]);
    }
    const reps: RepresentativeSet = { slots: repSlots, start: repStart, count: total };

    const assignment = assignToRepresentatives(tile, partition, reps, leavesByCell(partition));
    const groups = groupByRepresentative(assignment, reps);

    // Merge (spec §9) — the shared moment match, on global splat indices.
    const { colorDim } = view;
    const pos = new Float32Array(total * 3);
    const geo = new Float32Array(total * 8);
    const color = new Float32Array(total * colorDim);
    const out: MergedOut = {
        pos: new Float64Array(3),
        geo: new Float64Array(8),
        color: new Float64Array(colorDim)
    };
    const scratch = createMergeScratch();
    const members = new Int32Array(n);

    for (let k = 0; k < total; k++) {
        const from = groups.start[k];
        const count = groups.start[k + 1] - from;
        for (let m = 0; m < count; m++) {
            members[m] = partition.order[groups.members[from + m]];
        }
        mergeGroup(view, members, count, out, scratch);

        for (let a = 0; a < 3; a++) pos[k * 3 + a] = out.pos[a];
        for (let a = 0; a < 8; a++) geo[k * 8 + a] = out.geo[a];
        for (let a = 0; a < colorDim; a++) color[k * colorDim + a] = out.color[a];
    }

    return { count: total, pos, geo, color, partition, tile, reps, assignment };
};

export { decimateReference, leafAggregates, type ReferenceResult };
