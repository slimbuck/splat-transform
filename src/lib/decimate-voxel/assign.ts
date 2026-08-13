/**
 * Assigning every splat to a surviving representative (spec §8).
 *
 * A splat joins the most *similar* representative, not the nearest one, and the
 * search is restricted to its own voxel plus the voxels touching it. The
 * restriction is what keeps this linear-ish instead of a global nearest search,
 * and it is safe because a splat that would prefer a far-away representative
 * would be badly represented by it anyway.
 */

import { neighbourLeaves, type VoxelPartition } from './partition';
import { logSimilarity, type SimTile } from './similarity';

/**
 * Representatives grouped by leaf: leaf `g` owns `slots[start[g]..start[g+1])`,
 * each entry a tile slot.
 */
type RepresentativeSet = {
    slots: Int32Array;
    start: Int32Array;
    count: number;
};

/**
 * Map every slot to the representative it merges into.
 *
 * Representatives map to themselves. Ties on log similarity resolve to the
 * lower representative slot, and a splat whose entire neighbourhood scores
 * -Infinity falls back to its own voxel's first representative — so the result
 * is total: no splat is ever dropped for want of a target.
 *
 * @param tile - Tile filled in partition order, so leaf runs are slot ranges.
 * @param partition - The partition the tile was filled from.
 * @param reps - Representatives per leaf.
 * @param byCell - Base-cell index from `leavesByCell`.
 * @returns Assignment per slot, as a tile slot.
 */
const assignToRepresentatives = (
    tile: SimTile,
    partition: VoxelPartition,
    reps: RepresentativeSet,
    byCell: Map<number, number[]>
): Int32Array => {
    const out = new Int32Array(tile.count).fill(-1);

    for (let g = 0; g < partition.leafCount; g++) {
        const ownFirst = reps.slots[reps.start[g]];
        const neighbours = neighbourLeaves(partition, g, byCell);

        // Mark this leaf's representatives before scanning members, so they are
        // never reassigned to a neighbour that happens to score higher.
        for (let r = reps.start[g]; r < reps.start[g + 1]; r++) {
            out[reps.slots[r]] = reps.slots[r];
        }

        const end = partition.start[g + 1];
        for (let slot = partition.start[g]; slot < end; slot++) {
            if (out[slot] >= 0) continue;

            let best = -1;
            let bestLog = -Infinity;
            for (const h of neighbours) {
                for (let r = reps.start[h]; r < reps.start[h + 1]; r++) {
                    const rep = reps.slots[r];
                    const l = logSimilarity(tile, slot, rep);
                    if (l > bestLog || (l === bestLog && rep < best)) {
                        best = rep;
                        bestLog = l;
                    }
                }
            }

            out[slot] = best >= 0 ? best : ownFirst;
        }
    }

    return out;
};

/**
 * Group an assignment into per-representative member lists.
 *
 * @param assignment - Per-slot representative slot, from {@link assignToRepresentatives}.
 * @param reps - The representative set that produced it.
 * @returns Member slots per representative: `members[start[k]..start[k+1])`,
 * with each representative first in its own run.
 */
const groupByRepresentative = (
    assignment: Int32Array,
    reps: RepresentativeSet
): { members: Int32Array; start: Int32Array } => {
    // Representative slot -> its index in `reps.slots`.
    const indexOf = new Map<number, number>();
    for (let k = 0; k < reps.count; k++) indexOf.set(reps.slots[k], k);

    const counts = new Int32Array(reps.count + 1);
    for (let slot = 0; slot < assignment.length; slot++) {
        counts[indexOf.get(assignment[slot])! + 1]++;
    }
    for (let k = 0; k < reps.count; k++) counts[k + 1] += counts[k];

    const start = counts.slice();
    const members = new Int32Array(assignment.length);
    const cursor = counts.slice(0, reps.count);

    // The representative itself goes first, so a singleton group needs no
    // special case downstream.
    for (let k = 0; k < reps.count; k++) members[cursor[k]++] = reps.slots[k];
    for (let slot = 0; slot < assignment.length; slot++) {
        const k = indexOf.get(assignment[slot])!;
        if (reps.slots[k] !== slot) members[cursor[k]++] = slot;
    }

    return { members, start };
};

export { assignToRepresentatives, groupByRepresentative, type RepresentativeSet };
