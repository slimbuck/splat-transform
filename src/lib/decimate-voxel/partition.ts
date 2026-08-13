/**
 * Building the adaptive voxel partition (spec §3): bin into a space-uniform
 * grid, then split overfull cells into octants until occupancy is bounded.
 *
 * The output is deliberately shaped like what the GPU produces rather than like
 * a tree: splat indices permuted so each leaf owns one contiguous run, plus a
 * per-leaf AABB. Every later pass — representative selection, budget, merge —
 * then walks runs, and the WGSL versions of those passes get the same layout
 * from a radix sort and a segmented refine.
 *
 * This is the reference path: single-threaded, whole-scene, Set-based occupancy
 * probing. It exists so the GPU partition can be asserted against something
 * exact on scenes small enough to check by hand, not to run at 200M.
 */

import { MAX_DEPTH, fitDims, gridDims, subdivideThreshold, voxelBudget, type GridDims } from './voxel-grid';

/** Upper bound of the normalized position range (spec §3.2). */
const NORM_MAX = 0.999999;

/**
 * Leaves of the adaptive partition. Leaf `g` owns `order[start[g]..start[g+1])`,
 * and its AABB is `lo[g*3..]`..`hi[g*3..]` in normalized [0,1) space.
 */
type VoxelPartition = {
    leafCount: number;
    /** Splat indices, permuted so every leaf is contiguous. */
    order: Int32Array;
    /** Run offsets, length `leafCount + 1`. */
    start: Int32Array;
    lo: Float64Array;
    hi: Float64Array;
    /** Base-grid cell of each leaf, the key neighbour queries bucket on. */
    baseCell: Int32Array;
    /** Base grid dimensions. */
    dims: GridDims;
    /** Subdivision threshold H actually used. */
    threshold: number;
};

/**
 * Normalized positions in [0, NORM_MAX], one 3-tuple per splat.
 *
 * @param pos - Positions, 3 f32 per splat.
 * @param n - Splat count.
 * @param min - Bounds minimum per axis.
 * @param ext - Bounds extent per axis; a zero axis collapses to 0.
 * @returns Normalized positions.
 */
const normalizePositions = (
    pos: Float32Array,
    n: number,
    min: [number, number, number],
    ext: [number, number, number]
): Float64Array => {
    const out = new Float64Array(n * 3);
    const inv = [
        ext[0] > 0 ? 1 / ext[0] : 0,
        ext[1] > 0 ? 1 / ext[1] : 0,
        ext[2] > 0 ? 1 / ext[2] : 0
    ];
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < 3; a++) {
            const v = (pos[i * 3 + a] - min[a]) * inv[a];
            out[i * 3 + a] = v <= 0 ? 0 : (v >= NORM_MAX ? NORM_MAX : v);
        }
    }
    return out;
};

const cellOf = (norm: Float64Array, i: number, d: GridDims): number => {
    const ix = Math.min(d.dx - 1, Math.floor(norm[i * 3] * d.dx));
    const iy = Math.min(d.dy - 1, Math.floor(norm[i * 3 + 1] * d.dy));
    const iz = Math.min(d.dz - 1, Math.floor(norm[i * 3 + 2] * d.dz));
    return ix + d.dx * (iy + d.dy * iz);
};

/**
 * Occupied base cells for a candidate grid — the probe {@link fitDims} shrinks
 * against.
 *
 * @param norm - Normalized positions.
 * @param n - Splat count.
 * @param d - Candidate dimensions.
 * @returns Number of non-empty cells.
 */
const countOccupied = (norm: Float64Array, n: number, d: GridDims): number => {
    const seen = new Set<number>();
    for (let i = 0; i < n; i++) seen.add(cellOf(norm, i, d));
    return seen.size;
};

/**
 * Build the adaptive partition.
 *
 * Subdivision walks a FIFO queue seeded in base-cell order, so which cells win
 * the last slots under the leaf cap is deterministic and depth-first bias is
 * avoided: a cell splits only after every equally overfull cell ahead of it
 * has. Splitting stops at {@link MAX_DEPTH}, when a cell is at or under the
 * threshold, or when the leaf count reaches `target` — the partition can never
 * be finer than the output budget it feeds.
 *
 * @param pos - Positions, 3 f32 per splat.
 * @param n - Splat count.
 * @param min - Bounds minimum per axis.
 * @param ext - Bounds extent per axis.
 * @param target - Output budget T.
 * @returns The partition.
 */
const buildPartition = (
    pos: Float32Array,
    n: number,
    min: [number, number, number],
    ext: [number, number, number],
    target: number
): VoxelPartition => {
    const norm = normalizePositions(pos, n, min, ext);

    const budget = voxelBudget(target, n);
    const dims = fitDims(gridDims(ext, budget), budget, d => countOccupied(norm, n, d));
    const threshold = subdivideThreshold(n, budget);

    // Counting sort into base cells.
    const cellCount = dims.dx * dims.dy * dims.dz;
    const counts = new Int32Array(cellCount + 1);
    const cells = new Int32Array(n);
    for (let i = 0; i < n; i++) {
        const c = cellOf(norm, i, dims);
        cells[i] = c;
        counts[c + 1]++;
    }
    for (let c = 0; c < cellCount; c++) counts[c + 1] += counts[c];

    const order = new Int32Array(n);
    const cursor = counts.slice(0, cellCount);
    for (let i = 0; i < n; i++) order[cursor[cells[i]]++] = i;

    // Seed one leaf per occupied base cell, in cell order.
    type Leaf = {
        cell: number;
        start: number;
        end: number;
        depth: number;
        lo: [number, number, number];
        hi: [number, number, number];
    };
    const leaves: Leaf[] = [];
    const queue: number[] = [];
    const cw = 1 / dims.dx, ch = 1 / dims.dy, cd = 1 / dims.dz;
    for (let c = 0; c < cellCount; c++) {
        const start = counts[c];
        const end = counts[c + 1];
        if (start === end) continue;
        const ix = c % dims.dx;
        const iy = Math.floor(c / dims.dx) % dims.dy;
        const iz = Math.floor(c / (dims.dx * dims.dy));
        const leaf: Leaf = {
            cell: c,
            start,
            end,
            depth: 0,
            lo: [ix * cw, iy * ch, iz * cd],
            hi: [(ix + 1) * cw, (iy + 1) * ch, (iz + 1) * cd]
        };
        leaves.push(leaf);
        if (end - start > threshold) queue.push(leaves.length - 1);
    }

    // Split overfull cells at their midpoints, partitioning members in place.
    const scratch = new Int32Array(n);
    const octantCounts = new Int32Array(8);
    const octantCursor = new Int32Array(8);
    let head = 0;
    while (head < queue.length && leaves.length < target) {
        const idx = queue[head++];
        const leaf = leaves[idx];
        if (leaf.depth >= MAX_DEPTH || leaf.end - leaf.start <= threshold) continue;

        const mx = 0.5 * (leaf.lo[0] + leaf.hi[0]);
        const my = 0.5 * (leaf.lo[1] + leaf.hi[1]);
        const mz = 0.5 * (leaf.lo[2] + leaf.hi[2]);

        octantCounts.fill(0);
        for (let k = leaf.start; k < leaf.end; k++) {
            const i = order[k];
            const o = (norm[i * 3] >= mx ? 1 : 0) |
                      (norm[i * 3 + 1] >= my ? 2 : 0) |
                      (norm[i * 3 + 2] >= mz ? 4 : 0);
            octantCounts[o]++;
        }

        // The cap is checked against the whole split, not per octant: a partial
        // split would leave members inside no leaf's AABB, and every later pass
        // assumes a leaf's run and its box agree.
        let nonEmpty = 0;
        for (let o = 0; o < 8; o++) if (octantCounts[o] > 0) nonEmpty++;
        if (leaves.length + nonEmpty - 1 > target) continue;

        let acc = leaf.start;
        for (let o = 0; o < 8; o++) {
            octantCursor[o] = acc;
            acc += octantCounts[o];
        }
        for (let k = leaf.start; k < leaf.end; k++) {
            const i = order[k];
            const o = (norm[i * 3] >= mx ? 1 : 0) |
                      (norm[i * 3 + 1] >= my ? 2 : 0) |
                      (norm[i * 3 + 2] >= mz ? 4 : 0);
            scratch[octantCursor[o]++] = i;
        }
        order.set(scratch.subarray(leaf.start, leaf.end), leaf.start);

        // Rewrite this leaf as its first non-empty octant and append the rest,
        // so a split that keeps every member together costs no leaf and can
        // keep descending (spec §3.3).
        // The parent box, captured before the first octant overwrites it.
        const plo = leaf.lo;
        const phi = leaf.hi;
        const childDepth = leaf.depth + 1;

        let cursorPos = leaf.start;
        let first = true;
        for (let o = 0; o < 8; o++) {
            const count = octantCounts[o];
            if (count === 0) continue;
            const lo: [number, number, number] = [
                (o & 1) ? mx : plo[0],
                (o & 2) ? my : plo[1],
                (o & 4) ? mz : plo[2]
            ];
            const hi: [number, number, number] = [
                (o & 1) ? phi[0] : mx,
                (o & 2) ? phi[1] : my,
                (o & 4) ? phi[2] : mz
            ];

            if (first) {
                first = false;
                leaf.start = cursorPos;
                leaf.end = cursorPos + count;
                leaf.depth = childDepth;
                leaf.lo = lo;
                leaf.hi = hi;
                if (count > threshold) queue.push(idx);
            } else {
                leaves.push({
                    cell: leaf.cell,
                    start: cursorPos,
                    end: cursorPos + count,
                    depth: childDepth,
                    lo,
                    hi
                });
                if (count > threshold) queue.push(leaves.length - 1);
            }
            cursorPos += count;
        }
    }

    // Flatten, keeping leaves in member order so runs are ascending.
    leaves.sort((a, b) => a.start - b.start);
    const leafCount = leaves.length;
    const startOut = new Int32Array(leafCount + 1);
    const loOut = new Float64Array(leafCount * 3);
    const hiOut = new Float64Array(leafCount * 3);
    const baseCell = new Int32Array(leafCount);
    for (let g = 0; g < leafCount; g++) {
        const leaf = leaves[g];
        startOut[g] = leaf.start;
        baseCell[g] = leaf.cell;
        for (let a = 0; a < 3; a++) {
            loOut[g * 3 + a] = leaf.lo[a];
            hiOut[g * 3 + a] = leaf.hi[a];
        }
    }
    startOut[leafCount] = n;

    return { leafCount, order, start: startOut, lo: loOut, hi: hiOut, baseCell, dims, threshold };
};

/**
 * Do two leaves share a face, edge or corner (spec §3.3/§8)? True for a leaf
 * against itself.
 *
 * Leaves are half-open boxes, so touching means the closed boxes intersect:
 * equality of a face coordinate counts, with a tolerance for the accumulated
 * error in repeated midpoint splits. That error is ~1e-19 absolute, while the
 * finest possible leaf is ~1e-8 wide, so the tolerance cannot merge distinct
 * boxes.
 *
 * @param p - Partition.
 * @param a - First leaf.
 * @param b - Second leaf.
 * @returns Whether they touch.
 */
const leavesTouch = (p: VoxelPartition, a: number, b: number): boolean => {
    const eps = 1e-12;
    for (let axis = 0; axis < 3; axis++) {
        if (p.lo[a * 3 + axis] > p.hi[b * 3 + axis] + eps) return false;
        if (p.lo[b * 3 + axis] > p.hi[a * 3 + axis] + eps) return false;
    }
    return true;
};

/**
 * Leaves in a base cell, for neighbour queries.
 *
 * Subdivision only refines within a base cell, so every leaf touching leaf `g`
 * lives in one of the 27 base cells around `g`'s. That bounds the neighbour
 * search without any tree traversal — the property the GPU assignment pass
 * relies on.
 *
 * @param p - Partition.
 * @returns Leaf indices per base cell, keyed by cell index.
 */
const leavesByCell = (p: VoxelPartition): Map<number, number[]> => {
    const map = new Map<number, number[]>();
    for (let g = 0; g < p.leafCount; g++) {
        const cell = p.baseCell[g];
        const list = map.get(cell);
        if (list) list.push(g); else map.set(cell, [g]);
    }
    return map;
};

/**
 * Leaves touching leaf `g`, including `g` itself, in ascending leaf order.
 *
 * @param p - Partition.
 * @param g - Leaf index.
 * @param byCell - Index from {@link leavesByCell}.
 * @returns Touching leaf indices.
 */
const neighbourLeaves = (p: VoxelPartition, g: number, byCell: Map<number, number[]>): number[] => {
    const { dims } = p;
    const cell = p.baseCell[g];
    const cx = cell % dims.dx;
    const cy = Math.floor(cell / dims.dx) % dims.dy;
    const cz = Math.floor(cell / (dims.dx * dims.dy));

    const out: number[] = [];
    for (let dz = -1; dz <= 1; dz++) {
        const z = cz + dz;
        if (z < 0 || z >= dims.dz) continue;
        for (let dy = -1; dy <= 1; dy++) {
            const y = cy + dy;
            if (y < 0 || y >= dims.dy) continue;
            for (let dx = -1; dx <= 1; dx++) {
                const x = cx + dx;
                if (x < 0 || x >= dims.dx) continue;
                const list = byCell.get(x + dims.dx * (y + dims.dy * z));
                if (!list) continue;
                for (const h of list) {
                    if (leavesTouch(p, g, h)) out.push(h);
                }
            }
        }
    }
    out.sort((a, b) => a - b);
    return out;
};

export {
    NORM_MAX,
    normalizePositions,
    countOccupied,
    buildPartition,
    leavesTouch,
    leavesByCell,
    neighbourLeaves,
    type VoxelPartition
};
