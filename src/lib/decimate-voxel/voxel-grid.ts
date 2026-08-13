/**
 * Adaptive voxel grouping: the spatial partition the voxel decimator allocates
 * over. Engine-free (bundled into workers), and deliberately free of GPU types
 * so the same arithmetic can be asserted against the GPU path.
 *
 * The grid is uniform in SPACE, which is what distinguishes this allocator from
 * the other two: uniform decimation removes the same *fraction* everywhere and
 * adaptive removal follows local error, whereas here every occupied cell is
 * guaranteed at least one survivor. That floor is the coverage guarantee — no
 * region of space can be emptied — and dense cells are then subdivided so the
 * grid follows content density.
 */

/** Target representatives per initial voxel, before subdivision. */
const REPS_PER_VOXEL = 24;

/** Occupancy multiple above the mean that triggers subdivision. */
const SUBDIVIDE_SLACK = 8;

/** Octree depth ceiling for subdivision. */
const MAX_DEPTH = 16;

/** Shrink factor applied to every axis when the grid yields too many cells. */
const SHRINK = 0.85;

type GridDims = { dx: number; dy: number; dz: number };

/**
 * Voxel budget: the density of the initial grouping, distinct from the output
 * budget. Targets {@link REPS_PER_VOXEL} representatives per cell.
 *
 * @param targetCount - Output gaussian count.
 * @param n - Input gaussian count.
 * @returns The number of cells to aim for, clamped to [1, n].
 */
const voxelBudget = (targetCount: number, n: number): number =>
    Math.max(1, Math.min(n, Math.floor(targetCount / REPS_PER_VOXEL)));

/**
 * Per-axis cell counts, proportional to the bounds extent on each axis so cells
 * stay roughly cubic, with the product close to `budget`.
 *
 * @param ext - Bounds extent per axis.
 * @param budget - Target cell count from {@link voxelBudget}.
 * @returns Positive integer dimensions.
 */
const gridDims = (ext: [number, number, number], budget: number): GridDims => {
    const maxExt = Math.max(ext[0], ext[1], ext[2], 1e-20);
    const cbrt = Math.cbrt(budget);
    const axis = (e: number) => Math.max(1, Math.round(cbrt * (e / maxExt)));
    return { dx: axis(ext[0]), dy: axis(ext[1]), dz: axis(ext[2]) };
};

/**
 * Shrink every axis by {@link SHRINK} until the non-empty cell count fits the
 * budget. Operates on a caller-supplied occupancy probe so this stays free of
 * any particular storage layout, and so the GPU path can supply a counted
 * histogram rather than re-binning on the CPU.
 *
 * Iterates without a fixed limit (as the reference does) but every step strictly
 * reduces at least one dimension or bottoms out at 1 on all three, at which
 * point the count is 1 and the loop must exit.
 *
 * @param dims - Starting dimensions.
 * @param budget - Cell budget.
 * @param countNonEmpty - Returns the occupied-cell count for given dimensions.
 * @returns Dimensions whose occupied-cell count is within budget.
 */
const fitDims = (
    dims: GridDims,
    budget: number,
    countNonEmpty: (d: GridDims) => number
): GridDims => {
    let cur = dims;
    while (countNonEmpty(cur) > budget) {
        const next = {
            dx: Math.max(1, Math.floor(cur.dx * SHRINK)),
            dy: Math.max(1, Math.floor(cur.dy * SHRINK)),
            dz: Math.max(1, Math.floor(cur.dz * SHRINK))
        };
        if (next.dx === cur.dx && next.dy === cur.dy && next.dz === cur.dz) return cur;
        cur = next;
    }
    return cur;
};

/**
 * Occupancy above which a cell is split into octants.
 *
 * @param n - Input gaussian count.
 * @param budget - Cell budget.
 * @returns The subdivision threshold, at least 2.
 */
const subdivideThreshold = (n: number, budget: number): number =>
    Math.max(2, Math.ceil((SUBDIVIDE_SLACK * n) / Math.max(1, budget)));

export {
    REPS_PER_VOXEL,
    SUBDIVIDE_SLACK,
    MAX_DEPTH,
    SHRINK,
    voxelBudget,
    gridDims,
    fitDims,
    subdivideThreshold,
    type GridDims
};
