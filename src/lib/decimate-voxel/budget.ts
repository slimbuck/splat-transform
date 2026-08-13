/**
 * How the output budget is divided between voxels (spec §6).
 *
 * This is the allocation axis, and it is what makes this decimator behave
 * unlike the other two: every occupied voxel gets one representative
 * unconditionally — the coverage floor — and only the surplus is shared out,
 * proportional to each voxel's total opacity. So budget follows *presence*
 * first and accumulated opacity second, where uniform decimation follows a
 * fixed ratio and adaptive decimation follows local error.
 *
 * Pure integer/float arithmetic over per-voxel aggregates, which is exactly
 * what the GPU path produces from a segmented reduce, so this stays the
 * reference for both.
 */

import { REPS_PER_VOXEL } from './voxel-grid';

/**
 * Total representatives the local-fill stage may place (spec §6).
 *
 * @param target - Output budget T.
 * @param n - Input gaussian count.
 * @param voxelCount - Number of occupied voxels G.
 * @returns min(T, n, 24G).
 */
const localFillBudget = (target: number, n: number, voxelCount: number): number => Math.min(target, n, REPS_PER_VOXEL * voxelCount);

/**
 * Extra representatives per voxel, beyond the one each occupied voxel keeps.
 *
 * The ideal share `E_g = (B_local - G)·O_g / ΣO_h` is floored and clamped to
 * each voxel's remaining capacity; the shortfall from flooring is then handed
 * out one slot at a time in descending fractional-remainder order, cycling
 * until the budget is spent or every voxel is full. Voxels with a single member
 * are ineligible — there is nothing left in them to keep.
 *
 * @param counts - Member count per voxel.
 * @param opacitySums - Σα per voxel, the allocation weight.
 * @param voxelCount - Number of voxels; `counts`/`opacitySums` must be at least this long.
 * @param target - Output budget T.
 * @param n - Input gaussian count.
 * @returns Extra quota per voxel, summing to `localFillBudget - voxelCount` unless capacity runs out.
 */
const allocateExtras = (
    counts: ArrayLike<number>,
    opacitySums: ArrayLike<number>,
    voxelCount: number,
    target: number,
    n: number
): Int32Array => {
    const quota = new Int32Array(voxelCount);
    let remaining = localFillBudget(target, n, voxelCount) - voxelCount;
    if (remaining <= 0) return quota;

    // Eligible voxels and their opacity mass.
    const order: number[] = [];
    let totalOpacity = 0;
    for (let g = 0; g < voxelCount; g++) {
        if (counts[g] > 1) {
            order.push(g);
            totalOpacity += Math.max(opacitySums[g], 0);
        }
    }
    if (order.length === 0) return quota;

    // Floor the proportional share, clamped to what each voxel can hold. With
    // no opacity anywhere the shares are all 0 and the pass below degenerates
    // to round-robin, which is the sane reading of an all-transparent scene.
    const frac = new Float64Array(voxelCount);
    for (const g of order) {
        const exact = totalOpacity > 0 ?
            (remaining * Math.max(opacitySums[g], 0)) / totalOpacity :
            0;
        const floored = Math.min(Math.floor(exact), counts[g] - 1);
        quota[g] = floored;
        frac[g] = exact - Math.floor(exact);
        remaining -= floored;
    }

    // Largest remainder first, ties by lower voxel index so the result does not
    // depend on the sort's stability.
    order.sort((a, b) => (frac[b] - frac[a]) || (a - b));

    while (remaining > 0) {
        let placed = 0;
        for (const g of order) {
            if (remaining === 0) break;
            if (quota[g] < counts[g] - 1) {
                quota[g]++;
                remaining--;
                placed++;
            }
        }
        if (placed === 0) break;   // every voxel full: budget cannot be spent
    }

    return quota;
};

export { localFillBudget, allocateExtras };
