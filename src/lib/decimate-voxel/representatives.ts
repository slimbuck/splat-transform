/**
 * Which splats survive inside a voxel: the coverage-biased weighted medoid
 * (spec §5) that every voxel keeps, then the saturating-novelty detail
 * representatives (spec §7) that fill its quota.
 *
 * The policy is deliberately not "keep the biggest" or "keep the most central".
 * The first survivor is central *and* visually significant, and each extra one
 * is chosen for how little the survivors already cover it — so a voxel spends
 * its budget on genuinely different content instead of near-duplicates. That is
 * what stops a dense uniform surface from consuming slots a detailed edge in
 * the same voxel needs.
 *
 * Every pass here works on a half-open slot range `[from, from + count)` of a
 * tile, because that is what a voxel is once the partition has permuted the
 * splats: one contiguous segment. Selection state is indexed by absolute slot
 * and allocated once for the whole scene, so the cross-voxel global completion
 * pass (spec §7 stage 2) can rank candidates from different voxels against each
 * other without rebuilding anything — each candidate's distance is already
 * measured against its own voxel's representatives.
 */

import { blendedRadius, dissimilarity, logSimilarity, type SimTile } from './similarity';

/** Novelty saturation length (spec §7): ν(d) = 1 - exp(-d / 3). */
const NOVELTY_SCALE = 3;

/**
 * Log-similarity magnitude below which a candidate counts as fully covered.
 *
 * The reference tests for exactly 0. Our determinant comes from an adjugate
 * expansion rather than a cofactor one, so an exact duplicate lands within a
 * few ulps of 0 instead of on it; the tolerance restores the intent.
 */
const COVERED_EPS = 1e-12;

/**
 * Selection state, indexed by absolute tile slot. `nearest[j]` is the best log
 * similarity from slot `j` to any representative chosen from `j`'s own voxel,
 * so adding a representative is one max-update pass rather than a rescan.
 */
type Selection = {
    /** Slots chosen for the voxel most recently passed to {@link selectVoxel}. */
    reps: Int32Array;
    repCount: number;
    /** Best log similarity to a chosen representative, -Infinity if none yet. */
    nearest: Float64Array;
    /** Chosen or fully covered — ineligible for further selection. */
    done: Uint8Array;
};

/**
 * Allocate selection state.
 *
 * @param capacity - Slot capacity; the driver sizes this to the whole scene.
 * @returns Empty state.
 */
const createSelection = (capacity: number): Selection => ({
    reps: new Int32Array(capacity),
    repCount: 0,
    nearest: new Float64Array(capacity),
    done: new Uint8Array(capacity)
});

/**
 * Saturating novelty (spec §7): 0 for a redundant candidate, approaching 1 for
 * an outlier, so an unboundedly distant splat cannot produce an unbounded
 * priority.
 *
 * @param d - Dissimilarity to the nearest representative.
 * @returns Novelty in [0, 1].
 */
const novelty = (d: number): number => (d === Infinity ? 1 : 1 - Math.exp(-d / NOVELTY_SCALE));

/**
 * Candidate priority (spec §7): novelty x representative weight. Among equally
 * novel candidates the more visible one wins; among equally visible ones the
 * more different one wins.
 *
 * @param tile - Filled tile.
 * @param state - Selection state whose voxel already has a representative.
 * @param slot - Candidate slot.
 * @returns The priority, or -Infinity if the candidate is already done.
 */
const priority = (tile: SimTile, state: Selection, slot: number): number => {
    if (state.done[slot]) return -Infinity;
    return novelty(dissimilarity(state.nearest[slot])) * tile.w[slot];
};

/**
 * First representative of a voxel (spec §5): the minimizer of
 * `S_i = M_i / ŵ_i`, where `M_i` is the ŵ-weighted mean dissimilarity from `i`
 * to every member. Dividing by the candidate's own weight is what separates
 * this from a plain weighted medoid — it biases toward a splat that is both
 * central and visually significant, and since every later novelty score is
 * measured against the survivors, this choice steers the whole sequence.
 *
 * O(count²) similarity evaluations, which is why the caller caps voxel
 * occupancy by subdividing (spec §3.3).
 *
 * Ties resolve to the larger {@link blendedRadius}, then the lower slot.
 *
 * @param tile - Filled tile.
 * @param from - First slot of the voxel.
 * @param count - Members in the voxel.
 * @returns The winning slot, or -1 when `count` is 0.
 */
const firstRepresentative = (tile: SimTile, from: number, count: number): number => {
    if (count <= 0) return -1;
    if (count === 1) return from;

    const { w } = tile;
    const end = from + count;
    let wSum = 0;
    for (let j = from; j < end; j++) wSum += w[j];

    let best = -1;
    let bestScore = Infinity;
    let bestRadius = -Infinity;

    for (let i = from; i < end; i++) {
        let acc = 0;
        for (let j = from; j < end; j++) {
            if (j === i) continue;
            const d = dissimilarity(logSimilarity(tile, i, j));
            if (d === Infinity) {
                acc = Infinity;
                break;
            }
            acc += w[j] * d;
        }
        const score = acc / wSum / w[i];

        // `best < 0` also covers the all-degenerate case, where every score is
        // +Infinity and the radius tiebreak decides.
        if (best < 0 || score < bestScore) {
            best = i;
            bestScore = score;
            bestRadius = blendedRadius(tile, i);
        } else if (score === bestScore) {
            const radius = blendedRadius(tile, i);
            if (radius > bestRadius) {
                best = i;
                bestRadius = radius;
            }
        }
    }

    return best;
};

/**
 * Add `slot` to the representative set and refresh every remaining candidate's
 * distance to the set.
 *
 * @param tile - Filled tile.
 * @param state - Selection state, mutated.
 * @param from - First slot of the voxel.
 * @param count - Members in the voxel.
 * @param slot - Slot to promote.
 */
const addRepresentative = (
    tile: SimTile,
    state: Selection,
    from: number,
    count: number,
    slot: number
): void => {
    const { reps, nearest, done } = state;
    reps[state.repCount++] = slot;
    done[slot] = 1;

    const end = from + count;
    for (let j = from; j < end; j++) {
        if (done[j]) continue;
        const l = logSimilarity(tile, j, slot);
        if (l > nearest[j]) nearest[j] = l;
        // Fully covered by an existing representative: keeping it would spend a
        // slot on a duplicate, which is why identical input can finish under
        // target.
        if (nearest[j] > -COVERED_EPS) done[j] = 1;
    }
};

/**
 * Highest-priority remaining candidate in a voxel (spec §7), ties broken by
 * {@link blendedRadius} then lower slot — the same order §5 uses.
 *
 * @param tile - Filled tile.
 * @param state - Selection state.
 * @param from - First slot of the voxel.
 * @param count - Members in the voxel.
 * @returns The best slot, or -1 when nothing is eligible.
 */
const bestCandidate = (
    tile: SimTile,
    state: Selection,
    from: number,
    count: number
): number => {
    let best = -1;
    let bestPriority = -Infinity;
    let bestRadius = -Infinity;

    const end = from + count;
    for (let i = from; i < end; i++) {
        if (state.done[i]) continue;
        const p = priority(tile, state, i);
        if (p > bestPriority) {
            best = i;
            bestPriority = p;
            bestRadius = blendedRadius(tile, i);
        } else if (p === bestPriority) {
            const radius = blendedRadius(tile, i);
            if (radius > bestRadius) {
                best = i;
                bestRadius = radius;
            }
        }
    }

    return best;
};

/**
 * Select a voxel's survivors: the §5 representative plus up to `quota` §7
 * detail representatives.
 *
 * Fewer than `1 + quota` come back when candidates run out or become fully
 * covered; the caller's global completion pass can then reuse the same state to
 * spend the leftover budget elsewhere.
 *
 * @param tile - Filled tile.
 * @param from - First slot of the voxel.
 * @param count - Members in the voxel.
 * @param quota - Additional representatives allowed beyond the first.
 * @param state - State from {@link createSelection}, reset over this range.
 * @returns `state.reps[0..repCount)`, the chosen slots.
 */
const selectVoxel = (
    tile: SimTile,
    from: number,
    count: number,
    quota: number,
    state: Selection
): Selection => {
    if (from + count > state.done.length) {
        throw new Error(`selection state too small: need ${from + count}, have ${state.done.length}`);
    }

    state.repCount = 0;
    state.done.fill(0, from, from + count);
    state.nearest.fill(-Infinity, from, from + count);

    const first = firstRepresentative(tile, from, count);
    if (first < 0) return state;
    addRepresentative(tile, state, from, count, first);

    for (let k = 0; k < quota; k++) {
        const next = bestCandidate(tile, state, from, count);
        if (next < 0) break;
        addRepresentative(tile, state, from, count, next);
    }

    return state;
};

export {
    NOVELTY_SCALE,
    COVERED_EPS,
    createSelection,
    novelty,
    priority,
    firstRepresentative,
    addRepresentative,
    bestCandidate,
    selectVoxel,
    type Selection
};
