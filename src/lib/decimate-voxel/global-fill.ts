/**
 * Spending the leftover budget across voxels (spec §7 stage 2).
 *
 * The reference does this sequentially: take the highest-priority candidate
 * anywhere, refresh its voxel, repeat. That is not a rare tail — `fitDims`
 * guarantees `24G <= T`, and on a surface-like scene most base cells are empty,
 * so `24G` undershoots the target badly and this pass allocates the majority of
 * the output.
 *
 * It does not have to be sequential. Adding a representative can only lower the
 * remaining candidates' priorities, so each voxel's own sequence of picks is
 * non-increasing, and the global greedy order is just a merge of `G`
 * non-increasing sequences. A merge of sorted sequences taking `K` elements is
 * exactly a threshold: there is a single priority τ where every voxel takes all
 * of its picks at or above it. So the whole pass reduces to "compute each
 * voxel's full pick sequence, then find τ" — which is O(n) and parallel, rather
 * than K round trips.
 *
 * Selecting τ uses the fact that the IEEE-754 bit pattern of a non-negative
 * float is monotonic in its value: a 65,536-bucket histogram over the high half
 * of the bits brackets τ in one pass, and only the straddling bucket needs
 * exact ordering.
 */

/** Priority stamped on each voxel's first (spec §5) pick, so it always survives. */
const ALWAYS_TAKE = Infinity;

type GlobalFillInput = {
    /** Voxel count. */
    voxelCount: number;
    /** Slot offset of each voxel's pick run, length `voxelCount + 1`. */
    runStart: ArrayLike<number>;
    /** Picks available per voxel; run `g` holds `pickCount[g]` valid entries. */
    pickCount: ArrayLike<number>;
    /** Priority of each pick, indexed like the runs. */
    pickPriority: ArrayLike<number>;
    /** Picks each voxel is entitled to before the global pass (1 + its quota). */
    localTake: ArrayLike<number>;
    /** Total output budget T. */
    target: number;
};

/**
 * Decide how many picks each voxel keeps.
 *
 * Every voxel first keeps its local entitlement — capped by what it actually has,
 * which is how the coverage floor survives — and the remainder of the budget goes
 * to the globally highest-priority picks left over.
 *
 * @param input - Per-voxel pick sequences and the budget.
 * @returns Picks to keep per voxel, and the total.
 */
const globalFill = (input: GlobalFillInput): { take: Int32Array; total: number } => {
    const { voxelCount, runStart, pickCount, pickPriority, localTake, target } = input;

    const take = new Int32Array(voxelCount);
    let total = 0;
    for (let g = 0; g < voxelCount; g++) {
        take[g] = Math.min(pickCount[g], Math.max(0, localTake[g]));
        total += take[g];
    }

    let remaining = target - total;
    if (remaining <= 0) return { take, total };

    // Histogram the tail priorities by the high 16 bits of their float pattern.
    const BUCKETS = 1 << 16;
    const hist = new Int32Array(BUCKETS);
    const bits = new Float32Array(1);
    const asU32 = new Uint32Array(bits.buffer);
    const bucketOf = (priority: number): number => {
        if (!(priority > 0)) return 0;
        if (priority === ALWAYS_TAKE) return BUCKETS - 1;
        bits[0] = priority;
        return asU32[0] >>> 16;
    };

    let tailTotal = 0;
    for (let g = 0; g < voxelCount; g++) {
        const from = runStart[g];
        for (let k = take[g]; k < pickCount[g]; k++) {
            hist[bucketOf(pickPriority[from + k])]++;
            tailTotal++;
        }
    }
    if (tailTotal === 0) return { take, total };
    if (remaining >= tailTotal) {
        // Everything left fits; the budget simply cannot be filled.
        for (let g = 0; g < voxelCount; g++) {
            total += pickCount[g] - take[g];
            take[g] = pickCount[g];
        }
        return { take, total };
    }

    // Walk buckets from the top until the budget is met; that bucket straddles τ.
    let bucket = BUCKETS - 1;
    let above = 0;
    for (; bucket >= 0; bucket--) {
        if (above + hist[bucket] >= remaining) break;
        above += hist[bucket];
    }

    // Take every tail pick above the straddling bucket outright. Because each
    // voxel's tail is non-increasing, its survivors are always a prefix.
    const straddlers: { g: number; k: number; priority: number }[] = [];
    for (let g = 0; g < voxelCount; g++) {
        const from = runStart[g];
        for (let k = take[g]; k < pickCount[g]; k++) {
            const priority = pickPriority[from + k];
            const b = bucketOf(priority);
            if (b > bucket) {
                take[g]++;
                total++;
                remaining--;
            } else if (b === bucket) {
                straddlers.push({ g, k, priority });
            }
        }
    }

    // Resolve the straddling bucket exactly. Ties fall to the lower voxel then
    // the earlier pick, so the result does not depend on the sort's stability.
    // Compared rather than subtracted so equal infinities cannot yield NaN.
    straddlers.sort((a, b) => {
        if (a.priority !== b.priority) return a.priority < b.priority ? 1 : -1;
        return (a.g - b.g) || (a.k - b.k);
    });
    for (const s of straddlers) {
        if (remaining === 0) break;
        // A voxel's picks must stay a prefix: only extend by exactly one past
        // what it already holds.
        if (s.k !== take[s.g]) continue;
        take[s.g]++;
        total++;
        remaining--;
    }

    return { take, total };
};

export { ALWAYS_TAKE, globalFill, type GlobalFillInput };
