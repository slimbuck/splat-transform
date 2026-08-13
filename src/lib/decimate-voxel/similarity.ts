/**
 * Gaussian + RGB similarity (spec §4), with the tile decode every pass over a
 * voxel shares. Selection policy built on it lives in ./representatives.
 *
 * This is the metric the voxel decimator selects and assigns with, and it is
 * not a position distance: two splats are close only when their centres are
 * close *relative to their own scales*, their covariances agree in magnitude
 * and orientation, and their DC colours match. That is why a voxel of large
 * blurry background splats collapses hard while a voxel of small detailed ones
 * keeps more survivors at the same occupancy.
 *
 * Engine-free and GPU-type-free, like {@link ./voxel-grid}: the arithmetic here
 * is the ground truth the WGSL segmented kernels are asserted against. Work is
 * organized as a *tile* — one contiguous run of splats, which is what a voxel
 * becomes after the binning sort — because both the score (§5) and later the
 * novelty pass (§7) are O(n²) within a voxel and want their inputs decoded
 * once, not per pair.
 */

import { alphaDecode, det3, quatToRotmat, sigmaFromRotVar, type SplatView } from '../decimate/moment-match';
import { SH_C0 } from '../value-transforms';

/** Representative-weight floor (spec §5, ε_w). */
const EPS_W = 1e-12;

/**
 * Decoded per-splat similarity inputs for one contiguous run of splats.
 *
 * Slot order is the caller's; the voxel's own members must occupy the leading
 * slots, so a neighbourhood can be appended after them (spec §8) without
 * disturbing the per-voxel passes.
 */
type SimTile = {
    /** Filled slot count. */
    count: number;
    /** Row-major covariance, 9 per slot. */
    sig: Float64Array;
    /** Centre, 3 per slot. */
    mu: Float64Array;
    /** Display RGB clamped to [0,1], 3 per slot — the colour term's domain. */
    rgb: Float64Array;
    /** log|Σ|, or -Infinity where the covariance is degenerate. */
    logDet: Float64Array;
    /** Representative weight ŵ = max(α·max(s), {@link EPS_W}), 1 per slot. */
    w: Float64Array;
};

/**
 * Allocate a tile.
 *
 * @param capacity - Slots to make room for; {@link fillSimTile} grows as needed.
 * @returns An empty tile.
 */
const createSimTile = (capacity: number): SimTile => ({
    count: 0,
    sig: new Float64Array(capacity * 9),
    mu: new Float64Array(capacity * 3),
    rgb: new Float64Array(capacity * 3),
    logDet: new Float64Array(capacity),
    w: new Float64Array(capacity)
});

/**
 * Decode `count` splats into `tile`, growing it if it is too small.
 *
 * The DC coefficients are converted to display RGB and clamped to [0,1], which
 * is the domain the reference colour term is calibrated for: on raw SH
 * coefficients the same squared distance is ~12x larger and would swamp the
 * Gaussian term.
 *
 * @param view - Splat columns.
 * @param indices - Global splat index per slot.
 * @param count - Number of slots to fill.
 * @param tile - Tile to fill, from {@link createSimTile}.
 * @returns `tile`, or a larger replacement holding the same data.
 */
const fillSimTile = (
    view: SplatView,
    indices: ArrayLike<number>,
    count: number,
    tile: SimTile
): SimTile => {
    const t = count > tile.logDet.length ? createSimTile(count) : tile;
    const { pos, geo, color, colorDim } = view;
    const { sig, mu, rgb, logDet, w } = t;

    for (let slot = 0; slot < count; slot++) {
        const i = indices[slot];
        const i8 = i * 8;
        const o9 = slot * 9;
        const o3 = slot * 3;

        let qw = geo[i8], qx = geo[i8 + 1], qy = geo[i8 + 2], qz = geo[i8 + 3];
        const qn = 1 / Math.max(Math.hypot(qw, qx, qy, qz), 1e-12);
        qw *= qn; qx *= qn; qy *= qn; qz *= qn;
        const sx = Math.max(Math.exp(geo[i8 + 4]), 1e-12);
        const sy = Math.max(Math.exp(geo[i8 + 5]), 1e-12);
        const sz = Math.max(Math.exp(geo[i8 + 6]), 1e-12);

        quatToRotmat(qw, qx, qy, qz, sig, o9);
        sigmaFromRotVar(sig, o9, sx * sx, sy * sy, sz * sz, sig, o9);

        const det = det3(sig, o9);
        logDet[slot] = det > 0 && Number.isFinite(det) ? Math.log(det) : -Infinity;

        mu[o3] = pos[i * 3];
        mu[o3 + 1] = pos[i * 3 + 1];
        mu[o3 + 2] = pos[i * 3 + 2];

        const c = i * colorDim;
        rgb[o3] = Math.min(1, Math.max(0, 0.5 + color[c] * SH_C0));
        rgb[o3 + 1] = Math.min(1, Math.max(0, 0.5 + color[c + 1] * SH_C0));
        rgb[o3 + 2] = Math.min(1, Math.max(0, 0.5 + color[c + 2] * SH_C0));

        // §2.2: representative weight is opacity x longest axis, deliberately
        // not the area-based merge weight — it ranks candidates, it does not
        // conserve mass.
        w[slot] = Math.max(alphaDecode(geo[i8 + 7]) * Math.max(sx, sy, sz), EPS_W);
    }

    t.count = count;
    return t;
};

/**
 * Log similarity L = log B + log C (spec §4): the Bhattacharyya log coefficient
 * of the two Gaussians, clamped to at most 0, plus the negative squared RGB
 * distance.
 *
 * Returns -Infinity for degenerate input — a non-positive or non-finite
 * determinant on either splat or on their mean covariance. The reference relies
 * on the same guard rather than regularizing the diagonal, so no EPS_COV is
 * added here; scales are floored at 1e-12 on decode, which keeps well-formed
 * data far away from the guard.
 *
 * @param tile - Filled tile.
 * @param a - First slot.
 * @param b - Second slot.
 * @returns L in (-Infinity, 0].
 */
const logSimilarity = (tile: SimTile, a: number, b: number): number => {
    const { sig, mu, rgb, logDet } = tile;

    const la = logDet[a];
    const lb = logDet[b];
    if (la === -Infinity || lb === -Infinity) return -Infinity;

    const ao = a * 9;
    const bo = b * 9;

    // Mean covariance (symmetric, so six unique components).
    const m00 = 0.5 * (sig[ao] + sig[bo]);
    const m01 = 0.5 * (sig[ao + 1] + sig[bo + 1]);
    const m02 = 0.5 * (sig[ao + 2] + sig[bo + 2]);
    const m11 = 0.5 * (sig[ao + 4] + sig[bo + 4]);
    const m12 = 0.5 * (sig[ao + 5] + sig[bo + 5]);
    const m22 = 0.5 * (sig[ao + 8] + sig[bo + 8]);

    // Adjugate, reused for both the determinant and the quadratic form.
    const c00 = m11 * m22 - m12 * m12;
    const c01 = m02 * m12 - m01 * m22;
    const c02 = m01 * m12 - m02 * m11;
    const c11 = m00 * m22 - m02 * m02;
    const c12 = m01 * m02 - m00 * m12;
    const c22 = m00 * m11 - m01 * m01;

    const detM = m00 * c00 + m01 * c01 + m02 * c02;
    if (!(detM > 0) || !Number.isFinite(detM)) return -Infinity;

    const a3 = a * 3;
    const b3 = b * 3;
    const dx = mu[a3] - mu[b3];
    const dy = mu[a3 + 1] - mu[b3 + 1];
    const dz = mu[a3 + 2] - mu[b3 + 2];

    // Δμᵀ Σ̄⁻¹ Δμ, via adj(Σ̄)/|Σ̄|.
    const quad = (
        dx * dx * c00 + dy * dy * c11 + dz * dz * c22 +
        2 * (dx * dy * c01 + dx * dz * c02 + dy * dz * c12)
    ) / detM;

    const logB = 0.25 * la + 0.25 * lb - 0.5 * Math.log(detM) - 0.125 * quad;
    if (!Number.isFinite(logB)) return -Infinity;

    const dr = rgb[a3] - rgb[b3];
    const dg = rgb[a3 + 1] - rgb[b3 + 1];
    const db = rgb[a3 + 2] - rgb[b3 + 2];

    return Math.min(logB, 0) - (dr * dr + dg * dg + db * db);
};

/**
 * Dissimilarity from a log similarity (spec §4).
 *
 * @param logSim - Value from {@link logSimilarity}.
 * @returns max(0, -L), or +Infinity for degenerate pairs.
 */
const dissimilarity = (logSim: number): number => (logSim === -Infinity ? Infinity : Math.max(0, -logSim));

/**
 * Blended 3σ radius (spec §2.1): the geometric mean of the volume-equivalent
 * and RMS radii of the axis-aligned 3σ box. Used only to break score ties
 * deterministically, with a milder bias toward elongated splats than the
 * longest axis would give.
 *
 * @param tile - Filled tile.
 * @param slot - Slot index.
 * @returns The blended radius.
 */
const blendedRadius = (tile: SimTile, slot: number): number => {
    const o = slot * 9;
    const ex = 3 * Math.sqrt(Math.max(tile.sig[o], 0));
    const ey = 3 * Math.sqrt(Math.max(tile.sig[o + 4], 0));
    const ez = 3 * Math.sqrt(Math.max(tile.sig[o + 8], 0));
    const rv = Math.cbrt(ex * ey * ez);
    const rrms = Math.sqrt((ex * ex + ey * ey + ez * ez) / 3);
    return Math.sqrt(rv * rrms);
};

export {
    EPS_W,
    createSimTile,
    fillSimTile,
    logSimilarity,
    dissimilarity,
    blendedRadius,
    type SimTile
};
