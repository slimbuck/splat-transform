/**
 * Merge/cost math for chunk-native decimation, generalized to n-ary groups.
 *
 * Engine-free: this module is imported by worker tasks, so it must not import
 * DataTable or playcanvas (see the note at the top of workers/tasks.ts).
 *
 * The n = 2 path of {@link mergeGroup} is arithmetic-identical to the legacy
 * `momentMatch` in the pre-3.0 `data-table/decimate.ts` (area·α weighted,
 * law-of-total-variance covariance, mass-conserving opacity capped at 1) —
 * enforced by test/moment-match.test.mjs against a verbatim reference copy.
 */

const LOG2PI = Math.log(2 * Math.PI);

/** Covariance diagonal regularizer, matching legacy EPS_COV. */
const EPS_COV = 1e-8;

/**
 * What to do with merged mass that a unit-alpha Gaussian cannot carry. A merge
 * of n splats owes the summed `alpha * area` of all of them; when the merged
 * footprint is too small to carry it at `alpha <= 1`, the excess has to go
 * somewhere, and this is that choice. Orthogonal to how the group was chosen,
 * so it applies identically to every allocator (uniform, adaptive, voxel).
 *
 * - `none`   — discard it. `min(1, W/area)`. The shipped behaviour: cheapest,
 *              and the cause of the darkening and gaps at coarse levels.
 * - `alpha`  — raise the Gaussian's peak above 1 and let the renderer draw the
 *              over-unity profile. Costs FEWER fragments than `none`, because a
 *              more opaque splat terminates rays sooner.
 * - `scale`  — grow the footprint until the mass fits at `alpha <= 1`. Needs no
 *              format or renderer change, but costs 2-3x the fragments and
 *              softens detail, because coverage is bought with geometry.
 *
 * `scale` and `none` are self-contained: any reader renders them correctly.
 * `alpha` is not — a consumer assuming 0..1 draws the scene `alphaMax` times too
 * transparent with no error, so it needs a format marker to be safe. The
 * encoding here is a range reinterpretation (`alpha = alphaMax * sigmoid`),
 * which is NOT what Spark reads; matching Spark's `[1,2]` half-float mapping is
 * deferred follow-up work.
 */
type CompensationMode = 'none' | 'alpha' | 'scale';

type Compensation = {
    mode: CompensationMode;
    /**
     * Opacity range when `mode` is `alpha`; 1 elsewhere. The stored value stays
     * a plain logit and only the range it maps onto changes, so at 1 every path
     * is bit-identical to the shipped behaviour.
     */
    alphaMax: number;
    /**
     * Opacity range the INPUT is written in, which is not the same question as
     * `alphaMax` (the range we write). A stock PLY stores plain 0..1 opacity, so
     * a sub-unity splat carries mass = alpha and this must be 1. Only a file we
     * ourselves wrote under `alpha` compensation is in an over-unity range, and
     * only then should this match the range it was written with.
     *
     * Kept separate because conflating the two silently over-reads every source
     * scene: decoding a plain 0..1 file at alphaMax 4 inflates mass by ×3.8 at
     * alpha 0.5 and ×6.6 at alpha 0.99 — non-uniformly, so it does not cancel in
     * normalised merge weights. Measured effect before this split: 97.5% of
     * survivors pushed over unity on bad-cloud and its sky losing 34 dB.
     */
    inputAlphaMax: number;
    /**
     * Scalar on the mass target before solving for the profile's shape
     * parameter; 1 = exact 2D-integral match. Integral match is not the right
     * invariant under alpha compositing — a wide opaque plateau occludes more
     * completely than the same mass spread softly, so front-to-back
     * accumulation saturates early and the frame brightens. Below 1 shrinks the
     * plateau to compensate. Only meaningful when `mode` is `alpha`.
     */
    massCal: number;
};

const DEFAULT_COMPENSATION: Compensation = { mode: 'none', alphaMax: 1, inputAlphaMax: 1, massCal: 1 };

// ---------- sigmoid / logit ----------

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

const logit = (p: number) => {
    p = Math.max(1e-7, Math.min(1 - 1e-7, p));
    return Math.log(p / (1 - p));
};

/**
 * Module-level rather than a parameter because the mass convention has to be
 * consistent across every consumer in a run — {@link alphaDecode} is reached
 * from the edge cost and the priority queue as well as the merge site, and an
 * inconsistency between "what a merge produces" and "what the cost predicts"
 * would be silent. Set once per worker task (cheap, idempotent) rather than
 * imported, keeping this module free of engine and option types.
 */
let active: Compensation = DEFAULT_COMPENSATION;

const setCompensation = (c?: Partial<Compensation>): void => {
    active = c ? { ...DEFAULT_COMPENSATION, ...c } : DEFAULT_COMPENSATION;
    if (active.mode !== 'alpha') active.alphaMax = 1;
    active.alphaMax = Math.max(1, active.alphaMax);
    // Independent of `mode`: the range the input was written in is a property of
    // the file, not of what we are about to do to it. A cascade level reading an
    // over-unity file still has to decode it correctly even under `none`.
    active.inputAlphaMax = Math.max(1, active.inputAlphaMax);
};

const getCompensation = (): Compensation => active;

/**
 * Integrated mass of the renderer's over-unity profile, in units where an
 * unclamped amplitude-A Gaussian carries A. For the smooth (Spark) profile
 * `exp(-D/2 * (max(0, r - (D-1)))^2)`:
 *
 *   I(D) = (D-1)^2/2 + 1/D + (D-1)/2 * sqrt(2*pi/D)
 *
 * I(1) = 1 exactly, and I grows ~D^2/2 — so unlike the clamped form
 * `min(1, A*exp(-r^2/2))`, whose mass ceiling grows only logarithmically
 * (2.39 at A = 4, 2.79 at A = 6), this profile can actually carry the mass a
 * heavy merge needs.
 *
 * @param D - Profile shape parameter.
 * @returns The mass I(D) the profile carries.
 */
const profileMass = (D: number): number => {
    if (D <= 1) return D;
    const p = D - 1;
    return (p * p) / 2 + 1 / D + (p / 2) * Math.sqrt((2 * Math.PI) / D);
};

/**
 * Inverse of {@link profileMass}: the shape parameter D whose profile carries
 * mass `m`. Bisection — monotone, a handful of iterations, and only ever runs
 * on merges that exceed unit mass.
 *
 * @param m - Target mass.
 * @returns The shape parameter D with I(D) = m.
 */
const profileParamForMass = (m: number): number => {
    if (m <= 1) return m;
    let lo = 1, hi = 2;
    while (profileMass(hi) < m && hi < 1e6) hi *= 2;
    for (let i = 0; i < 40; i++) {
        const mid = 0.5 * (lo + hi);
        if (profileMass(mid) < m) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
};

/**
 * Stored logit -> the MASS the splat carries (not the raw shape parameter).
 * Internal consumers — merge weights, edge costs — want mass, so an over-unity
 * splat must report I(D), otherwise a merged splat's weight silently
 * under-counts the energy it actually emits.
 *
 * Reads {@link Compensation.inputAlphaMax}, NOT `alphaMax`: this decodes what we
 * were given, while `alphaEncode` writes what we produce, and the two ranges are
 * only the same in the middle of a cascade whose earlier levels we wrote.
 *
 * @param stored - Stored opacity logit.
 * @returns The mass the splat carries.
 */
const alphaDecode = (stored: number) => profileMass(active.inputAlphaMax * sigmoid(stored));

/**
 * Mass -> stored logit. Solves for the shape parameter first, so what lands in
 * the file is D (what the renderer needs), while callers keep thinking in mass.
 *
 * @param mass - Mass to encode.
 * @returns The stored opacity logit.
 */
const alphaEncode = (mass: number) => {
    const D = profileParamForMass(mass * active.massCal);
    return logit(Math.max(0, Math.min(1, D / active.alphaMax)));
};

/**
 * True when the input and output opacity conventions differ, so a splat copied
 * verbatim out of the input would be misread on the way back in.
 *
 * Merged splats are always written through {@link alphaEncode} and so are
 * automatically in the output convention. Pass-through survivors are not: a
 * decimator that block-copies an untouched row is copying the INPUT's range into
 * a file declared to be in ours, which reads ~`alphaMax` times too bright.
 *
 * @returns Whether {@link convertStoredOpacity} is required.
 */
const needsOpacityConversion = (): boolean => active.alphaMax !== active.inputAlphaMax || active.massCal !== 1;

/**
 * Re-encode one stored opacity from the input convention into the output one.
 * Mathematically the identity when the two agree, but not bit-exact through
 * logit/sigmoid, so callers gate on {@link needsOpacityConversion} to leave the
 * unaffected paths byte-for-byte unchanged.
 *
 * @param stored - Stored opacity logit in the INPUT convention.
 * @returns Stored opacity logit in the OUTPUT convention.
 */
const convertStoredOpacity = (stored: number): number => alphaEncode(alphaDecode(stored));

const logAddExp = (a: number, b: number) => {
    if (a === -Infinity) return b;
    if (b === -Infinity) return a;
    const m = Math.max(a, b);
    return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
};

// ---------- PRNG (MC samples for the edge cost; seed 0 matches legacy) ----------

const mulberry32 = (seed: number) => {
    return () => {
        let t = (seed += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
};

const makeGaussianSamples = (n: number, seed: number): Float64Array[] => {
    const rand = mulberry32(seed >>> 0);
    const out: Float64Array[] = [];
    while (out.length < n) {
        const u1 = Math.max(rand(), 1e-12);
        const u2 = rand();
        const u3 = Math.max(rand(), 1e-12);
        const u4 = rand();
        const r1 = Math.sqrt(-2 * Math.log(u1));
        const t1 = 2 * Math.PI * u2;
        const r2 = Math.sqrt(-2 * Math.log(u3));
        const t2 = 2 * Math.PI * u4;
        out.push(new Float64Array([r1 * Math.cos(t1), r1 * Math.sin(t1), r2 * Math.cos(t2)]));
    }
    return out;
};

// ---------- 3x3 matrix helpers (row-major, 9 floats) ----------

type Mat9 = Float32Array | Float64Array;

const quatToRotmat = (qw: number, qx: number, qy: number, qz: number, out: Mat9, o: number) => {
    const xx = qx * qx, yy = qy * qy, zz = qz * qz;
    const wx = qw * qx, wy = qw * qy, wz = qw * qz;
    const xy = qx * qy, xz = qx * qz, yz = qy * qz;
    out[o] = 1 - 2 * (yy + zz);
    out[o + 1] = 2 * (xy - wz);
    out[o + 2] = 2 * (xz + wy);
    out[o + 3] = 2 * (xy + wz);
    out[o + 4] = 1 - 2 * (xx + zz);
    out[o + 5] = 2 * (yz - wx);
    out[o + 6] = 2 * (xz - wy);
    out[o + 7] = 2 * (yz + wx);
    out[o + 8] = 1 - 2 * (xx + yy);
};

const sigmaFromRotVar = (R: Mat9, r: number, vx: number, vy: number, vz: number, out: Mat9, o: number) => {
    const r00 = R[r], r01 = R[r + 1], r02 = R[r + 2];
    const r10 = R[r + 3], r11 = R[r + 4], r12 = R[r + 5];
    const r20 = R[r + 6], r21 = R[r + 7], r22 = R[r + 8];
    out[o] = r00 * r00 * vx + r01 * r01 * vy + r02 * r02 * vz;
    out[o + 1] = r00 * r10 * vx + r01 * r11 * vy + r02 * r12 * vz;
    out[o + 2] = r00 * r20 * vx + r01 * r21 * vy + r02 * r22 * vz;
    out[o + 3] = out[o + 1];
    out[o + 4] = r10 * r10 * vx + r11 * r11 * vy + r12 * r12 * vz;
    out[o + 5] = r10 * r20 * vx + r11 * r21 * vy + r12 * r22 * vz;
    out[o + 6] = out[o + 2];
    out[o + 7] = out[o + 5];
    out[o + 8] = r20 * r20 * vx + r21 * r21 * vy + r22 * r22 * vz;
};

const det3 = (A: Mat9, o: number) => {
    return (
        A[o] * (A[o + 4] * A[o + 8] - A[o + 5] * A[o + 7]) -
        A[o + 1] * (A[o + 3] * A[o + 8] - A[o + 5] * A[o + 6]) +
        A[o + 2] * (A[o + 3] * A[o + 7] - A[o + 4] * A[o + 6])
    );
};

const gaussLogpdfDiagrot = (
    x: number, y: number, z: number,
    mx: number, my: number, mz: number,
    R: Mat9, ro: number,
    invx: number, invy: number, invz: number, logdet: number
) => {
    const dx = x - mx, dy = y - my, dz = z - mz;
    const y0 = dx * R[ro] + dy * R[ro + 3] + dz * R[ro + 6];
    const y1 = dx * R[ro + 1] + dy * R[ro + 4] + dz * R[ro + 7];
    const y2 = dx * R[ro + 2] + dy * R[ro + 5] + dz * R[ro + 8];
    const quad = y0 * y0 * invx + y1 * y1 * invy + y2 * y2 * invz;
    return -0.5 * (3 * LOG2PI + logdet + quad);
};

// Jacobi eigendecomposition for 3x3 symmetric matrix; caller-provided scratch,
// eigenvalues land on A's diagonal, eigenvectors in V's columns.
const eigenSymmetric3x3 = (Ain: Float64Array, A: Float64Array, V: Float64Array) => {
    A.set(Ain);
    V[0] = 1; V[1] = 0; V[2] = 0;
    V[3] = 0; V[4] = 1; V[5] = 0;
    V[6] = 0; V[7] = 0; V[8] = 1;

    for (let iter = 0; iter < 24; iter++) {
        let p = 0, q = 1;
        let maxAbs = Math.abs(A[1]);
        if (Math.abs(A[2]) > maxAbs) {
            p = 0; q = 2; maxAbs = Math.abs(A[2]);
        }
        if (Math.abs(A[5]) > maxAbs) {
            p = 1; q = 2; maxAbs = Math.abs(A[5]);
        }
        if (maxAbs < 1e-12) break;

        const pp = 3 * p + p, qq = 3 * q + q, pq = 3 * p + q;
        const app = A[pp], aqq = A[qq], apq = A[pq];
        const tau = (aqq - app) / (2 * apq);
        const t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        for (let k = 0; k < 3; k++) {
            if (k === p || k === q) continue;
            const kp = 3 * k + p, kq = 3 * k + q;
            const pk = 3 * p + k, qk = 3 * q + k;
            const akp = A[kp], akq = A[kq];
            A[kp] = c * akp - s * akq;
            A[pk] = A[kp];
            A[kq] = s * akp + c * akq;
            A[qk] = A[kq];
        }
        A[pp] = c * c * app - 2 * s * c * apq + s * s * aqq;
        A[qq] = s * s * app + 2 * s * c * apq + c * c * aqq;
        A[pq] = 0; A[3 * q + p] = 0;

        for (let k = 0; k < 3; k++) {
            const kp = 3 * k + p, kq = 3 * k + q;
            const vkp = V[kp], vkq = V[kq];
            V[kp] = c * vkp - s * vkq;
            V[kq] = s * vkp + c * vkq;
        }
    }
};

const rotmatToQuat = (R: Float64Array, o: number, out: Float64Array, oo: number) => {
    const m00 = R[o], m11 = R[o + 4], m22 = R[o + 8];
    const tr = m00 + m11 + m22;
    let qw: number, qx: number, qy: number, qz: number;

    if (tr > 0) {
        const S = Math.sqrt(tr + 1) * 2;
        qw = 0.25 * S;
        qx = (R[o + 7] - R[o + 5]) / S;
        qy = (R[o + 2] - R[o + 6]) / S;
        qz = (R[o + 3] - R[o + 1]) / S;
    } else if (R[o] > R[o + 4] && R[o] > R[o + 8]) {
        const S = Math.sqrt(1 + R[o] - R[o + 4] - R[o + 8]) * 2;
        qw = (R[o + 7] - R[o + 5]) / S;
        qx = 0.25 * S;
        qy = (R[o + 1] + R[o + 3]) / S;
        qz = (R[o + 2] + R[o + 6]) / S;
    } else if (R[o + 4] > R[o + 8]) {
        const S = Math.sqrt(1 + R[o + 4] - R[o] - R[o + 8]) * 2;
        qw = (R[o + 2] - R[o + 6]) / S;
        qx = (R[o + 1] + R[o + 3]) / S;
        qy = 0.25 * S;
        qz = (R[o + 5] + R[o + 7]) / S;
    } else {
        const S = Math.sqrt(1 + R[o + 8] - R[o] - R[o + 4]) * 2;
        qw = (R[o + 3] - R[o + 1]) / S;
        qx = (R[o + 2] + R[o + 6]) / S;
        qy = (R[o + 5] + R[o + 7]) / S;
        qz = 0.25 * S;
    }

    const n = Math.hypot(qw, qx, qy, qz);
    const inv = 1 / Math.max(n, 1e-12);
    out[oo] = qw * inv;
    out[oo + 1] = qx * inv;
    out[oo + 2] = qy * inv;
    out[oo + 3] = qz * inv;
};

// ---------- ellipsoid area (Knud Thomsen p=1.6075) ----------

const ELLIPSOID_P = 1.6075;
const ellipsoidArea = (sx: number, sy: number, sz: number): number => {
    const a = Math.pow(sx * sy, ELLIPSOID_P);
    const b = Math.pow(sx * sz, ELLIPSOID_P);
    const c = Math.pow(sy * sz, ELLIPSOID_P);
    return 4 * Math.PI * Math.pow((a + b + c) / 3, 1 / ELLIPSOID_P);
};

// ---------- splat views ----------

/**
 * Column-tight view over a batch of splats, the working representation for
 * block processing. `pos` is 3 f32/splat (x, y, z); `geo` is 8 f32/splat in
 * geometric-layer order (rot_0..3, scale_0..2 log-space, opacity logit);
 * `color` is `colorDim` f32/splat (dc0..2 then f_rest coefficients).
 */
type SplatView = {
    pos: Float32Array;
    geo: Float32Array;
    color: Float32Array;
    colorDim: number;
};

/** Merged-splat output: pos 3, geo 8 (same encoding as SplatView), color colorDim. */
type MergedOut = {
    pos: Float64Array;
    geo: Float64Array;
    color: Float64Array;
};

/**
 * Reusable scratch shared by {@link mergeGroup} and the CPU edge cost.
 * Allocated once per pass, not per call — the merge loop runs hundreds of
 * millions of times at scale.
 */
type MergeScratch = {
    sigm: Float64Array;
    sigI: Float64Array;
    sigJ: Float64Array;
    rI: Float64Array;
    rJ: Float64Array;
    sig: Float64Array;
    rM: Float64Array;
    eigA: Float64Array;
    eigV: Float64Array;
    /** Per-member normalized weights (grown on demand; groups cap at ~4). */
    weights: Float64Array;
};

const createMergeScratch = (): MergeScratch => ({
    sigm: new Float64Array(9),
    sigI: new Float64Array(9),
    sigJ: new Float64Array(9),
    rI: new Float64Array(9),
    rJ: new Float64Array(9),
    sig: new Float64Array(9),
    rM: new Float64Array(9),
    eigA: new Float64Array(9),
    eigV: new Float64Array(9),
    weights: new Float64Array(8)
});

/**
 * Merge weight of one splat: area·α "ink" mass (+1e-30, the merge-path
 * epsilon; the cost-path cache uses +1e-12 — both match their legacy
 * counterparts exactly).
 * @param geo - Geometric-layer view (8 f32/splat).
 * @param i - Splat index.
 * @returns The merge weight.
 */
const splatMass = (geo: Float32Array, i: number): number => {
    const i8 = i * 8;
    const sx = Math.max(Math.exp(geo[i8 + 4]), 1e-12);
    const sy = Math.max(Math.exp(geo[i8 + 5]), 1e-12);
    const sz = Math.max(Math.exp(geo[i8 + 6]), 1e-12);
    return alphaDecode(geo[i8 + 7]) * ellipsoidArea(sx, sy, sz) + 1e-30;
};

/**
 * n-ary moment match: merge `count` splats into one Gaussian. Weights are
 * area·α; merged covariance is the weighted sum of (δδᵀ + Σₖ) (law of total
 * variance); opacity is mass-conserving capped at 1; color/SH is the weighted
 * average. For n = 2 this is arithmetic-identical to the legacy pairwise
 * `momentMatch`.
 *
 * @param view - Splat columns.
 * @param members - Indices of the splats to merge.
 * @param count - Number of members.
 * @param out - Output splat (geo encoded ready-to-store: quat, log scales, logit opacity).
 * @param scratch - Reusable scratch from {@link createMergeScratch}.
 */
const mergeGroup = (
    view: SplatView,
    members: ArrayLike<number>,
    count: number,
    out: MergedOut,
    scratch: MergeScratch
): void => {
    const { pos, geo, color, colorDim } = view;

    // Per-member normalized weights, computed once — splatMass is
    // transcendental-heavy and was previously re-evaluated per loop (and per
    // coefficient in the color loop); caching the identical values is
    // bit-exact.
    let weights = scratch.weights;
    if (count > weights.length) {
        weights = scratch.weights = new Float64Array(count);
    }
    let W = 0;
    for (let m = 0; m < count; m++) {
        const mass = splatMass(geo, members[m]);
        weights[m] = mass;
        W += mass;
    }
    for (let m = 0; m < count; m++) weights[m] /= W;

    // Merged mean (weighted).
    let mux = 0, muy = 0, muz = 0;
    for (let m = 0; m < count; m++) {
        const i = members[m];
        const p = weights[m];
        mux += p * pos[i * 3];
        muy += p * pos[i * 3 + 1];
        muz += p * pos[i * 3 + 2];
    }

    // Merged covariance: Σ pₖ (δₖδₖᵀ + Σₖ), accumulated member-by-member.
    const Sig = scratch.sig;
    Sig.fill(0);
    const SigI = scratch.sigI;
    const Ri = scratch.rI;
    for (let m = 0; m < count; m++) {
        const i = members[m];
        const i8 = i * 8;
        const p = weights[m];

        let qw = geo[i8], qx = geo[i8 + 1], qy = geo[i8 + 2], qz = geo[i8 + 3];
        const qn = 1 / Math.max(Math.hypot(qw, qx, qy, qz), 1e-12);
        qw *= qn; qx *= qn; qy *= qn; qz *= qn;
        const sx = Math.max(Math.exp(geo[i8 + 4]), 1e-12);
        const sy = Math.max(Math.exp(geo[i8 + 5]), 1e-12);
        const sz = Math.max(Math.exp(geo[i8 + 6]), 1e-12);

        quatToRotmat(qw, qx, qy, qz, Ri, 0);
        sigmaFromRotVar(Ri, 0, sx * sx, sy * sy, sz * sz, SigI, 0);

        const dx = pos[i * 3] - mux, dy = pos[i * 3 + 1] - muy, dz = pos[i * 3 + 2] - muz;
        Sig[0] += p * (dx * dx + SigI[0]);
        Sig[1] += p * (dx * dy + SigI[1]);
        Sig[2] += p * (dx * dz + SigI[2]);
        Sig[4] += p * (dy * dy + SigI[4]);
        Sig[5] += p * (dy * dz + SigI[5]);
        Sig[8] += p * (dz * dz + SigI[8]);
    }
    Sig[3] = Sig[1];
    Sig[6] = Sig[2];
    Sig[7] = Sig[5];
    Sig[0] += EPS_COV;
    Sig[4] += EPS_COV;
    Sig[8] += EPS_COV;

    // Eigendecompose → scales (√λ, descending) + right-handed rotation → quat.
    const eigA = scratch.eigA;
    const eigV = scratch.eigV;
    eigenSymmetric3x3(Sig, eigA, eigV);
    const vecs = eigV;

    const v0 = eigA[0], v1 = eigA[4], v2 = eigA[8];
    let o0: number, o1: number, o2: number;
    if (v0 >= v1) {
        if (v1 >= v2)      {
            o0 = 0; o1 = 1; o2 = 2;
        } else if (v0 >= v2) {
            o0 = 0; o1 = 2; o2 = 1;
        } else               {
            o0 = 2; o1 = 0; o2 = 1;
        }
    } else {
        if (v0 >= v2)      {
            o0 = 1; o1 = 0; o2 = 2;
        } else if (v1 >= v2) {
            o0 = 1; o1 = 2; o2 = 0;
        } else               {
            o0 = 2; o1 = 1; o2 = 0;
        }
    }
    const ev0 = Math.max(eigA[3 * o0 + o0], 1e-18);
    const ev1 = Math.max(eigA[3 * o1 + o1], 1e-18);
    const ev2 = Math.max(eigA[3 * o2 + o2], 1e-18);

    let s0 = Math.sqrt(ev0);
    let s1 = Math.sqrt(ev1);
    let s2 = Math.sqrt(ev2);

    // Compensation mode `scale`: grow the footprint until the owed mass fits at
    // the profile's ceiling, instead of letting the clamp below discard it.
    //
    // `ellipsoidArea` is homogeneous of degree 2 in the scales, so the factor is
    // exact rather than iterative: area(f*s) = f^2 * area(s). Scales grow
    // isotropically, preserving shape and orientation — which is precisely what
    // trades blur for coverage, since the gap is filled with geometry.
    //
    // Composes with `alphaMax` rather than replacing it: at the default 1 the
    // ceiling is 1 and this is pure inflation, while a higher `alphaMax` leaves
    // only the residual the peak cannot carry for the footprint to absorb.
    if (active.mode === 'scale') {
        const ceiling = profileMass(active.alphaMax);
        const area = Math.max(ellipsoidArea(s0, s1, s2), 1e-30);
        const needed = W / ceiling;
        if (needed > area) {
            const f = Math.sqrt(needed / area);
            s0 *= f; s1 *= f; s2 *= f;
        }
    }

    // Mass-conserving opacity, capped at the profile ceiling. Under `none` the
    // ceiling is 1 and this clamp discards mass whenever the merged footprint is
    // too small to carry it; splats that hit it land at the logit saturation
    // ceiling in the output, which is how the discarded-mass rate is measured
    // offline. Under `alpha` the ceiling rises and the excess is kept as peak.
    const alphaM = Math.min(profileMass(active.alphaMax), W / Math.max(ellipsoidArea(s0, s1, s2), 1e-30));

    const Rm = scratch.rM;
    Rm[0] = vecs[o0]; Rm[1] = vecs[o1]; Rm[2] = vecs[o2];
    Rm[3] = vecs[3 + o0]; Rm[4] = vecs[3 + o1]; Rm[5] = vecs[3 + o2];
    Rm[6] = vecs[6 + o0]; Rm[7] = vecs[6 + o1]; Rm[8] = vecs[6 + o2];
    if (det3(Rm, 0) < 0) {
        Rm[2] *= -1; Rm[5] *= -1; Rm[8] *= -1;
    }
    rotmatToQuat(Rm, 0, out.geo, 0);

    out.pos[0] = mux; out.pos[1] = muy; out.pos[2] = muz;
    out.geo[4] = Math.log(s0);
    out.geo[5] = Math.log(s1);
    out.geo[6] = Math.log(s2);
    out.geo[7] = alphaEncode(alphaM);

    // Color: weight-normalized (area·α weighted) average.
    for (let c = 0; c < colorDim; c++) {
        let acc = 0;
        for (let m = 0; m < count; m++) {
            acc += weights[m] * color[members[m] * colorDim + c];
        }
        out.color[c] = acc;
    }
};

export {
    EPS_COV,
    LOG2PI,
    setCompensation,
    getCompensation,
    DEFAULT_COMPENSATION,
    alphaDecode,
    alphaEncode,
    needsOpacityConversion,
    convertStoredOpacity,
    sigmoid,
    logit,
    logAddExp,
    makeGaussianSamples,
    quatToRotmat,
    sigmaFromRotVar,
    det3,
    gaussLogpdfDiagrot,
    eigenSymmetric3x3,
    rotmatToQuat,
    ellipsoidArea,
    splatMass,
    mergeGroup,
    createMergeScratch,
    type SplatView,
    type MergedOut,
    type MergeScratch,
    type Compensation,
    type CompensationMode
};
