/**
 * Shared WGSL for voxel decimation: the per-slot tile layout and the Gaussian +
 * RGB similarity of Splat-Simplify.md §4, interpolated into the select and
 * assign kernels.
 *
 * These functions read a storage binding named `tile` directly rather than
 * taking it as a pointer: storage-address-space pointer parameters are not base
 * WGSL. Every consumer must therefore bind its tile under that name.
 *
 * Mirrors `decimate-voxel/similarity.ts` — keep them in lockstep. The CPU side
 * is f64 and is the parity oracle; this is f32, so expect agreement to ~1e-5
 * absolute on the log similarity rather than bit equality.
 */

/** Per-slot floats in the tile buffer. */
const VOXEL_TILE_STRIDE = 14;

const voxelSimilarityWgsl = /* wgsl */`
// Tile layout, ${VOXEL_TILE_STRIDE} f32 per slot:
//   [0..2]   mean
//   [3..8]   covariance (xx, xy, xz, yy, yz, zz)
//   [9..11]  display RGB in [0,1]
//   [12]     representative weight w = alpha * max(scale)
//   [13]     log|covariance| = 2*(log sx + log sy + log sz), or the sentinel
const TILE_STRIDE: u32 = ${VOXEL_TILE_STRIDE}u;
const EPS_W: f32 = 1e-12;
const NOVELTY_SCALE: f32 = 3.0;
// Must match COVERED_EPS in decimate-voxel/representatives.ts: it is sized for
// this kernel's f32 noise floor on log|Sigma|, not for the CPU's f64.
const COVERED_EPS: f32 = 1e-4;

// WGSL has no -inf literal, so degeneracy is flagged with sentinels far below
// any real value and tested by comparison.
const LOG_DET_DEGENERATE: f32 = -1e30;
const LOG_SIM_DEGENERATE: f32 = -1e30;
const BIG_DISSIMILARITY: f32 = 1e30;

fn tileMean(slot: u32) -> vec3f {
    let o = slot * TILE_STRIDE;
    return vec3f(tile[o], tile[o + 1u], tile[o + 2u]);
}

fn tileWeight(slot: u32) -> f32 {
    return tile[slot * TILE_STRIDE + 12u];
}

// Blended 3-sigma radius (spec §2.1), the deterministic tiebreak. Only ever
// evaluated on an exact tie, so the pow/sqrt cost is off the hot path.
fn tileRadius(slot: u32) -> f32 {
    let o = slot * TILE_STRIDE;
    let ex = 3.0 * sqrt(max(tile[o + 3u], 0.0));
    let ey = 3.0 * sqrt(max(tile[o + 6u], 0.0));
    let ez = 3.0 * sqrt(max(tile[o + 8u], 0.0));
    let rv = pow(max(ex * ey * ez, 0.0), 1.0 / 3.0);
    let rrms = sqrt((ex * ex + ey * ey + ez * ez) / 3.0);
    return sqrt(rv * rrms);
}

// Log similarity L = min(log B, 0) + log C (spec §4).
//
// The Bhattacharyya term is evaluated on covariances rescaled by the mean
// covariance's trace. That is algebraically an identity — the determinant ratio
// is scale-invariant — but it keeps every determinant near 1. Without it, a
// scene with 1e-5 scales has |Sigma| ~ 1e-30 and the f32 determinant collapses
// into denormals.
fn logSimilarity(a: u32, b: u32) -> f32 {
    let ao = a * TILE_STRIDE;
    let bo = b * TILE_STRIDE;

    if (tile[ao + 13u] <= LOG_DET_DEGENERATE || tile[bo + 13u] <= LOG_DET_DEGENERATE) {
        return LOG_SIM_DEGENERATE;
    }

    // Mean covariance.
    let m0 = 0.5 * (tile[ao + 3u] + tile[bo + 3u]);
    let m1 = 0.5 * (tile[ao + 4u] + tile[bo + 4u]);
    let m2 = 0.5 * (tile[ao + 5u] + tile[bo + 5u]);
    let m3 = 0.5 * (tile[ao + 6u] + tile[bo + 6u]);
    let m4 = 0.5 * (tile[ao + 7u] + tile[bo + 7u]);
    let m5 = 0.5 * (tile[ao + 8u] + tile[bo + 8u]);

    // Rescale so the mean covariance has unit mean diagonal.
    let s2 = max((m0 + m3 + m5) / 3.0, 1e-30);
    let inv = 1.0 / s2;

    let n0 = m0 * inv; let n1 = m1 * inv; let n2 = m2 * inv;
    let n3 = m3 * inv; let n4 = m4 * inv; let n5 = m5 * inv;

    // Adjugate of the normalized mean covariance, reused for det and quad form.
    let c00 = n3 * n5 - n4 * n4;
    let c01 = n2 * n4 - n1 * n5;
    let c02 = n1 * n4 - n2 * n3;
    let c11 = n0 * n5 - n2 * n2;
    let c12 = n1 * n2 - n0 * n4;
    let c22 = n0 * n3 - n1 * n1;
    let detM = n0 * c00 + n1 * c01 + n2 * c02;
    if (!(detM > 0.0)) { return LOG_SIM_DEGENERATE; }

    // Scaling a 3x3 by 1/s2 divides its determinant by s2^3, so the stored logs
    // shift by -3 log s2. The three shifts cancel exactly in the sum below,
    // which is why this rescaling changes nothing but the exponent range.
    let logS2 = log(s2);
    let la = tile[ao + 13u] - 3.0 * logS2;
    let lb = tile[bo + 13u] - 3.0 * logS2;

    let d = (tileMean(a) - tileMean(b)) / sqrt(s2);
    let quad = (
        c00 * d.x * d.x + c11 * d.y * d.y + c22 * d.z * d.z +
        2.0 * (c01 * d.x * d.y + c02 * d.x * d.z + c12 * d.y * d.z)
    ) / detM;

    let logB = 0.25 * la + 0.25 * lb - 0.5 * log(detM) - 0.125 * quad;

    let dr = tile[ao + 9u] - tile[bo + 9u];
    let dg = tile[ao + 10u] - tile[bo + 10u];
    let db = tile[ao + 11u] - tile[bo + 11u];

    return min(logB, 0.0) - (dr * dr + dg * dg + db * db);
}

// Dissimilarity d = max(0, -L); degenerate pairs report a large finite value
// rather than an infinity WGSL cannot express.
fn dissimilarity(logSim: f32) -> f32 {
    return select(max(0.0, -logSim), BIG_DISSIMILARITY, logSim <= LOG_SIM_DEGENERATE);
}

// Saturating novelty (spec §7).
fn novelty(d: f32) -> f32 {
    return 1.0 - exp(-d / NOVELTY_SCALE);
}
`;

export { VOXEL_TILE_STRIDE, voxelSimilarityWgsl };
