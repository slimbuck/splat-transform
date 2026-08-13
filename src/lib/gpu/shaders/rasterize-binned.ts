/**
 * Binned rasterize shader. Each workgroup handles one tile and only walks
 * the splats that have been pre-binned into it (tile-bin pre-pass on CPU
 * or GPU). Replaces the "walk all splats per pixel" loop in a non-binned
 * rasterizer with "walk this tile's slice", which is the asymptotic
 * fix for performance at high splat counts.
 *
 * The slice is stored in two buffers:
 *   - `tileOffsets[T + 1]` — exclusive prefix sum: tile T's slice is
 *     `tileData[tileOffsets[T] .. tileOffsets[T + 1])`.
 *   - `tileData[]` — splat indices, grouped by tile, depth-sorted within
 *     each tile (the orchestrator's CPU pre-sort + stable per-splat
 *     binning produces this layout for free).
 *
 * Projection-mode variation: `PROJECTION_EQUIRECT` wraps the per-pixel
 * `dx = px - splat.x` into `[-W/2, W/2]` so a tile on the opposite side
 * of the ±π longitude seam evaluates against the splat's nearer copy.
 * Without the flag the raw delta is used.
 *
 * @returns WGSL source for the binned-rasterize compute shader.
 */
const rasterizeBinnedWgsl = () => /* wgsl */`
#include "uniformsStruct"
#include "constants"

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> projected: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> runningState: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> tileOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> sortedSplatIndices: array<u32>;

#ifdef FRAG_STATS
// PROTOTYPE (ST_FRAG_STATS): measured fragment cost, to compare saturation
// strategies without trusting an analytical footprint estimate. Three counters
// mirror the loop's early-out ladder: candidates considered, fragments inside
// the footprint, fragments actually blended. The middle one is what a larger
// truncation radius directly inflates.
@group(0) @binding(5) var<storage, read_write> fragStats: array<atomic<u32>>;

// WGSL has no atomic u64 and a full-frame fragment count overflows u32 (a
// 1280x720 frame over a few thousand overlapping splats per pixel passes 2^32),
// so each counter is a (lo, hi) pair. atomicAdd returns the pre-add value,
// which is what makes the carry detectable without a lock.
fn statAdd(slot: u32, d: u32) {
    if (d == 0u) { return; }
    let old = atomicAdd(&fragStats[slot * 2u], d);
    if (old > 0xffffffffu - d) { atomicAdd(&fragStats[slot * 2u + 1u], 1u); }
}
#endif

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(workgroup_id) wgId: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    if (wgId.x >= uniforms.groupTilesX || wgId.y >= uniforms.groupTilesY) { return; }

    let tileIdx = wgId.y * uniforms.groupTilesX + wgId.x;
    let sliceStart = tileOffsets[tileIdx];
    let sliceEnd = tileOffsets[tileIdx + 1u];

    let localPixelX = wgId.x * TILE_SIZE + lid.x;
    let localPixelY = wgId.y * TILE_SIZE + lid.y;
    let groupPixelW = uniforms.groupTilesX * TILE_SIZE;

    let imagePixelX = uniforms.groupPixelOriginX + localPixelX;
    let imagePixelY = uniforms.groupPixelOriginY + localPixelY;
    if (imagePixelX >= uniforms.imageWidth || imagePixelY >= uniforms.imageHeight) { return; }

    let pixelIdx = localPixelY * groupPixelW + localPixelX;
    var state = runningState[pixelIdx];
    var color = state.rgb;
    var T = state.a;

    if (T < MIN_TRANSMITTANCE) { return; }

    let px = f32(imagePixelX) + 0.5;
    let py = f32(imagePixelY) + 0.5;

#ifdef PROJECTION_EQUIRECT
    let imgWf2 = f32(uniforms.imageWidth);
    let halfImgW = imgWf2 * 0.5;
#endif
#ifdef FRAG_STATS
    // Accumulated per invocation and flushed once at the end: an atomicAdd per
    // fragment would serialise the whole loop and make the timing meaningless.
    var nCand = 0u;
    var nBox = 0u;
    var nBlend = 0u;
#endif
    for (var i: u32 = sliceStart; i < sliceEnd; i = i + 1u) {
        if (T < MIN_TRANSMITTANCE) { break; }
#ifdef FRAG_STATS
        nCand = nCand + 1u;
#endif
        let splatIdx = sortedSplatIndices[i];
        let v0 = projected[splatIdx * 3u + 0u];
#ifdef PROJECTION_EQUIRECT
        // Equirect: a splat near the ±π longitude seam is tile-binned on
        // both sides of the image. Wrap dx into [-W/2, W/2] so a tile on
        // the opposite side of the seam pulls the splat's footprint from
        // the correct (nearer) copy.
        var dx = px - v0.x;
        if (dx > halfImgW) { dx = dx - imgWf2; }
        else if (dx < -halfImgW) { dx = dx + imgWf2; }
#else
        let dx = px - v0.x;
#endif
        let dy = py - v0.y;
        let r = v0.z;
        if (r <= 0.0 || abs(dx) > r || abs(dy) > r) { continue; }
#ifdef FRAG_STATS
        nBox = nBox + 1u;
#endif
        let v1 = projected[splatIdx * 3u + 1u];
        let power = -0.5 * (v1.x * dx * dx + 2.0 * v1.y * dx * dy + v1.z * dy * dy);
        if (power > 0.0) { continue; }
        // Subtract GAUSSIAN_FLOOR so each splat's alpha reaches 0 exactly
        // at the 3σ truncation radius instead of clipping at ~1.1% —
        // eliminates faint ring artifacts at splat edges. Matches the
        // PlayCanvas engine.
        // Over-unity splats (v0.w > 1) use Spark's smooth over-unity profile:
        //   opacity(x) = exp(-0.5 * D * (max(0, |x| - (D-1)))^2)
        // a unit-peak plateau of radius (D-1) joined C1-smoothly to a
        // Gaussian whose slope steepens by D. Kerbl et al.'s
        // min(1, D*exp(-x^2/2)) has the same plateau but meets it at a
        // nonzero slope, leaving a visible corner at the saturation ring.
        // v1.w carries D * radiusFade * dofAlphaScale, so dividing recovers
        // the modulation alone (the profile already peaks at 1).
        let ampD = v0.w;
        var prof = exp(power);
        var amp = v1.w;
        if (ampD > 1.0) {
            let xr = sqrt(max(0.0, -2.0 * power));
            let sh = max(0.0, xr - (ampD - 1.0));
            prof = exp(-0.5 * ampD * sh * sh);
            amp = v1.w / ampD;
        }
        let alpha = min(OPACITY_CAP, amp * max(0.0, prof - GAUSSIAN_FLOOR));
        if (alpha < MIN_ALPHA) { continue; }
#ifdef FRAG_STATS
        nBlend = nBlend + 1u;
#endif
        let weight = T * alpha;
        let v2 = projected[splatIdx * 3u + 2u];
        color = color + weight * v2.rgb;
        T = T * (1.0 - alpha);
    }

    runningState[pixelIdx] = vec4<f32>(color, T);

#ifdef FRAG_STATS
    statAdd(0u, nCand);
    statAdd(1u, nBox);
    statAdd(2u, nBlend);
#endif
}
`;

export { rasterizeBinnedWgsl };
