import {
    BUFFERUSAGE_COPY_DST,
    BUFFERUSAGE_COPY_SRC,
    GraphicsDevice,
    StorageBuffer
} from 'playcanvas';

import { makeKernel, type Kernel } from './compute-kernel';
import { getCompensation, type SplatView } from '../decimate/moment-match';
import { globalFill } from '../decimate-voxel/global-fill';
import { type VoxelPartition } from '../decimate-voxel/partition';
import { SCALE_FLOOR_RATIO } from '../decimate-voxel/similarity';
import { VOXEL_TILE_STRIDE, voxelSimilarityWgsl } from './shaders/chunks/voxel-similarity';

/** Threads per workgroup; the select kernel runs one workgroup per voxel. */
const WORKGROUP = 64;

/** Floats per leaf in the packed leaf buffer: cell (bitcast), lo3, hi3, pad. */
const LEAF_STRIDE = 8;

/**
 * WGSL: decode splats into the similarity tile, in partition order.
 *
 * One thread per slot, and arithmetically identical to the CPU decode including
 * the EPS_COV diagonal regularizer — which is also what keeps the f32
 * determinant out of the denormals a zero-thickness splat would otherwise
 * produce.
 *
 * @param inputAlphaMax - Opacity range the INPUT file is written in (1 for a
 * stock PLY). Decoding, not encoding, so this is not the compensation ceiling.
 * @returns WGSL source.
 */
const decodeWgsl = (inputAlphaMax: number) => /* wgsl */`
struct Uniforms {
    slotBase: u32,
    slotCount: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> geo: array<f32>;
@group(0) @binding(3) var<storage, read> dc: array<f32>;
@group(0) @binding(4) var<storage, read> order: array<u32>;
@group(0) @binding(5) var<storage, read_write> tile: array<f32>;

const TILE_STRIDE: u32 = ${VOXEL_TILE_STRIDE}u;
const ALPHA_MAX: f32 = ${inputAlphaMax.toFixed(6)};
const SH_C0: f32 = 0.28209479177387814;
const EPS_W: f32 = 1e-12;
const EPS_COV: f32 = 1e-8;
const SCALE_FLOOR_RATIO: f32 = ${SCALE_FLOOR_RATIO};
const TWO_PI: f32 = 6.283185307179586;
const LOG_DET_DEGENERATE: f32 = -1e30;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Integrated mass of the renderer's over-unity profile (matches profileMass).
fn profileMass(D: f32) -> f32 {
    if (D <= 1.0) { return D; }
    let p = D - 1.0;
    return p * p * 0.5 + 1.0 / D + (p * 0.5) * sqrt(TWO_PI / D);
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= uniforms.slotCount) { return; }
    let slot = uniforms.slotBase + gid.x;
    let i = order[slot];

    let g8 = i * 8u;
    var q = vec4f(geo[g8], geo[g8 + 1u], geo[g8 + 2u], geo[g8 + 3u]);
    q = q / max(length(q), 1e-12);

    let rx = max(exp(geo[g8 + 4u]), 1e-12);
    let ry = max(exp(geo[g8 + 5u]), 1e-12);
    let rz = max(exp(geo[g8 + 6u]), 1e-12);

    // Ratio floor on the scales, matching the CPU decode — this is what keeps
    // the covariance condition number inside what f32 can carry through the
    // quadratic form. See SCALE_FLOOR_RATIO in decimate-voxel/similarity.ts.
    let sFloor = max(rx, max(ry, rz)) * SCALE_FLOOR_RATIO;
    let sx = max(rx, sFloor);
    let sy = max(ry, sFloor);
    let sz = max(rz, sFloor);

    // Rotation matrix from the (w, x, y, z) quaternion.
    let xx = q.y * q.y; let yy = q.z * q.z; let zz = q.w * q.w;
    let wx = q.x * q.y; let wy = q.x * q.z; let wz = q.x * q.w;
    let xy = q.y * q.z; let xz = q.y * q.w; let yz = q.z * q.w;
    let r0 = vec3f(1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy));
    let r1 = vec3f(2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx));
    let r2 = vec3f(2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy));

    let v = vec3f(sx * sx, sy * sy, sz * sz);
    let o = slot * TILE_STRIDE;

    tile[o] = pos[i * 3u];
    tile[o + 1u] = pos[i * 3u + 1u];
    tile[o + 2u] = pos[i * 3u + 2u];

    // EPS_COV on the diagonal, matching the CPU decode and the merge site: 2D
    // captures have an exactly-zero third scale and leave Sigma singular.
    let m00 = dot(r0 * r0, v) + EPS_COV;
    let m01 = dot(r0 * r1, v);
    let m02 = dot(r0 * r2, v);
    let m11 = dot(r1 * r1, v) + EPS_COV;
    let m12 = dot(r1 * r2, v);
    let m22 = dot(r2 * r2, v) + EPS_COV;

    tile[o + 3u] = m00;
    tile[o + 4u] = m01;
    tile[o + 5u] = m02;
    tile[o + 6u] = m11;
    tile[o + 7u] = m12;
    tile[o + 8u] = m22;

    let c = i * 3u;
    tile[o + 9u] = clamp(0.5 + dc[c] * SH_C0, 0.0, 1.0);
    tile[o + 10u] = clamp(0.5 + dc[c + 1u] * SH_C0, 0.0, 1.0);
    tile[o + 11u] = clamp(0.5 + dc[c + 2u] * SH_C0, 0.0, 1.0);

    let alpha = profileMass(ALPHA_MAX * sigmoid(geo[g8 + 7u]));
    tile[o + 12u] = max(alpha * max(sx, max(sy, sz)), EPS_W);

    // The regularized determinant, computed the same way the CPU does. It cannot
    // underflow f32: EPS_COV floors it at 1e-24.
    let det = m00 * (m11 * m22 - m12 * m12) -
              m01 * (m01 * m22 - m12 * m02) +
              m02 * (m01 * m12 - m11 * m02);
    tile[o + 13u] = select(LOG_DET_DEGENERATE, log(det), det > 0.0);
}
`;

/**
 * WGSL: per-voxel survivor selection — the §5 coverage-biased weighted medoid,
 * then §7's greedy novelty sequence, recorded in full with the priority each
 * pick was taken at.
 *
 * One workgroup per voxel, which is what makes this worth moving: §5 is
 * O(occupancy²) similarity evaluations and every voxel is independent, so the
 * quadratic term parallelizes perfectly while the sequential dependence between
 * successive picks stays inside a single workgroup.
 *
 * Recording the whole sequence rather than stopping at the local quota is what
 * lets the global pass (§7 stage 2) be a threshold instead of K round trips —
 * see decimate-voxel/global-fill.ts.
 *
 * Barriers sit only at loop positions whose trip count is workgroup-uniform
 * (`quota` and the reduction ladder), never under a data-dependent branch.
 *
 * @returns WGSL source.
 */
const selectWgsl = () => /* wgsl */`
struct Uniforms {
    voxelBase: u32,
    voxelTotal: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> tile: array<f32>;
@group(0) @binding(2) var<storage, read> voxelStart: array<u32>;
@group(0) @binding(3) var<storage, read_write> pickSlot: array<u32>;
@group(0) @binding(4) var<storage, read_write> pickPriority: array<f32>;
@group(0) @binding(5) var<storage, read_write> pickCount: array<u32>;
@group(0) @binding(6) var<storage, read_write> nearest: array<f32>;
@group(0) @binding(7) var<storage, read_write> done: array<u32>;

const WG: u32 = ${WORKGROUP}u;
const INVALID: u32 = 0xFFFFFFFFu;
// A "worse than anything real" key; the INVALID slot check is what actually
// guards the comparators, so this only has to be large, not maximal.
const BIG_SCORE: f32 = 1e30;
// Stamped on the first pick so the host's threshold always keeps it — this is
// the coverage floor, and it is not up for negotiation against a budget.
const ALWAYS_TAKE: f32 = 1e30;

${voxelSimilarityWgsl}

var<workgroup> redKey: array<f32, ${WORKGROUP}>;
var<workgroup> redAux: array<f32, ${WORKGROUP}>;
var<workgroup> redIdx: array<u32, ${WORKGROUP}>;
var<workgroup> shWeight: f32;
var<workgroup> shPick: u32;
var<workgroup> shPriority: f32;
var<workgroup> shPlaced: u32;

// Total order on candidates so the tree reduction is order-independent:
// smaller key wins, then larger radius, then lower slot.
fn betterMin(kA: f32, rA: f32, sA: u32, kB: f32, rB: f32, sB: u32) -> bool {
    if (sA == INVALID) { return false; }
    if (sB == INVALID) { return true; }
    if (kA != kB) { return kA < kB; }
    if (rA != rB) { return rA > rB; }
    return sA < sB;
}

fn betterMax(kA: f32, rA: f32, sA: u32, kB: f32, rB: f32, sB: u32) -> bool {
    if (sA == INVALID) { return false; }
    if (sB == INVALID) { return true; }
    if (kA != kB) { return kA > kB; }
    if (rA != rB) { return rA > rB; }
    return sA < sB;
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let v = uniforms.voxelBase + wid.x;
    let t = lid.x;

    // Uniform across the workgroup, so the barriers below stay in uniform flow.
    if (v >= uniforms.voxelTotal) { return; }

    let start = voxelStart[v];
    let count = voxelStart[v + 1u] - start;

    // Reset this voxel's scratch and accumulate the weight denominator.
    var wPartial = 0.0;
    for (var i = t; i < count; i += WG) {
        let slot = start + i;
        nearest[slot] = LOG_SIM_DEGENERATE;
        done[slot] = 0u;
        wPartial += tileWeight(slot);
    }
    redKey[t] = wPartial;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (t < s) { redKey[t] += redKey[t + s]; }
        workgroupBarrier();
    }
    if (t == 0u) {
        shWeight = redKey[0];
        shPlaced = 0u;
    }
    workgroupBarrier();
    let wSum = shWeight;

    // ---- §5: minimize S_i = M_i / w_i ----
    var bKey = BIG_SCORE;
    var bAux = -1.0;
    var bSlot = INVALID;
    for (var i = t; i < count; i += WG) {
        let si = start + i;
        var acc = 0.0;
        for (var j = 0u; j < count; j++) {
            if (j == i) { continue; }
            let sj = start + j;
            acc += tileWeight(sj) * dissimilarity(logSimilarity(si, sj));
        }
        let score = acc / wSum / tileWeight(si);
        let radius = tileRadius(si);
        if (betterMin(score, radius, si, bKey, bAux, bSlot)) {
            bKey = score;
            bAux = radius;
            bSlot = si;
        }
    }

    workgroupBarrier();
    redKey[t] = bKey;
    redAux[t] = bAux;
    redIdx[t] = bSlot;
    workgroupBarrier();
    for (var s = WG / 2u; s > 0u; s >>= 1u) {
        if (t < s) {
            if (betterMin(redKey[t + s], redAux[t + s], redIdx[t + s], redKey[t], redAux[t], redIdx[t])) {
                redKey[t] = redKey[t + s];
                redAux[t] = redAux[t + s];
                redIdx[t] = redIdx[t + s];
            }
        }
        workgroupBarrier();
    }
    if (t == 0u) { shPick = redIdx[0]; }
    workgroupBarrier();

    // ---- promote the winner, then refresh the voxel's novelty distances ----
    var rep = shPick;
    if (rep != INVALID) {
        if (t == 0u) {
            pickSlot[start] = rep;
            pickPriority[start] = ALWAYS_TAKE;
            done[rep] = 1u;
            shPlaced = 1u;
        }
        for (var i = t; i < count; i += WG) {
            let slot = start + i;
            if (slot == rep || done[slot] != 0u) { continue; }
            let l = logSimilarity(slot, rep);
            if (l > nearest[slot]) { nearest[slot] = l; }
            if (nearest[slot] > -COVERED_EPS) { done[slot] = 1u; }
        }
    }
    storageBarrier();

    // ---- §7: the voxel's full greedy sequence, highest novelty x weight first ----
    //
    // Every pick this voxel could ever make is recorded with the priority it was
    // taken at, not just the ones its local quota pays for. Those priorities are
    // non-increasing, so the host can spend the leftover global budget by
    // thresholding them instead of replaying the greedy loop pick by pick.
    for (var k = 1u; k < count; k++) {
        var pKey = -1.0;
        var pAux = -1.0;
        var pSlot = INVALID;
        for (var i = t; i < count; i += WG) {
            let slot = start + i;
            if (done[slot] != 0u) { continue; }
            let p = novelty(dissimilarity(nearest[slot])) * tileWeight(slot);
            let radius = tileRadius(slot);
            if (betterMax(p, radius, slot, pKey, pAux, pSlot)) {
                pKey = p;
                pAux = radius;
                pSlot = slot;
            }
        }

        workgroupBarrier();
        redKey[t] = pKey;
        redAux[t] = pAux;
        redIdx[t] = pSlot;
        workgroupBarrier();
        for (var s = WG / 2u; s > 0u; s >>= 1u) {
            if (t < s) {
                if (betterMax(redKey[t + s], redAux[t + s], redIdx[t + s], redKey[t], redAux[t], redIdx[t])) {
                    redKey[t] = redKey[t + s];
                    redAux[t] = redAux[t + s];
                    redIdx[t] = redIdx[t + s];
                }
            }
            workgroupBarrier();
        }
        if (t == 0u) {
            shPick = redIdx[0];
            shPriority = redKey[0];
        }
        workgroupBarrier();

        // Nothing eligible: the loop still runs its remaining trips (cheap, and
        // it keeps the barriers above out of data-dependent control flow).
        rep = shPick;
        if (rep != INVALID) {
            if (t == 0u) {
                pickSlot[start + shPlaced] = rep;
                pickPriority[start + shPlaced] = shPriority;
                done[rep] = 1u;
                shPlaced = shPlaced + 1u;
            }
            for (var i = t; i < count; i += WG) {
                let slot = start + i;
                if (slot == rep || done[slot] != 0u) { continue; }
                let l = logSimilarity(slot, rep);
                if (l > nearest[slot]) { nearest[slot] = l; }
                if (nearest[slot] > -COVERED_EPS) { done[slot] = 1u; }
            }
        }
        storageBarrier();
    }

    if (t == 0u) { pickCount[v] = shPlaced; }
}
`;

/**
 * WGSL: assign every splat to the most similar representative in its own voxel
 * or a touching one (spec §8).
 *
 * One thread per splat. Subdivision only refines inside a base cell, so the
 * candidate leaves all live in the 27 base cells around the splat's own — no
 * tree descent, just a bounded scan. This is the pass that dominates the CPU
 * reference, which is why it is here.
 *
 * @returns WGSL source.
 */
const assignWgsl = () => /* wgsl */`
struct Uniforms {
    slotBase: u32,
    slotCount: u32,
    dimX: u32,
    dimY: u32,
    dimZ: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> tile: array<f32>;
@group(0) @binding(2) var<storage, read> leafData: array<f32>;
@group(0) @binding(3) var<storage, read> cellRanges: array<u32>;
@group(0) @binding(4) var<storage, read> repRanges: array<u32>;
@group(0) @binding(5) var<storage, read> repSlots: array<u32>;
@group(0) @binding(6) var<storage, read> leafOfSlot: array<u32>;
@group(0) @binding(7) var<storage, read_write> assignment: array<u32>;

const LEAF_STRIDE: u32 = ${LEAF_STRIDE}u;
const INVALID: u32 = 0xFFFFFFFFu;
const TOUCH_EPS: f32 = 1e-12;

${voxelSimilarityWgsl}

fn leavesTouch(a: u32, b: u32) -> bool {
    let ao = a * LEAF_STRIDE;
    let bo = b * LEAF_STRIDE;
    for (var axis = 0u; axis < 3u; axis++) {
        if (leafData[ao + 1u + axis] > leafData[bo + 4u + axis] + TOUCH_EPS) { return false; }
        if (leafData[bo + 1u + axis] > leafData[ao + 4u + axis] + TOUCH_EPS) { return false; }
    }
    return true;
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= uniforms.slotCount) { return; }
    let slot = uniforms.slotBase + gid.x;
    let g = leafOfSlot[slot];

    let ownStart = repRanges[g * 2u];
    let ownCount = repRanges[g * 2u + 1u];

    // A representative keeps itself.
    for (var r = 0u; r < ownCount; r++) {
        if (repSlots[ownStart + r] == slot) {
            assignment[slot] = slot;
            return;
        }
    }

    let cell = bitcast<u32>(leafData[g * LEAF_STRIDE]);
    let cx = cell % uniforms.dimX;
    let cy = (cell / uniforms.dimX) % uniforms.dimY;
    let cz = cell / (uniforms.dimX * uniforms.dimY);

    var best = INVALID;
    var bestLog = 0.0;

    for (var dz = 0u; dz < 3u; dz++) {
        let z = cz + dz;
        if (z == 0u || z > uniforms.dimZ) { continue; }
        for (var dy = 0u; dy < 3u; dy++) {
            let y = cy + dy;
            if (y == 0u || y > uniforms.dimY) { continue; }
            for (var dx = 0u; dx < 3u; dx++) {
                let x = cx + dx;
                if (x == 0u || x > uniforms.dimX) { continue; }

                let c = (x - 1u) + uniforms.dimX * ((y - 1u) + uniforms.dimY * (z - 1u));
                let leafStart = cellRanges[c * 2u];
                let leafCount = cellRanges[c * 2u + 1u];

                for (var li = 0u; li < leafCount; li++) {
                    let h = leafStart + li;
                    if (!leavesTouch(g, h)) { continue; }
                    let rs = repRanges[h * 2u];
                    let rc = repRanges[h * 2u + 1u];
                    for (var r = 0u; r < rc; r++) {
                        let rep = repSlots[rs + r];
                        let l = logSimilarity(slot, rep);
                        if (best == INVALID || l > bestLog || (l == bestLog && rep < best)) {
                            best = rep;
                            bestLog = l;
                        }
                    }
                }
            }
        }
    }

    assignment[slot] = select(best, repSlots[ownStart], best == INVALID);
}
`;

type VoxelSelectResult = {
    /** Final representative slots per leaf, ascending: `repStart[g]..repStart[g+1]`. */
    repSlots: Uint32Array;
    /** Run offsets, length `leafCount + 1`. */
    repStart: Uint32Array;
    /** Representatives kept per leaf. */
    repCount: Uint32Array;
    /** Total representatives, i.e. the output splat count. */
    count: number;
    /** Representative slot per slot. */
    assignment: Uint32Array;
};

/**
 * GPU voxel selection and assignment: spec §5, §7 (both stages), and §8.
 *
 * A run is decode → select → global fill → assign, all against one
 * device-resident tile, with one readback in the middle so the host can pick the
 * global threshold. The select kernel records each voxel's *entire* greedy
 * sequence with per-pick priorities, which is what turns the reference's
 * sequential global pass into that single threshold.
 *
 * The partition (spec §3) stays on the CPU: it is a counting sort plus a shallow
 * refine, cheap next to the quadratic per-voxel work, and the streaming layer
 * needs it in host memory anyway.
 */
class GpuVoxelSelect {
    execute: (
        view: SplatView,
        n: number,
        partition: VoxelPartition,
        quotas: Int32Array,
        target: number
    ) => Promise<VoxelSelectResult>;

    destroy: () => void;

    /**
     * @param device - PlayCanvas GraphicsDevice (WebGPU).
     * @param maxN - Maximum splats per run.
     * @param maxLeaves - Maximum voxels per run.
     */
    constructor(device: GraphicsDevice, maxN: number, maxLeaves: number) {
        const limits = (device as any).limits;
        const maxBinding = typeof limits?.maxStorageBufferBindingSize === 'number' ?
            limits.maxStorageBufferBindingSize :
            Infinity;
        const tileBytes = maxN * VOXEL_TILE_STRIDE * 4;
        if (tileBytes > maxBinding) {
            throw new Error(
                `GpuVoxelSelect: tile buffer (${maxN} splats × ${VOXEL_TILE_STRIDE} floats = ` +
                `${tileBytes} bytes) exceeds device maxStorageBufferBindingSize (${maxBinding})`
            );
        }

        // Compensation is fixed for the run, same as the merge site. This kernel
        // DECODES the input, so it takes inputAlphaMax — the range the source
        // file is written in — not alphaMax, which is the range we will write.
        const { inputAlphaMax } = getCompensation();

        const posBuf = new StorageBuffer(device, maxN * 3 * 4, BUFFERUSAGE_COPY_DST);
        const geoBuf = new StorageBuffer(device, maxN * 8 * 4, BUFFERUSAGE_COPY_DST);
        const dcBuf = new StorageBuffer(device, maxN * 3 * 4, BUFFERUSAGE_COPY_DST);
        const orderBuf = new StorageBuffer(device, maxN * 4, BUFFERUSAGE_COPY_DST);
        const tileBuf = new StorageBuffer(device, tileBytes);

        const voxelStartBuf = new StorageBuffer(device, (maxLeaves + 1) * 4, BUFFERUSAGE_COPY_DST);
        // One pick per member at most, so the sequences fit in n entries.
        const pickSlotBuf = new StorageBuffer(device, maxN * 4, BUFFERUSAGE_COPY_SRC);
        const pickPriorityBuf = new StorageBuffer(device, maxN * 4, BUFFERUSAGE_COPY_SRC);
        const pickCountBuf = new StorageBuffer(device, maxLeaves * 4, BUFFERUSAGE_COPY_SRC);
        const nearestBuf = new StorageBuffer(device, maxN * 4);
        const doneBuf = new StorageBuffer(device, maxN * 4);

        const leafBuf = new StorageBuffer(device, maxLeaves * LEAF_STRIDE * 4, BUFFERUSAGE_COPY_DST);
        const cellBuf = new StorageBuffer(device, 4, BUFFERUSAGE_COPY_DST);   // resized per run
        const repRangeBuf = new StorageBuffer(device, maxLeaves * 2 * 4, BUFFERUSAGE_COPY_DST);
        const repSlotsBuf = new StorageBuffer(device, maxN * 4, BUFFERUSAGE_COPY_DST);
        const leafOfSlotBuf = new StorageBuffer(device, maxN * 4, BUFFERUSAGE_COPY_DST);
        const assignBuf = new StorageBuffer(device, maxN * 4, BUFFERUSAGE_COPY_SRC);

        let cellBufActive = cellBuf;

        const decode: Kernel = makeKernel(
            device, 'voxel-decode', decodeWgsl(inputAlphaMax),
            ['slotBase', 'slotCount'],
            [['pos', true], ['geo', true], ['dc', true], ['order', true], ['tile', false]]
        );
        decode.compute.setParameter('pos', posBuf);
        decode.compute.setParameter('geo', geoBuf);
        decode.compute.setParameter('dc', dcBuf);
        decode.compute.setParameter('order', orderBuf);
        decode.compute.setParameter('tile', tileBuf);

        const select: Kernel = makeKernel(
            device, 'voxel-select', selectWgsl(),
            ['voxelBase', 'voxelTotal'],
            [
                ['tile', true], ['voxelStart', true], ['pickSlot', false],
                ['pickPriority', false], ['pickCount', false],
                ['nearest', false], ['done', false]
            ]
        );
        select.compute.setParameter('tile', tileBuf);
        select.compute.setParameter('voxelStart', voxelStartBuf);
        select.compute.setParameter('pickSlot', pickSlotBuf);
        select.compute.setParameter('pickPriority', pickPriorityBuf);
        select.compute.setParameter('pickCount', pickCountBuf);
        select.compute.setParameter('nearest', nearestBuf);
        select.compute.setParameter('done', doneBuf);

        const assign: Kernel = makeKernel(
            device, 'voxel-assign', assignWgsl(),
            ['slotBase', 'slotCount', 'dimX', 'dimY', 'dimZ'],
            [
                ['tile', true], ['leafData', true], ['cellRanges', true],
                ['repRanges', true], ['repSlots', true], ['leafOfSlot', true],
                ['assignment', false]
            ]
        );
        assign.compute.setParameter('tile', tileBuf);
        assign.compute.setParameter('leafData', leafBuf);
        assign.compute.setParameter('repRanges', repRangeBuf);
        assign.compute.setParameter('repSlots', repSlotsBuf);
        assign.compute.setParameter('leafOfSlot', leafOfSlotBuf);
        assign.compute.setParameter('assignment', assignBuf);

        // Dispatch caps: 65,535 workgroups per dimension.
        const slotsPerBatch = 65535 * WORKGROUP;
        const voxelsPerBatch = 65535;

        // Uniform writes are snapshotted once per submit, so back-to-back
        // dispatches of the same Compute would all see the last batch's values.
        // Every batched loop below submits per dispatch.
        const submit = () => (device as unknown as { submit: () => void }).submit();

        this.execute = async (
            view: SplatView,
            n: number,
            partition: VoxelPartition,
            quotas: Int32Array,
            target: number
        ): Promise<VoxelSelectResult> => {
            const leafCount = partition.leafCount;
            if (n > maxN) throw new Error(`GpuVoxelSelect: n=${n} exceeds maxN=${maxN}`);
            if (leafCount > maxLeaves) {
                throw new Error(`GpuVoxelSelect: leafCount=${leafCount} exceeds maxLeaves=${maxLeaves}`);
            }

            // Only the DC colour reaches the metric, so pack 3 floats per splat
            // rather than uploading whole SH rows.
            const dcData = new Float32Array(n * 3);
            const { colorDim } = view;
            for (let i = 0; i < n; i++) {
                dcData[i * 3] = view.color[i * colorDim];
                dcData[i * 3 + 1] = view.color[i * colorDim + 1];
                dcData[i * 3 + 2] = view.color[i * colorDim + 2];
            }

            const voxelStart = new Uint32Array(leafCount + 1);
            const leafOfSlot = new Uint32Array(n);
            for (let g = 0; g < leafCount; g++) {
                voxelStart[g] = partition.start[g];
                leafOfSlot.fill(g, partition.start[g], partition.start[g + 1]);
            }
            voxelStart[leafCount] = n;

            posBuf.write(0, view.pos, 0, n * 3);
            geoBuf.write(0, view.geo, 0, n * 8);
            dcBuf.write(0, dcData, 0, n * 3);
            orderBuf.write(0, new Uint32Array(partition.order.buffer, partition.order.byteOffset, n), 0, n);
            voxelStartBuf.write(0, voxelStart, 0, voxelStart.length);
            leafOfSlotBuf.write(0, leafOfSlot, 0, n);

            // Decode the tile.
            for (let base = 0; base < n; base += slotsPerBatch) {
                const slotCount = Math.min(slotsPerBatch, n - base);
                decode.compute.setParameter('slotBase', base);
                decode.compute.setParameter('slotCount', slotCount);
                decode.compute.setupDispatch(Math.ceil(slotCount / WORKGROUP));
                device.computeDispatch([decode.compute], `voxel-decode-${base}`);
                submit();
            }

            // Select per voxel.
            for (let base = 0; base < leafCount; base += voxelsPerBatch) {
                const dispatch = Math.min(voxelsPerBatch, leafCount - base);
                select.compute.setParameter('voxelBase', base);
                select.compute.setParameter('voxelTotal', leafCount);
                select.compute.setupDispatch(dispatch);
                device.computeDispatch([select.compute], `voxel-select-${base}`);
                submit();
            }

            // Read the pick sequences back and choose the global threshold: each
            // voxel keeps its local entitlement, then the leftover budget goes to
            // the highest-priority picks anywhere (spec §7 stage 2).
            const pickCount = new Uint32Array(leafCount);
            await pickCountBuf.read(0, leafCount * 4, pickCount, true);
            const pickSlot = new Uint32Array(n);
            await pickSlotBuf.read(0, n * 4, pickSlot, true);
            const pickPriority = new Float32Array(n);
            await pickPriorityBuf.read(0, n * 4, pickPriority, true);

            const localTake = new Int32Array(leafCount);
            for (let g = 0; g < leafCount; g++) {
                localTake[g] = 1 + Math.max(0, quotas[g]);
            }
            const { take, total } = globalFill({
                voxelCount: leafCount,
                runStart: partition.start,
                pickCount,
                pickPriority,
                localTake,
                target
            });

            // Compact the kept picks into per-leaf runs, ascending within a leaf
            // so the assignment scan order matches the reference's.
            const repStart = new Uint32Array(leafCount + 1);
            for (let g = 0; g < leafCount; g++) repStart[g + 1] = repStart[g] + take[g];
            const repCount = new Uint32Array(leafCount);
            const repSlots = new Uint32Array(total);
            for (let g = 0; g < leafCount; g++) {
                const from = partition.start[g];
                const kept = pickSlot.subarray(from, from + take[g]);
                const sorted = Array.from(kept).sort((a, b) => a - b);
                repSlots.set(sorted, repStart[g]);
                repCount[g] = take[g];
            }
            repSlotsBuf.write(0, repSlots, 0, Math.max(total, 1));

            // Leaf boxes and per-cell leaf ranges for the neighbourhood scan.
            // Leaves are ordered by member offset, and subdivision only splits
            // within a base cell, so a cell's leaves are always contiguous.
            const { dims } = partition;
            const cellCount = dims.dx * dims.dy * dims.dz;
            const cellRanges = new Uint32Array(cellCount * 2);
            for (let g = 0; g < leafCount; g++) {
                const cell = partition.baseCell[g];
                if (cellRanges[cell * 2 + 1] === 0) cellRanges[cell * 2] = g;
                cellRanges[cell * 2 + 1]++;
            }

            const leafData = new Float32Array(leafCount * LEAF_STRIDE);
            const leafCellView = new Uint32Array(leafData.buffer);
            const repRanges = new Uint32Array(leafCount * 2);
            for (let g = 0; g < leafCount; g++) {
                leafCellView[g * LEAF_STRIDE] = partition.baseCell[g];
                for (let a = 0; a < 3; a++) {
                    leafData[g * LEAF_STRIDE + 1 + a] = partition.lo[g * 3 + a];
                    leafData[g * LEAF_STRIDE + 4 + a] = partition.hi[g * 3 + a];
                }
                repRanges[g * 2] = repStart[g];
                repRanges[g * 2 + 1] = repCount[g];
            }

            if (cellBufActive.byteSize < cellRanges.byteLength) {
                cellBufActive.destroy();
                cellBufActive = new StorageBuffer(device, cellRanges.byteLength, BUFFERUSAGE_COPY_DST);
            }
            cellBufActive.write(0, cellRanges, 0, cellRanges.length);
            assign.compute.setParameter('cellRanges', cellBufActive);

            leafBuf.write(0, leafData, 0, leafData.length);
            repRangeBuf.write(0, repRanges, 0, repRanges.length);

            assign.compute.setParameter('dimX', dims.dx);
            assign.compute.setParameter('dimY', dims.dy);
            assign.compute.setParameter('dimZ', dims.dz);
            for (let base = 0; base < n; base += slotsPerBatch) {
                const slotCount = Math.min(slotsPerBatch, n - base);
                assign.compute.setParameter('slotBase', base);
                assign.compute.setParameter('slotCount', slotCount);
                assign.compute.setupDispatch(Math.ceil(slotCount / WORKGROUP));
                device.computeDispatch([assign.compute], `voxel-assign-${base}`);
                submit();
            }

            const assignment = new Uint32Array(n);
            await assignBuf.read(0, n * 4, assignment, true);

            return { repSlots, repStart, repCount, count: total, assignment };
        };

        this.destroy = () => {
            posBuf.destroy();
            geoBuf.destroy();
            dcBuf.destroy();
            orderBuf.destroy();
            tileBuf.destroy();
            voxelStartBuf.destroy();
            pickSlotBuf.destroy();
            pickPriorityBuf.destroy();
            pickCountBuf.destroy();
            repSlotsBuf.destroy();
            nearestBuf.destroy();
            doneBuf.destroy();
            leafBuf.destroy();
            cellBufActive.destroy();
            repRangeBuf.destroy();
            leafOfSlotBuf.destroy();
            assignBuf.destroy();
            decode.destroy();
            select.destroy();
            assign.destroy();
        };
    }
}

export { GpuVoxelSelect, VOXEL_TILE_STRIDE, type VoxelSelectResult };
