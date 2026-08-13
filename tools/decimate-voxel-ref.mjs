/**
 * Whole-scene runner for the voxel-decimation reference (Splat-Simplify.md
 * §3-§9), the oracle the GPU path is checked against.
 *
 * It loads a PLY resident, decimates to a target count, and writes a PLY, so
 * output can be rendered and scored with the usual gap metric. Resident and
 * single-threaded by design — the point is exactness, not throughput. Budget
 * roughly 140 bytes per input gaussian plus the reference's own working set.
 *
 * Usage:
 *   node --import tsx tools/decimate-voxel-ref.mjs <in.ply> <out.ply> <target>
 */
import { closeSync, openSync, writeSync } from 'node:fs';

import { NodeReadFileSystem } from '../src/cli/node-file-system.js';
import { decimateReference } from '../src/lib/decimate-voxel/reference.js';
import { createChunkDataPool } from '../src/lib/index.js';
import { readPly } from '../src/lib/readers/read-ply.js';

const log = msg => process.stderr.write(`${msg}\n`);

const loadPly = async (filename) => {
    const pool = createChunkDataPool();
    const fs = new NodeReadFileSystem();
    const src = await readPly(await fs.createSource(filename), pool);
    const { meta } = src;
    if (meta.numLods !== 1) throw new Error('single-LOD input required');

    const n = meta.numGaussians;
    const colorDim = meta.layouts.color.stride >> 2;
    log(`loading ${filename}: ${n} gaussians · colorDim ${colorDim}`);

    const view = {
        pos: new Float32Array(n * 3),
        geo: new Float32Array(n * 8),
        color: new Float32Array(n * colorDim),
        colorDim
    };

    let base = 0;
    for (let c = 0; c < meta.numChunks[0]; c++) {
        const count = Math.min(meta.chunkSize, n - c * meta.chunkSize);
        const pcd = pool.acquire('position', meta.layouts.position, count);
        const gcd = pool.acquire('geometric', meta.layouts.geometric, count);
        const ccd = pool.acquire('color', meta.layouts.color, count);
        await src.read({ chunkIndex: c, position: pcd, geometric: gcd, color: ccd });
        view.pos.set(new Float32Array(pcd.data, 0, count * 3), base * 3);
        view.geo.set(new Float32Array(gcd.data, 0, count * 8), base * 8);
        view.color.set(new Float32Array(ccd.data, 0, count * colorDim), base * colorDim);
        base += count;
        pcd.release(); gcd.release(); ccd.release();
    }
    await src.close();

    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < 3; a++) {
            const v = view.pos[i * 3 + a];
            if (v < min[a]) min[a] = v;
            if (v > max[a]) max[a] = v;
        }
    }
    const ext = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    log(`bounds ${min.map(v => v.toFixed(2))} .. ${max.map(v => v.toFixed(2))}`);

    return { view, n, min, ext };
};

const writePly = (path, result, colorDim) => {
    const props = ['x', 'y', 'z', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
    for (let r = 0; r < colorDim - 3; r++) props.push(`f_rest_${r}`);
    props.push('opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3');

    const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${result.count}\n` +
        `${props.map(p => `property float ${p}`).join('\n')}\nend_header\n`;
    const floats = props.length;

    const fd = openSync(path, 'w');
    writeSync(fd, Buffer.from(header, 'ascii'));

    const ROWS = 65536;
    const buf = Buffer.allocUnsafe(ROWS * floats * 4);
    const f32 = new Float32Array(buf.buffer, buf.byteOffset, ROWS * floats);
    let rows = 0;

    for (let k = 0; k < result.count; k++) {
        const o = rows * floats;
        f32[o] = result.pos[k * 3];
        f32[o + 1] = result.pos[k * 3 + 1];
        f32[o + 2] = result.pos[k * 3 + 2];
        for (let c = 0; c < colorDim; c++) f32[o + 3 + c] = result.color[k * colorDim + c];

        // Reference geo order is quat, log scales, logit opacity; PLY wants
        // opacity, scales, quat.
        const g = k * 8;
        const oo = o + 3 + colorDim;
        f32[oo] = result.geo[g + 7];
        f32[oo + 1] = result.geo[g + 4];
        f32[oo + 2] = result.geo[g + 5];
        f32[oo + 3] = result.geo[g + 6];
        f32[oo + 4] = result.geo[g];
        f32[oo + 5] = result.geo[g + 1];
        f32[oo + 6] = result.geo[g + 2];
        f32[oo + 7] = result.geo[g + 3];

        if (++rows === ROWS) {
            writeSync(fd, buf, 0, rows * floats * 4);
            rows = 0;
        }
    }
    if (rows > 0) writeSync(fd, buf, 0, rows * floats * 4);
    closeSync(fd);
    log(`wrote ${path} (${result.count} gaussians)`);
};

const [input, output, targetArg] = process.argv.slice(2);
if (!input || !output || !targetArg) {
    log('usage: decimate-voxel-ref.mjs <in.ply> <out.ply> <target>');
    process.exit(1);
}

const target = Number(targetArg);
const { view, n, min, ext } = await loadPly(input);

const t0 = performance.now();
const result = decimateReference(view, n, min, ext, target);
const dt = (performance.now() - t0) / 1000;

const { partition, reps } = result;
let maxOcc = 0;
for (let g = 0; g < partition.leafCount; g++) {
    maxOcc = Math.max(maxOcc, partition.start[g + 1] - partition.start[g]);
}
log(`leaves ${partition.leafCount} · dims ${partition.dims.dx}x${partition.dims.dy}x${partition.dims.dz} · ` +
    `H ${partition.threshold} · max occupancy ${maxOcc}`);
log(`kept ${reps.count}/${target} (${(100 * reps.count / n).toFixed(2)}% of input) in ${dt.toFixed(1)}s`);

writePly(output, result, view.colorDim);
