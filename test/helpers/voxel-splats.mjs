/**
 * Splat builders for the voxel-decimator tests: turn plain descriptors into the
 * SplatView layout (pos 3, geo 8, color 3) the similarity tile decodes from.
 */
import { createSimTile, fillSimTile } from '../../src/lib/decimate-voxel/similarity.js';

const SH_C0 = 0.28209479177387814;

/** Display RGB -> the DC coefficient that decodes back to it. */
const dcFor = rgb => (rgb - 0.5) / SH_C0;

const logit = p => Math.log(p / (1 - p));

/**
 * Build a SplatView from descriptors: `p` position, `s` linear scale (scalar or
 * per-axis), `a` opacity, `rgb` display colour (default mid grey), `q`
 * quaternion (default identity).
 */
const makeView = (splats) => {
    const n = splats.length;
    const pos = new Float32Array(n * 3);
    const geo = new Float32Array(n * 8);
    const color = new Float32Array(n * 3);

    splats.forEach((sp, i) => {
        const s = Array.isArray(sp.s) ? sp.s : [sp.s, sp.s, sp.s];
        const q = sp.q ?? [1, 0, 0, 0];
        const rgb = sp.rgb ?? [0.5, 0.5, 0.5];

        pos[i * 3] = sp.p[0];
        pos[i * 3 + 1] = sp.p[1];
        pos[i * 3 + 2] = sp.p[2];

        geo[i * 8] = q[0];
        geo[i * 8 + 1] = q[1];
        geo[i * 8 + 2] = q[2];
        geo[i * 8 + 3] = q[3];
        geo[i * 8 + 4] = Math.log(s[0]);
        geo[i * 8 + 5] = Math.log(s[1]);
        geo[i * 8 + 6] = Math.log(s[2]);
        geo[i * 8 + 7] = logit(sp.a);

        color[i * 3] = dcFor(rgb[0]);
        color[i * 3 + 1] = dcFor(rgb[1]);
        color[i * 3 + 2] = dcFor(rgb[2]);
    });

    return { pos, geo, color, colorDim: 3 };
};

/** Descriptors -> a filled similarity tile, slots in descriptor order. */
const tileOf = (splats) => {
    const view = makeView(splats);
    const indices = splats.map((_, i) => i);
    return fillSimTile(view, indices, splats.length, createSimTile(splats.length));
};

export { SH_C0, dcFor, logit, makeView, tileOf };
