import { type GraphicsDevice } from 'playcanvas';

import { allocateExtras } from './budget';
import { buildPartition } from './partition';
import { leafAggregates } from './reference';
import {
    type ChunkDataPool,
    type ChunkSource,
    type ChunkSourceMetadata
} from '../chunk';
import {
    createMergeScratch,
    mergeGroup,
    setCompensation,
    type Compensation,
    type MergedOut,
    type SplatView
} from '../decimate/moment-match';
import { createBlockProducerSource, type ChunkPayload } from '../decimate-uniform/block-producer';
import { GpuVoxelSelect } from '../gpu/gpu-voxel-select';
import { bakeTransform } from '../ops';
import { type DeviceCreator } from '../types';
import { fmtBytes, fmtCount, logger, Transform } from '../utils';

/** Resident bytes per gaussian for the selection pass: position, geometry, DC. */
const SELECTION_BYTES = 12 + 32 + 12;

/**
 * Input rows per gather read.
 *
 * Output chunking is fixed by the source's chunk size, but the number of *input*
 * rows behind one output chunk is the decimation ratio times that — which on a
 * deep target is most of the scene. Gathering is therefore batched under this
 * cap independently of the output chunk boundary, so peak gather memory stays
 * bounded by the cap rather than by the ratio.
 */
const MAX_GATHER_ROWS = 1 << 18;

type DecimateVoxelOptions = {
    /** Upper bound on the gaussians to keep (≥ 1). */
    targetCount: number;
    /** GPU device factory. Required — this decimator has no CPU path. */
    createDevice: DeviceCreator;
    /**
     * What to do with merged mass a unit-alpha Gaussian cannot carry: discard it
     * (`none`, the default), keep it as peak opacity above 1 (`alpha`), or grow
     * the footprint to fit it (`scale`). Orthogonal to allocation, so the same
     * choice applies to every decimator.
     */
    compensation?: Compensation;
};

/**
 * Read position, geometry and DC colour into resident columns.
 *
 * Only the DC coefficients reach the similarity metric, so the view is built at
 * `colorDim = 3` no matter how many SH bands the input carries — on a band-3
 * scene that is 56 bytes per gaussian resident instead of 236.
 *
 * @param source - Input source.
 * @param pool - Chunk-data pool.
 * @returns The selection view.
 */
const readSelectionView = async (source: ChunkSource, pool: ChunkDataPool): Promise<SplatView> => {
    const { meta } = source;
    const n = meta.numGaussians;
    const inColorDim = meta.layouts.color!.stride >> 2;
    const view: SplatView = {
        pos: new Float32Array(n * 3),
        geo: new Float32Array(n * 8),
        color: new Float32Array(n * 3),
        colorDim: 3
    };

    const numChunks = meta.numChunks[0] ?? 0;
    const bar = logger.bar('reading', numChunks);
    let base = 0;
    for (let c = 0; c < numChunks; c++) {
        const count = Math.min(meta.chunkSize, n - c * meta.chunkSize);
        const pcd = pool.acquire('position', meta.layouts.position!, count);
        const gcd = pool.acquire('geometric', meta.layouts.geometric!, count);
        const ccd = pool.acquire('color', meta.layouts.color!, count);
        await source.read({ chunkIndex: c, position: pcd, geometric: gcd, color: ccd });

        view.pos.set(new Float32Array(pcd.data, 0, count * 3), base * 3);
        view.geo.set(new Float32Array(gcd.data, 0, count * 8), base * 8);
        const color = new Float32Array(ccd.data, 0, count * inColorDim);
        for (let i = 0; i < count; i++) {
            view.color[(base + i) * 3] = color[i * inColorDim];
            view.color[(base + i) * 3 + 1] = color[i * inColorDim + 1];
            view.color[(base + i) * 3 + 2] = color[i * inColorDim + 2];
        }

        base += count;
        pcd.release();
        gcd.release();
        ccd.release();
        bar.tick();
    }
    bar.end();
    return view;
};

/**
 * Bounds of the resident positions.
 *
 * @param pos - Positions, 3 f32 per gaussian.
 * @param n - Gaussian count.
 * @returns Bounds minimum and extent per axis.
 */
const boundsOf = (pos: Float32Array, n: number) => {
    const min: [number, number, number] = [Infinity, Infinity, Infinity];
    const max: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < n; i++) {
        for (let a = 0; a < 3; a++) {
            const v = pos[i * 3 + a];
            if (v < min[a]) min[a] = v;
            if (v > max[a]) max[a] = v;
        }
    }
    const ext: [number, number, number] = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
    return { min, ext };
};

/**
 * Merge stream: walk output groups in order, gather each chunk's member rows
 * from the input, and moment-match.
 *
 * Members of one group are spatially local but scattered in input row order, so
 * the heavy layers are pulled with a gather read per output chunk rather than
 * held resident. Positions come from the resident columns.
 *
 * @param source - Input source (gather-capable).
 * @param pool - Chunk-data pool.
 * @param view - Resident selection view (positions and geometry).
 * @param members - Input rows per group, grouped by `offsets`.
 * @param offsets - Group offsets into `members`, length `groups + 1`.
 * @param groups - Output group count.
 * @param chunkSize - Output rows per payload.
 * @param compensation - Merged-mass policy.
 * @param tick - Progress callback (groups emitted).
 * @yields One payload per output chunk.
 */
async function *voxelMergeStream(
    source: ChunkSource,
    pool: ChunkDataPool,
    view: SplatView,
    members: Int32Array,
    offsets: Int32Array,
    groups: number,
    chunkSize: number,
    compensation: Compensation | undefined,
    tick: (n: number) => void
): AsyncGenerator<ChunkPayload> {
    const { meta } = source;
    const colorDim = meta.layouts.color!.stride >> 2;
    const hasOther = meta.availableLayers.has('other') && (meta.layouts.other?.stride ?? 0) > 0;
    const otherDim = hasOther ? meta.layouts.other!.stride >> 2 : 0;

    const outPos = new Float32Array(chunkSize * 3);
    const outGeo = new Float32Array(chunkSize * 8);
    const outColor = new Float32Array(chunkSize * colorDim);
    const outOther = hasOther ? new Uint32Array(chunkSize * otherDim) : undefined;

    const out: MergedOut = {
        pos: new Float64Array(3),
        geo: new Float64Array(8),
        color: new Float64Array(colorDim)
    };
    const scratch = createMergeScratch();
    let memberRows = new Int32Array(64);

    // The merge honours the compensation mode through module state, which the
    // worker path sets per task; here mergeGroup runs inline, so set it once.
    setCompensation(compensation);

    // Merge groups `[from, to)` into output rows starting at `rowBase`.
    const mergeBatch = async (from: number, to: number, rowBase: number): Promise<void> => {
        const gatherCount = offsets[to] - offsets[from];

        // Ascending order is friendlier to the reader, so sort a copy and index
        // back through the sorted position.
        const sorted = new Uint32Array(gatherCount);
        for (let i = 0; i < gatherCount; i++) sorted[i] = members[offsets[from] + i];
        sorted.sort();
        const rowOf = new Map<number, number>();
        for (let i = 0; i < gatherCount; i++) rowOf.set(sorted[i], i);

        const gcd = pool.acquire('geometric', meta.layouts.geometric!, gatherCount);
        const ccd = pool.acquire('color', meta.layouts.color!, gatherCount);
        const ocd = hasOther ? pool.acquire('other', meta.layouts.other!, gatherCount) : undefined;
        await source.read({
            indices: sorted,
            indexOffset: 0,
            count: gatherCount,
            geometric: gcd,
            color: ccd,
            other: ocd
        });

        // A view over the gathered rows, so mergeGroup addresses them directly.
        const gathered: SplatView = {
            pos: new Float32Array(gatherCount * 3),
            geo: new Float32Array(gcd.data, 0, gatherCount * 8),
            color: new Float32Array(ccd.data, 0, gatherCount * colorDim),
            colorDim
        };
        for (let i = 0; i < gatherCount; i++) {
            const row = sorted[i];
            gathered.pos[i * 3] = view.pos[row * 3];
            gathered.pos[i * 3 + 1] = view.pos[row * 3 + 1];
            gathered.pos[i * 3 + 2] = view.pos[row * 3 + 2];
        }
        const otherData = ocd ? new Uint32Array(ocd.data, 0, gatherCount * otherDim) : undefined;

        for (let g = from; g < to; g++) {
            const r = rowBase + (g - from);
            const count = offsets[g + 1] - offsets[g];
            if (count > memberRows.length) memberRows = new Int32Array(count);
            for (let m = 0; m < count; m++) {
                memberRows[m] = rowOf.get(members[offsets[g] + m])!;
            }

            mergeGroup(gathered, memberRows, count, out, scratch);

            for (let a = 0; a < 3; a++) outPos[r * 3 + a] = out.pos[a];
            for (let a = 0; a < 8; a++) outGeo[r * 8 + a] = out.geo[a];
            for (let a = 0; a < colorDim; a++) outColor[r * colorDim + a] = out.color[a];
            // `other` holds opaque per-splat columns with no merge semantics, so
            // the representative's row carries through.
            if (outOther && otherData) {
                const rep = memberRows[0];
                outOther.set(otherData.subarray(rep * otherDim, (rep + 1) * otherDim), r * otherDim);
            }
        }

        gcd.release();
        ccd.release();
        ocd?.release();
    };

    for (let base = 0; base < groups; base += chunkSize) {
        const rows = Math.min(chunkSize, groups - base);

        // Sub-batch inside the output chunk so no single gather exceeds the cap.
        let batchFrom = base;
        while (batchFrom < base + rows) {
            let batchTo = batchFrom + 1;      // at least one group, however large
            while (
                batchTo < base + rows &&
                offsets[batchTo + 1] - offsets[batchFrom] <= MAX_GATHER_ROWS
            ) {
                batchTo++;
            }
            await mergeBatch(batchFrom, batchTo, batchFrom - base);
            batchFrom = batchTo;
        }

        const payload: ChunkPayload = {
            count: rows,
            position: outPos.subarray(0, rows * 3),
            geometric: outGeo.subarray(0, rows * 8),
            color: outColor.subarray(0, rows * colorDim)
        };
        if (outOther) payload.other = outOther.subarray(0, rows * otherDim);
        yield payload;
        tick(rows);
    }
}

/**
 * Voxel decimation over a chunk source (spec §3-§9).
 *
 * Space-uniform allocation: every occupied voxel keeps at least one survivor —
 * the coverage floor neither the uniform nor the adaptive decimator has — and
 * the rest of the budget goes to the most novel content anywhere. Selection and
 * assignment run on the GPU; the partition, the global threshold and the merge
 * run on the host.
 *
 * Unlike the other two decimators this is single-pass: the allocator reaches any
 * target directly, with no cascade of generations.
 *
 * Resident cost is 56 bytes per input gaussian for selection, independent of SH
 * band count, plus the same again on the device for the similarity tile. The
 * heavy colour layer is never resident — it is gathered per output chunk — so a
 * gather-capable input is required.
 *
 * @param source - Input (consumed: the returned source owns it). Single LOD, gaussian layers required.
 * @param pool - Chunk-data pool; its chunk size must match the source's.
 * @param opts - Options.
 * @returns The decimated stream-once source.
 */
const decimateSourceVoxel = async (
    source: ChunkSource,
    pool: ChunkDataPool,
    opts: DecimateVoxelOptions
): Promise<ChunkSource> => {
    const { targetCount } = opts;
    const inputMeta = source.meta;

    if (inputMeta.numLods > 1) {
        throw new Error(
            `decimate requires a single-LOD source (got ${inputMeta.numLods} LODs); select a level first (--select-lod / selectLod)`
        );
    }
    for (const layer of ['position', 'geometric', 'color'] as const) {
        if (!inputMeta.availableLayers.has(layer)) {
            throw new Error(`decimate requires gaussian splat data (missing '${layer}' layer)`);
        }
    }
    if (targetCount < 1) {
        throw new Error(`decimate target must be at least 1 (got ${targetCount})`);
    }
    if (targetCount >= inputMeta.numGaussians) {
        return source;
    }

    const device: GraphicsDevice = await opts.createDevice();

    // Bake up front: the partition and the metric consume geometry, so they must
    // see PLY-space values (identity fast-path when there is no transform).
    const src = bakeTransform(source, Transform.PLY);
    const n = src.meta.numGaussians;

    const group = logger.group('Voxel decimate');
    logger.info(
        `${fmtCount(n)} → ${fmtCount(targetCount)} · ${fmtBytes(n * SELECTION_BYTES)} resident`
    );

    const view = await readSelectionView(src, pool);
    const { min, ext } = boundsOf(view.pos, n);

    const partition = buildPartition(view.pos, n, min, ext, targetCount);
    const { counts, opacity } = leafAggregates(view, partition);
    const quotas = allocateExtras(counts, opacity, partition.leafCount, targetCount, n);

    const gpu = new GpuVoxelSelect(device, n, partition.leafCount);
    let result;
    try {
        result = await gpu.execute(view, n, partition, quotas, targetCount);
    } finally {
        gpu.destroy();
    }

    const outCount = result.count;

    // Group input rows by the representative they merge into. Representatives
    // are emitted in slot order, which is the partition's spatial order, so the
    // output stays spatially coherent.
    const repIndex = new Int32Array(n).fill(-1);
    for (let k = 0; k < outCount; k++) repIndex[result.repSlots[k]] = k;

    const offsets = new Int32Array(outCount + 1);
    for (let slot = 0; slot < n; slot++) offsets[repIndex[result.assignment[slot]] + 1]++;
    for (let k = 0; k < outCount; k++) offsets[k + 1] += offsets[k];

    const members = new Int32Array(n);
    const cursor = offsets.slice(0, outCount);
    // Representative first in its own group, matching the reference's ordering.
    for (let k = 0; k < outCount; k++) members[cursor[k]++] = partition.order[result.repSlots[k]];
    for (let slot = 0; slot < n; slot++) {
        const k = repIndex[result.assignment[slot]];
        if (result.repSlots[k] !== slot) members[cursor[k]++] = partition.order[slot];
    }

    let maxGroup = 0;
    for (let k = 0; k < outCount; k++) maxGroup = Math.max(maxGroup, offsets[k + 1] - offsets[k]);
    logger.info(
        `voxels ${fmtCount(partition.leafCount)} · kept ${fmtCount(outCount)} · largest group ${maxGroup}`
    );
    group.end();

    const outMeta: ChunkSourceMetadata = {
        numGaussians: outCount,
        numLods: 1,
        lodCounts: [outCount],
        chunkSize: src.meta.chunkSize,
        numChunks: [Math.ceil(outCount / src.meta.chunkSize)],
        shBands: src.meta.shBands,
        model: src.meta.model,
        extraColumns: src.meta.extraColumns,
        transform: src.meta.transform,
        availableLayers: src.meta.availableLayers,
        layouts: src.meta.layouts
    };

    const mergeBar = logger.bar('merging', outCount);
    const producer = createBlockProducerSource(outMeta, () => voxelMergeStream(
        src, pool, view, members, offsets, outCount,
        src.meta.chunkSize, opts.compensation, count => mergeBar.tick(count)
    ));

    let closed = false;
    return {
        meta: producer.meta,
        read: request => producer.read(request),
        close: async () => {
            if (closed) return;
            closed = true;
            mergeBar.end();
            await producer.close();
            await src.close();
        }
    };
};

export { decimateSourceVoxel, type DecimateVoxelOptions };
