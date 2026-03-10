/**
 * Spark-style voxel octree merging for discrete LoD generation.
 *
 * Places each splat into its "natural" octree level based on size,
 * then merges bottom-up to produce coarser representations. Discrete
 * LoD levels are extracted as "cuts" at different octree depths.
 */

import { Column, DataTable } from '../data-table/data-table.js';
import { combine } from '../data-table/combine.js';
import { sigmoid } from '../utils/math.js';
import { logger } from '../utils/logger.js';
import {
    quatScaleToCovariance,
    covarianceToQuatScale,
    mergeGaussians,
    ellipsoidSurfaceArea,
    type SymMat3
} from './covariance-utils.js';

/**
 * Compute the octree level at which a splat naturally belongs.
 * A splat belongs to the level where its size falls within [0.5*step, step)
 * for that level's grid spacing (base^level).
 *
 * @param maxScale - The maximum linear scale of the splat: max(sx, sy, sz)
 * @param base - The octree base (default 1.5, range 1.1 to 2.0)
 * @returns The natural octree level (can be negative)
 */
const computeNaturalLevel = (maxScale: number, base: number): number => {
    if (maxScale <= 0) return -100;
    const splatSize = 2 * maxScale;
    return Math.floor(Math.log(splatSize) / Math.log(base));
};

/**
 * Create a hash key for a voxel cell at a given level and integer coordinates.
 */
const voxelKey = (level: number, ix: number, iy: number, iz: number): string => {
    return `${level}:${ix}:${iy}:${iz}`;
};

interface VoxelCell {
    level: number;
    ix: number;
    iy: number;
    iz: number;
    indices: number[];
}

/**
 * Build a voxel grid by placing each splat into its natural level cell.
 */
const buildVoxelGrid = (
    dataTable: DataTable,
    base: number
): { cells: Map<string, VoxelCell>; minLevel: number; maxLevel: number } => {
    const x = dataTable.getColumnByName('x').data;
    const y = dataTable.getColumnByName('y').data;
    const z = dataTable.getColumnByName('z').data;
    const s0 = dataTable.getColumnByName('scale_0').data;
    const s1 = dataTable.getColumnByName('scale_1').data;
    const s2 = dataTable.getColumnByName('scale_2').data;

    const cells = new Map<string, VoxelCell>();
    let minLevel = Infinity;
    let maxLevel = -Infinity;
    const n = dataTable.numRows;

    for (let i = 0; i < n; i++) {
        const sx = Math.exp(s0[i]);
        const sy = Math.exp(s1[i]);
        const sz = Math.exp(s2[i]);
        const maxScale = Math.max(sx, sy, sz);

        const level = computeNaturalLevel(maxScale, base);
        const step = Math.pow(base, level);

        const ix = Math.floor(x[i] / step);
        const iy = Math.floor(y[i] / step);
        const iz = Math.floor(z[i] / step);

        const key = voxelKey(level, ix, iy, iz);
        let cell = cells.get(key);
        if (!cell) {
            cell = { level, ix, iy, iz, indices: [] };
            cells.set(key, cell);
        }
        cell.indices.push(i);

        minLevel = Math.min(minLevel, level);
        maxLevel = Math.max(maxLevel, level);
    }

    return { cells, minLevel, maxLevel };
};

/**
 * Compute the extended opacity parameter D for a merged splat.
 *
 * Following Spark: D = sqrt(1 + e * ln(A)) where A = totalWeight / mergedArea.
 * For A <= 1, D equals A directly (standard opacity range).
 */
const computeExtendedOpacity = (totalWeight: number, mergedArea: number): number => {
    if (mergedArea <= 0) return 0;
    const A = totalWeight / mergedArea;
    if (A <= 1) return A;
    return Math.sqrt(1 + Math.E * Math.log(A));
};

/**
 * Merge a group of splats from the DataTable into a single merged splat row.
 *
 * Returns an object with all column values for the merged splat.
 */
const mergeSplatGroup = (
    dataTable: DataTable,
    indices: number[]
): Record<string, number> => {
    if (indices.length === 1) {
        const row: Record<string, number> = {};
        for (const col of dataTable.columns) {
            row[col.name] = col.data[indices[0]];
        }
        return row;
    }

    const xCol = dataTable.getColumnByName('x').data;
    const yCol = dataTable.getColumnByName('y').data;
    const zCol = dataTable.getColumnByName('z').data;
    const opCol = dataTable.getColumnByName('opacity').data;
    const s0Col = dataTable.getColumnByName('scale_0').data;
    const s1Col = dataTable.getColumnByName('scale_1').data;
    const s2Col = dataTable.getColumnByName('scale_2').data;
    const r0Col = dataTable.getColumnByName('rot_0').data;
    const r1Col = dataTable.getColumnByName('rot_1').data;
    const r2Col = dataTable.getColumnByName('rot_2').data;
    const r3Col = dataTable.getColumnByName('rot_3').data;

    // Collect positions, covariances, and weights
    const positions: number[][] = [];
    const covariances: SymMat3[] = [];
    const weights: number[] = [];
    const linearOpacities: number[] = [];

    for (const idx of indices) {
        const sx = Math.exp(s0Col[idx]);
        const sy = Math.exp(s1Col[idx]);
        const sz = Math.exp(s2Col[idx]);
        const opacity = sigmoid(opCol[idx]);
        const area = ellipsoidSurfaceArea(sx, sy, sz);
        const weight = opacity * area;

        positions.push([xCol[idx], yCol[idx], zCol[idx]]);
        covariances.push(
            quatScaleToCovariance(r0Col[idx], r1Col[idx], r2Col[idx], r3Col[idx], sx, sy, sz)
        );
        weights.push(weight);
        linearOpacities.push(opacity);
    }

    // Merge center and covariance
    const merged = mergeGaussians(positions, covariances, weights);
    const { qw, qx, qy, qz, sx, sy, sz } = covarianceToQuatScale(merged.covariance);

    // Extended opacity D
    const mergedArea = ellipsoidSurfaceArea(sx, sy, sz);
    const D = computeExtendedOpacity(merged.totalWeight, mergedArea);

    // Weighted SH color blending
    const row: Record<string, number> = {};
    const shKeys: string[] = [];
    for (const col of dataTable.columns) {
        if (col.name.startsWith('f_dc') || col.name.startsWith('f_rest')) {
            shKeys.push(col.name);
        }
    }

    for (const key of shKeys) {
        const colData = dataTable.getColumnByName(key).data;
        let weightedSum = 0;
        for (let i = 0; i < indices.length; i++) {
            weightedSum += colData[indices[i]] * weights[i];
        }
        row[key] = merged.totalWeight > 0 ? weightedSum / merged.totalWeight : 0;
    }

    // Set core properties
    row['x'] = merged.center[0];
    row['y'] = merged.center[1];
    row['z'] = merged.center[2];
    row['scale_0'] = Math.log(sx);
    row['scale_1'] = Math.log(sy);
    row['scale_2'] = Math.log(sz);
    row['rot_0'] = qw;
    row['rot_1'] = qx;
    row['rot_2'] = qy;
    row['rot_3'] = qz;

    // Store D in a special column; the SOG writer will handle encoding
    row['d_opacity'] = D;

    // Store logit(min(D, 1)) for the standard opacity column
    const clampedAlpha = Math.min(D, 0.999);
    row['opacity'] = Math.log(clampedAlpha / (1 - clampedAlpha));

    // Zero out any other columns not handled above
    for (const col of dataTable.columns) {
        if (!(col.name in row)) {
            row[col.name] = 0;
        }
    }

    return row;
};

/**
 * Extract a discrete LoD level from the octree hierarchy.
 *
 * For a given cut level, we collect:
 * - All original (unmerged) splats at levels >= cutLevel
 * - All merged nodes at levels < cutLevel (these replace their children)
 *
 * The result is a self-contained set of splats at a coarser resolution.
 */
const extractLodLevel = (
    dataTable: DataTable,
    cells: Map<string, VoxelCell>,
    minLevel: number,
    maxLevel: number,
    cutLevel: number,
    base: number
): DataTable => {
    // Collect original splat indices that are at or above the cut level
    const keepIndices: number[] = [];
    const mergedRows: Record<string, number>[] = [];

    for (const [, cell] of cells) {
        if (cell.level >= cutLevel) {
            // These splats are at a coarse enough level, keep them as-is
            keepIndices.push(...cell.indices);
        }
    }

    // For splats below the cut level, merge them in voxel cells at the cut level
    const fineIndicesMap = new Map<string, number[]>();
    const cutStep = Math.pow(base, cutLevel);

    const xCol = dataTable.getColumnByName('x').data;
    const yCol = dataTable.getColumnByName('y').data;
    const zCol = dataTable.getColumnByName('z').data;

    for (const [, cell] of cells) {
        if (cell.level >= cutLevel) continue;

        for (const idx of cell.indices) {
            const ix = Math.floor(xCol[idx] / cutStep);
            const iy = Math.floor(yCol[idx] / cutStep);
            const iz = Math.floor(zCol[idx] / cutStep);
            const key = `${ix}:${iy}:${iz}`;

            let group = fineIndicesMap.get(key);
            if (!group) {
                group = [];
                fineIndicesMap.set(key, group);
            }
            group.push(idx);
        }
    }

    // Merge each group into a single splat
    for (const [, group] of fineIndicesMap) {
        if (group.length === 1) {
            keepIndices.push(group[0]);
        } else {
            mergedRows.push(mergeSplatGroup(dataTable, group));
        }
    }

    // Build the output DataTable
    // Start with the kept original splats
    let result: DataTable;
    if (keepIndices.length > 0) {
        result = dataTable.permuteRows(new Uint32Array(keepIndices));
    } else {
        // Edge case: all splats were merged
        result = new DataTable(dataTable.columns.map(c => new Column(c.name, new (c.data.constructor as any)(0))));
    }

    if (mergedRows.length > 0) {
        // Ensure d_opacity column exists in result
        if (!result.hasColumn('d_opacity')) {
            const dOpData = new Float32Array(result.numRows);
            // Original splats have D = sigmoid(opacity)
            const opCol = result.getColumnByName('opacity')?.data;
            if (opCol) {
                for (let i = 0; i < result.numRows; i++) {
                    dOpData[i] = sigmoid(opCol[i]);
                }
            }
            result.addColumn(new Column('d_opacity', dOpData));
        }

        // Build a DataTable from merged rows
        const mergedColumns = result.columns.map(c => {
            const constructor = c.data.constructor as new (length: number) => any;
            const data = new constructor(mergedRows.length);
            for (let i = 0; i < mergedRows.length; i++) {
                data[i] = mergedRows[i][c.name] ?? 0;
            }
            return new Column(c.name, data);
        });

        const mergedTable = new DataTable(mergedColumns);
        result = combine([result, mergedTable]);
    }

    return result;
};

export interface DecimateOptions {
    /** Number of octree levels to merge through (higher = more aggressive). */
    levels: number;
    /** Octree base for level spacing (default 1.5, range 1.1-2.0). */
    base: number;
}

/**
 * Decimate splats using Spark-style voxel octree merging.
 *
 * Nearby splats are grouped by their natural voxel level and merged into
 * single representative splats. The `levels` parameter controls how many
 * octree levels above the minimum are merged, providing a simple knob for
 * decimation aggressiveness.
 *
 * @param dataTable - Input splat data
 * @param options - Decimation parameters
 * @returns A decimated DataTable with fewer splats
 */
const decimateSplats = (
    dataTable: DataTable,
    options: DecimateOptions
): DataTable => {
    const { levels, base } = options;

    logger.log(`Building voxel grid with base ${base}...`);
    const { cells, minLevel, maxLevel } = buildVoxelGrid(dataTable, base);

    logger.log(`Voxel grid: ${cells.size} cells, levels ${minLevel} to ${maxLevel}`);

    const cutLevel = minLevel + levels;
    logger.log(`Decimating with cut at octree level ${cutLevel} (merging ${levels} levels above minimum ${minLevel})...`);
    logger.log(`Input: ${dataTable.numRows} splats`);

    const result = extractLodLevel(dataTable, cells, minLevel, maxLevel, cutLevel, base);

    logger.log(`Output: ${result.numRows} splats (${((result.numRows / dataTable.numRows) * 100).toFixed(1)}% of original)`);

    return result;
};

export { decimateSplats, computeNaturalLevel, mergeSplatGroup };
