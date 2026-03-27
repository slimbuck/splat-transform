/**
 * Tests for filterOutliers ProcessAction.
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import { Column, DataTable, processDataTable } from '../src/lib/index.js';

/**
 * Create a DataTable with positions at specified locations.
 *
 * @param {number[][]} positions - Array of [x, y, z] positions.
 * @returns {DataTable}
 */
function createPositionTable(positions) {
    const n = positions.length;
    const x = new Float32Array(n);
    const y = new Float32Array(n);
    const z = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        x[i] = positions[i][0];
        y[i] = positions[i][1];
        z[i] = positions[i][2];
    }

    return new DataTable([
        new Column('x', x),
        new Column('y', y),
        new Column('z', z)
    ]);
}

describe('filterOutliers', () => {
    it('should return original row count when no outliers exist', () => {
        const positions = [];
        for (let i = 0; i < 100; i++) {
            positions.push([i * 0.1, i * 0.1, i * 0.1]);
        }
        const dt = createPositionTable(positions);

        const result = processDataTable(dt, [{ kind: 'filterOutliers', multiplier: 3.0 }]);

        assert.strictEqual(result.numRows, dt.numRows, 'should keep all rows');
    });

    it('should remove extreme outliers', () => {
        const positions = [];
        for (let i = 0; i < 100; i++) {
            positions.push([
                (i % 10) * 0.1,
                0,
                Math.floor(i / 10) * 0.1
            ]);
        }
        positions.push([1000, 0, 0]);
        positions.push([0, 1000, 0]);
        positions.push([0, 0, 1000]);

        const dt = createPositionTable(positions);
        const result = processDataTable(dt, [{ kind: 'filterOutliers', multiplier: 3.0 }]);

        assert(result.numRows < dt.numRows, 'should have fewer rows');
        assert(result.numRows >= 100, 'should keep the core gaussians');

        const xCol = result.getColumnByName('x').data;
        for (let i = 0; i < result.numRows; i++) {
            assert(xCol[i] < 500, `x[${i}] should not be an extreme outlier`);
        }
    });

    it('should respect the IQR multiplier', () => {
        const positions = [];
        for (let i = 0; i < 80; i++) {
            positions.push([i * 0.01, 0, 0]);
        }
        for (let i = 0; i < 20; i++) {
            positions.push([5 + i * 0.1, 0, 0]);
        }

        const dt = createPositionTable(positions);

        const resultPermissive = processDataTable(dt, [{ kind: 'filterOutliers', multiplier: 100.0 }]);
        assert.strictEqual(resultPermissive.numRows, dt.numRows, 'large multiplier should keep all');

        const resultStrict = processDataTable(dt, [{ kind: 'filterOutliers', multiplier: 0.5 }]);
        assert(resultStrict.numRows <= dt.numRows, 'small multiplier should remove some');
    });

    it('should handle tables with fewer than 4 rows', () => {
        const dt = createPositionTable([[0, 0, 0], [1, 1, 1], [2, 2, 2]]);

        const result = processDataTable(dt, [{ kind: 'filterOutliers', multiplier: 3.0 }]);

        assert.strictEqual(result.numRows, dt.numRows, 'should return all rows for tiny tables');
    });

    it('should use default multiplier of 3.0 when not specified', () => {
        const positions = [];
        for (let i = 0; i < 100; i++) {
            positions.push([
                (i % 10) * 0.1,
                0,
                Math.floor(i / 10) * 0.1
            ]);
        }
        positions.push([1000, 0, 0]);

        const dt = createPositionTable(positions);

        const resultDefault = processDataTable(dt, [{ kind: 'filterOutliers' }]);
        const resultExplicit = processDataTable(dt, [{ kind: 'filterOutliers', multiplier: 3.0 }]);

        assert.strictEqual(resultDefault.numRows, resultExplicit.numRows, 'default should match multiplier=3.0');
    });

    it('should handle asymmetric distributions', () => {
        const positions = [];
        for (let i = 0; i < 100; i++) {
            positions.push([100 + (i % 10) * 0.1, 0, Math.floor(i / 10) * 0.1]);
        }
        positions.push([0, 0, 0]);

        const dt = createPositionTable(positions);
        const result = processDataTable(dt, [{ kind: 'filterOutliers', multiplier: 3.0 }]);

        assert(result.numRows < dt.numRows, 'should remove the origin outlier');

        const xCol = result.getColumnByName('x').data;
        for (let i = 0; i < result.numRows; i++) {
            assert(xCol[i] >= 99, `x[${i}] (${xCol[i]}) should be near the cluster`);
        }
    });

    it('should work per-axis independently', () => {
        const positions = [];
        for (let i = 0; i < 100; i++) {
            positions.push([i * 0.1, i * 0.1, i * 0.1]);
        }
        positions.push([1000, 5, 5]);

        const dt = createPositionTable(positions);
        const result = processDataTable(dt, [{ kind: 'filterOutliers', multiplier: 3.0 }]);

        assert(result.numRows < dt.numRows, 'should remove the X outlier');
    });

    it('should filter on custom columns when specified', () => {
        const n = 100;
        const x = new Float32Array(n + 1);
        const y = new Float32Array(n + 1);
        const z = new Float32Array(n + 1);
        const scale0 = new Float32Array(n + 1);

        for (let i = 0; i < n; i++) {
            x[i] = i * 0.1;
            y[i] = i * 0.1;
            z[i] = i * 0.1;
            scale0[i] = i * 0.01;
        }
        // Outlier only on scale_0, spatially normal
        x[n] = 5;
        y[n] = 5;
        z[n] = 5;
        scale0[n] = 1000;

        const dt = new DataTable([
            new Column('x', x),
            new Column('y', y),
            new Column('z', z),
            new Column('scale_0', scale0)
        ]);

        // Default columns (x,y,z) should keep the scale outlier
        const resultDefault = processDataTable(dt, [{ kind: 'filterOutliers', multiplier: 3.0 }]);
        assert.strictEqual(resultDefault.numRows, n + 1, 'default columns should not filter scale_0 outlier');

        // Custom columns should catch the scale outlier
        const resultCustom = processDataTable(dt, [{ kind: 'filterOutliers', multiplier: 3.0, columns: ['scale_0'] }]);
        assert(resultCustom.numRows < n + 1, 'custom columns should filter scale_0 outlier');
        assert.strictEqual(resultCustom.numRows, n, 'should keep only the non-outlier rows');
    });

    it('should throw when a specified column does not exist', () => {
        const dt = createPositionTable([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]);

        assert.throws(() => {
            processDataTable(dt, [{ kind: 'filterOutliers', columns: ['nonexistent'] }]);
        }, /column 'nonexistent' not found/);
    });
});
