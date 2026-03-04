/**
 * Tests for connected component filtering (keep only largest 6-connected component).
 *
 * Bit layout: bitIdx = lx + ly*4 + lz*16
 * lo = bits 0-31 (lz=0: bits 0-15, lz=1: bits 16-31)
 * hi = bits 32-63 (lz=2: bits 0-15, lz=3: bits 16-31)
 */

import { describe, it } from 'node:test';
import assert from 'node:assert';

import {
    BlockAccumulator,
    xyzToMorton,
    popcount
} from '../src/lib/voxel/sparse-octree.js';
import { filterConnectedComponents } from '../src/lib/voxel/voxel-connected-components.js';

const SOLID_LO = 0xFFFFFFFF >>> 0;
const SOLID_HI = 0xFFFFFFFF >>> 0;

/**
 * Set a single voxel bit in a 4x4x4 block.
 * Returns [lo, hi] mask pair.
 */
function voxelBit(lx, ly, lz) {
    const bitIdx = lx + ly * 4 + lz * 16;
    if (bitIdx < 32) {
        return [1 << bitIdx, 0];
    }
    return [0, 1 << (bitIdx - 32)];
}

/**
 * Combine multiple voxel positions into a single [lo, hi] mask.
 */
function voxelMask(...positions) {
    let lo = 0, hi = 0;
    for (const [lx, ly, lz] of positions) {
        const [blo, bhi] = voxelBit(lx, ly, lz);
        lo |= blo;
        hi |= bhi;
    }
    return [lo, hi];
}

/**
 * Count total set voxels across lo and hi.
 */
function countVoxels(lo, hi) {
    return popcount(lo) + popcount(hi);
}

/**
 * Check if a specific voxel is set in a [lo, hi] mask.
 */
function isVoxelSet(lo, hi, lx, ly, lz) {
    const bitIdx = lx + ly * 4 + lz * 16;
    if (bitIdx < 32) {
        return (lo & (1 << bitIdx)) !== 0;
    }
    return (hi & (1 << (bitIdx - 32))) !== 0;
}

// ============================================================================
// Basic connectivity
// ============================================================================

describe('filterConnectedComponents', function () {
    describe('single component', function () {
        it('should preserve a single connected cluster', function () {
            const acc = new BlockAccumulator();
            // Line of 3 voxels along X — all connected
            const [lo, hi] = voxelMask([0, 0, 0], [1, 0, 0], [2, 0, 0]);
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterConnectedComponents(acc);

            const mixed = result.getMixedBlocks();
            assert.strictEqual(mixed.morton.length, 1);
            assert.strictEqual(countVoxels(mixed.masks[0], mixed.masks[1]), 3,
                'All connected voxels should be preserved');
        });

        it('should return original accumulator when only one component exists', function () {
            const acc = new BlockAccumulator();
            // 2x2x2 cube — all connected
            const [lo, hi] = voxelMask(
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]
            );
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result, acc, 'Should return original accumulator (no filtering needed)');
        });

        it('should handle empty accumulator', function () {
            const acc = new BlockAccumulator();
            const result = filterConnectedComponents(acc);
            assert.strictEqual(result.count, 0);
        });
    });

    // ============================================================================
    // Multiple components
    // ============================================================================

    describe('multiple components', function () {
        it('should keep the larger of two disconnected clusters', function () {
            const acc = new BlockAccumulator();
            // Large cluster: 4 connected voxels along X
            // Small cluster: 2 connected voxels, isolated from the large one
            const [lo, hi] = voxelMask(
                [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0],  // large (4 voxels)
                [0, 3, 3], [1, 3, 3]                            // small (2 voxels)
            );
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterConnectedComponents(acc);

            const mixed = result.getMixedBlocks();
            assert.strictEqual(mixed.morton.length, 1);
            const rlo = mixed.masks[0];
            const rhi = mixed.masks[1];

            assert.strictEqual(countVoxels(rlo, rhi), 4, 'Only the larger cluster should remain');
            assert.ok(isVoxelSet(rlo, rhi, 0, 0, 0), '(0,0,0) should be preserved');
            assert.ok(isVoxelSet(rlo, rhi, 3, 0, 0), '(3,0,0) should be preserved');
            assert.ok(!isVoxelSet(rlo, rhi, 0, 3, 3), '(0,3,3) should be removed');
            assert.ok(!isVoxelSet(rlo, rhi, 1, 3, 3), '(1,3,3) should be removed');
        });

        it('should remove a single disconnected voxel when a larger cluster exists', function () {
            const acc = new BlockAccumulator();
            // 2 connected voxels + 1 isolated voxel
            const [lo, hi] = voxelMask(
                [1, 1, 1], [2, 1, 1],  // connected pair (2 voxels)
                [0, 3, 3]               // isolated (1 voxel)
            );
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterConnectedComponents(acc);

            const mixed = result.getMixedBlocks();
            assert.strictEqual(mixed.morton.length, 1);
            const rlo = mixed.masks[0];
            const rhi = mixed.masks[1];

            assert.strictEqual(countVoxels(rlo, rhi), 2);
            assert.ok(isVoxelSet(rlo, rhi, 1, 1, 1));
            assert.ok(isVoxelSet(rlo, rhi, 2, 1, 1));
            assert.ok(!isVoxelSet(rlo, rhi, 0, 3, 3));
        });

        it('should handle block becoming empty after filtering', function () {
            const acc = new BlockAccumulator();
            // Block 0: large cluster
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

            // Block 1: small disconnected cluster (not adjacent to block 0)
            const [lo, hi] = voxelMask([1, 1, 1], [2, 1, 1]);
            acc.addBlock(xyzToMorton(5, 5, 5), lo, hi);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result.solidCount, 1, 'Solid block should remain');
            assert.strictEqual(result.mixedCount, 0, 'Disconnected mixed block should be removed');
        });
    });

    // ============================================================================
    // Solid blocks
    // ============================================================================

    describe('solid blocks', function () {
        it('should keep connected solid blocks', function () {
            const acc = new BlockAccumulator();
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            acc.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result.solidCount, 2, 'Adjacent solid blocks should be preserved');
        });

        it('should remove disconnected solid block when smaller', function () {
            const acc = new BlockAccumulator();
            // Large cluster: 2 adjacent solid blocks (128 voxels)
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            acc.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);

            // Small cluster: 1 isolated solid block (64 voxels)
            acc.addBlock(xyzToMorton(10, 10, 10), SOLID_LO, SOLID_HI);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result.solidCount, 2, 'Only the 2 connected blocks should remain');
        });

        it('should keep single solid block when it is the only block', function () {
            const acc = new BlockAccumulator();
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result.solidCount, 1);
        });
    });

    // ============================================================================
    // Cross-block connectivity
    // ============================================================================

    describe('cross-block connectivity', function () {
        it('should connect voxels across block boundaries in +X direction', function () {
            const acc = new BlockAccumulator();
            // Block (0,0,0): voxels at lx=3 face, connected to block (1,0,0)
            const [lo0, hi0] = voxelMask([3, 0, 0], [3, 1, 0]);
            acc.addBlock(xyzToMorton(0, 0, 0), lo0, hi0);

            // Block (1,0,0): voxels at lx=0 face, touching block (0,0,0)
            const [lo1, hi1] = voxelMask([0, 0, 0], [0, 1, 0]);
            acc.addBlock(xyzToMorton(1, 0, 0), lo1, hi1);

            // Block (5,5,5): isolated single voxel
            const [lo2, hi2] = voxelBit(1, 1, 1);
            acc.addBlock(xyzToMorton(5, 5, 5), lo2, hi2);

            const result = filterConnectedComponents(acc);

            // The two cross-block-connected blocks form the larger component (4 voxels)
            assert.strictEqual(result.count, 2, 'Two connected blocks should remain');
            assert.strictEqual(result.mixedCount, 2);
        });

        it('should connect voxels across block boundaries in +Y direction', function () {
            const acc = new BlockAccumulator();
            // Block (0,0,0): voxels at ly=3
            const [lo0, hi0] = voxelMask([1, 3, 0], [2, 3, 0]);
            acc.addBlock(xyzToMorton(0, 0, 0), lo0, hi0);

            // Block (0,1,0): voxels at ly=0
            const [lo1, hi1] = voxelMask([1, 0, 0], [2, 0, 0]);
            acc.addBlock(xyzToMorton(0, 1, 0), lo1, hi1);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result, acc, 'Single component, should return original');
        });

        it('should connect voxels across block boundaries in +Z direction', function () {
            const acc = new BlockAccumulator();
            // Block (0,0,0): voxels at lz=3
            const [lo0, hi0] = voxelMask([1, 1, 3], [2, 1, 3]);
            acc.addBlock(xyzToMorton(0, 0, 0), lo0, hi0);

            // Block (0,0,1): voxels at lz=0
            const [lo1, hi1] = voxelMask([1, 1, 0], [2, 1, 0]);
            acc.addBlock(xyzToMorton(0, 0, 1), lo1, hi1);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result, acc, 'Single component, should return original');
        });

        it('should connect mixed block to adjacent solid block', function () {
            const acc = new BlockAccumulator();
            // Solid block at (0,0,0)
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);

            // Mixed block at (1,0,0) with voxels on lx=0 face (touching solid)
            const [lo, hi] = voxelMask([0, 1, 1], [0, 2, 1]);
            acc.addBlock(xyzToMorton(1, 0, 0), lo, hi);

            // Isolated voxel far away
            const [lo2, hi2] = voxelBit(2, 2, 2);
            acc.addBlock(xyzToMorton(10, 10, 10), lo2, hi2);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result.solidCount, 1, 'Solid block preserved');
            assert.strictEqual(result.mixedCount, 1, 'Connected mixed block preserved');
            // The isolated block at (10,10,10) should be removed
            const mixed = result.getMixedBlocks();
            assert.strictEqual(mixed.morton.length, 1);
        });

        it('should not connect non-touching face voxels across blocks', function () {
            const acc = new BlockAccumulator();
            // Block (0,0,0): voxel at lx=3, ly=0, lz=0
            const [lo0, hi0] = voxelBit(3, 0, 0);
            acc.addBlock(xyzToMorton(0, 0, 0), lo0, hi0);

            // Block (1,0,0): voxel at lx=0, ly=3, lz=3 — same face but different position
            const [lo1, hi1] = voxelBit(0, 3, 3);
            acc.addBlock(xyzToMorton(1, 0, 0), lo1, hi1);

            const result = filterConnectedComponents(acc);

            // These voxels are on the same face plane but NOT 6-connected neighbors
            // (different ly, lz), so they form 2 separate components of equal size.
            // The filter keeps the largest; with a tie, one is kept.
            assert.strictEqual(result.count, 1, 'Only one component survives');
        });
    });

    // ============================================================================
    // Connectivity patterns
    // ============================================================================

    describe('connectivity patterns', function () {
        it('should treat diagonal voxels as disconnected (6-connected only)', function () {
            const acc = new BlockAccumulator();
            // Two diagonally adjacent voxels — NOT 6-connected
            const [lo, hi] = voxelMask([0, 0, 0], [1, 1, 1]);
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterConnectedComponents(acc);

            // Two equal-size components; filter keeps one
            assert.strictEqual(result.count, 1);
            const mixed = result.getMixedBlocks();
            assert.strictEqual(countVoxels(mixed.masks[0], mixed.masks[1]), 1);
        });

        it('should handle L-shaped cluster as one component', function () {
            const acc = new BlockAccumulator();
            // L-shape: (0,0,0)-(1,0,0)-(2,0,0)-(2,1,0)
            const [lo, hi] = voxelMask(
                [0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0]
            );
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result, acc, 'L-shape is one component');
        });

        it('should preserve the Z-connected chain across lo/hi word boundary', function () {
            const acc = new BlockAccumulator();
            // Chain along Z: lz=0 -> lz=1 -> lz=2 -> lz=3 at lx=1, ly=1
            // This crosses the lo/hi boundary (lz=1 is in lo, lz=2 is in hi)
            const [lo, hi] = voxelMask(
                [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 3]
            );
            acc.addBlock(xyzToMorton(0, 0, 0), lo, hi);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result, acc, 'Z-chain crossing lo/hi is one component');
        });
    });

    // ============================================================================
    // Multi-block scenarios
    // ============================================================================

    describe('multi-block scenarios', function () {
        it('should handle a chain of blocks forming one component', function () {
            const acc = new BlockAccumulator();
            // 3 blocks in a row along X, each with face voxels touching the next
            const [lo0, hi0] = voxelMask([3, 1, 1], [2, 1, 1]);
            acc.addBlock(xyzToMorton(0, 0, 0), lo0, hi0);

            const [lo1, hi1] = voxelMask([0, 1, 1], [1, 1, 1], [2, 1, 1], [3, 1, 1]);
            acc.addBlock(xyzToMorton(1, 0, 0), lo1, hi1);

            const [lo2, hi2] = voxelMask([0, 1, 1], [1, 1, 1]);
            acc.addBlock(xyzToMorton(2, 0, 0), lo2, hi2);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result, acc, 'Chain of 3 blocks is one component');
        });

        it('should separate two disconnected multi-block clusters', function () {
            const acc = new BlockAccumulator();
            // Cluster A: 2 solid blocks (128 voxels)
            acc.addBlock(xyzToMorton(0, 0, 0), SOLID_LO, SOLID_HI);
            acc.addBlock(xyzToMorton(1, 0, 0), SOLID_LO, SOLID_HI);

            // Cluster B: 1 mixed block with 4 voxels
            const [lo, hi] = voxelMask([0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]);
            acc.addBlock(xyzToMorton(20, 20, 20), lo, hi);

            const result = filterConnectedComponents(acc);

            assert.strictEqual(result.solidCount, 2, 'Larger solid cluster kept');
            assert.strictEqual(result.mixedCount, 0, 'Smaller mixed cluster removed');
        });
    });
});
