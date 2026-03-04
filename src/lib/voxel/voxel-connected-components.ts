import {
    BlockAccumulator,
    xyzToMorton,
    mortonToXYZ,
    popcount
} from './sparse-octree';
import { logger } from '../utils/logger';

const SOLID_MASK = 0xFFFFFFFF >>> 0;

// ============================================================================
// Union-Find with path compression and union-by-rank
// ============================================================================

class UnionFind {
    private _parent: Int32Array;

    private _rank: Uint8Array;

    constructor(size: number) {
        this._parent = new Int32Array(size);
        this._rank = new Uint8Array(size);
        for (let i = 0; i < size; i++) {
            this._parent[i] = i;
        }
    }

    find(x: number): number {
        let root = x;
        while (this._parent[root] !== root) root = this._parent[root];
        while (this._parent[x] !== root) {
            const next = this._parent[x];
            this._parent[x] = root;
            x = next;
        }
        return root;
    }

    union(a: number, b: number): void {
        const ra = this.find(a);
        const rb = this.find(b);
        if (ra === rb) return;
        if (this._rank[ra] < this._rank[rb]) {
            this._parent[ra] = rb;
        } else if (this._rank[ra] > this._rank[rb]) {
            this._parent[rb] = ra;
        } else {
            this._parent[rb] = ra;
            this._rank[ra]++;
        }
    }
}

// ============================================================================
// Face voxel pair tables for cross-block unions
// ============================================================================
// Bit layout: bitIdx = lx + ly*4 + lz*16
// Each table stores 16 pairs: [ourBit0, adjBit0, ourBit1, adjBit1, ...] (32 entries)

const FACE_X = new Uint8Array(32);
const FACE_Y = new Uint8Array(32);
const FACE_Z = new Uint8Array(32);

for (let b = 0; b < 4; b++) {
    for (let a = 0; a < 4; a++) {
        const idx = (b * 4 + a) * 2;
        // +X: our lx=3 <-> adj lx=0, varying ly=a, lz=b
        FACE_X[idx] = 3 + a * 4 + b * 16;
        FACE_X[idx + 1] = 0 + a * 4 + b * 16;
        // +Y: our ly=3 <-> adj ly=0, varying lx=a, lz=b
        FACE_Y[idx] = a + 12 + b * 16;
        FACE_Y[idx + 1] = a + b * 16;
        // +Z: our lz=3 <-> adj lz=0, varying lx=a, ly=b
        FACE_Z[idx] = a + b * 4 + 48;
        FACE_Z[idx + 1] = a + b * 4;
    }
}

// ============================================================================
// Main function
// ============================================================================

/**
 * Remove small disconnected voxel clusters, keeping only the largest
 * 6-connected component.
 *
 * Uses Union-Find with path compression and union-by-rank. Connectivity
 * is computed both within 4x4x4 blocks (via bit-index arithmetic) and
 * across block boundaries (via adjacent block lookups). Solid blocks are
 * treated as fully connected internally (all 64 voxels share one component).
 *
 * @param accumulator - BlockAccumulator with voxelization results
 * @returns New BlockAccumulator containing only the largest connected component
 */
function filterConnectedComponents(accumulator: BlockAccumulator): BlockAccumulator {
    const mixed = accumulator.getMixedBlocks();
    const solid = accumulator.getSolidBlocks();
    const masks = mixed.masks;

    const totalBlocks = mixed.morton.length + solid.length;
    if (totalBlocks === 0) return new BlockAccumulator();

    // Build lookup: morton -> index in the respective array
    const solidIndexMap = new Map<number, number>();
    for (let i = 0; i < solid.length; i++) {
        solidIndexMap.set(solid[i], i);
    }

    const mixedMap = new Map<number, number>();
    for (let i = 0; i < mixed.morton.length; i++) {
        mixedMap.set(mixed.morton[i], i);
    }

    // --- Phase 1: Count voxels, assign base IDs, precompute bit-to-offset tables ---

    const mixedBaseIds = new Int32Array(mixed.morton.length);
    const solidBaseIds = new Int32Array(solid.length);
    let totalVoxels = 0;

    for (let i = 0; i < mixed.morton.length; i++) {
        mixedBaseIds[i] = totalVoxels;
        totalVoxels += popcount(masks[i * 2]) + popcount(masks[i * 2 + 1]);
    }
    for (let i = 0; i < solid.length; i++) {
        solidBaseIds[i] = totalVoxels;
        totalVoxels += 64;
    }

    if (totalVoxels === 0) return new BlockAccumulator();

    // For each mixed block, map bitIdx -> compact offset (or -1 if bit unset).
    // This avoids repeated popcount calls during union operations.
    const bitOffsets = new Int8Array(mixed.morton.length * 64);
    bitOffsets.fill(-1);

    for (let i = 0; i < mixed.morton.length; i++) {
        const lo = masks[i * 2];
        const hi = masks[i * 2 + 1];
        const base = i * 64;
        let off = 0;
        for (let b = 0; b < 32; b++) {
            if (lo & (1 << b)) bitOffsets[base + b] = off++;
        }
        for (let b = 0; b < 32; b++) {
            if (hi & (1 << b)) bitOffsets[base + 32 + b] = off++;
        }
    }

    const uf = new UnionFind(totalVoxels);

    // --- Phase 2: Intra-block unions ---

    // Mixed blocks: for each set voxel, union with its +X, +Y, +Z neighbor if set
    for (let i = 0; i < mixed.morton.length; i++) {
        const base = mixedBaseIds[i];
        const offBase = i * 64;

        for (let lz = 0; lz < 4; lz++) {
            for (let ly = 0; ly < 4; ly++) {
                for (let lx = 0; lx < 4; lx++) {
                    const bitIdx = lx + ly * 4 + lz * 16;
                    const off = bitOffsets[offBase + bitIdx];
                    if (off === -1) continue;

                    const id = base + off;

                    if (lx < 3) {
                        const nOff = bitOffsets[offBase + bitIdx + 1];
                        if (nOff !== -1) uf.union(id, base + nOff);
                    }
                    if (ly < 3) {
                        const nOff = bitOffsets[offBase + bitIdx + 4];
                        if (nOff !== -1) uf.union(id, base + nOff);
                    }
                    if (lz < 3) {
                        const nOff = bitOffsets[offBase + bitIdx + 16];
                        if (nOff !== -1) uf.union(id, base + nOff);
                    }
                }
            }
        }
    }

    // Solid blocks: all 64 voxels are connected — chain-union into one component
    for (let i = 0; i < solid.length; i++) {
        const base = solidBaseIds[i];
        for (let j = 1; j < 64; j++) {
            uf.union(base, base + j);
        }
    }

    // --- Phase 3: Cross-block unions (only +X, +Y, +Z to cover each edge once) ---

    const processFaceUnions = (
        ourMorton: number, adjMorton: number,
        facePairs: Uint8Array
    ): void => {
        const ourSolidIdx = solidIndexMap.get(ourMorton);
        const ourMixedIdx = mixedMap.get(ourMorton);
        const adjSolidIdx = solidIndexMap.get(adjMorton);
        const adjMixedIdx = mixedMap.get(adjMorton);

        if (adjSolidIdx === undefined && adjMixedIdx === undefined) return;

        const ourIsSolid = ourSolidIdx !== undefined;
        const adjIsSolid = adjSolidIdx !== undefined;

        // Both solid: one union suffices since each block is already one component
        if (ourIsSolid && adjIsSolid) {
            uf.union(solidBaseIds[ourSolidIdx], solidBaseIds[adjSolidIdx]);
            return;
        }

        for (let p = 0; p < 32; p += 2) {
            const ourBit = facePairs[p];
            const adjBit = facePairs[p + 1];

            let ourId: number;
            if (ourIsSolid) {
                ourId = solidBaseIds[ourSolidIdx!] + ourBit;
            } else {
                const off = bitOffsets[ourMixedIdx! * 64 + ourBit];
                if (off === -1) continue;
                ourId = mixedBaseIds[ourMixedIdx!] + off;
            }

            let adjId: number;
            if (adjIsSolid) {
                adjId = solidBaseIds[adjSolidIdx!] + adjBit;
            } else {
                const off = bitOffsets[adjMixedIdx! * 64 + adjBit];
                if (off === -1) continue;
                adjId = mixedBaseIds[adjMixedIdx!] + off;
            }

            uf.union(ourId, adjId);
        }
    };

    for (let i = 0; i < mixed.morton.length; i++) {
        const morton = mixed.morton[i];
        const [bx, by, bz] = mortonToXYZ(morton);
        processFaceUnions(morton, xyzToMorton(bx + 1, by, bz), FACE_X);
        processFaceUnions(morton, xyzToMorton(bx, by + 1, bz), FACE_Y);
        processFaceUnions(morton, xyzToMorton(bx, by, bz + 1), FACE_Z);
    }
    for (let i = 0; i < solid.length; i++) {
        const morton = solid[i];
        const [bx, by, bz] = mortonToXYZ(morton);
        processFaceUnions(morton, xyzToMorton(bx + 1, by, bz), FACE_X);
        processFaceUnions(morton, xyzToMorton(bx, by + 1, bz), FACE_Y);
        processFaceUnions(morton, xyzToMorton(bx, by, bz + 1), FACE_Z);
    }

    // --- Phase 4: Find largest component ---

    const componentSizes = new Map<number, number>();
    for (let i = 0; i < totalVoxels; i++) {
        const root = uf.find(i);
        componentSizes.set(root, (componentSizes.get(root) ?? 0) + 1);
    }

    let largestRoot = -1;
    let largestSize = 0;
    for (const [root, size] of componentSizes) {
        if (size > largestSize) {
            largestSize = size;
            largestRoot = root;
        }
    }

    if (componentSizes.size <= 1) {
        logger.log(`connected components: 1 component, ${totalVoxels} voxels, no filtering needed`);
        return accumulator;
    }

    // --- Phase 5: Rebuild accumulator with only the largest component ---

    const result = new BlockAccumulator();
    let voxelsRemoved = 0;

    for (let i = 0; i < mixed.morton.length; i++) {
        const origLo = masks[i * 2];
        const origHi = masks[i * 2 + 1];
        const base = mixedBaseIds[i];
        let newLo = 0;
        let newHi = 0;
        let offset = 0;

        for (let b = 0; b < 32; b++) {
            if (origLo & (1 << b)) {
                if (uf.find(base + offset) === largestRoot) {
                    newLo |= (1 << b);
                } else {
                    voxelsRemoved++;
                }
                offset++;
            }
        }
        for (let b = 0; b < 32; b++) {
            if (origHi & (1 << b)) {
                if (uf.find(base + offset) === largestRoot) {
                    newHi |= (1 << b);
                } else {
                    voxelsRemoved++;
                }
                offset++;
            }
        }

        result.addBlock(mixed.morton[i], newLo, newHi);
    }

    // Solid blocks: all-or-nothing since all 64 voxels share one component
    for (let i = 0; i < solid.length; i++) {
        if (uf.find(solidBaseIds[i]) === largestRoot) {
            result.addBlock(solid[i], SOLID_MASK, SOLID_MASK);
        } else {
            voxelsRemoved += 64;
        }
    }

    logger.log(
        `connected components: ${componentSizes.size} components, ` +
        `largest has ${largestSize} voxels, ${voxelsRemoved} voxels removed`
    );

    return result;
}

export { filterConnectedComponents };
