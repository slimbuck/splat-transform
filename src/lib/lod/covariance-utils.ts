/**
 * Utility functions for 3D Gaussian covariance manipulation.
 *
 * Handles conversion between quaternion+scale representation and 3x3
 * covariance matrices, including merging of multiple Gaussians into
 * a single representative Gaussian.
 */

// 3x3 symmetric matrix stored as [a00, a01, a02, a11, a12, a22]
type SymMat3 = [number, number, number, number, number, number];

/**
 * Convert quaternion (w,x,y,z) and scale (sx,sy,sz) to a 3x3 covariance matrix.
 * Covariance = R * S * S^T * R^T where R is the rotation matrix and S is the diagonal scale matrix.
 */
const quatScaleToCovariance = (qw: number, qx: number, qy: number, qz: number, sx: number, sy: number, sz: number): SymMat3 => {
    // Rotation matrix from quaternion (column-major conceptually)
    const r00 = 1 - 2 * (qy * qy + qz * qz);
    const r01 = 2 * (qx * qy - qz * qw);
    const r02 = 2 * (qx * qz + qy * qw);
    const r10 = 2 * (qx * qy + qz * qw);
    const r11 = 1 - 2 * (qx * qx + qz * qz);
    const r12 = 2 * (qy * qz - qx * qw);
    const r20 = 2 * (qx * qz - qy * qw);
    const r21 = 2 * (qy * qz + qx * qw);
    const r22 = 1 - 2 * (qx * qx + qy * qy);

    // M = R * S (scale applied to columns of R)
    const m00 = r00 * sx, m01 = r01 * sy, m02 = r02 * sz;
    const m10 = r10 * sx, m11 = r11 * sy, m12 = r12 * sz;
    const m20 = r20 * sx, m21 = r21 * sy, m22 = r22 * sz;

    // Covariance = M * M^T (symmetric)
    return [
        m00 * m00 + m01 * m01 + m02 * m02,   // [0,0]
        m00 * m10 + m01 * m11 + m02 * m12,   // [0,1]
        m00 * m20 + m01 * m21 + m02 * m22,   // [0,2]
        m10 * m10 + m11 * m11 + m12 * m12,   // [1,1]
        m10 * m20 + m11 * m21 + m12 * m22,   // [1,2]
        m20 * m20 + m21 * m21 + m22 * m22    // [2,2]
    ];
};

/**
 * Eigendecomposition of a 3x3 symmetric matrix using the analytical method.
 * Returns eigenvalues (sorted descending) and corresponding eigenvectors.
 */
const symmetricEigen3 = (s: SymMat3): { values: [number, number, number]; vectors: number[][] } => {
    const a00 = s[0], a01 = s[1], a02 = s[2];
    const a11 = s[3], a12 = s[4], a22 = s[5];

    // Characteristic equation: λ³ - p1·λ² + p2·λ - p3 = 0
    const p1 = a00 + a11 + a22;  // trace
    const q = p1 / 3;

    const p2 = a00 * a11 + a00 * a22 + a11 * a22 - a01 * a01 - a02 * a02 - a12 * a12;
    const p3 = a00 * a11 * a22 + 2 * a01 * a02 * a12 - a00 * a12 * a12 - a11 * a02 * a02 - a22 * a01 * a01; // determinant

    const p = (p1 * p1 - 3 * p2) / 9;
    const r = (2 * p1 * p1 * p1 - 9 * p1 * p2 + 27 * p3) / 54;

    const epsilon = 1e-12;

    let e0: number, e1: number, e2: number;

    if (p < epsilon) {
        // Matrix is essentially a scalar multiple of identity
        e0 = e1 = e2 = q;
    } else {
        const sqrtP = Math.sqrt(p);
        const phi = Math.acos(Math.max(-1, Math.min(1, r / (p * sqrtP)))) / 3;

        e0 = q + 2 * sqrtP * Math.cos(phi);
        e1 = q + 2 * sqrtP * Math.cos(phi - 2 * Math.PI / 3);
        e2 = q + 2 * sqrtP * Math.cos(phi - 4 * Math.PI / 3);
    }

    // Sort descending
    const values: [number, number, number] = [e0, e1, e2];
    values.sort((a, b) => b - a);

    // Clamp eigenvalues to be non-negative
    values[0] = Math.max(values[0], epsilon);
    values[1] = Math.max(values[1], epsilon);
    values[2] = Math.max(values[2], epsilon);

    // Compute eigenvectors via (A - λI) null space
    const vectors: number[][] = [];
    for (let i = 0; i < 3; i++) {
        const lam = values[i];
        const b00 = a00 - lam, b11 = a11 - lam, b22 = a22 - lam;

        // Try cross product of two rows of (A - λI) for the null space vector
        const row0 = [b00, a01, a02];
        const row1 = [a01, b11, a12];
        const row2 = [a02, a12, b22];

        // Cross product of row0 and row1
        let vx = row0[1] * row1[2] - row0[2] * row1[1];
        let vy = row0[2] * row1[0] - row0[0] * row1[2];
        let vz = row0[0] * row1[1] - row0[1] * row1[0];
        let len = Math.sqrt(vx * vx + vy * vy + vz * vz);

        if (len < epsilon) {
            // Try row0 x row2
            vx = row0[1] * row2[2] - row0[2] * row2[1];
            vy = row0[2] * row2[0] - row0[0] * row2[2];
            vz = row0[0] * row2[1] - row0[1] * row2[0];
            len = Math.sqrt(vx * vx + vy * vy + vz * vz);
        }

        if (len < epsilon) {
            // Try row1 x row2
            vx = row1[1] * row2[2] - row1[2] * row2[1];
            vy = row1[2] * row2[0] - row1[0] * row2[2];
            vz = row1[0] * row2[1] - row1[1] * row2[0];
            len = Math.sqrt(vx * vx + vy * vy + vz * vz);
        }

        if (len < epsilon) {
            // Fallback: use axis-aligned vector
            if (i === 0) vectors.push([1, 0, 0]);
            else if (i === 1) vectors.push([0, 1, 0]);
            else vectors.push([0, 0, 1]);
        } else {
            vectors.push([vx / len, vy / len, vz / len]);
        }
    }

    // Gram-Schmidt orthogonalization to ensure orthonormal basis
    gramSchmidt(vectors);

    return { values, vectors };
};

const gramSchmidt = (vectors: number[][]) => {
    for (let i = 0; i < vectors.length; i++) {
        for (let j = 0; j < i; j++) {
            const dot = vectors[i][0] * vectors[j][0] + vectors[i][1] * vectors[j][1] + vectors[i][2] * vectors[j][2];
            vectors[i][0] -= dot * vectors[j][0];
            vectors[i][1] -= dot * vectors[j][1];
            vectors[i][2] -= dot * vectors[j][2];
        }
        const len = Math.sqrt(vectors[i][0] ** 2 + vectors[i][1] ** 2 + vectors[i][2] ** 2);
        if (len > 1e-12) {
            vectors[i][0] /= len;
            vectors[i][1] /= len;
            vectors[i][2] /= len;
        }
    }
};

/**
 * Decompose a symmetric 3x3 covariance matrix back into quaternion (w,x,y,z) and scale (sx,sy,sz).
 * Scale values are the square roots of eigenvalues. Quaternion is derived from the eigenvector rotation matrix.
 */
const covarianceToQuatScale = (cov: SymMat3): { qw: number; qx: number; qy: number; qz: number; sx: number; sy: number; sz: number } => {
    const { values, vectors } = symmetricEigen3(cov);

    const sx = Math.sqrt(values[0]);
    const sy = Math.sqrt(values[1]);
    const sz = Math.sqrt(values[2]);

    // Build rotation matrix from eigenvectors (columns)
    const r00 = vectors[0][0], r10 = vectors[0][1], r20 = vectors[0][2];
    const r01 = vectors[1][0], r11 = vectors[1][1], r21 = vectors[1][2];
    const r02 = vectors[2][0], r12 = vectors[2][1], r22 = vectors[2][2];

    // Ensure proper rotation (det = +1)
    const det = r00 * (r11 * r22 - r12 * r21) -
                r01 * (r10 * r22 - r12 * r20) +
                r02 * (r10 * r21 - r11 * r20);

    const sign = det < 0 ? -1 : 1;
    const s00 = r00 * sign, s10 = r10 * sign, s20 = r20 * sign;

    // Convert rotation matrix to quaternion (Shepperd's method)
    const trace = s00 + r11 + r22;
    let qw: number, qx: number, qy: number, qz: number;

    if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1);
        qw = 0.25 / s;
        qx = (r21 - r12) * s;
        qy = (r02 * sign - r20) * s;
        qz = (s10 - r01 * sign) * s;
    } else if (s00 > r11 && s00 > r22) {
        const s = 2 * Math.sqrt(1 + s00 - r11 - r22);
        qw = (r21 - r12) / s;
        qx = 0.25 * s;
        qy = (r01 * sign + s10) / s;
        qz = (r02 * sign + s20) / s;
    } else if (r11 > r22) {
        const s = 2 * Math.sqrt(1 + r11 - s00 - r22);
        qw = (r02 * sign - r20) / s;
        qx = (r01 * sign + s10) / s;
        qy = 0.25 * s;
        qz = (r12 + r21) / s;
    } else {
        const s = 2 * Math.sqrt(1 + r22 - s00 - r11);
        qw = (s10 - r01 * sign) / s;
        qx = (r02 * sign + s20) / s;
        qy = (r12 + r21) / s;
        qz = 0.25 * s;
    }

    // Normalize quaternion
    const qlen = Math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    return {
        qw: qw / qlen, qx: qx / qlen, qy: qy / qlen, qz: qz / qlen,
        sx, sy, sz
    };
};

/**
 * Merge multiple Gaussians into a single Gaussian.
 *
 * Follows Kerbl et al.: the merged covariance is the weighted sum of
 * individual covariances plus the outer product of position offsets from
 * the merged center (law of total variance).
 *
 * @param positions - Array of [x,y,z] positions
 * @param covariances - Array of SymMat3 covariance matrices
 * @param weights - Array of per-Gaussian weights (opacity * surface_area)
 * @returns Merged center, covariance, and total weight
 */
const mergeGaussians = (
    positions: number[][],
    covariances: SymMat3[],
    weights: number[]
): { center: [number, number, number]; covariance: SymMat3; totalWeight: number } => {
    const n = positions.length;
    let totalWeight = 0;

    // Weighted center
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) {
        const w = weights[i];
        totalWeight += w;
        cx += positions[i][0] * w;
        cy += positions[i][1] * w;
        cz += positions[i][2] * w;
    }

    const invW = totalWeight > 0 ? 1 / totalWeight : 0;
    cx *= invW;
    cy *= invW;
    cz *= invW;

    // Merged covariance = weighted sum of (individual cov + outer product of offset)
    let c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0;
    for (let i = 0; i < n; i++) {
        const w = weights[i] * invW;
        const dx = positions[i][0] - cx;
        const dy = positions[i][1] - cy;
        const dz = positions[i][2] - cz;
        const cov = covariances[i];

        c00 += w * (cov[0] + dx * dx);
        c01 += w * (cov[1] + dx * dy);
        c02 += w * (cov[2] + dx * dz);
        c11 += w * (cov[3] + dy * dy);
        c12 += w * (cov[4] + dy * dz);
        c22 += w * (cov[5] + dz * dz);
    }

    return {
        center: [cx, cy, cz],
        covariance: [c00, c01, c02, c11, c12, c22],
        totalWeight
    };
};

/**
 * Compute the surface area of a Gaussian modeled as an ellipsoid.
 * Uses Knud Thomsen's approximation: S ≈ 4π * ((a^p*b^p + a^p*c^p + b^p*c^p) / 3)^(1/p)
 * where p ≈ 1.6075.
 */
const ellipsoidSurfaceArea = (sx: number, sy: number, sz: number): number => {
    const p = 1.6075;
    const ap = Math.pow(sx, p);
    const bp = Math.pow(sy, p);
    const cp = Math.pow(sz, p);
    return 4 * Math.PI * Math.pow((ap * bp + ap * cp + bp * cp) / 3, 1 / p);
};

export { quatScaleToCovariance, covarianceToQuatScale, mergeGaussians, ellipsoidSurfaceArea };
export type { SymMat3 };
