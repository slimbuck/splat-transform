import { Mat3 } from 'playcanvas';

/* eslint-disable indent */

const kSqrt03_02  = Math.sqrt(3.0 /  2.0);
const kSqrt01_03  = Math.sqrt(1.0 /  3.0);
const kSqrt02_03  = Math.sqrt(2.0 /  3.0);
const kSqrt04_03  = Math.sqrt(4.0 /  3.0);
const kSqrt01_04  = Math.sqrt(1.0 /  4.0);
const kSqrt03_04  = Math.sqrt(3.0 /  4.0);
const kSqrt01_05  = Math.sqrt(1.0 /  5.0);
const kSqrt03_05  = Math.sqrt(3.0 /  5.0);
const kSqrt06_05  = Math.sqrt(6.0 /  5.0);
const kSqrt08_05  = Math.sqrt(8.0 /  5.0);
const kSqrt09_05  = Math.sqrt(9.0 /  5.0);
const kSqrt01_06  = Math.sqrt(1.0 /  6.0);
const kSqrt05_06  = Math.sqrt(5.0 /  6.0);
const kSqrt03_08  = Math.sqrt(3.0 /  8.0);
const kSqrt05_08  = Math.sqrt(5.0 /  8.0);
const kSqrt09_08  = Math.sqrt(9.0 /  8.0);
const kSqrt05_09  = Math.sqrt(5.0 /  9.0);
const kSqrt08_09  = Math.sqrt(8.0 /  9.0);
const kSqrt01_10  = Math.sqrt(1.0 / 10.0);
const kSqrt03_10  = Math.sqrt(3.0 / 10.0);
const kSqrt01_12  = Math.sqrt(1.0 / 12.0);
const kSqrt04_15  = Math.sqrt(4.0 / 15.0);
const kSqrt01_16  = Math.sqrt(1.0 / 16.0);
const kSqrt15_16  = Math.sqrt(15.0 / 16.0);
const kSqrt01_18  = Math.sqrt(1.0 / 18.0);
const kSqrt01_60  = Math.sqrt(1.0 / 60.0);

const dp = (n: number, start: number, a: number[] | Float32Array, b: number[] | Float32Array) => {
    let sum = 0;
    for (let i = 0; i < n; i++) {
        sum += a[start + i] * b[i];
    }
    return sum;
};

const coeffsIn = new Float32Array(15);

// Rotate spherical harmonics up to band 3 based on https://github.com/andrewwillmott/sh-lib
//
// This implementation calculates the rotation factors during construction which can then
// be used to rotate multiple spherical harmonics cheaply.
class RotateSH {
    apply: (result: Float32Array | number[], src?: Float32Array | number[]) => void;

    constructor(mat: Mat3) {
        const rot = mat.data;

        // band 1
        const sh1 = [
            [rot[4], -rot[7], rot[1]],
            [-rot[5], rot[8], -rot[2]],
            [rot[3], -rot[6], rot[0]]
        ];

        // band 2
        const sh2 = [[
            kSqrt01_04 * ((sh1[2][2] * sh1[0][0] + sh1[2][0] * sh1[0][2]) + (sh1[0][2] * sh1[2][0] + sh1[0][0] * sh1[2][2])),
                          (sh1[2][1] * sh1[0][0] + sh1[0][1] * sh1[2][0]),
            kSqrt03_04 *  (sh1[2][1] * sh1[0][1] + sh1[0][1] * sh1[2][1]),
                          (sh1[2][1] * sh1[0][2] + sh1[0][1] * sh1[2][2]),
            kSqrt01_04 * ((sh1[2][2] * sh1[0][2] - sh1[2][0] * sh1[0][0]) + (sh1[0][2] * sh1[2][2] - sh1[0][0] * sh1[2][0]))
        ], [
            kSqrt01_04 * ((sh1[1][2] * sh1[0][0] + sh1[1][0] * sh1[0][2]) + (sh1[0][2] * sh1[1][0] + sh1[0][0] * sh1[1][2])),
                           sh1[1][1] * sh1[0][0] + sh1[0][1] * sh1[1][0],
            kSqrt03_04 *  (sh1[1][1] * sh1[0][1] + sh1[0][1] * sh1[1][1]),
                           sh1[1][1] * sh1[0][2] + sh1[0][1] * sh1[1][2],
            kSqrt01_04 * ((sh1[1][2] * sh1[0][2] - sh1[1][0] * sh1[0][0]) + (sh1[0][2] * sh1[1][2] - sh1[0][0] * sh1[1][0]))
        ], [
            kSqrt01_03 * (sh1[1][2] * sh1[1][0] + sh1[1][0] * sh1[1][2]) - kSqrt01_12 * ((sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) + (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2])),
            kSqrt04_03 *  sh1[1][1] * sh1[1][0] - kSqrt01_03 * (sh1[2][1] * sh1[2][0] + sh1[0][1] * sh1[0][0]),
                          sh1[1][1] * sh1[1][1] - kSqrt01_04 * (sh1[2][1] * sh1[2][1] + sh1[0][1] * sh1[0][1]),
            kSqrt04_03 *  sh1[1][1] * sh1[1][2] - kSqrt01_03 * (sh1[2][1] * sh1[2][2] + sh1[0][1] * sh1[0][2]),
            kSqrt01_03 * (sh1[1][2] * sh1[1][2] - sh1[1][0] * sh1[1][0]) - kSqrt01_12 * ((sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) + (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]))
        ], [
            kSqrt01_04 * ((sh1[1][2] * sh1[2][0] + sh1[1][0] * sh1[2][2]) + (sh1[2][2] * sh1[1][0] + sh1[2][0] * sh1[1][2])),
                           sh1[1][1] * sh1[2][0] + sh1[2][1] * sh1[1][0],
            kSqrt03_04 *  (sh1[1][1] * sh1[2][1] + sh1[2][1] * sh1[1][1]),
                           sh1[1][1] * sh1[2][2] + sh1[2][1] * sh1[1][2],
            kSqrt01_04 * ((sh1[1][2] * sh1[2][2] - sh1[1][0] * sh1[2][0]) + (sh1[2][2] * sh1[1][2] - sh1[2][0] * sh1[1][0]))
        ], [
            kSqrt01_04 * ((sh1[2][2] * sh1[2][0] + sh1[2][0] * sh1[2][2]) - (sh1[0][2] * sh1[0][0] + sh1[0][0] * sh1[0][2])),
                          (sh1[2][1] * sh1[2][0] - sh1[0][1] * sh1[0][0]),
            kSqrt03_04 *  (sh1[2][1] * sh1[2][1] - sh1[0][1] * sh1[0][1]),
                          (sh1[2][1] * sh1[2][2] - sh1[0][1] * sh1[0][2]),
            kSqrt01_04 * ((sh1[2][2] * sh1[2][2] - sh1[2][0] * sh1[2][0]) - (sh1[0][2] * sh1[0][2] - sh1[0][0] * sh1[0][0]))
        ]];

        // band 3
        const sh3 = [[
            kSqrt01_04 * ((sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) + (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4])),
            kSqrt03_02 *  (sh1[2][1] * sh2[0][0] + sh1[0][1] * sh2[4][0]),
            kSqrt15_16 *  (sh1[2][1] * sh2[0][1] + sh1[0][1] * sh2[4][1]),
            kSqrt05_06 *  (sh1[2][1] * sh2[0][2] + sh1[0][1] * sh2[4][2]),
            kSqrt15_16 *  (sh1[2][1] * sh2[0][3] + sh1[0][1] * sh2[4][3]),
            kSqrt03_02 *  (sh1[2][1] * sh2[0][4] + sh1[0][1] * sh2[4][4]),
            kSqrt01_04 * ((sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) + (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]))
        ], [
            kSqrt01_06 * (sh1[1][2] * sh2[0][0] + sh1[1][0] * sh2[0][4]) + kSqrt01_06 * ((sh1[2][2] * sh2[1][0] + sh1[2][0] * sh2[1][4]) + (sh1[0][2] * sh2[3][0] + sh1[0][0] * sh2[3][4])),
                          sh1[1][1] * sh2[0][0]                          +               (sh1[2][1] * sh2[1][0] + sh1[0][1] * sh2[3][0]),
            kSqrt05_08 *  sh1[1][1] * sh2[0][1]                          + kSqrt05_08 *  (sh1[2][1] * sh2[1][1] + sh1[0][1] * sh2[3][1]),
            kSqrt05_09 *  sh1[1][1] * sh2[0][2]                          + kSqrt05_09 *  (sh1[2][1] * sh2[1][2] + sh1[0][1] * sh2[3][2]),
            kSqrt05_08 *  sh1[1][1] * sh2[0][3]                          + kSqrt05_08 *  (sh1[2][1] * sh2[1][3] + sh1[0][1] * sh2[3][3]),
                          sh1[1][1] * sh2[0][4]                          +               (sh1[2][1] * sh2[1][4] + sh1[0][1] * sh2[3][4]),
            kSqrt01_06 * (sh1[1][2] * sh2[0][4] - sh1[1][0] * sh2[0][0]) + kSqrt01_06 * ((sh1[2][2] * sh2[1][4] - sh1[2][0] * sh2[1][0]) + (sh1[0][2] * sh2[3][4] - sh1[0][0] * sh2[3][0]))
        ], [
            kSqrt04_15 * (sh1[1][2] * sh2[1][0] + sh1[1][0] * sh2[1][4]) + kSqrt01_05 * (sh1[0][2] * sh2[2][0] + sh1[0][0] * sh2[2][4]) - kSqrt01_60 * ((sh1[2][2] * sh2[0][0] + sh1[2][0] * sh2[0][4]) - (sh1[0][2] * sh2[4][0] + sh1[0][0] * sh2[4][4])),
            kSqrt08_05 *  sh1[1][1] * sh2[1][0]                          + kSqrt06_05 *  sh1[0][1] * sh2[2][0] - kSqrt01_10 * (sh1[2][1] * sh2[0][0] - sh1[0][1] * sh2[4][0]),
                          sh1[1][1] * sh2[1][1]                          + kSqrt03_04 *  sh1[0][1] * sh2[2][1] - kSqrt01_16 * (sh1[2][1] * sh2[0][1] - sh1[0][1] * sh2[4][1]),
            kSqrt08_09 *  sh1[1][1] * sh2[1][2]                          + kSqrt02_03 *  sh1[0][1] * sh2[2][2] - kSqrt01_18 * (sh1[2][1] * sh2[0][2] - sh1[0][1] * sh2[4][2]),
                          sh1[1][1] * sh2[1][3]                          + kSqrt03_04 *  sh1[0][1] * sh2[2][3] - kSqrt01_16 * (sh1[2][1] * sh2[0][3] - sh1[0][1] * sh2[4][3]),
            kSqrt08_05 *  sh1[1][1] * sh2[1][4]                          + kSqrt06_05 *  sh1[0][1] * sh2[2][4] - kSqrt01_10 * (sh1[2][1] * sh2[0][4] - sh1[0][1] * sh2[4][4]),
            kSqrt04_15 * (sh1[1][2] * sh2[1][4] - sh1[1][0] * sh2[1][0]) + kSqrt01_05 * (sh1[0][2] * sh2[2][4] - sh1[0][0] * sh2[2][0]) - kSqrt01_60 * ((sh1[2][2] * sh2[0][4] - sh1[2][0] * sh2[0][0]) - (sh1[0][2] * sh2[4][4] - sh1[0][0] * sh2[4][0]))
        ], [
            kSqrt03_10 * (sh1[1][2] * sh2[2][0] + sh1[1][0] * sh2[2][4]) - kSqrt01_10 * ((sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) + (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4])),
            kSqrt09_05 *  sh1[1][1] * sh2[2][0]                          - kSqrt03_05 *  (sh1[2][1] * sh2[3][0] + sh1[0][1] * sh2[1][0]),
            kSqrt09_08 *  sh1[1][1] * sh2[2][1]                          - kSqrt03_08 *  (sh1[2][1] * sh2[3][1] + sh1[0][1] * sh2[1][1]),
                          sh1[1][1] * sh2[2][2]                          - kSqrt01_03 *  (sh1[2][1] * sh2[3][2] + sh1[0][1] * sh2[1][2]),
            kSqrt09_08 *  sh1[1][1] * sh2[2][3]                          - kSqrt03_08 *  (sh1[2][1] * sh2[3][3] + sh1[0][1] * sh2[1][3]),
            kSqrt09_05 *  sh1[1][1] * sh2[2][4]                          - kSqrt03_05 *  (sh1[2][1] * sh2[3][4] + sh1[0][1] * sh2[1][4]),
            kSqrt03_10 * (sh1[1][2] * sh2[2][4] - sh1[1][0] * sh2[2][0]) - kSqrt01_10 * ((sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) + (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]))
        ], [
            kSqrt04_15 * (sh1[1][2] * sh2[3][0] + sh1[1][0] * sh2[3][4]) + kSqrt01_05 * (sh1[2][2] * sh2[2][0] + sh1[2][0] * sh2[2][4]) - kSqrt01_60 * ((sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) + (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4])),
            kSqrt08_05 *  sh1[1][1] * sh2[3][0]                          + kSqrt06_05 *  sh1[2][1] * sh2[2][0] - kSqrt01_10 * (sh1[2][1] * sh2[4][0] + sh1[0][1] * sh2[0][0]),
                          sh1[1][1] * sh2[3][1]                          + kSqrt03_04 *  sh1[2][1] * sh2[2][1] - kSqrt01_16 * (sh1[2][1] * sh2[4][1] + sh1[0][1] * sh2[0][1]),
            kSqrt08_09 *  sh1[1][1] * sh2[3][2]                          + kSqrt02_03 *  sh1[2][1] * sh2[2][2] - kSqrt01_18 * (sh1[2][1] * sh2[4][2] + sh1[0][1] * sh2[0][2]),
                          sh1[1][1] * sh2[3][3]                          + kSqrt03_04 *  sh1[2][1] * sh2[2][3] - kSqrt01_16 * (sh1[2][1] * sh2[4][3] + sh1[0][1] * sh2[0][3]),
            kSqrt08_05 *  sh1[1][1] * sh2[3][4]                          + kSqrt06_05 *  sh1[2][1] * sh2[2][4] - kSqrt01_10 * (sh1[2][1] * sh2[4][4] + sh1[0][1] * sh2[0][4]),
            kSqrt04_15 * (sh1[1][2] * sh2[3][4] - sh1[1][0] * sh2[3][0]) + kSqrt01_05 * (sh1[2][2] * sh2[2][4] - sh1[2][0] * sh2[2][0]) - kSqrt01_60 * ((sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) + (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]))
        ], [
            kSqrt01_06 * (sh1[1][2] * sh2[4][0] + sh1[1][0] * sh2[4][4]) + kSqrt01_06 * ((sh1[2][2] * sh2[3][0] + sh1[2][0] * sh2[3][4]) - (sh1[0][2] * sh2[1][0] + sh1[0][0] * sh2[1][4])),
                          sh1[1][1] * sh2[4][0]                          +               (sh1[2][1] * sh2[3][0] - sh1[0][1] * sh2[1][0]),
            kSqrt05_08 *  sh1[1][1] * sh2[4][1]                          + kSqrt05_08 *  (sh1[2][1] * sh2[3][1] - sh1[0][1] * sh2[1][1]),
            kSqrt05_09 *  sh1[1][1] * sh2[4][2]                          + kSqrt05_09 *  (sh1[2][1] * sh2[3][2] - sh1[0][1] * sh2[1][2]),
            kSqrt05_08 *  sh1[1][1] * sh2[4][3]                          + kSqrt05_08 *  (sh1[2][1] * sh2[3][3] - sh1[0][1] * sh2[1][3]),
                          sh1[1][1] * sh2[4][4]                          +               (sh1[2][1] * sh2[3][4] - sh1[0][1] * sh2[1][4]),
            kSqrt01_06 * (sh1[1][2] * sh2[4][4] - sh1[1][0] * sh2[4][0]) + kSqrt01_06 * ((sh1[2][2] * sh2[3][4] - sh1[2][0] * sh2[3][0]) - (sh1[0][2] * sh2[1][4] - sh1[0][0] * sh2[1][0]))
        ], [
            kSqrt01_04 * ((sh1[2][2] * sh2[4][0] + sh1[2][0] * sh2[4][4]) - (sh1[0][2] * sh2[0][0] + sh1[0][0] * sh2[0][4])),
            kSqrt03_02 *  (sh1[2][1] * sh2[4][0] - sh1[0][1] * sh2[0][0]),
            kSqrt15_16 *  (sh1[2][1] * sh2[4][1] - sh1[0][1] * sh2[0][1]),
            kSqrt05_06 *  (sh1[2][1] * sh2[4][2] - sh1[0][1] * sh2[0][2]),
            kSqrt15_16 *  (sh1[2][1] * sh2[4][3] - sh1[0][1] * sh2[0][3]),
            kSqrt03_02 *  (sh1[2][1] * sh2[4][4] - sh1[0][1] * sh2[0][4]),
            kSqrt01_04 * ((sh1[2][2] * sh2[4][4] - sh1[2][0] * sh2[4][0]) - (sh1[0][2] * sh2[0][4] - sh1[0][0] * sh2[0][0]))
        ]];

        // rotate spherical harmonic coefficients, up to band 3
        this.apply = (result: Float32Array | number[], src?: Float32Array | number[]) => {
            if (!src || src === result) {
                coeffsIn.set(result);
                src = coeffsIn;
            }

            // band 1
            if (result.length < 3) {
                return;
            }
            result[0] = dp(3, 0, src, sh1[0]);
            result[1] = dp(3, 0, src, sh1[1]);
            result[2] = dp(3, 0, src, sh1[2]);

            // band 2
            if (result.length < 8) {
                return;
            }
            result[3] = dp(5, 3, src, sh2[0]);
            result[4] = dp(5, 3, src, sh2[1]);
            result[5] = dp(5, 3, src, sh2[2]);
            result[6] = dp(5, 3, src, sh2[3]);
            result[7] = dp(5, 3, src, sh2[4]);

            // band 3
            if (result.length < 15) {
                return;
            }
            result[8]  = dp(7, 8, src, sh3[0]);
            result[9]  = dp(7, 8, src, sh3[1]);
            result[10] = dp(7, 8, src, sh3[2]);
            result[11] = dp(7, 8, src, sh3[3]);
            result[12] = dp(7, 8, src, sh3[4]);
            result[13] = dp(7, 8, src, sh3[5]);
            result[14] = dp(7, 8, src, sh3[6]);
        };
    }
}

export { RotateSH };
