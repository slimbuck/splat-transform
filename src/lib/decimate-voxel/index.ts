// The voxel decimator, reached through `--decimate-voxel` /
// `decimateSourceVoxel()`. Space-uniform allocation with a per-voxel coverage
// floor, GPU-only. The uniform path is exported from ../decimate-uniform/ and
// the adaptive one from ../decimate/.
export { decimateSourceVoxel, type DecimateVoxelOptions } from './decimate-source';
