# decimate-voxel — the voxel decimator

Reached through `--decimate-voxel` / `decimateSourceVoxel()`. The other two are
`src/lib/decimate-uniform/` (`--decimate`) and `src/lib/decimate/`
(`--decimate-adaptive`). The names describe how each allocates removal, not a
ranking.

What distinguishes this one is a **coverage floor**: allocation runs over a grid
that is uniform in *space*, and every occupied cell keeps at least one survivor,
so no region can be emptied. Uniform decimation removes the same *fraction*
everywhere; adaptive decimation follows local error and will leave a difficult
region almost untouched while collapsing an easy one. Neither can promise that a
given piece of space still has something in it afterwards.

It is also **single-pass** — the allocator reaches any target directly, with no
cascade of generations — and **GPU-only**: there is no CPU fallback, and the CLI
fails loudly rather than silently running a different algorithm.

## Where it comes from

The algorithm follows the published formulas of WilliamLiu-1997's
`3DGS-PLY-3DTiles-Converter` `simplify` (its `Splat-Simplify.md`), implemented
from the write-up rather than ported. Section numbers in the source comments —
§3 partition, §4 similarity, §5 first representative, §6 budget, §7 detail
representatives, §8 assignment, §9 merge — refer to that document.

§9 is not reimplemented here: it is `mergeGroup` from `../decimate/moment-match`,
the same moment match the other two decimators use, so `--decimate-compensate`
applies identically.

## Layout

| file | role |
| --- | --- |
| `voxel-grid.ts` | §3 arithmetic: voxel budget, grid dims, subdivision threshold |
| `partition.ts` | §3 build: binning, octant subdivision, leaf neighbourhoods |
| `similarity.ts` | §4 metric and the per-splat tile decode |
| `representatives.ts` | §5 coverage-biased medoid, §7 novelty sequence |
| `budget.ts` | §6 opacity-proportional local allocation |
| `global-fill.ts` | §7 stage 2, as a threshold rather than a greedy loop |
| `assign.ts` | §8 neighbourhood argmax and grouping |
| `reference.ts` | whole-scene CPU driver — the oracle, not the shipping path |
| `decimate-source.ts` | the shipping path: chunk source in, chunk source out |

The kernels live in `../gpu/gpu-voxel-select.ts` and
`../gpu/shaders/chunks/voxel-similarity.ts`.

## Why there are two implementations

`reference.ts` is a straight, resident, f64 transcription of §3–§9 that nothing
in production calls. It exists so the GPU path can be checked against something
exact, and the two deliberately differ in *approach* where they can:

- the reference spends the global budget with a sequential lazy heap; the GPU
  path records every voxel's full pick sequence and thresholds the priorities
- the reference is f64 throughout; the kernels are f32

Two unrelated algorithms agreeing is a much stronger signal than an
implementation compared against a copy of itself — in particular it is what
validates the threshold argument in `global-fill.ts`. `test/gpu-voxel-select.*`
runs that comparison; agreement is exact on structural results (per-voxel counts,
the coverage floor, assignment locality) and ~100% on the picks themselves.

## Numerical notes

Both paths floor each splat's scales at a *fraction* of its own longest axis
(`SCALE_FLOOR_RATIO`). This is not cosmetic. The Bhattacharyya term divides by
`|Σ̄|`, so an ill-conditioned covariance amplifies rounding error, and 2D-gaussian
captures arrive with an exactly-zero third scale. An absolute floor cannot fix it
portably: it caps the condition number at `maxVariance/EPS_COV`, which depends on
the scene's units. Measured on `scenes/2dgs.ply`, GPU/CPU assignment agreement
went 34.6% → 94.3% (adding `EPS_COV`) → 99.96% (adding the ratio floor).

The clamp only affects *comparison*; `mergeGroup` still sees the real scales.

## Cost

Resident cost is 56 bytes per input gaussian regardless of SH band count — only
the DC coefficients reach the metric — plus the same again on the device for the
similarity tile. The colour layer is never resident; it is gathered per output
chunk, so the input must support gather reads.

Measured on `scenes/castle.ply`, 5M → 500K: 5.6 s and 1.34 GB peak, against
53.3 s and 4.58 GB for `--decimate` at the same target. Scenes large enough to
exceed the device's storage-buffer limit are not yet supported — the constructor
fails with the limit in the message rather than degrading.
