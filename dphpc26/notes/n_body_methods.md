# n_body_methods: Triton ports

Written 2026-11-13. Four kernels: `particle_drift_kick`, `gem`, `force_lj`, `warpx_boris_push` (the module and Triton file
carry the `warpx_` prefix). CUDA-event medians from the ledger, RTX 5060 Ti, all validated unless marked.

## Results

| Kernel | Preset | float64 | float32 |
|---|---|---|---|
| particle_drift_kick | M | 8.18 ms (372 GB/s) | 4.28 ms (355 GB/s) |
| | L | 13.36 ms (377 GB/s) | 6.92 ms (364 GB/s) |
| | XL | 21.97 ms (379 GB/s) | not run |
| gem | M | 58.10 ms | 1.37 ms |
| | L | 130.70 ms | 2.92 ms |
| | XL | 280.11 ms | 6.10 ms |
| | fuzzed | 165.92 ms | 3.56 ms |
| force_lj | M | 57.40 ms | 1.50 ms |
| | L | 1307.09 ms | 27.53 ms |
| warpx_boris_push (fp64 only) | M | 4.90 ms | n/a |
| | L | 89.61 ms | n/a |
| | XL (88M particles) | 1531.40 ms | n/a |
| | fuzzed | 5.94 ms (a small draw) | n/a |

S is launch-bound for all four (0.2 to 0.6 ms) and left out. Bandwidth counts 8 loads and 8 stores per particle.

`force_lj` at XL (405,224 atoms) and at the fuzzed preset (anchored on XL) is NOT measured: both runs hit the harness's
default 200 s limit on the first execution (`first_execution took too long`; autotuning six configs at about 10^11 pairs
each is the likely cause, not checked). The ledger holds two `timeout` rows under run label `claude-nbody-heavy`. To
measure them, rerun with `-t 1500` and `--no-validate`.

## Findings

- **force_lj: the "simple" loop beat the 2D tile.** One lane per atom walking the partners one at a time (scalar,
  shared loads) is 1.4x to 4x faster than a `[block_i, block_j]` tile with `tl.sum` over the partner axis (M float32
  1.50 vs 5.37 ms, L float32 27.5 vs 113.6 ms, M float64 57.4 vs 83.4 ms, L float64 1307 vs 1889 ms). Not profiled; the
  tile version pays for the reduction over the partner axis on every tile.
- **force_lj float32 needs the reference's summation order.** The tiled version failed float32 validation at M and L
  (1 of 52,728 and 5 of 253,179 elements, all near-zero results; "LAPACK ratio" 0.04, threshold 30) while the harness's
  NumPy column passed: NumPy adds the partners in order j = 0..N-1. The strict-j-order kernel passes. Float64 passed
  either way. Fused multiply-add is off (`enable_fp_fusion=False`) because numpy rounds each product before adding it.
- **warpx_boris_push is bitwise identical to NumPy** for all three push types (0, 1, 2) at M (262,144 particles, 40
  steps): same operation order, FMA off, and `econst = 0.5*q*dt/m` and `1/c^2` combined on the host in doubles. Every
  lane keeps its momentum in registers across all steps, so the arrays are read and written once, not once per step.
  The harness validates only the default push type; the other two were checked with a direct comparison.
- **float64 costs 40x to 50x on this card** for the compute-bound kernels (gem M 58.1 vs 1.37 ms, force_lj L 1307 vs
  27.5 ms), as expected of a consumer GPU's fp64 rate. The streaming kernel only pays 2x (bytes).
- **particle_drift_kick runs at about 80 to 85 percent of the card's memory bandwidth** (spec figure 448 GB/s, quoted from
  memory), so there is little to gain without changing the traffic.

## Reminders (from earlier notes)

A constant bound to a name inside a kernel becomes fp32; write it inline or as a module-level `tl.constexpr`. Float
kernel arguments (`kappa`, `dt`, ...) arrive as fp32: pass them in a tensor and `tl.load` them.
