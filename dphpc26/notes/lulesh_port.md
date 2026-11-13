# lulesh: Triton port, design and what went wrong on the way

Written 2026-11-13. Kernel: `unstructured_grids/lulesh/lulesh_triton.py` (about 780 lines).

## Result

Validates at every preset in float32 and float64 (CUDA-event medians from the ledger, RTX 5060 Ti, run label
`claude-lulesh`; XL not run):

| Preset | float64 | float32 |
|---|---|---|
| S (64 elements, 30 cycles) | 33.97 ms | 32.56 ms |
| M (91,125, 30 cycles) | 100.35 ms | 35.14 ms |
| L (262,144, 67 cycles) | 608.66 ms | 90.03 ms |
| fuzzed | 67.36 ms | 21.54 ms |

For scale: the harness's NumPy column takes 19.1 s at M in float64. S is dominated by launch latency (about 1 ms per
cycle: ten launches and one device-to-host read).

## Design

The time loop and variable time step stay on the host, as in the reference: the next dt comes from two global minima, so
each cycle ends with one device-to-host read (`torch.stack([courant.min(), hydro.min()]).tolist()`). One kernel per
phase: `_force` (per element: stress + Flanagan-Belytschko hourglass, atomically added into the node forces),
`_accel`, `_symmetry`, `_velocity_position` (per node), `_kinematics`, `_gradients`, `_q_region`, `_eos` (per element) and
`_constraints` (per-block minima). Closed-form geometry uses eight per-corner vectors; the hourglass modes use
`[block, 8]` tiles with two small lookup tables (gamma and the volume-derivative permutation). The stress force does not
need the jacobian: the reference overwrites it with `volo * v` before use.

Not done: no autotuning (fixed block 64, 4 warps); node forces use atomics, so the sums are not bit-reproducible;
`enable_fp_fusion=False` is set on every launch and I did not test whether it still matters.

## Pitfalls (all cost time)

1. **A global read inside a kernel must be `tl.constexpr`.** A plain module constant raises `NameError` at compile time.
2. **A constant bound to a NAME inside a kernel becomes an fp32 value; written inline it stays exact.**
   `twelfth = 1.0 / 12.0` made every volume derivative 3e-8 too large (fp32 rounding of 1/12), which the hourglass force
   amplified through cancellation to a 6.5e-9 error in `e` and a validation failure at M (S still passed). A literal in
   an expression (`x * (1.0 / 12.0)`) is exact in fp64, as is a module-level `tl.constexpr`.
3. **Triton cannot loop over a Python tuple of tensors** (`for a in (t0, t1):` fails); unroll by hand.
4. **A tree sum differs from NumPy's sequential sum** (`tl.sum` vs `matmul`) in 64,793 of 64,989 nonzero hourglass mode
   sums. I switched to sequential sums, which reproduce `matmul` bit for bit, but that was not the cause of the error.

## How the bug was found (reusable)

- Compare every output array after a few cycles (`dev_lulesh`-style): at S everything agreed, at M after 30 cycles the
  forces were off by 1e-9.
- **Check whether the reference amplifies noise before blaming rounding**: a one-ulp change of every nodal mass moved
  the NumPy outputs by only 1e-15 after 30 cycles, so a 1e-9 gap is a real difference, not roundoff.
- **Compare one cycle from a shared mid-simulation state** (run NumPy K cycles, copy the state, take one cycle in each):
  any remaining error is not amplification. It showed 2.7e-10 in the forces, 1e-16 with stress only.
- **Dump per-element intermediates from a copy of the kernel** and compare them against a long-double evaluation of the
  same formula. NumPy's `dvdx` was accurate to 1e-19, Triton's was off by 3e-8 relative, which is the fp32 rounding of 1/12.
- The summation-order tests are only meaningful on NONZERO entries: on a mostly quiescent mesh nearly every entry is
  exactly 0 and any order "matches".
