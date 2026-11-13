# edge_laplacian: why float32 on the fuzzed preset is not gradeable

Written 2026-11-12. Kernel: `unstructured_grids/edge_laplacian/edge_laplacian_triton.py`.

## Observation

`run-framework -f triton -b edge_laplacian -p fuzzed -d float32` times the kernel (195 ms) but validation raises:

    UngradeableTolerance: eps_acc*sqrt(l) = 1.227e-03 >= rtol 1.000e-03 at l=105888413

The ledger stores this run as `validated = 0`, `status = ok`, with no failure text, so it looks like a wrong answer.
It is not one. Float64 passes at the same size, and so do float32 at presets S and M.

| Preset | dtype | Validation | Median |
|---|---|---|---|
| S | float32 | pass | 0.228 ms |
| M | float32 | pass | 1.757 ms |
| fuzzed | float32 | ungradeable | 195.164 ms |
| M | float64 | pass | 6.691 ms |
| fuzzed | float64 | pass | 208.684 ms |

## Cause

The harness refuses to grade when `eps * sqrt(l) >= rtol` (`precision.ungradeable`). `l` is the output's accumulation
length, derived in `harness/grading.py: contracted_extent` as the product of the shape symbols that an input has and the
output lacks. `Lx` is `(N,)`; `src`, `dst` and `w` are `(E,)`; so `l = E`, i.e. every node is assumed to sum every edge.
For float32 `eps = 1.19e-7`, and `E = 105,888,413` at the fuzzed preset gives 1.227e-3.

The real accumulation is short. Fuzzed inputs (harness generator, `fuzz_iteration=0`): `N = 22,262,073`,
`E = 105,888,413`; each node has 9.5 incident edges on average, 34 at most.

## The kernel is accurate

Triton float32 result against a float64 reference computed from the same inputs:

- largest absolute error 3.2e-6 for a largest value of 16.3: 1.9e-7 relative, float32 rounding level;
- on elements with |reference| above 1e-3 of the maximum, the largest relative error is 1.5e-5.

## Why not declare a smaller `chain_length`

The manifest accepts `chain_length` for an output and it overrides the derived `l`. We did not change it (shared
benchmark definition, and it changes the tolerance for every framework). Applying the grading formula by hand with
`l = 34`, the maximum degree: 143 of 22 million elements would fail, all near-zero results where terms cancel, because
the formula requires such elements to be accurate to about `rtol * atol`. This is our reconstruction of the formula,
not a harness run. The current huge `l` only passes because it makes `atol` about 0.02, which is what the refusal
exists to prevent.

## Decision

- Report this kernel at the fuzzed preset in float64; report float32 at S and M.
- State that float32 at the fuzzed size is ungradeable by the harness for any implementation.
- Open question for the advisor: should scatter-add and other large-reduction kernels get a declared `chain_length`
  or a different band for float32?
- Ledger gap: store the validation error text in `failure` so "ungradeable" differs from "wrong".

## Reproduce

Inside a GPU allocation, in the repo root (needs the cluster environment from `hpcagent_bench/cluster/env.sh`):

```python
import numpy as np, torch
from hpcagent_bench.categories import Preset
from hpcagent_bench.frameworks import Benchmark
from hpcagent_bench.benchmarks.scientific_computing.unstructured_grids.edge_laplacian.edge_laplacian_triton import kernel

data = Benchmark("edge_laplacian").get_data(Preset.FUZZED, "float32", fuzz_iteration=0)
src, dst, w, x = (np.asarray(data[k]) for k in ("src", "dst", "w", "x"))
N, E = x.shape[0], src.shape[0]
deg = np.bincount(src, minlength=N) + np.bincount(dst, minlength=N)
print(N, E, deg.mean(), deg.max())

flux = w.astype(np.float64) * (x.astype(np.float64)[src] - x.astype(np.float64)[dst])
ref = np.bincount(src, flux, minlength=N) - np.bincount(dst, flux, minlength=N)

to_gpu = lambda a: torch.from_numpy(np.ascontiguousarray(a)).cuda()
Lx = torch.zeros(N, dtype=torch.float32, device="cuda")
kernel(to_gpu(src), to_gpu(dst), to_gpu(w), to_gpu(x), Lx)
err = np.abs(Lx.cpu().numpy().astype(np.float64) - ref)
print(err.max(), np.abs(ref).max(), err.max() / np.abs(ref).max())
```
