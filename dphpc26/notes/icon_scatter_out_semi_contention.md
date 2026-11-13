# icon_scatter: out_semi atomic contention (optimization lead, not measured)

Written 2026-11-12. Kernel: `unstructured_grids/icon_scatter/icon_scatter_triton.py`.

## Status

The port validates at every preset in float32 and float64 (CUDA-event medians, RTX 5060 Ti, ledger run label
`claude-icon-scatter`):

| Preset | float32 | float64 |
|---|---|---|
| S | 0.261 ms | 0.218 ms |
| M | 1.191 ms | 1.441 ms |
| L | 6.091 ms | 8.274 ms |
| fuzzed | 22.85 ms | 29.686 ms |

Autotune configs are untuned (a guessed order of 8 block-size and warp pairs).

## Hypothesis

`out_semi` has its block axis fixed at 0, so every (neighbour, block) pair adds into the same row of `nproma` slots.
Each slot receives `nnbr * nblks` atomic adds: 288 at preset M, about 15,000 at XL. That contention probably
dominates the cost, unlike `out`, whose targets are spread over all blocks.

## To test

- Time the kernel with the `out_semi` atomic removed (result invalid, timing only) to see how much of the time it is.
- If it matters: reduce within a program before the atomic (sum the `NBLKS`-many contributions to one slot locally),
  or give each block a private partial row and sum the rows in a second small kernel.
- Check validation again in float32; the summation order changes.
