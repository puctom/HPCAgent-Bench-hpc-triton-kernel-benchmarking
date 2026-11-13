# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Triton dwt2d: one kernel per level, each program a 2-D tile of output pairs; a second kernel per step.

A level reads the s x s block and writes its four h x h quadrants. Each output (i, j) needs only the 2 x 2 input
patch at (2i, 2j), so the row and the column pass fuse into one kernel. The level reads the previous level's LL
quadrant, which it also overwrites, so the LL quadrant is double-buffered in two scratch arrays (and written to
``out`` as well, where the next level overwrites it unless it is the last).
"""

import torch
import triton
import triton.language as tl

BLOCK = 32


@triton.jit
def haar_level_kernel(src, src_ld, out, ll, N, h, BLOCK: tl.constexpr):
    """One Haar level: src (row stride src_ld) -> out quadrants, and LL also into ll (row stride h)."""
    i = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)[:, None]
    j = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)[None, :]
    mask = (i < h) & (j < h)
    p = src + (2 * i) * src_ld + 2 * j
    a = tl.load(p, mask=mask)
    b = tl.load(p + 1, mask=mask)
    c = tl.load(p + src_ld, mask=mask)
    d = tl.load(p + src_ld + 1, mask=mask)
    # Same operation order as the reference: rows first (L, H), then columns.
    l0 = (a + b) * 0.5
    l1 = (c + d) * 0.5
    h0 = (a - b) * 0.5
    h1 = (c - d) * 0.5
    lowlow = (l0 + l1) * 0.5
    q = out + i * N + j
    tl.store(q, lowlow, mask=mask)
    tl.store(q + h, (h0 + h1) * 0.5, mask=mask)
    tl.store(q + h * N, (l0 - l1) * 0.5, mask=mask)
    tl.store(q + h * N + h, (h0 - h1) * 0.5, mask=mask)
    tl.store(ll + i * h + j, lowlow, mask=mask)


@triton.jit
def blend_kernel(image, out, total, BLOCK: tl.constexpr):
    """image = 0.5 * (image + out)."""
    k = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = k < total
    tl.store(image + k, 0.5 * (tl.load(image + k, mask=mask) + tl.load(out + k, mask=mask)), mask=mask)


def dwt2d(image, nlevels, out, N, nsteps):
    N, nlevels, nsteps = int(N), int(nlevels), int(nsteps)
    # Level l writes an (N >> l+1)^2 LL quadrant into scratch[l % 2]; even levels are at most N^2/4, odd N^2/16.
    scratch = (
        torch.empty(max(N * N // 4, 1), dtype=image.dtype, device=image.device),
        torch.empty(max(N * N // 16, 1), dtype=image.dtype, device=image.device),
    )
    for _step in range(nsteps):
        # out[:] = image: entries outside every level's 2h x 2h block keep the image value.
        out.copy_(image)
        src, src_ld = image, N
        for lvl in range(nlevels):
            h = (N >> lvl) // 2
            if h == 0:
                break
            ll = scratch[lvl % 2]
            haar_level_kernel[(triton.cdiv(h, BLOCK), triton.cdiv(h, BLOCK))](src, src_ld, out, ll, N, h, BLOCK=BLOCK)
            src, src_ld = ll, h
        blend_kernel[(triton.cdiv(N * N, 1024),)](image, out, N * N, BLOCK=1024)
