import torch
import triton
import triton.language as tl


@triton.jit
def _pack(kDivM, star, metadata, NB: tl.constexpr, NQ: tl.constexpr,
          K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
          S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
          KR: tl.constexpr, KC: tl.constexpr, SC: tl.constexpr):
    row = tl.arange(0, KR)
    col = tl.arange(0, KC)
    value = tl.load(kDivM + (row[:, None] // NB) * K0 + (row[:, None] % NB) * K1 + col[None, :] * K2,
                    (row[:, None] < 3 * NB) & (col[None, :] < NB), other=0)
    live = (col[None, :] < NB) & (value != 0)
    position = tl.cumsum(live.to(tl.int32), axis=1) - 1
    count = tl.sum(live.to(tl.int32), axis=1)
    tl.store(metadata + 5 + row[:, None] * NB + position, col[None, :], live)
    tl.store(metadata + 5 + 3 * NB * NB + row, count, row < 3 * NB)
    tl.store(metadata, tl.sum(count, axis=0) == 3 * NB * NB)

    p = tl.arange(0, SC)
    q = tl.arange(0, SC)
    dense = tl.full((), True, tl.int1)
    for d in tl.static_range(3):
        coefficient = tl.load(star + d * S0 + q[None, :] * S1 + p[:, None] * S2,
                              (p[:, None] < NQ) & (q[None, :] < NQ), other=0)
        nonzero = (q[None, :] < NQ) & (coefficient != 0)
        slot = tl.cumsum(nonzero.to(tl.int32), axis=1) - 1
        length = tl.sum(nonzero.to(tl.int32), axis=1)
        offset = 5 + 3 * NB * NB + 3 * NB
        tl.store(metadata + offset + (d * NQ + p[:, None]) * NQ + slot, q[None, :], nonzero)
        tl.store(metadata + offset + 3 * NQ * NQ + d * NQ + p, length, p < NQ)
        tl.store(metadata + 2 + d, tl.max(length, axis=0))
        dense = dense & (tl.sum(length, axis=0) == NQ * NQ)
    tl.store(metadata + 1, dense)


@triton.jit
def _stiffness(I, kDivM, tmp, metadata, BATCH: tl.constexpr, NB: tl.constexpr, NQ: tl.constexpr,
               I0: tl.constexpr, I1: tl.constexpr, I2: tl.constexpr,
               K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr):
    if tl.load(metadata):
        dense_idx = (tl.program_id(0) // 3) * 256 + tl.arange(0, 256)
        dense_d = tl.program_id(0) % 3
        dense_b, dense_k, dense_q = dense_idx // (NB * NQ), (dense_idx // NQ) % NB, dense_idx % NQ
        dense_value = tl.zeros((256,), tmp.dtype.element_ty)
        for dense_l in range(NB):
            dense_value += tl.load(kDivM + dense_d * K0 + dense_k * K1 + dense_l * K2) * tl.load(
                I + dense_b * I0 + dense_l * I1 + dense_q * I2, dense_idx < BATCH * NB * NQ, other=0
            )
        tl.store(tmp + dense_d * BATCH * NB * NQ + dense_idx, dense_value, dense_idx < BATCH * NB * NQ)
    else:
        program = tl.program_id(0)
        k, d = program % NB, (program // NB) % 3
        idx = (program // (3 * NB)) * 256 + tl.arange(0, 256)
        b, q = idx // NQ, idx % NQ
        row = d * NB + k
        count = tl.load(metadata + 5 + 3 * NB * NB + row)
        value = tl.zeros((256,), tmp.dtype.element_ty)
        for entry in range(count):
            l = tl.load(metadata + 5 + row * NB + entry)
            coefficient = tl.load(kDivM + d * K0 + k * K1 + l * K2)
            value += coefficient * tl.load(I + b * I0 + l * I1 + q * I2, b < BATCH, other=0)
        tl.store(tmp + d * BATCH * NB * NQ + b * NB * NQ + k * NQ + q, value, b < BATCH)


@triton.jit
def _flux(Q, tmp, star, metadata, BATCH: tl.constexpr, NB: tl.constexpr, NQ: tl.constexpr,
          Q0: tl.constexpr, Q1: tl.constexpr, Q2: tl.constexpr,
          S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr):
    idx = tl.program_id(0) * 256 + tl.arange(0, 256)
    b, k, p = idx // (NB * NQ), (idx // NQ) % NB, idx % NQ
    value = tl.zeros((256,), Q.dtype.element_ty)
    if tl.load(metadata + 1):
        for d in range(3):
            for dense_q in range(NQ):
                value += tl.load(tmp + d * BATCH * NB * NQ + (idx // NQ) * NQ + dense_q,
                                 idx < BATCH * NB * NQ, other=0) * tl.load(star + d * S0 + dense_q * S1 + p * S2)
    else:
        offset = 5 + 3 * NB * NB + 3 * NB
        for d in range(3):
            count = tl.load(metadata + offset + 3 * NQ * NQ + d * NQ + p)
            maximum = tl.load(metadata + 2 + d)
            for entry in range(maximum):
                active = (idx < BATCH * NB * NQ) & (entry < count)
                q = tl.load(metadata + offset + (d * NQ + p) * NQ + entry, active, other=0)
                coefficient = tl.load(star + d * S0 + q * S1 + p * S2, active, other=0)
                value += tl.load(tmp + d * BATCH * NB * NQ + (idx // NQ) * NQ + q, active, other=0) * coefficient
    dst = Q + b * Q0 + k * Q1 + p * Q2
    tl.store(dst, tl.load(dst, idx < BATCH * NB * NQ, other=0) + value, idx < BATCH * NB * NQ)


def kernel(Q, I, kDivM, star):
    batch, nb, nq = Q.shape
    metadata = torch.empty(5 + 3 * nb * (nb + 1) + 3 * nq * (nq + 1), dtype=torch.int32, device=Q.device)
    tmp = torch.empty((3, *I.shape), dtype=I.dtype, device=I.device)
    _pack[(1,)](kDivM, star, metadata, nb, nq, *kDivM.stride(), *star.stride(),
                 triton.next_power_of_2(3 * nb), triton.next_power_of_2(nb), triton.next_power_of_2(nq), num_warps=8)
    _stiffness[(triton.cdiv(batch * nq, 256) * 3 * nb,)](
        I, kDivM, tmp, metadata, batch, nb, nq, *I.stride(), *kDivM.stride(), enable_fp_fusion=True
    )
    _flux[(triton.cdiv(batch * nb * nq, 256),)](
        Q, tmp, star, metadata, batch, nb, nq, *Q.stride(), *star.stride(), enable_fp_fusion=True
    )


def prepare(Q, I, kDivM, star):
    """Capture repeated updates on fixed GPU tensors; keep the input tensors alive during replay."""
    with torch.cuda.device(Q.device):
        warmup = Q.clone()
        kernel(warmup, I, kDivM, star)
        torch.cuda.synchronize(Q.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel(Q, I, kDivM, star)
    return graph
