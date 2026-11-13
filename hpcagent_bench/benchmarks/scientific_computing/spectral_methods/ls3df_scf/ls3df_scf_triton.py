# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Triton ls3df_scf: the grid-sized operators in Triton; the small dense linear algebra and the 3-D FFTs are torch.

A plain (unoptimized) port, the reference's SCF loop line for line. Triton kernels, one lane per grid point:
    gather_kernel:   vloc = V_tot on the fragment's Lb^3 box (periodic offsets)                (Gen_VF)
    hpsi_kernel:     hx = -half_inv_h2 * (periodic 8th-order Laplacian) X + vloc * X            (the stencil part of hpsi)
    recurrence_kernel: ynew = (hy - c y) * alpha - beta * x                                    (the Chebyshev step)
    density_kernel:  rho_out[box] += alpha * sum_k occ_k X^2, a scatter-add (no two points of one box share a cell)
    poisson_kernel:  rho_g * 4 pi / |G|^2 on the FFT grid
    lda_kernel:      V = V_H + V_ion + the LDA exchange-correlation potential                  (genpot)
The projector term of hpsi, both Rayleigh-Ritz solves (Cholesky, eigh), the Lanczos bound and the FFTs are torch
(cuBLAS / cuSOLVER / cuFFT). The fragments run one after another, as in the reference. fp64 scalars travel in tensors of
the data type: a Python float would reach a kernel as fp32.
"""

import numpy as np
import torch
import triton
import triton.language as tl

BLOCK = 256

# 8th-order central 2nd-derivative weights for -1/2 nabla^2, and the LDA constants, as in the reference.
_C0 = -205.0 / 72.0
_CW = (8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0)
_NLANC = 6
AX = tl.constexpr(0.9847450218426965)
GAMMA = tl.constexpr(-0.1423)
B1 = tl.constexpr(1.0529)
B2 = tl.constexpr(0.3334)
CA = tl.constexpr(0.0311)
CB = tl.constexpr(-0.0480)
CC = tl.constexpr(0.0020)
CD = tl.constexpr(-0.0116)
THIRD = tl.constexpr(1.0 / 3.0)
FOUR_PI = tl.constexpr(4.0 * np.pi)
LOG_RS_NUM = tl.constexpr(float(np.log(3.0 / (4.0 * np.pi))))


@triton.jit
def gather_kernel(vtot, vloc, ox, oy, oz, N, Lb, total, BLOCK: tl.constexpr):
    """vloc[x, y, z] = vtot[(ox + x) % N, (oy + y) % N, (oz + z) % N]."""
    p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = p < total
    z = p % Lb
    y = (p // Lb) % Lb
    x = p // (Lb * Lb)
    cell = (((ox + x) % N) * N + (oy + y) % N) * N + (oz + z) % N
    tl.store(vloc + p, tl.load(vtot + cell, mask=mask), mask=mask)


@triton.jit
def _pair(y, p, pos, m, stride, N, mask):
    """y[pos + m] + y[pos - m] along one periodic axis, for the element at flat offset p sitting at pos on it."""
    up = (pos + m) % N
    dn = (pos - m + N) % N
    return tl.load(y + p + (up - pos) * stride, mask=mask) + tl.load(y + p + (dn - pos) * stride, mask=mask)


@triton.jit
def _axis(y, p, pos, stride, N, mask, centre, coef):
    """The 9-point stencil along one axis: C0 * y + sum_m w_m * (y[+m] + y[-m])."""
    acc = tl.load(coef + 0) * centre
    acc += tl.load(coef + 1) * _pair(y, p, pos, 1, stride, N, mask)
    acc += tl.load(coef + 2) * _pair(y, p, pos, 2, stride, N, mask)
    acc += tl.load(coef + 3) * _pair(y, p, pos, 3, stride, N, mask)
    acc += tl.load(coef + 4) * _pair(y, p, pos, 4, stride, N, mask)
    return acc


@triton.jit
def hpsi_kernel(y, vloc, hy, coef, Lb, K, total, BLOCK: tl.constexpr):
    """hy = -half_inv_h2 * lap(y) + vloc * y over the (Lb, Lb, Lb, K) block; coef = [C0, w1, w2, w3, w4, half_inv_h2]."""
    p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = p < total
    # p = ((i * Lb + j) * Lb + l) * K + col
    rest = p // K
    l = rest % Lb
    rest = rest // Lb
    j = rest % Lb
    i = rest // Lb
    centre = tl.load(y + p, mask=mask)
    t0 = _axis(y, p, i, Lb * Lb * K, Lb, mask, centre, coef)
    t1 = _axis(y, p, j, Lb * K, Lb, mask, centre, coef)
    t2 = _axis(y, p, l, K, Lb, mask, centre, coef)
    v = tl.load(vloc + (p // K), mask=mask)
    tl.store(hy + p, -tl.load(coef + 5) * (t0 + t1 + t2) + v * centre, mask=mask)


@triton.jit
def recurrence_kernel(hy, y, x, ynew, scal, total, HAS_PREV: tl.constexpr, BLOCK: tl.constexpr):
    """ynew = (hy - c * y) * alpha - beta * x; scal = [c, alpha, beta]."""
    p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = p < total
    res = (tl.load(hy + p, mask=mask) - tl.load(scal + 0) * tl.load(y + p, mask=mask)) * tl.load(scal + 1)
    if HAS_PREV:
        res -= tl.load(scal + 2) * tl.load(x + p, mask=mask)
    tl.store(ynew + p, res, mask=mask)


@triton.jit
def density_kernel(x, occ, alpha, f, rho_out, ox, oy, oz, N, Lb, K, total, BLOCK: tl.constexpr):
    """rho_out[box] += alpha[f] * sum_k occ_k x_k^2, x the (Lb, Lb, Lb, K) block of one fragment."""
    p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = p < total
    dens = tl.zeros((BLOCK,), dtype=rho_out.dtype.element_ty)
    for k in range(K):
        xv = tl.load(x + p * K + k, mask=mask)
        dens += tl.load(occ + k) * xv * xv
    z = p % Lb
    y = (p // Lb) % Lb
    xx = p // (Lb * Lb)
    cell = (((ox + xx) % N) * N + (oy + y) % N) * N + (oz + z) % N
    tl.store(rho_out + cell, tl.load(rho_out + cell, mask=mask) + tl.load(alpha + f) * dens, mask=mask)


@triton.jit
def poisson_kernel(rho_g, inv_gsq, out, total, BLOCK: tl.constexpr):
    """out = 4 pi rho_g * inv_gsq (rho_g, out complex as (re, im) pairs; inv_gsq real)."""
    p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = p < total
    g = tl.load(inv_gsq + p, mask=mask)
    tl.store(out + 2 * p, FOUR_PI * tl.load(rho_g + 2 * p, mask=mask) * g, mask=mask)
    tl.store(out + 2 * p + 1, FOUR_PI * tl.load(rho_g + 2 * p + 1, mask=mask) * g, mask=mask)


@triton.jit
def lda_kernel(rho, vh, vion, out, total, BLOCK: tl.constexpr):
    """out = V_H + V_ion + v_xc(rho): Slater exchange + Perdew-Zunger correlation (Hartree units)."""
    p = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = p < total
    n = tl.maximum(tl.load(rho + p, mask=mask, other=1.0), 1.0e-12)
    ln_n = tl.log(n)
    rs = tl.exp((LOG_RS_NUM - ln_n) * THIRD)
    n13 = tl.exp(ln_n * THIRD)
    v_x = -AX * n13
    sqrt_rs = tl.sqrt(rs)
    ln_rs = tl.log(rs)
    denom = 1.0 + B1 * sqrt_rs + B2 * rs
    v_c_ge1 = (GAMMA / denom) * (1.0 + (7.0 / 6.0) * B1 * sqrt_rs + (4.0 / 3.0) * B2 * rs) / denom
    v_c_lt1 = CA * ln_rs + (CB - CA / 3.0) + (2.0 / 3.0) * CC * rs * ln_rs + (2.0 * CD - CC) / 3.0 * rs
    v_c = tl.where(rs < 1.0, v_c_lt1, v_c_ge1)
    tl.store(out + p, tl.load(vh + p, mask=mask) + tl.load(vion + p, mask=mask) + (v_x + v_c), mask=mask)


class Fragment:
    """What one hpsi call needs besides the block: the potential on the box, the projectors, the stencil constants."""

    def __init__(self, vloc, proj_f, dij_f, coef):
        self.vloc, self.proj_f, self.dij_f, self.coef = vloc, proj_f, dij_f, coef


def hpsi(X, frag):
    """H X = -1/2 nabla^2 X + V_local X + sum_pq beta_p D_pq <beta_q|X>, for X of shape (Lb, Lb, Lb, k)."""
    Lb, K = X.shape[0], X.shape[-1]
    total = X.numel()
    hx = torch.empty_like(X)
    hpsi_kernel[(triton.cdiv(total, BLOCK),)](X, frag.vloc, hx, frag.coef, Lb, K, total, BLOCK=BLOCK)
    overlap = frag.proj_f.T @ X.reshape(-1, K)  # <beta_q|X>   (nproj, k)
    return hx + (frag.proj_f @ (frag.dij_f @ overlap)).reshape(X.shape)


def upper_bound(frag, v):
    """k-step Lanczos upper bound: theta_max plus the last residual beta."""
    Lb = v.shape[0]
    v = v / (torch.linalg.norm(v) + 1.0e-30)
    v_prev = torch.zeros_like(v)
    alphas, betas = [], []
    beta = 0.0
    for _ in range(_NLANC):
        w = hpsi(v.reshape(Lb, Lb, Lb, 1).contiguous(), frag)[:, :, :, 0]
        alpha = float(v.ravel() @ w.ravel())
        w = w - alpha * v - beta * v_prev
        beta = float(torch.linalg.norm(w))
        alphas.append(alpha)
        if beta < 1.0e-12:
            break
        v_prev, v = v, w / beta
        betas.append(beta)
    T = torch.diag(torch.tensor(alphas, dtype=v.dtype, device=v.device))
    if betas[: len(alphas) - 1]:
        off = torch.tensor(betas[: len(alphas) - 1], dtype=v.dtype, device=v.device)
        T = T + torch.diag(off, 1) + torch.diag(off, -1)
    return float(torch.linalg.eigvalsh(T).max()) + beta


def cheb_filter(frag, X, m, a, b, a0):
    """Degree-m scaled Chebyshev filter p_m(H) X damping the interval [a, b] (CheFSI)."""
    total = X.numel()
    grid = (triton.cdiv(total, BLOCK),)
    e = 0.5 * (b - a)
    c = 0.5 * (b + a)
    sigma = e / (a0 - c)
    sigma1 = sigma

    def step(y_in, x_in, alpha, beta, has_prev):
        hy = hpsi(y_in, frag)
        out = torch.empty_like(y_in)
        scal = torch.tensor([c, alpha, beta], dtype=X.dtype, device=X.device)
        recurrence_kernel[grid](hy, y_in, x_in, out, scal, total, HAS_PREV=has_prev, BLOCK=BLOCK)
        return out

    Y = step(X, X, sigma1 / e, 0.0, False)
    for _ in range(2, int(m) + 1):
        sigma_new = 1.0 / (2.0 / sigma1 - sigma)
        Ynew = step(Y, X, 2.0 * sigma_new / e, sigma * sigma_new, True)
        X, Y, sigma = Y, Ynew, sigma_new
    return Y


def rayleigh_ritz(frag, Y):
    """Orthonormalize Y in its own metric, rotate to Ritz vectors of H_F; returns the block and the sorted Ritz values."""
    shp = Y.shape
    k = shp[-1]
    Yf = Y.reshape(-1, k)
    Wf = hpsi(Y, frag).reshape(-1, k)
    h_sub = 0.5 * (Yf.T @ Wf + (Yf.T @ Wf).T)
    s_sub = 0.5 * (Yf.T @ Yf + (Yf.T @ Yf).T) + 1.0e-12 * torch.eye(k, dtype=Yf.dtype, device=Yf.device)
    L = torch.linalg.cholesky(s_sub)
    Linv = torch.linalg.inv(L)
    w, U = torch.linalg.eigh(Linv @ h_sub @ Linv.T)
    # Ritz vectors are fixed only up to a sign; pin it as the reference does.
    U = U * torch.sign(U[torch.argmax(torch.abs(U), dim=0), torch.arange(U.shape[1], device=U.device)])
    C = Linv.T @ U
    return (Yf @ C).reshape(shp), w


def inverse_gsq(n, h, dtype, device):
    """1 / |G|^2 on the FFT grid with the G = 0 cell left at zero."""
    kx = (2.0 * np.pi * torch.fft.fftfreq(n, d=h, dtype=torch.float64, device=device)).to(dtype)
    gsq = (
        (kx[:, None, None] * kx[:, None, None])
        + (kx[None, :, None] * kx[None, :, None])
        + (kx[None, None, :] * kx[None, None, :])
    )
    gsq[0, 0, 0] = 1.0
    inv = 1.0 / gsq
    inv[0, 0, 0] = 0.0
    return inv.contiguous()


def genpot(rho, V_ion, inv, out):
    """V_tot = V_H + V_ion + V_xc, gauge-fixed to zero mean, written to out."""
    total = rho.numel()
    rho_g = torch.fft.fftn(rho - rho.mean())
    w = torch.empty_like(rho_g)
    grid = (triton.cdiv(total, BLOCK),)
    poisson_kernel[grid](torch.view_as_real(rho_g), inv, torch.view_as_real(w), total, BLOCK=BLOCK)
    vh = torch.fft.ifftn(w).real.contiguous()
    lda_kernel[grid](rho, vh, V_ion, out, total, BLOCK=BLOCK)
    out -= out.mean()


def ls3df_scf(dvol, half_inv_h2, tol, nscf, mix, m, offsets, alpha, occ, V_ion, proj, dij, psi_frag, rho, V_tot):
    N = rho.shape[0]
    nfrag, Lb = psi_frag.shape[0], psi_frag.shape[1]
    nproj = proj.shape[-1]
    dtype, device = rho.dtype, rho.device
    h = float(np.sqrt(0.5 / half_inv_h2))
    coef = torch.tensor([_C0, *_CW, float(half_inv_h2)], dtype=dtype, device=device)
    inv = inverse_gsq(N, h, dtype, device)
    proj_flat = proj.reshape(nfrag, Lb * Lb * Lb, nproj)
    offs = offsets.tolist()
    box_total = Lb * Lb * Lb
    box_grid = (triton.cdiv(box_total, BLOCK),)

    rho_in = rho.clone()
    nelec = float(rho_in.sum()) * float(dvol)  # electrons to conserve while patching
    genpot(rho_in, V_ion, inv, V_tot)  # potential of the seed density
    b_frag = [0.0] * nfrag  # per-fragment upper bound (set once)
    b_valid = [False] * nfrag
    vloc = torch.empty((Lb, Lb, Lb), dtype=dtype, device=device)

    for _ in range(int(nscf)):
        rho_out = torch.zeros((N, N, N), dtype=dtype, device=device)
        for f in range(nfrag):
            ox, oy, oz = offs[f]
            gather_kernel[box_grid](V_tot, vloc, ox, oy, oz, N, Lb, box_total, BLOCK=BLOCK)  # Gen_VF
            frag = Fragment(vloc, proj_flat[f], dij[f], coef)
            # PEtot_F: one CheFSI filter + Rayleigh-Ritz sweep of the fragment KS problem.
            if not b_valid[f]:
                b_frag[f] = 1.2 * upper_bound(frag, psi_frag[f][..., 0].contiguous())
                b_valid[f] = True
            X, w = rayleigh_ritz(frag, psi_frag[f])
            # keep the damping window strictly above the wanted band, as the reference does
            b_hi = max(b_frag[f], float(w[-1]) * 1.1 + 1.0)
            Y = cheb_filter(frag, X, m, float(w[-1]), b_hi, float(w[0]))
            X, w = rayleigh_ritz(frag, Y)
            psi_frag[f] = X
            K = X.shape[-1]
            density_kernel[box_grid](X, occ, alpha, f, rho_out, ox, oy, oz, N, Lb, K, box_total, BLOCK=BLOCK)  # Gen_dens
        rho_out = torch.clamp(rho_out, min=0.0)
        q = float(rho_out.sum()) * float(dvol)
        if q > 0.0:
            rho_out *= nelec / q  # restore the electron count
        rho_error = float((rho_out - rho_in).abs().sum()) / (float(rho_in.abs().sum()) + 1.0e-30)
        rho_in = rho_in + float(mix) * (rho_out - rho_in)  # linear density mixing
        genpot(rho_in, V_ion, inv, V_tot)  # GENPOT: rebuild the potential
        if rho_error < float(tol):
            break

    rho.copy_(rho_in)
