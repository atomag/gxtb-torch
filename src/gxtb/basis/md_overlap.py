"""McMurchie–Davidson overlap for contracted Gaussians with metric‑orthonormal
real‑spherical transforms (s, p, d, f, g) per doc/theory.

Implements one‑electron overlap integrals for contracted Cartesian Gaussians
via Hermite Gaussian recursion (McMurchie–Davidson 1D recurrence) and maps to
the real‑spherical ordering used by diatomic rotations:
    p: (p_y, p_z, p_x)
    d: (d_xy, d_yz, d_z^2, d_xz, d_{x^2-y^2})
    f: 7 standard real spherical functions
    g: 9 real functions (m = −4..+4) from orthonormalized polynomials

Crucially, the spherical transforms are left‑whitened with the on‑center
Cartesian metric of the contracted shell so that, for every shell A with
contraction vector c and primitive exponents α,
    T_A S_cc(A) T_A^T = I,
ensuring exact contracted normalization and eliminating near‑singular overlap
matrices in multi‑center assemblies.

__doc_refs__ = {
    # Overlap recursion and contracted accumulation
    "overlap": {"file": "doc/theory/8_diatomic_frame_scaled_overlap.md", "eqs": [31]},
    # Spherical ordering and rotation consistency
    "rotation": {"file": "doc/theory/8_diatomic_frame_scaled_overlap.md", "eqs": [35, 36, 37, 38]},
    # Contracted normalization in the spherical metric
    "norm": {"file": "doc/theory/7_q-vSZP_basis_set.md", "eqs": [8, 9, 10, 11]},
}
"""
from __future__ import annotations

import torch
from math import sqrt, pi, exp
from typing import List, Tuple, Dict, Tuple as Tup
from functools import lru_cache

Tensor = torch.Tensor
__all__ = ["overlap_shell_pair"]

_HAVE_DXTB = False  # explicit: no dependency on external dxtb package

# Optional CPU acceleration via numba (must be algebraically identical).
# Disabled by default until full parity with the reference Python kernel
# is verified for all on‑center/off‑center cases and angular momenta.
try:
    from .md_numba import overlap_cart_block_numba as _overlap_cart_block_nb  # type: ignore
    _HAVE_NUMBA = False  # keep disabled by default until parity verified
except Exception:
    _HAVE_NUMBA = False


@lru_cache(maxsize=None)
def _cart_list(l: int) -> Tuple[Tuple[int,int,int], ...]:
    """Deterministic Cartesian exponent list for angular momentum l (cached)."""
    out: List[Tuple[int,int,int]] = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            lz = l - lx - ly
            out.append((lx, ly, lz))
    return tuple(out)


_TRAFO_CACHE: Dict[Tup[int, torch.dtype, torch.device], torch.Tensor] = {}

# Cache for metric‑orthonormalized transforms keyed by (l, alphas, coeffs, dtype, device)
_METRIC_TRAFO_CACHE: Dict[Tup[int, Tup[float, ...], Tup[float, ...], torch.dtype, torch.device], torch.Tensor] = {}


def _spherical_transform(l: int, dtype, device) -> torch.Tensor:
    key = (l, dtype, device)
    if key in _TRAFO_CACHE:
        return _TRAFO_CACHE[key]
    if l == 0:
        M = torch.eye(1, dtype=dtype, device=device)
    elif l == 1:
        M = torch.tensor([
            [0.0, 1.0, 0.0],  # py
            [0.0, 0.0, 1.0],  # pz
            [1.0, 0.0, 0.0],  # px
        ], dtype=dtype, device=device)
    elif l == 2:
        cart = _cart_list(2)
        def c(t): return cart.index(t)
        T = torch.zeros((5,6), dtype=dtype, device=device)
        rt2 = sqrt(2.0); rt6 = sqrt(6.0)
        T[0, c((1,1,0))] = rt2              # d_xy
        T[1, c((0,1,1))] = rt2              # d_yz
        T[2, c((0,0,2))] = 2.0/rt6          # d_z2
        T[2, c((2,0,0))] = -1.0/rt6
        T[2, c((0,2,0))] = -1.0/rt6
        T[3, c((1,0,1))] = rt2              # d_xz
        T[4, c((2,0,0))] = 1.0/rt2          # d_x2-y2
        T[4, c((0,2,0))] = -1.0/rt2
        M = T
    elif l == 3:
        # f transform (adapted & normalized). Original dxtb ordering columns:
        # fxxx,fyyy,fzzz,fxxy,fxxz,fxyy,fyyz,fxzz,fyzz,fxyz
        dxtb = torch.tensor([
            [  0.0, -sqrt(5.0/8.0), 0.0, sqrt(45.0/8.0),   0.0,    0.0,    0.0, 0.0, 0.0, 0.0],
            [  0.0,   0.0, 0.0,   0.0,   0.0,    0.0,    0.0, 0.0, 0.0, sqrt(15.0)],
            [  0.0, -sqrt(3.0/8.0), 0.0, -sqrt(3.0/8.0),   0.0,    0.0,    0.0, 0.0,  sqrt(6.0), 0.0],
            [  0.0,   0.0, 1.0,   0.0,  -1.5,    0.0,   -1.5, 0.0, 0.0, 0.0],
            [-sqrt(3.0/8.0),   0.0, 0.0,   0.0,   0.0,  -sqrt(3.0/8.0),    0.0,  sqrt(6.0), 0.0, 0.0],
            [  0.0,   0.0, 0.0,   0.0, sqrt(15.0/4.0),    0.0, -sqrt(15.0/4.0), 0.0, 0.0, 0.0],
            [ sqrt(5.0/8.0),   0.0, 0.0,   0.0,   0.0, -sqrt(45.0/8.0),    0.0, 0.0, 0.0, 0.0],
        ], dtype=dtype, device=device)
        # Our cart order: fxxx, fxxy, fxxz, fxyy, fxyz, fxzz, fyyy, fyyz, fyzz, fzzz
        perm = [0,3,4,5,9,7,1,6,8,2]
        M = dxtb[:, perm]
    elif l == 4:
        # Full (non‑placeholder) 9 real g functions generated from polynomial forms
        # in the (m=-4..+4) style ordering analogous to d/f ordering used above.
        # Base (unnormalized) polynomial set (each row as dict(expt_tuple)->coef):
        cart = _cart_list(4)  # length 15
        idx = {exp:i for i,exp in enumerate(cart)}
        polys = [
            { (3,1,0): 1.0, (1,3,0): -1.0 },                                    # g_{-4} ~ x^3 y - x y^3 = xy(x^2 - y^2)
            { (2,1,1): 3.0, (0,3,1): -1.0 },                                    # g_{-3} ~ 3 x^2 y z - y^3 z = y z (3 x^2 - y^2)
            { (1,1,2): 6.0, (3,1,0): -1.0, (1,3,0): -1.0 },                     # g_{-2} ~ x y (6 z^2 - x^2 - y^2)
            { (0,1,3): 4.0, (2,1,1): -3.0, (0,3,1): -3.0 },                     # g_{-1} ~ y z (4 z^2 -3 x^2 -3 y^2)
            { (4,0,0): 3.0, (2,2,0): 6.0, (2,0,2): -24.0, (0,4,0): 3.0, (0,2,2): -24.0, (0,0,4): 8.0 },  # g_0
            { (1,0,3): 4.0, (3,0,1): -3.0, (1,2,1): -3.0 },                     # g_{+1} ~ x z (4 z^2 -3 x^2 -3 y^2)
            { (2,0,2): 6.0, (4,0,0): -1.0, (0,2,2): -6.0, (0,4,0): 1.0 },       # g_{+2} ~ 6 x^2 z^2 - x^4 -6 y^2 z^2 + y^4
            { (3,0,1): 1.0, (1,2,1): -3.0 },                                    # g_{+3} ~ x^3 z -3 x y^2 z = x z (x^2 - 3 y^2)
            { (4,0,0): 1.0, (2,2,0): -6.0, (0,4,0): 1.0 },                      # g_{+4} ~ x^4 -6 x^2 y^2 + y^4
        ]
        C = torch.zeros((9, len(cart)), dtype=dtype, device=device)
        for r, poly in enumerate(polys):
            for expt, coef in poly.items():
                C[r, idx[expt]] = coef
        # Orthonormalize rows via QR on transpose (deterministic ordering)
        QT, R = torch.linalg.qr(C.T, mode='reduced')  # QT: (15,9)
        T = QT.T  # 9 x 15 with orthonormal rows
        # Fix row signs for determinism: ensure first non‑zero coefficient > 0
        for r in range(T.shape[0]):
            row = T[r]
            nz = torch.nonzero(row.abs() > 1e-12, as_tuple=False)
            if nz.numel():
                first_idx = nz[0,0]
                if row[first_idx] < 0:
                    T[r] = -row
        M = T
    else:
        raise ValueError(f"Angular momentum l={l} not supported (max g)")
    _TRAFO_CACHE[key] = M
    return M


def _metric_orthonormal_sph_transform(l: int, Scc: torch.Tensor) -> torch.Tensor:
    """Return M' with M' Scc M'^T = I using base transform for angular momentum l.

    - Scc must be the on-center Cartesian overlap (n_cart x n_cart) for the shell.
    - We compute G = M0 Scc M0^T and take its inverse square root to left-whiten M0.
    """
    dtype = Scc.dtype
    device = Scc.device
    # Base transform (rows span the intended real‑spherical subspace).
    M0 = _spherical_transform(l, dtype, device)  # (n_sph x n_cart)
    # Symmetrize and robustify eigen-decomposition with small ridge if needed
    G = M0 @ (0.5 * (Scc + Scc.T)) @ M0.T
    ridge = [0.0, 1e-14, 1e-12, 1e-10, 1e-8]
    evals = None; evecs = None
    for r in ridge:
        try:
            Greg = G if r == 0.0 else (G + r * torch.eye(G.shape[0], dtype=G.dtype, device=G.device))
            _evals, _evecs = torch.linalg.eigh(Greg)
            evals, evecs = _evals, _evecs
            break
        except Exception:
            continue
    if evals is None or evecs is None:
        # Fallback to SVD-based whitening
        try:
            U, S, Vh = torch.linalg.svd(G, full_matrices=False)
            ev_clamped = S.clamp_min(1e-30)
            inv_sqrt = U @ torch.diag(ev_clamped.rsqrt()) @ U.T
            return inv_sqrt @ M0
        except Exception:
            raise RuntimeError("Failed to diagonalize on-center metric for spherical transform")
    # Left‑whiten with G^{-1/2} to satisfy T Scc T^T = I exactly (within tol)
    ev_clamped = evals.clamp_min(1e-30)
    inv_sqrt = (evecs * ev_clamped.rsqrt()) @ evecs.T
    return inv_sqrt @ M0


def _metric_transform_for_shell(l: int, alpha: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """Compute or retrieve metric‑orthonormal spherical transform for one shell.

    We build the on‑center Cartesian metric S_cc from contracted primitives and
    then whiten the base transform as T = G^{-1/2} M0 to enforce T S_cc T^T = I.

    Caching uses exact tuples of (alpha, c) to guarantee determinism and avoid
    recomputation across shell‑pair evaluations.
    """
    key = (
        l,
        tuple(float(x) for x in alpha.detach().cpu().tolist()),
        tuple(float(x) for x in c.detach().cpu().tolist()),
        alpha.dtype,
        alpha.device,
    )
    if key in _METRIC_TRAFO_CACHE:
        return _METRIC_TRAFO_CACHE[key]
    # On‑center Cartesian metric for this contracted shell
    R0 = alpha.new_zeros(3)
    Scc = _overlap_cart_block(l, l, alpha, c, alpha, c, R0)
    T = _metric_orthonormal_sph_transform(l, Scc)
    _METRIC_TRAFO_CACHE[key] = T
    return T


def _one_d_recur(l1: int, l2: int, PA: float, PB: float, gamma: float, K: float):
    """Hermite 1D recurrence (Obara–Saika form) for overlap integrals.

    E[0,0] = K
    E[i,0] = PA E[i-1,0] + (i-1)/(2γ) E[i-2,0]
    E[0,j] = PB E[0,j-1] + (j-1)/(2γ) E[0,j-2]
    E[i,j] = PA E[i-1,j] + (i-1)/(2γ) E[i-2,j] + (j)/(2γ) E[i-1,j-1]
    """
    # Torch implementation (CPU/GPU) faithful to the Obara–Saika 1D recurrence.
    # eq: Hermite recurrence used in MD overlap (doc refs above). No algebra change.
    dtype = torch.float64
    S = torch.zeros((l1 + 1, l2 + 1), dtype=dtype)
    S[0, 0] = K
    inv2g = 1.0 / (2.0 * gamma)
    # First column
    for i in range(1, l1 + 1):
        S[i, 0] = PA * S[i - 1, 0]
        if i > 1:
            S[i, 0] = S[i, 0] + (i - 1) * inv2g * S[i - 2, 0]
    # First row
    for j in range(1, l2 + 1):
        S[0, j] = PB * S[0, j - 1]
        if j > 1:
            S[0, j] = S[0, j] + (j - 1) * inv2g * S[0, j - 2]
    # Interior
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            S[i, j] = PA * S[i - 1, j] + (i - 1) * inv2g * (S[i - 2, j] if i > 1 else 0.0) + j * inv2g * S[i - 1, j - 1]
    return S.tolist()


def _overlap_cart_block(li: int, lj: int, alpha_i: torch.Tensor, c_i: torch.Tensor, alpha_j: torch.Tensor, c_j: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    # Cartesian overlap matrix (ncart_i, ncart_j)
    cart_i = _cart_list(li)
    cart_j = _cart_list(lj)
    nci, ncj = len(cart_i), len(cart_j)
    out = torch.zeros((nci, ncj), dtype=alpha_i.dtype, device=alpha_i.device)
    # Fast path: CPU + numba available -> use compiled kernel
    if _HAVE_NUMBA and alpha_i.device.type == 'cpu' and alpha_j.device.type == 'cpu':
        RB = -R
        Rx, Ry, Rz = float(RB[0].item()), float(RB[1].item()), float(RB[2].item())
        block_np = _overlap_cart_block_nb(
            int(li), int(lj),
            alpha_i.detach().cpu().numpy(), c_i.detach().cpu().numpy(),
            alpha_j.detach().cpu().numpy(), c_j.detach().cpu().numpy(),
            Rx, Ry, Rz,
        )
        return torch.from_numpy(block_np).to(dtype=alpha_i.dtype, device=alpha_i.device)
    # place A at origin, B at R_B = Rj - Ri => R vector is difference passed as R (Ri - Rj) so RB = -R
    RB = -R
    Rx,Ry,Rz = [float(x) for x in RB.tolist()]
    # Precompute 1D recurrence extents once per (li,lj)
    max_ix = max(ci[0] for ci in cart_i); max_jx = max(cj[0] for cj in cart_j)
    max_iy = max(ci[1] for ci in cart_i); max_jy = max(cj[1] for cj in cart_j)
    max_iz = max(ci[2] for ci in cart_i); max_jz = max(cj[2] for cj in cart_j)
    # Precompute Cartesian index arrays for vectorized gather
    lix_idx = torch.tensor([ci[0] for ci in cart_i], dtype=torch.long, device=alpha_i.device)
    liy_idx = torch.tensor([ci[1] for ci in cart_i], dtype=torch.long, device=alpha_i.device)
    liz_idx = torch.tensor([ci[2] for ci in cart_i], dtype=torch.long, device=alpha_i.device)
    ljx_idx = torch.tensor([cj[0] for cj in cart_j], dtype=torch.long, device=alpha_i.device)
    ljy_idx = torch.tensor([cj[1] for cj in cart_j], dtype=torch.long, device=alpha_i.device)
    ljz_idx = torch.tensor([cj[2] for cj in cart_j], dtype=torch.long, device=alpha_i.device)

    for ip in range(alpha_i.shape[0]):
        a = float(alpha_i[ip].item())
        ci_val = float(c_i[ip].item())
        for jp in range(alpha_j.shape[0]):
            b = float(alpha_j[jp].item())
            cj_val = float(c_j[jp].item())
            gamma = a + b
            mu = a*b/gamma
            # Gaussian product center P relative to A(0): P = (a*0 + b*RB)/gamma = (b/gamma)*RB
            Px = (b/gamma)*Rx; Py = (b/gamma)*Ry; Pz = (b/gamma)*Rz
            PAx, PAy, PAz = Px, Py, Pz
            PBx, PBy, PBz = Px - Rx, Py - Ry, Pz - Rz
            # Scalar Gaussian prefactor (identical expression without 0-d tensor construction)
            K = (pi / gamma) ** 1.5 * exp(-mu * (Rx * Rx + Ry * Ry + Rz * Rz))
            Sx = _one_d_recur(max_ix, max_jx, PAx, PBx, gamma, K)
            Sy = _one_d_recur(max_iy, max_jy, PAy, PBy, gamma, 1.0)
            Sz = _one_d_recur(max_iz, max_jz, PAz, PBz, gamma, 1.0)
            # Convert small Python lists to tensors once per primitive pair
            Sx_t = torch.tensor(Sx, dtype=alpha_i.dtype, device=alpha_i.device)
            Sy_t = torch.tensor(Sy, dtype=alpha_i.dtype, device=alpha_i.device)
            Sz_t = torch.tensor(Sz, dtype=alpha_i.dtype, device=alpha_i.device)
            # Gather rows/cols for all (ii,jj) pairs vectorized
            X = Sx_t.index_select(0, lix_idx).index_select(1, ljx_idx)
            Y = Sy_t.index_select(0, liy_idx).index_select(1, ljy_idx)
            Z = Sz_t.index_select(0, liz_idx).index_select(1, ljz_idx)
            out = out + (ci_val * cj_val) * (X * Y * Z)
    return out


def _cart_to_sph_matrix(l: int, dtype, device) -> Tensor:
    return _spherical_transform(l, dtype, device)


def overlap_shell_pair(l_i: int, l_j: int, alpha_i: Tensor, c_i: Tensor, alpha_j: Tensor, c_j: Tensor, R: Tensor) -> Tensor:
    """Real‑spherical overlap sub‑block for a contracted shell pair i–j.

    - Cartesian block S_cart is accumulated from contracted primitives via MD
      recursion (doc/theory/8, Eq. 31 context for S).
    - Each shell uses its own metric‑orthonormal spherical transform T so that
      on‑center blocks are exactly orthonormal in the contracted metric
      (doc/theory/7 Eqs. 8–11 for contraction + normalization rationale).
    """
    device = alpha_i.device
    dtype = alpha_i.dtype
    # Cartesian overlap block
    S_cart = _overlap_cart_block(l_i, l_j, alpha_i, c_i, alpha_j, c_j, R)
    # Metric‑orthonormal spherical transforms per shell
    Ti = _metric_transform_for_shell(l_i, alpha_i, c_i)
    Tj = _metric_transform_for_shell(l_j, alpha_j, c_j)
    return Ti @ S_cart @ Tj.T


def _one_d_recur_torch(l1: int, l2: int, PA: Tensor, PB: Tensor, gamma: Tensor, K0: Tensor) -> Tensor:
    """Hermite 1D recurrence with torch tensors (autograd-friendly, no in-place writes).

    Returns tensor of shape (l1+1, l2+1).
    """
    dev = PA.device
    dtype = PA.dtype
    inv2g = 1.0 / (2.0 * gamma)
    # Build as Python list-of-lists of tensors to avoid in-place on views
    S_list: list[list[Tensor]] = [[None for _ in range(l2 + 1)] for _ in range(l1 + 1)]  # type: ignore
    S_list[0][0] = K0
    # First column (j=0)
    for i in range(1, l1 + 1):
        prev2 = S_list[i - 2][0] if i > 1 else torch.zeros((), dtype=dtype, device=dev)
        S_list[i][0] = PA * S_list[i - 1][0] + (i - 1) * inv2g * prev2
    # First row (i=0)
    for j in range(1, l2 + 1):
        prev2 = S_list[0][j - 2] if j > 1 else torch.zeros((), dtype=dtype, device=dev)
        S_list[0][j] = PB * S_list[0][j - 1] + (j - 1) * inv2g * prev2
    # General case
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            term1 = PA * S_list[i - 1][j]
            term2 = (i - 1) * inv2g * (S_list[i - 2][j] if i > 1 else torch.zeros((), dtype=dtype, device=dev))
            term3 = j * inv2g * S_list[i - 1][j - 1]
            S_list[i][j] = term1 + term2 + term3
    rows = [torch.stack(S_list[i], dim=0) for i in range(l1 + 1)]
    return torch.stack(rows, dim=0)


def _overlap_cart_block_torch(li: int, lj: int, alpha_i: Tensor, c_i: Tensor, alpha_j: Tensor, c_j: Tensor, R: Tensor) -> Tensor:
    """Cartesian overlap block (autograd-friendly tensor path)."""
    cart_i = _cart_list(li)
    cart_j = _cart_list(lj)
    nci, ncj = len(cart_i), len(cart_j)
    out = torch.zeros((nci, ncj), dtype=alpha_i.dtype, device=alpha_i.device)
    # Components
    Rx, Ry, Rz = R[0], R[1], R[2]
    R2 = Rx * Rx + Ry * Ry + Rz * Rz
    # Exponents/coefficients kept as tensors for autograd
    max_ix = max(ci[0] for ci in cart_i); max_jx = max(cj[0] for cj in cart_j)
    max_iy = max(ci[1] for ci in cart_i); max_jy = max(cj[1] for cj in cart_j)
    max_iz = max(ci[2] for ci in cart_i); max_jz = max(cj[2] for cj in cart_j)
    for ip in range(alpha_i.shape[0]):
        a = alpha_i[ip]
        ci_val = c_i[ip]
        for jp in range(alpha_j.shape[0]):
            b = alpha_j[jp]
            cj_val = c_j[jp]
            gamma = a + b
            mu = a * b / gamma
            Px = (b / gamma) * Rx; Py = (b / gamma) * Ry; Pz = (b / gamma) * Rz
            PAx, PAy, PAz = Px, Py, Pz
            PBx, PBy, PBz = Px - Rx, Py - Ry, Pz - Rz
            K0 = (torch.tensor(pi, dtype=alpha_i.dtype, device=alpha_i.device) / gamma) ** 1.5 * torch.exp(-mu * R2)
            Sx = _one_d_recur_torch(max_ix, max_jx, PAx, PBx, gamma, K0)
            Sy = _one_d_recur_torch(max_iy, max_jy, PAy, PBy, gamma, torch.tensor(1.0, dtype=alpha_i.dtype, device=alpha_i.device))
            Sz = _one_d_recur_torch(max_iz, max_jz, PAz, PBz, gamma, torch.tensor(1.0, dtype=alpha_i.dtype, device=alpha_i.device))
            for ii, (lix, liy, liz) in enumerate(cart_i):
                for jj, (ljx, ljy, ljz) in enumerate(cart_j):
                    val = ci_val * cj_val * Sx[lix, ljx] * Sy[liy, ljy] * Sz[liz, ljz]
                    basis = torch.zeros_like(out)
                    basis[ii, jj] = 1.0
                    out = out + basis * val
    return out


def overlap_shell_pair_torch(l_i: int, l_j: int, alpha_i: Tensor, c_i: Tensor, alpha_j: Tensor, c_j: Tensor, R: Tensor) -> Tensor:
    """Real-spherical overlap for a shell pair with autograd support w.r.t. R.

    This mirrors overlap_shell_pair but keeps all arithmetic in torch to allow
    differentiation with respect to the intercenter vector R.
    """
    S_cart = _overlap_cart_block_torch(l_i, l_j, alpha_i, c_i, alpha_j, c_j, R)
    Ti = _metric_transform_for_shell(l_i, alpha_i, c_i)
    Tj = _metric_transform_for_shell(l_j, alpha_j, c_j)
    return Ti @ S_cart @ Tj.T
