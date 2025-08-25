from __future__ import annotations

"""AO multipole moment integrals (monopole, dipole, quadrupole) for contracted Gaussians.

Implements real spherical moment sub-blocks for shell pairs using McMurchie–Davidson
Hermite recurrences, reusing the 1D tables from overlap. Equations (doc/theory/16):
 - Moment integrals definitions: S_{κλ}, D^α_{κλ}, Q^{αβ}_{κλ} (Eq. 111a–c)
 - The construction uses 1D Hermite integrals and their first/second moment forms
   derived via standard MD/Obara–Saika relations.

No placeholders: exact closed-form 1D relations are used; angular transforms map to
the same real spherical ordering as overlap.
"""

from typing import Tuple
import torch
from math import pi, exp

Tensor = torch.Tensor

# Reuse helpers from overlap implementation
from .md_overlap import _cart_list, _one_d_recur, _metric_transform_for_shell

__all__ = ["moment_shell_pair"]


def _f1(PA: float, i: int, j: int, inv2g: float, S: list[list[float]]) -> float:
    """First moment 1D integral 〈i | x | j〉 relation (Eq. 111b 1D building block).

    〈i|x|j〉 = PA * S_{ij} + (i/(2γ)) S_{i-1,j} + (j/(2γ)) S_{i,j-1}
    inv2g = 1/(2γ)
    """
    term = PA * S[i][j]
    if i > 0:
        term += i * inv2g * S[i - 1][j]
    if j > 0:
        term += j * inv2g * S[i][j - 1]
    return term


def _g2(PA: float, i: int, j: int, inv2g: float, S: list[list[float]]) -> float:
    """Second moment 1D integral 〈i | x^2 | j〉.

    Derived from Hermite relations (standard Obara–Saika):
    〈i|x^2|j〉 = (PA^2 + 1/(2γ)) S_{ij}
               + (i*PA/γ) S_{i-1,j} + (j*PA/γ) S_{i,j-1}
               + (i(i-1)/(4γ^2)) S_{i-2,j} + (j(j-1)/(4γ^2)) S_{i,j-2} + (ij/(2γ^2)) S_{i-1,j-1}
    Using inv2g = 1/(2γ), so 1/γ = 2*inv2g, 1/(4γ^2) = inv2g^2, 1/(2γ^2)=2*inv2g^2.
    """
    invg = 2.0 * inv2g
    inv4g2 = inv2g * inv2g
    inv2g2 = 2.0 * inv4g2
    val = (PA * PA + inv2g) * S[i][j]
    if i > 0:
        val += (i * PA * invg) * S[i - 1][j]
    if j > 0:
        val += (j * PA * invg) * S[i][j - 1]
    if i > 1:
        val += (i * (i - 1) * inv4g2) * S[i - 2][j]
    if j > 1:
        val += (j * (j - 1) * inv4g2) * S[i][j - 2]
    if i > 0 and j > 0:
        val += (i * j * inv2g2) * S[i - 1][j - 1]
    return val


def _moment_cart_block(
    li: int,
    lj: int,
    alpha_i: Tensor,
    c_i: Tensor,
    alpha_j: Tensor,
    c_j: Tensor,
    R: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Cartesian moment matrices for a shell pair: (Dx, Dy, Dz, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz).

    Uses Hermite recurrences with 1D tables Sx,Sy,Sz and first/second moment forms.
    """
    cart_i = _cart_list(li)
    cart_j = _cart_list(lj)
    nci, ncj = len(cart_i), len(cart_j)
    dev = alpha_i.device
    dtype = alpha_i.dtype
    Dx = torch.zeros((nci, ncj), dtype=dtype, device=dev)
    Dy = torch.zeros_like(Dx)
    Dz = torch.zeros_like(Dx)
    Qxx = torch.zeros_like(Dx)
    Qxy = torch.zeros_like(Dx)
    Qxz = torch.zeros_like(Dx)
    Qyy = torch.zeros_like(Dx)
    Qyz = torch.zeros_like(Dx)
    Qzz = torch.zeros_like(Dx)

    RB = -R
    Rx, Ry, Rz = [float(x) for x in RB.tolist()]
    # Precompute 1D recurrence extents once
    max_ix = max(ci[0] for ci in cart_i); max_jx = max(cj[0] for cj in cart_j)
    max_iy = max(ci[1] for ci in cart_i); max_jy = max(cj[1] for cj in cart_j)
    max_iz = max(ci[2] for ci in cart_i); max_jz = max(cj[2] for cj in cart_j)
    for ip in range(alpha_i.shape[0]):
        a = float(alpha_i[ip].item())
        ci_val = float(c_i[ip].item())
        for jp in range(alpha_j.shape[0]):
            b = float(alpha_j[jp].item())
            cj_val = float(c_j[jp].item())
            gamma = a + b
            mu = a * b / gamma
            # Gaussian product center P relative to A(0)
            Px = (b / gamma) * Rx
            Py = (b / gamma) * Ry
            Pz = (b / gamma) * Rz
            PAx, PAy, PAz = Px, Py, Pz
            PBx, PBy, PBz = Px - Rx, Py - Ry, Pz - Rz
            # Scalar Gaussian prefactor (avoids device round-trip, identical math)
            K = (pi / gamma) ** 1.5 * exp(-mu * (Rx * Rx + Ry * Ry + Rz * Rz))
            # 1D Hermite tables
            Sx = _one_d_recur(max_ix, max_jx, PAx, PBx, gamma, K)
            Sy = _one_d_recur(max_iy, max_jy, PAy, PBy, gamma, 1.0)
            Sz = _one_d_recur(max_iz, max_jz, PAz, PBz, gamma, 1.0)
            inv2g = 1.0 / (2.0 * gamma)
            # Accumulate over Cartesian function pairs
            for ii, (lix, liy, liz) in enumerate(cart_i):
                for jj, (ljx, ljy, ljz) in enumerate(cart_j):
                    w = ci_val * cj_val
                    # Dipoles
                    fx = _f1(PAx, lix, ljx, inv2g, Sx)
                    fy = _f1(PAy, liy, ljy, inv2g, Sy)
                    fz = _f1(PAz, liz, ljz, inv2g, Sz)
                    sY = Sy[liy][ljy]; sZ = Sz[liz][ljz]; sX = Sx[lix][ljx]
                    Dx[ii, jj] += w * fx * sY * sZ
                    Dy[ii, jj] += w * sX * fy * sZ
                    Dz[ii, jj] += w * sX * sY * fz
                    # Quadrupoles
                    gx = _g2(PAx, lix, ljx, inv2g, Sx)
                    gy = _g2(PAy, liy, ljy, inv2g, Sy)
                    gz = _g2(PAz, liz, ljz, inv2g, Sz)
                    Qxx[ii, jj] += w * gx * sY * sZ
                    Qyy[ii, jj] += w * sX * gy * sZ
                    Qzz[ii, jj] += w * sX * sY * gz
                    Qxy[ii, jj] += w * fx * fy * sZ
                    Qxz[ii, jj] += w * fx * fz * sY
                    Qyz[ii, jj] += w * fy * fz * sX
    return Dx, Dy, Dz, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz


def moment_shell_pair(
    l_i: int,
    l_j: int,
    alpha_i: Tensor,
    c_i: Tensor,
    alpha_j: Tensor,
    c_j: Tensor,
    R: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Compute real-spherical dipole and quadrupole moment sub-blocks for a shell pair (Eq. 111b–c).

    Returns D_x, D_y, D_z (3 matrices) and Q_xx, Q_xy, Q_xz, Q_yy, Q_yz, Q_zz (6 matrices), each shaped
    (n_sph_i, n_sph_j), where the spherical ordering matches overlap’s transforms.
    """
    dtype = alpha_i.dtype
    device = alpha_i.device
    # Cartesian blocks
    Dx, Dy, Dz, Qxx, Qxy, Qxz, Qyy, Qyz, Qzz = _moment_cart_block(l_i, l_j, alpha_i, c_i, alpha_j, c_j, R)
    # Transform to the same metric‑orthonormal spherical basis as overlap
    # Ensures consistency: T S_cc T^T = I for on‑center contracted shells.
    Ti = _metric_transform_for_shell(l_i, alpha_i, c_i)
    Tj = _metric_transform_for_shell(l_j, alpha_j, c_j)
    def sph(M: Tensor) -> Tensor:
        return Ti @ M @ Tj.T
    return sph(Dx), sph(Dy), sph(Dz), sph(Qxx), sph(Qxy), sph(Qxz), sph(Qyy), sph(Qyz), sph(Qzz)


def _f1_torch(PA: torch.Tensor, i: int, j: int, inv2g: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Torch first-moment 1D integral 〈i | x | j〉 (autograd-friendly)."""
    dtype = PA.dtype; device = PA.device
    term = PA * S[i, j]
    if i > 0:
        term = term + i * inv2g * S[i - 1, j]
    if j > 0:
        term = term + j * inv2g * S[i, j - 1]
    return term


def _g2_torch(PA: torch.Tensor, i: int, j: int, inv2g: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Torch second moment 1D integral 〈i | x^2 | j〉 (autograd-friendly)."""
    dtype = PA.dtype; device = PA.device
    invg = 2.0 * inv2g
    inv4g2 = inv2g * inv2g
    inv2g2 = 2.0 * inv4g2
    val = (PA * PA + inv2g) * S[i, j]
    if i > 0:
        val = val + (i * PA * invg) * S[i - 1, j]
    if j > 0:
        val = val + (j * PA * invg) * S[i, j - 1]
    if i > 1:
        val = val + (i * (i - 1) * inv4g2) * S[i - 2, j]
    if j > 1:
        val = val + (j * (j - 1) * inv4g2) * S[i, j - 2]
    if i > 0 and j > 0:
        val = val + (i * j * inv2g2) * S[i - 1, j - 1]
    return val


def moment_shell_pair_torch(
    l_i: int,
    l_j: int,
    alpha_i: torch.Tensor,
    c_i: torch.Tensor,
    alpha_j: torch.Tensor,
    c_j: torch.Tensor,
    R: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Autograd-friendly real-spherical moment blocks for a shell pair (Eq. 111b–c).

    Uses torch Hermite recurrences from md_overlap._one_d_recur_torch and the same
    metric-orthonormal spherical transforms as overlap.
    """
    from .md_overlap import _cart_list, _metric_transform_for_shell, _one_d_recur_torch
    cart_i = _cart_list(l_i)
    cart_j = _cart_list(l_j)
    nci, ncj = len(cart_i), len(cart_j)
    dev = alpha_i.device; dtype = alpha_i.dtype
    Dx = torch.zeros((nci, ncj), dtype=dtype, device=dev)
    Dy = torch.zeros_like(Dx)
    Dz = torch.zeros_like(Dx)
    Qxx = torch.zeros_like(Dx)
    Qxy = torch.zeros_like(Dx)
    Qxz = torch.zeros_like(Dx)
    Qyy = torch.zeros_like(Dx)
    Qyz = torch.zeros_like(Dx)
    Qzz = torch.zeros_like(Dx)
    # Components
    Rx, Ry, Rz = R[0], R[1], R[2]
    # Precompute 1D extents
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
            R2 = Rx * Rx + Ry * Ry + Rz * Rz
            K0 = (torch.tensor(pi, dtype=dtype, device=dev) / gamma) ** 1.5 * torch.exp(-mu * R2)
            Sx = _one_d_recur_torch(max_ix, max_jx, PAx, PBx, gamma, K0)
            Sy = _one_d_recur_torch(max_iy, max_jy, PAy, PBy, gamma, torch.tensor(1.0, dtype=dtype, device=dev))
            Sz = _one_d_recur_torch(max_iz, max_jz, PAz, PBz, gamma, torch.tensor(1.0, dtype=dtype, device=dev))
            inv2g = 1.0 / (2.0 * gamma)
            for ii, (lix, liy, liz) in enumerate(cart_i):
                for jj, (ljx, ljy, ljz) in enumerate(cart_j):
                    fx = _f1_torch(PAx, lix, ljx, inv2g, Sx)
                    fy = _f1_torch(PAy, liy, ljy, inv2g, Sy)
                    fz = _f1_torch(PAz, liz, ljz, inv2g, Sz)
                    sX = Sx[lix, ljx]
                    sY = Sy[liy, ljy]
                    sZ = Sz[liz, ljz]
                    gx = _g2_torch(PAx, lix, ljx, inv2g, Sx)
                    gy = _g2_torch(PAy, liy, ljy, inv2g, Sy)
                    gz = _g2_torch(PAz, liz, ljz, inv2g, Sz)
                    w = ci_val * cj_val
                    Dx = Dx + w * (fx * sY * sZ) * _basis_ij(nci, ncj, ii, jj, dev, dtype)
                    Dy = Dy + w * (sX * fy * sZ) * _basis_ij(nci, ncj, ii, jj, dev, dtype)
                    Dz = Dz + w * (sX * sY * fz) * _basis_ij(nci, ncj, ii, jj, dev, dtype)
                    Qxx = Qxx + w * (gx * sY * sZ) * _basis_ij(nci, ncj, ii, jj, dev, dtype)
                    Qyy = Qyy + w * (sX * gy * sZ) * _basis_ij(nci, ncj, ii, jj, dev, dtype)
                    Qzz = Qzz + w * (sX * sY * gz) * _basis_ij(nci, ncj, ii, jj, dev, dtype)
                    Qxy = Qxy + w * (fx * fy * sZ) * _basis_ij(nci, ncj, ii, jj, dev, dtype)
                    Qxz = Qxz + w * (fx * fz * sY) * _basis_ij(nci, ncj, ii, jj, dev, dtype)
                    Qyz = Qyz + w * (fy * fz * sX) * _basis_ij(nci, ncj, ii, jj, dev, dtype)
    Ti = _metric_transform_for_shell(l_i, alpha_i, c_i)
    Tj = _metric_transform_for_shell(l_j, alpha_j, c_j)
    def sph(M: torch.Tensor) -> torch.Tensor:
        return Ti @ M @ Tj.T
    return sph(Dx), sph(Dy), sph(Dz), sph(Qxx), sph(Qxy), sph(Qxz), sph(Qyy), sph(Qyz), sph(Qzz)


def _basis_ij(nci: int, ncj: int, ii: int, jj: int, dev, dtype) -> torch.Tensor:
    """Return a matrix with 1.0 at (ii,jj) and 0 elsewhere (no in-place writes)."""
    out = torch.zeros((nci, ncj), dtype=dtype, device=dev)
    out[ii, jj] = 1.0
    return out
