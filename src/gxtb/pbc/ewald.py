from __future__ import annotations

"""3D tin‑foil Ewald lattice sum utilities for 1/r interactions.

Implements the identity (doc/theory/25_periodic_boundary_conditions.md):

  Σ_{R∈Z^3} 1/|r+R| = Σ_{R} erfc(η|r+R|)/|r+R|
                     + (4π/V) Σ_{G≠0} exp(−G^2/(4η^2)) cos(G·r)/G^2
                     − π/(V η^2),

with explicit real‑space and reciprocal‑space cutoffs provided by the caller.

All computations are deterministic and Torch‑native.
"""

from typing import List, Tuple
import math
import torch

Tensor = torch.Tensor

__all__ = [
    "reciprocal_vectors",
    "realspace_translations",
    "ewald_sum_1over_r",
    "ewald_grad_hess_1over_r",
]


def reciprocal_vectors(cell: Tensor, g_cut: float) -> List[Tensor]:
    """Enumerate reciprocal lattice vectors G within |G| <= g_cut, excluding G=0.

    Returns a list of (3,) tensors on the same device/dtype as cell.
    """
    device = cell.device
    dtype = cell.dtype
    # Reciprocal basis B = 2π A^{-T}
    B = 2.0 * math.pi * torch.linalg.inv(cell).T
    # Find conservative bounds on integer ranges by column norms of B
    b1, b2, b3 = B[0], B[1], B[2]
    L1 = float(torch.linalg.vector_norm(b1))
    L2 = float(torch.linalg.vector_norm(b2))
    L3 = float(torch.linalg.vector_norm(b3))
    eps = 1e-15
    n1 = max(0, math.ceil(g_cut / max(L1, eps)))
    n2 = max(0, math.ceil(g_cut / max(L2, eps)))
    n3 = max(0, math.ceil(g_cut / max(L3, eps)))
    out: List[Tensor] = []
    for i in range(-n1, n1 + 1):
        for j in range(-n2, n2 + 1):
            for k in range(-n3, n3 + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                G = i * b1 + j * b2 + k * b3
                if float(torch.linalg.vector_norm(G)) <= g_cut + 1e-12:
                    out.append(G.to(device=device, dtype=dtype))
    return out


def realspace_translations(cell: Tensor, r_cut: float) -> List[Tensor]:
    """Enumerate real‑space translation vectors R within |R| <= r_cut, including R=0.

    Returns a list of (3,) tensors on the same device/dtype as cell.
    """
    from .cell import build_lattice_translations
    trips = build_lattice_translations(r_cut, cell)
    a1, a2, a3 = cell[0], cell[1], cell[2]
    return [(i * a1 + j * a2 + k * a3) for (i, j, k) in trips]


def ewald_sum_1over_r(r: Tensor, cell: Tensor, eta: float, r_cut: float, g_cut: float) -> Tensor:
    """Compute Σ_R 1/|r+R| via Ewald decomposition with explicit cutoffs.

    - r: (3,) displacement vector in Angstrom
    - cell: (3,3) lattice
    - eta: splitting parameter (1/Angstrom)
    - r_cut: real‑space cutoff for erfc term
    - g_cut: reciprocal‑space |G| cutoff for G‑sum
    Returns scalar tensor on the same device/dtype.
    """
    device = cell.device
    dtype = cell.dtype
    V = torch.det(cell)
    if float(V.item()) <= 0.0:
        raise ValueError("Cell volume must be positive")
    r = r.to(device=device, dtype=dtype)
    # Real‑space sum
    eta_t = torch.tensor(float(eta), dtype=dtype, device=device)
    R_list = realspace_translations(cell, r_cut)
    acc_real = torch.tensor(0.0, dtype=dtype, device=device)
    eps = torch.finfo(dtype).eps
    for R in R_list:
        x = r + R
        d = torch.linalg.vector_norm(x)
        if d.item() < eps:
            # Exclude R=0 when r=0; handled implicitly by reciprocal + constant term
            continue
        acc_real = acc_real + torch.erfc(eta_t * d) / d
    # Reciprocal‑space sum
    G_list = reciprocal_vectors(cell, g_cut)
    four_pi_over_V = torch.tensor(4.0 * math.pi / float(V.item()), dtype=dtype, device=device)
    acc_recip = torch.tensor(0.0, dtype=dtype, device=device)
    for G in G_list:
        G2 = torch.dot(G, G)
        if G2.item() == 0.0:
            continue
        acc_recip = acc_recip + torch.exp(-G2 / (4.0 * eta_t * eta_t)) * torch.cos(torch.dot(G, r)) / G2
    acc_recip = four_pi_over_V * acc_recip
    # Constant term
    const = -math.pi / (float(V.item()) * (eta ** 2))
    return acc_real + acc_recip + torch.tensor(const, dtype=dtype, device=device)


def ewald_grad_hess_1over_r(rij: Tensor, cell: Tensor, eta: float, r_cut: float, g_cut: float) -> tuple[Tensor, Tensor]:
    """Ewald gradient and Hessian of Σ_R 1/|r+R| for a batch of displacements.

    Inputs
    - rij: (..., 3) tensor of displacement vectors in Angstrom
    - cell: (3,3) lattice (Angstrom)
    - eta: splitting parameter (1/Angstrom)
    - r_cut: real-space cutoff (Angstrom)
    - g_cut: reciprocal-space |G| cutoff (1/Angstrom)

    Returns (grad, hess) with shapes (...,3) and (...,3,3).

    Equations (doc/theory/25_periodic_boundary_conditions.md):
    - Real-space: ∇[erfc(ηR)/R] and ∇∇[erfc(ηR)/R] summed over |R|<=r_cut (excluding singular terms).
    - Reciprocal-space: ∇F = -(4π/V) Σ_{G≠0} e^{−G^2/(4η^2)} sin(G·r) (G/G^2);
      ∇∇F = -(4π/V) Σ_{G≠0} e^{−G^2/(4η^2)} cos(G·r) (G⊗G)/G^2.
    - Constant term has zero derivatives.
    """
    device = cell.device
    dtype = cell.dtype
    rij = rij.to(device=device, dtype=dtype)
    shape = rij.shape[:-1]
    V = torch.det(cell)
    if float(V.item()) <= 0.0:
        raise ValueError("Cell volume must be positive")
    two_over_sqrtpi = 2.0 / math.sqrt(math.pi)
    eta_t = torch.tensor(float(eta), dtype=dtype, device=device)
    # Real-space translations
    R_list = realspace_translations(cell, float(r_cut))
    # Initialize accumulators
    grad = torch.zeros((*shape, 3), dtype=dtype, device=device)
    hess = torch.zeros((*shape, 3, 3), dtype=dtype, device=device)
    eps = torch.finfo(dtype).eps
    I3 = torch.eye(3, dtype=dtype, device=device).view(*(1 for _ in shape), 3, 3)
    # Real-space sum
    for R in R_list:
        x = rij + R  # (...,3)
        d = torch.linalg.norm(x, dim=-1)  # (...,)
        mask = d > eps
        if not bool(mask.any()):
            continue
        # Unit vectors u = x / d
        u = torch.zeros_like(x)
        u[mask] = x[mask] / d[mask].unsqueeze(-1)
        # a(R) and a'(R) scalars
        d_masked = d[mask]
        erfc_term = torch.erfc(eta_t * d_masked)
        exp_term = torch.exp(-(eta_t * d_masked) ** 2)
        a = erfc_term / (d_masked * d_masked) + (eta_t * two_over_sqrtpi) * exp_term / d_masked
        # a'(R) = d/dR a(R)
        a_prime = (
            -2.0 * erfc_term / (d_masked ** 3)
            + (eta_t * two_over_sqrtpi) * exp_term * (-(2.0 * eta_t * eta_t) - (2.0 / (d_masked * d_masked)))
        )
        # Gradient: ∇f_real = -u * a(R)
        g = torch.zeros_like(x)
        g[mask] = -u[mask] * a.unsqueeze(-1)
        grad = grad + g
        # Hessian: H_real = -[ a/R (I - u u^T) + a'(R) u u^T ]
        H = torch.zeros_like(hess)
        # Build (I - u u^T)
        uuT = torch.einsum('...i,...j->...ij', u[mask], u[mask])
        term1 = (a / d_masked).view(*a.shape, 1, 1) * (I3.expand(*shape, 3, 3)[mask] - uuT)
        term2 = a_prime.view(*a_prime.shape, 1, 1) * uuT
        H_mask = -(term1 + term2)
        H[mask] = H_mask
        hess = hess + H
    # Reciprocal-space sum
    G_list = reciprocal_vectors(cell, float(g_cut))
    if G_list:
        G = torch.stack(G_list, dim=0)  # (nG,3)
        G2 = torch.sum(G * G, dim=1)  # (nG,)
        fac = torch.exp(-G2 / (4.0 * eta_t * eta_t))  # (nG,)
        fourpi_over_V = torch.tensor(4.0 * math.pi / float(V.item()), dtype=dtype, device=device)
        # dot products: (..., nG)
        dots = torch.tensordot(rij, G.T, dims=([rij.ndim - 1], [0]))
        sin_d = torch.sin(dots)  # (..., nG)
        cos_d = torch.cos(dots)  # (..., nG)
        # Gradient: -(4π/V) Σ fac * sin(G·r) * (G/G^2)
        w_grad = (fac / G2).view(1, *([1] * (rij.ndim - 1)), -1)  # broadcast to (..., nG)
        contrib_g = -fourpi_over_V * torch.tensordot(sin_d * w_grad.squeeze(0), G, dims=([rij.ndim - 1], [0]))
        grad = grad + contrib_g
        # Hessian: -(4π/V) Σ fac * cos(G·r) * (G⊗G)/G^2
        GG = torch.einsum('ga,gb->gab', G, G)  # (nG,3,3)
        w_h = (fac / G2).view(-1, 1, 1) * GG  # (nG,3,3)
        # sum over nG with cos weight
        contrib_H = -fourpi_over_V * torch.tensordot(cos_d, w_h, dims=([rij.ndim - 1], [0]))  # (...,3,3)
        hess = hess + contrib_H
    return grad, hess
