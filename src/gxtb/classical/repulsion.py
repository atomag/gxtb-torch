from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..cn import coordination_number

__all__ = ["RepulsionParams", "repulsion_energy"]


@dataclass(frozen=True)
class RepulsionParams:
    # element-wise
    z_eff0: torch.Tensor       # (Zmax+1,)
    alpha0: torch.Tensor       # (Zmax+1,)
    kq: torch.Tensor           # (Zmax+1,)
    kq2: torch.Tensor          # (Zmax+1,)
    kcn_elem: torch.Tensor     # (Zmax+1,)
    r0: torch.Tensor           # (Zmax+1,)

    # global
    kpen1_hhe: float           # for H/He
    kpen1_rest: float          # for others
    kpen2: float
    kpen3: float
    kpen4: float
    kexp: float                # typically 1.5

    # CN params (if CN not provided)
    r_cov: Optional[torch.Tensor] = None  # (Zmax+1,)
    k_cn: Optional[float] = None


def _alpha_ab(alpha_a: torch.Tensor, alpha_b: torch.Tensor) -> torch.Tensor:
    # Eq. (55)
    return (alpha_a * alpha_b) / torch.clamp(alpha_a + alpha_b, min=torch.finfo(alpha_a.dtype).eps)


def _alpha_a(alpha0_a: torch.Tensor, kcn_a: torch.Tensor, cn_a: torch.Tensor) -> torch.Tensor:
    # Eq. (56)
    return alpha0_a * torch.sqrt(1.0 + kcn_a * torch.sqrt(torch.clamp(cn_a, min=0.0)))


def _zeff(z0: torch.Tensor, kq: torch.Tensor, kq2: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    # Eq. (53)
    return z0 * (1.0 - kq * q + kq2 * q * q)


def repulsion_energy(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    params: RepulsionParams,
    charges_eeqbc: torch.Tensor,
    cn: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Semi-classical atomic charge-dependent repulsion per doc/theory/11_semi_classical_repulsion.md.
    Computes scalar energy for a single structure.

    positions: (nat, 3)
    numbers: (nat,)
    charges_eeqbc: (nat,) atomic EEQBC charges q^EEQBC_A
    cn: (nat,) optional CN_A; if None, computed using params.r_cov and params.k_cn
    """
    device = positions.device
    dtype = positions.dtype
    nat = positions.shape[-2]
    z = numbers.long()
    # CN
    if cn is None:
        if params.r_cov is None or params.k_cn is None:
            raise ValueError("CN not provided and (r_cov, k_cn) missing in parameters.")
        cn = coordination_number(positions[None, ...], numbers[None, ...], params.r_cov.to(device=device, dtype=dtype), float(params.k_cn))[0]

    # Per-atom parameters
    z0 = params.z_eff0[z].to(device=device, dtype=dtype)
    alpha0 = params.alpha0[z].to(device=device, dtype=dtype)
    kq = params.kq[z].to(device=device, dtype=dtype)
    kq2 = params.kq2[z].to(device=device, dtype=dtype)
    kcn_a = params.kcn_elem[z].to(device=device, dtype=dtype)
    r0 = params.r0[z].to(device=device, dtype=dtype)

    # Derived per-atom quantities
    q = charges_eeqbc.to(device=device, dtype=dtype)
    zeff = _zeff(z0, kq, kq2, q)
    alpha_a = _alpha_a(alpha0, kcn_a, cn)

    # Pairwise distances (avoid forming (nat,nat,3) tensor)
    dist = torch.cdist(positions, positions)  # (nat, nat)
    eps = torch.finfo(dtype).eps
    # Symmetric combinations
    alpha_b = alpha_a.unsqueeze(0) + torch.zeros_like(dist)
    alpha_a_mat = alpha_a.unsqueeze(1) + torch.zeros_like(dist)
    alpha_ab = _alpha_ab(alpha_a_mat, alpha_b)

    r0_ab = torch.sqrt(r0.unsqueeze(1) * r0.unsqueeze(0))
    # kpen1 depends on H/He or not
    is_hhe = (z <= 2).to(dtype)
    kpen1_a = is_hhe * params.kpen1_hhe + (1.0 - is_hhe) * params.kpen1_rest
    kpen1_b = kpen1_a.unsqueeze(0) + torch.zeros_like(dist)
    kpen1_a_mat = kpen1_a.unsqueeze(1) + torch.zeros_like(dist)
    inv_r = 1.0 / torch.clamp(dist, min=eps)
    series = (
        1.0
        + 0.5 * (kpen1_a_mat + kpen1_b) * inv_r
        + params.kpen2 * inv_r**2
        + params.kpen3 * inv_r**3
        + params.kpen4 * inv_r**4
    )
    expo = torch.exp(-alpha_ab * (dist + r0_ab) ** params.kexp)

    zeff_mat = zeff.unsqueeze(1) * zeff.unsqueeze(0)
    mask = ~torch.eye(nat, dtype=torch.bool, device=device)
    e_pair = 0.5 * (zeff_mat * series * expo)[mask]
    e = e_pair.sum()
    return e
