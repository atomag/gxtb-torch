from __future__ import annotations

import torch

__all__ = ["coordination_number", "coordination_number_grad"]


def coordination_number(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    r_cov: torch.Tensor,
    k_cn: float,
) -> torch.Tensor:
    """
    Error-function-based CN per doc/theory/9_cn.md Eq. (47).

    positions: (..., nat, 3)
    numbers: (..., nat) integer atomic numbers (1-based Z)
    r_cov: (Zmax+1,) tensor mapping Z -> R_cov (bohr or angstrom consistently)
    k_cn: scalar steepness
    returns: (..., nat) CN values
    """
    pos = positions
    device = pos.device
    dtype = pos.dtype
    nat = pos.shape[-2]
    rij = pos.unsqueeze(-3) - pos.unsqueeze(-2)  # (..., nat, nat, 3)
    dist = torch.linalg.norm(rij, dim=-1)  # (..., nat, nat)

    # R_cov_AB = 0.5 * (R_cov_A + R_cov_B)
    rca = r_cov[numbers.long()]  # (..., nat)
    rcb = r_cov[numbers.long()]  # (..., nat)
    rc_ab = 0.5 * (rca.unsqueeze(-1) + rcb.unsqueeze(-2))  # (..., nat, nat)

    # Avoid self terms
    eye = torch.eye(nat, device=device, dtype=dtype)
    mask_off = (1.0 - eye)

    arg = k_cn * (dist - rc_ab) / torch.clamp(rc_ab, min=torch.finfo(dtype).eps)
    contrib = 0.5 * (1.0 + torch.erf(arg)) * mask_off
    cn = contrib.sum(dim=-1)
    return cn


def coordination_number_grad(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    r_cov: torch.Tensor,
    k_cn: float,
) -> torch.Tensor:
    """
    d(CN_A)/dR_X per Eq. (48b). Returns (..., nat, nat, 3) where last two axes are (A,X,3).
    """
    pos = positions
    device = pos.device
    dtype = pos.dtype
    nat = pos.shape[-2]
    rij = pos.unsqueeze(-3) - pos.unsqueeze(-2)  # (..., nat, nat, 3)
    dist = torch.linalg.norm(rij, dim=-1)  # (..., nat, nat)
    # unit vectors and distance derivative
    eps = torch.finfo(dtype).eps
    inv_dist = 1.0 / torch.clamp(dist, min=eps)
    uvec = rij * inv_dist.unsqueeze(-1)  # (..., nat, nat, 3)

    rca = r_cov[numbers.long()]  # (..., nat)
    rcb = r_cov[numbers.long()]  # (..., nat)
    rc_ab = 0.5 * (rca.unsqueeze(-1) + rcb.unsqueeze(-2))  # (..., nat, nat)

    x = k_cn * (dist - rc_ab) / torch.clamp(rc_ab, min=eps)
    pref = k_cn / (torch.clamp(rc_ab, min=torch.finfo(dtype).eps) * torch.sqrt(torch.tensor(torch.pi, dtype=dtype, device=device)))
    gpair = pref * torch.exp(-x * x)  # (..., nat, nat)

    # dR_AB/dR_X = (delta_BX - delta_AX) * uvec
    eye = torch.eye(nat, device=device, dtype=dtype)
    delta_AX = eye.unsqueeze(-2)  # (..., nat, nat)
    delta_BX = eye.unsqueeze(-3)  # (..., nat, nat)
    factor = (delta_BX - delta_AX).unsqueeze(-1)  # (..., nat, nat, 1)
    dcn = 0.5 * gpair.unsqueeze(-1) * factor * uvec  # (..., nat, nat, 3)
    return dcn
