from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from ..cn import coordination_number, coordination_number_grad

__all__ = ["RepulsionParams", "repulsion_energy", "repulsion_energy_and_gradient"]


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


def repulsion_energy_and_gradient(
    positions: torch.Tensor,
    numbers: torch.Tensor,
    params: RepulsionParams,
    charges_eeqbc: torch.Tensor,
    *,
    cn: Optional[torch.Tensor] = None,
    dcn_dpos: Optional[torch.Tensor] = None,
    dq_dpos: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (E^{rep}, dE^{rep}/dR) per doc/theory/11_semi_classical_repulsion.md.

    Theory mapping:
      - Energy Eq. (52) with Z^{eff}(q^{EEQBC}) from Eq. (53)
      - Kernel f^{rep}(R) Eq. (54)
      - α_{AB} combination Eq. (55), α_A(CN_A) Eq. (56)
      - Gradient decomposition Eq. (57a–b)
      - ∂α_{AB}/∂R_X via Eq. (60b) and ∂α_A/∂R_X via Eq. (61)

    Notes:
      - The charge-derivative contribution ∂Z^{eff}/∂R_X (Eq. 58) requires ∂q^{EEQBC}/∂R_X.
        If any of k^q or k^{q,2} are non-zero for active elements and dq_dpos is None,
        this function raises ValueError (no hidden defaults per policy).
    """
    device = positions.device
    dtype = positions.dtype
    nat = numbers.shape[0]
    z = numbers.long()

    # CN and its gradient
    if cn is None:
        if params.r_cov is None or params.k_cn is None:
            raise ValueError("CN not provided and (r_cov, k_cn) missing in parameters.")
        cn = coordination_number(positions[None, ...], numbers[None, ...], params.r_cov.to(device=device, dtype=dtype), float(params.k_cn))[0]
    if dcn_dpos is None:
        if params.r_cov is None or params.k_cn is None:
            raise ValueError("CN gradient required but (r_cov, k_cn) missing in parameters.")
        dcn_dpos = coordination_number_grad(positions, numbers, params.r_cov.to(device=device, dtype=dtype), float(params.k_cn))  # (A,B,X,3)
    # Reduce neighbor index B to obtain ∂CN_A/∂R_X (sum over neighbors)
    if dcn_dpos.dim() == 4:
        dcn_AX = dcn_dpos.sum(dim=1)
    else:
        dcn_AX = dcn_dpos  # already (A,X,3)

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
    # Eq. (56)
    alpha_a = _alpha_a(alpha0, kcn_a, cn)

    # Distances and unit vectors
    rij = positions.unsqueeze(1) - positions.unsqueeze(0)  # (i,j,3)
    dist = torch.linalg.norm(rij, dim=-1)  # (i,j)
    eps = torch.finfo(dtype).eps
    inv_r = 1.0 / torch.clamp(dist, min=eps)
    uij = rij * inv_r.unsqueeze(-1)

    # Pairwise combinations
    alpha_i = alpha_a.unsqueeze(1) + torch.zeros_like(dist)
    alpha_j = alpha_a.unsqueeze(0) + torch.zeros_like(dist)
    denom = torch.clamp(alpha_i + alpha_j, min=eps)
    alpha_ab = (alpha_i * alpha_j) / denom  # Eq. (55)

    r0_ab = torch.sqrt(r0.unsqueeze(1) * r0.unsqueeze(0))
    # kpen1 per atom and per pair (A vs rest)
    is_hhe = (z <= 2).to(dtype)
    kpen1_a = is_hhe * params.kpen1_hhe + (1.0 - is_hhe) * params.kpen1_rest
    kpen1_i = kpen1_a.unsqueeze(1) + torch.zeros_like(dist)
    kpen1_j = kpen1_a.unsqueeze(0) + torch.zeros_like(dist)

    # Series and exponential (Eq. 54)
    r = dist
    series = (
        1.0
        + 0.5 * (kpen1_i + kpen1_j) * inv_r
        + params.kpen2 * inv_r**2
        + params.kpen3 * inv_r**3
        + params.kpen4 * inv_r**4
    )
    pow1 = torch.clamp(r + r0_ab, min=eps)
    gexp = torch.exp(-alpha_ab * pow1**params.kexp)

    zeff_i = zeff.unsqueeze(1)
    zeff_j = zeff.unsqueeze(0)
    Zprod = zeff_i * zeff_j
    mask = ~torch.eye(nat, dtype=torch.bool, device=device)

    # Energy (reuse existing path)
    e = 0.5 * (Zprod * series * gexp)[mask].sum()

    # --- Gradient contributions ---
    grad = torch.zeros((nat, 3), dtype=dtype, device=device)
    # Unique pairs mask (i<j)
    m_upper = torch.triu(torch.ones_like(dist, dtype=torch.bool), diagonal=1)

    # 1) Distance (radial) part of ∂f/∂R_X (Eq. 59 first term and the α_AB·kexp term via ∂R)
    ds_dr = (
        -0.5 * (kpen1_i + kpen1_j) * inv_r**2
        - 2.0 * params.kpen2 * inv_r**3
        - 3.0 * params.kpen3 * inv_r**4
        - 4.0 * params.kpen4 * inv_r**5
    )
    dlogg_dr = -alpha_ab * params.kexp * pow1**(params.kexp - 1.0)
    df_dr = ds_dr * gexp + series * gexp * dlogg_dr
    fpair = (Zprod * df_dr)  # scalar per pair; energy uses 0.5 Σ_{i≠j}; here we use i<j without 0.5
    vec = fpair.unsqueeze(-1) * uij  # (i,j,3)
    # Accumulate using upper-triangular pairs
    mu = m_upper.unsqueeze(-1)
    grad = grad + (vec * mu).sum(dim=1)  # i row gains +
    grad = grad - (vec * mu).sum(dim=0)  # j col gains -

    # 2) α-chain part via ∂α_AB/∂R_X (Eq. 59 second term with ∂α_AB)
    # df/dα_AB = - series * gexp * pow1^{kexp}
    df_dalpha = -series * gexp * (pow1**params.kexp)
    # ∂α_A/∂R_X (Eq. 61): α_A^0 · k^{CN}_A · (1/(2√CN_A)) · ∂CN_A/∂R_X
    sqrt_cn = torch.sqrt(torch.clamp(cn, min=eps))
    coef_a = alpha0 * kcn_a * (0.5 / torch.clamp(sqrt_cn, min=eps))
    # dcn_dpos: (A,X,3) -> make (A,1,X,3) and (1,B,X,3)
    dalpha_A = (coef_a.view(nat, 1, 1) * dcn_AX)  # (A,X,3)
    # ∂α_AB/∂R_X = (α_B^2/(α_A+α_B)^2) ∂α_A/∂R_X + (α_A^2/(α_A+α_B)^2) ∂α_B/∂R_X
    coefA = (alpha_j**2) / (denom**2)
    coefB = (alpha_i**2) / (denom**2)
    # Broadcast dα_i and dα_j over pair grid
    dalpha_i_rep = dalpha_A.unsqueeze(1).expand(nat, nat, nat, 3)  # (i,j,X,3) using i index
    dalpha_j_rep = dalpha_A.unsqueeze(0).expand(nat, nat, nat, 3)  # (i,j,X,3) using j index
    dalpha_AB = coefA.unsqueeze(2).unsqueeze(3) * dalpha_i_rep + coefB.unsqueeze(2).unsqueeze(3) * dalpha_j_rep
    # Pair scalar for chain term with unique pairs
    S_chain = (Zprod * df_dalpha)
    C_ABX3 = S_chain.unsqueeze(2).unsqueeze(3) * dalpha_AB  # (i,j,X,3)
    C_ABX3 = C_ABX3 * m_upper.unsqueeze(2).unsqueeze(3)
    grad = grad + C_ABX3.sum(dim=(0, 1))

    # 3) Z^{eff} chain via ∂q/∂R_X (Eq. 58)
    if ((kq[numbers.long()].abs() > 0).any() or (kq2[numbers.long()].abs() > 0).any()) and dq_dpos is None:
        raise ValueError("Repulsion gradient requires dq^{EEQBC}/dR when k^q or k^{q,2} are non-zero (doc/theory/11 Eq. 58)")
    if dq_dpos is not None:
        # dZ_A/∂R_X = z0_A ( -kq_A + 2 kq2_A q_A ) dq_A/∂R_X
        wA = z0 * ( -kq + 2.0 * kq2 * q )  # (A,)
        dZ_A = wA.view(nat, 1, 1) * dq_dpos.to(device=device, dtype=dtype)  # (A,X,3)
        f_AB = (series * gexp)  # (i,j)
        termZ = (
            f_AB.unsqueeze(2).unsqueeze(3)
            * (
                zeff_j.unsqueeze(2).unsqueeze(3) * dZ_A.unsqueeze(1)
                + zeff_i.unsqueeze(2).unsqueeze(3) * dZ_A.unsqueeze(0).transpose(0, 1)
            )
        )  # (i,j,X,3)
        termZ = termZ * m_upper.unsqueeze(2).unsqueeze(3)
        grad = grad + termZ.sum(dim=(0, 1))

    # Sanity: gradient shape (nat,3)
    if grad.shape != (nat, 3):  # pragma: no cover - defensive guard
        raise RuntimeError(f"repulsion_energy_and_gradient produced gradient with shape {tuple(grad.shape)}, expected ({nat},3)")
    return e, grad
