from __future__ import annotations

"""DFT-revD4 dispersion (doc/theory/22_dft_revd4.md).

Implements loaders for method-level parameters (s6,s8,s9,a1,alp,damping) from
`parameters/dftd4parameters.toml` and provides a strict interface for computing
E^{revD4} given C6/C8 tables and charge-dependent scaling ζ_A(q_A).

This module does not fabricate missing reference polarizabilities or C_n tables.
If required inputs are not given, a descriptive exception is raised.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch

from gxtb.params.loader import load_d4_parameters, select_d4_params

Tensor = torch.Tensor

__all__ = [
    "D4Method",
    "load_d4_method",
    "d4_energy",
    "d4_energy_components",
    "revd4_energy",
    "revd4_energy_from_mats",
    "build_revd4_c6_c8",
    "build_revd4_c9_from_c6",
    "compute_d4s_atomic_potential",
]

# Theory cross-reference for revD4 implementation
__doc_refs__ = {
    "file": "doc/theory/22_dft_revd4.md",
    "eqs": [161, 162, 163, 167, 169, 170, 171],
}


@dataclass(frozen=True)
class D4Method:
    s6: float
    s8: float
    s9: float
    a1: float
    a2: float  # retained for compatibility though doc removes it; if absent use 0.0
    alp: float
    damping: str
    mbd: str


def load_d4_method(
    toml_path: str,
    variant: str = "bj-eeq-atm",
    functional: Optional[str] = None,
) -> D4Method:
    """Select D4 parameters from TOML with tad-dftd4 semantics.

    - variant: e.g., 'bj-eeq-atm' or 'bj-eeq-two'.
    - functional: if None/'default', pull from [default.parameter]; otherwise
      case-insensitive lookup in [parameter.<functional>].

    Note: Per doc/theory/22_dft_revd4.md Eq. 170–171, revD4 uses BJ-type
    damping with a single global a1; the a2 term of legacy D4 is not used
    in revD4 energy here, even if present in the TOML.
    """
    params = load_d4_parameters(toml_path)
    block = select_d4_params(params, method="d4", functional=functional, variant=variant)
    s6 = float(block.get("s6", 1.0))
    s8 = float(block.get("s8", 0.0))
    s9 = float(block.get("s9", 0.0))
    a1 = float(block.get("a1", 0.0))
    a2 = float(block.get("a2", 0.0))  # unused in revD4 BJ, kept for completeness
    alp = float(block.get("alp", 16.0))
    damping = str(block.get("damping", "bj"))
    mbd = str(block.get("mbd", "none"))
    return D4Method(s6=s6, s8=s8, s9=s9, a1=a1, a2=a2, alp=alp, damping=damping, mbd=mbd)


# ===================== D4 (tad-dftd4) reimplementation =====================


def _trapzd_twobody(alpha: Tensor) -> Tensor:
    """Casimir–Polder integration of reference polarizabilities to C6 (pair-ref).

    Inputs
    - alpha: (nat, R, W) reference polarizabilities α_{Z,r}(iω)
    - weights: (nat, R) Gaussian+ζ weights per atom/reference (already masked)

    Output: rc6_ref: (nat, nat, R, R) reference C6 tensor (per pair and references)
    """
    # Using weights on α first yields per-atom weighted α; C6 reference tensor = ∫ α_A α_B dω
    # But tad-dftd4 computes rc6 = trapzd(alpha) producing (n,n,R,R) before weighting.
    # We mirror that: integrate α over ω, outer on references.
    # α: (n,R,W) -> integrate on W: t = ∫ α(ω) α(ω) dω per (A,B,rA,rB)
    thopi = 3.0 / 3.141592653589793238462643383279502884197
    # Fixed 23-point trapezoidal weights (copied from tad-dftd4 utils)
    w = torch.tensor([
        2.4999500000000000e-002,
        4.9999500000000000e-002,
        7.5000000000000010e-002,
        0.1000000000000000,
        0.1000000000000000,
        0.1000000000000000,
        0.1000000000000000,
        0.1000000000000000,
        0.1000000000000000,
        0.1000000000000000,
        0.1000000000000000,
        0.1500000000000000,
        0.2000000000000000,
        0.2000000000000000,
        0.2000000000000000,
        0.2000000000000000,
        0.3500000000000000,
        0.5000000000000000,
        0.7500000000000000,
        1.0000000000000000,
        1.7500000000000000,
        2.5000000000000000,
        1.2500000000000000,
    ], device=alpha.device, dtype=alpha.dtype)
    n, R, W = alpha.shape
    # (n,R,W) x (n,R,W) -> (n,n,R,R,W)
    Ai = alpha.unsqueeze(1).unsqueeze(3)
    Aj = alpha.unsqueeze(0).unsqueeze(2)
    prod = Ai * Aj  # (n,n,R,R,W)
    rc6 = thopi * torch.einsum('ijraw,w->ijra', prod, w)  # (n,n,R,R)
    return rc6


def _d4_weight_references(numbers: Tensor, cn: Tensor, q: Tensor, ref: dict, *, ga: float = 3.0, gc: float = 2.0, wf: float | Tensor = 6.0) -> tuple[Tensor, Tensor]:
    """Compute Gaussian weights times ζ scaling (zeta*gw) per tad-dftd4 D4 model.

    - ref must contain tensors on same device/dtype: 'refc','refcovcn','clsq','zeff','gam'
    - wf: weighting factor (scalar 6.0 by default)

    Returns: tuple (zeta, gw), each (nat, R); zeta masked to refc>0.
    """
    device, dtype = cn.device, cn.dtype
    Z = numbers.long()
    refc = ref['refc'][Z]           # (n,R)
    refcn = ref['refcovcn'][Z].to(dtype=torch.float64)  # double for stability
    mask = refc > 0
    # Gaussian weights per tad-dftd4 (D4Model.weight_references):
    # tmp = exp(-(cn - cn_ref)^2); sum_{i=1..Ns} tmp^(i*wf)
    wf_t = wf if torch.is_tensor(wf) else torch.tensor(float(wf), device=device, dtype=torch.float64)
    dcn = cn.to(torch.float64).unsqueeze(-1) - refcn  # (n, R)
    tmp = torch.exp(-dcn * dcn)  # (n,R)
    def refc_pow(ns: int) -> Tensor:
        return sum((torch.pow(tmp, i * wf_t) for i in range(1, ns + 1)), torch.tensor(0.0, device=device, dtype=torch.float64))
    expw = torch.where(refc == 1, refc_pow(1), tmp)
    expw = torch.where(refc == 3, refc_pow(3), expw)
    expw = torch.where(mask, expw, torch.tensor(0.0, device=device, dtype=torch.float64))
    norm = torch.where(mask, expw.sum(dim=-1, keepdim=True), torch.tensor(1e-300, device=device, dtype=torch.float64))
    gw = (expw / norm).to(dtype)
    # Charge scaling ζ(q) using EEQ reference charges (clsq)
    zeff = ref['zeff'][Z].to(dtype)
    gam = (ref['gam'][Z] * gc).to(dtype)
    refq = ref['clsq'][Z].to(dtype)
    qmod = q + zeff
    # zeta(q) = exp(ga * (1 - exp(gam * (1 - (refq+zeff)/(q+zeff))))) for q+zeff>0 else exp(ga)
    eps = torch.tensor(torch.finfo(dtype).eps, device=device, dtype=dtype)
    scale = torch.exp(gam.unsqueeze(-1) * (1.0 - (refq + zeff.unsqueeze(-1)) / (qmod.unsqueeze(-1) - eps)))
    zeta = torch.where(qmod.unsqueeze(-1) > 0.0, torch.exp(torch.tensor(ga, device=device, dtype=dtype) * (1.0 - scale)), torch.exp(torch.tensor(ga, device=device, dtype=dtype)))
    zeta = torch.where(mask, zeta, torch.tensor(0.0, device=device, dtype=dtype))
    return zeta, gw


def _d4_ref_alpha(numbers: Tensor, ref: dict, *, gc: float = 2.0, ga: float = 3.0) -> Tensor:
    """Build per-atom reference polarizabilities α(iω) per tad-dftd4 BaseModel._get_alpha (Eq. 164).

    Uses reference charges 'clsh' (EEQ) for ζ_ref, not the dynamic charges.

    Required ref keys: 'refsys','refascale','refalpha','refscount','secscale','secalpha','zeff','gam','clsh'.
    Returns α: (nat, R, W)
    """
    Z = numbers.long()
    device = ref['refalpha'].device
    dtype = ref['refalpha'].dtype
    refsys = ref['refsys'][Z]
    refascale = ref['refascale'][Z].to(dtype)
    refalpha = ref['refalpha'][Z].to(dtype)
    refscount = ref['refscount'][Z].to(dtype)
    secscale = ref['secscale'].to(dtype)
    secalpha = ref['secalpha'].to(dtype)
    # zeta_ref from clsh (reference charges)
    if 'clsh' not in ref:
        raise KeyError("ref dict missing 'clsh' required for reference ζ in α construction")
    zeff = ref['zeff'][refsys]
    gam = (ref['gam'][refsys] * gc).to(dtype)
    qref = (ref['clsh'][Z]).to(dtype)  # (nat, R)
    # Broadcast to match refsys indexing for zeff,gam
    zeta = torch.exp(torch.tensor(ga, device=device, dtype=dtype) * (1.0 - torch.exp(gam * (1.0 - zeff / (qref + zeff)))))
    sec_s = secscale[refsys]              # (n,R) or (n,R,1) depending on source
    if sec_s.ndim == 2:
        sec_s = sec_s.unsqueeze(-1)
    sec_a = secalpha[refsys]              # (n,R,W)
    aiw = sec_s * sec_a * zeta.unsqueeze(-1)  # (n,R,W)
    h = refalpha - refscount.unsqueeze(-1) * aiw
    alpha = refascale.unsqueeze(-1) * h
    return torch.where(alpha > 0.0, alpha, torch.tensor(0.0, device=device, dtype=dtype))


def d4_energy_components(numbers: Tensor, positions: Tensor, charges: Tensor, method: D4Method, ref: dict) -> tuple[Tensor, Tensor, Tensor]:
    """Compute D4 two-body and ATM energies per tad-dftd4.

    Returns (E2, EATM, E_total_disp)
    """
    device, dtype = positions.device, positions.dtype
    n = numbers.shape[0]
    # Coordination numbers
    cn = ref.get('cn', None)
    if cn is None:
        if 'r_cov' in ref and 'k_cn' in ref:
            from gxtb.cn import coordination_number
            cn = coordination_number(positions, numbers, ref['r_cov'].to(device=device, dtype=dtype), float(ref['k_cn']))
        else:
            raise ValueError("D4: 'ref' must supply 'cn' or both 'r_cov' and 'k_cn' to compute CN (no hidden defaults)")
    # Weights: ζ(q) and Gaussian gw(CN)
    zeta, gw = _d4_weight_references(numbers, cn, charges, ref)
    # Reference α(iω) built with reference ζ (clsh), Eq. 164
    alpha = _d4_ref_alpha(numbers, ref)
    # Reference C6
    rc6 = _trapzd_twobody(alpha)
    # Atomic pair C6 via weighting: C6 = Σ_{rA,rB} rc6 * W_A * W_B with W = ζ(q)·gw
    W = zeta * gw
    C6 = torch.einsum('ijab,ia,jb->ij', rc6, W, W)
    # Two-body energy
    rij = positions.unsqueeze(0) - positions.unsqueeze(1)
    R = torch.linalg.norm(rij, dim=-1)
    i = torch.arange(n, device=device)
    j = torch.arange(n, device=device)
    mask = (i[:, None] > j[None, :])
    eps = torch.finfo(dtype).eps
    R = R + torch.eye(n, device=device, dtype=dtype) * eps
    r4r2 = ref['r4r2'][numbers.long()].to(dtype)
    R0 = method.a1 * torch.sqrt(3.0 * r4r2.unsqueeze(1) * r4r2.unsqueeze(0)) + method.a2
    f6 = 1.0 / (R**6 + R0**6)
    f8 = 1.0 / (R**8 + R0**8)
    C8 = C6 * (3.0 * r4r2.unsqueeze(1) * r4r2.unsqueeze(0))
    E2 = -(
        method.s6 * torch.where(mask, C6 * f6, torch.zeros_like(R)) +
        method.s8 * torch.where(mask, C8 * f8, torch.zeros_like(R))
    ).sum()
    # ATM approximate C9 = sqrt(|C6_AB*C6_AC*C6_BC|)
    C9 = torch.zeros((n, n, n), device=device, dtype=dtype)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                if a == b or a == c or b == c:
                    continue
                C9[a, b, c] = torch.sqrt(torch.clamp(C6[a, b] * C6[a, c] * C6[b, c], min=0.0))
    # Critical radii for ATM via BJ radii (R0) triplet product
    R2 = R * R
    Rab2 = R2.unsqueeze(-1)
    Rac2 = R2.unsqueeze(-2)
    Rbc2 = R2.unsqueeze(-3)
    Rab = R.unsqueeze(-1)
    Rac = R.unsqueeze(-2)
    Rbc = R.unsqueeze(-3)
    R0ab = R0.unsqueeze(-1)
    R0ac = R0.unsqueeze(-2)
    R0bc = R0.unsqueeze(-3)
    R0prod = R0ab * R0ac * R0bc
    Rprod = Rab * Rac * Rbc + eps
    # Zero damping f9 = 1/(1 + 6 (R0bar / Rbar)^alp), but we operate on products -> use alp/3
    fdamp = 1.0 / (1.0 + 6.0 * (R0prod / Rprod) ** (method.alp / 3.0))
    s = (Rab2 + Rbc2 - Rac2) * (Rab2 - Rbc2 + Rac2) * (-Rab2 + Rbc2 + Rac2)
    Rprod3 = Rprod * Rprod * Rprod
    Rprod5 = Rprod3 * Rprod * Rprod
    ang = 0.375 * s / Rprod5 + 1.0 / Rprod3
    EATM = (method.s9 * (ang * fdamp * C9)).sum() / 6.0
    E = E2 + EATM
    return E2, EATM, E


def d4_energy(numbers: Tensor, positions: Tensor, charges: Tensor, method: D4Method, ref: dict) -> Tensor:
    """Convenience wrapper returning total D4 dispersion energy using reimplemented algorithm (no external imports)."""
    _, _, E = d4_energy_components(numbers, positions, charges, method, ref)
    return E


def d4_energy_with_grad(
    numbers: Tensor,
    positions: Tensor,
    charges: Tensor,
    method: D4Method,
    ref: dict,
) -> tuple[Tensor, Tensor]:
    """Return (E^{D4}, ∂E/∂R) for molecular (non-PBC) case via autograd.

    Computes E using d4_energy_components (two-body + ATM) and differentiates w.r.t. positions.
    All inputs must be torch tensors on the same device/dtype.
    """
    pos_req = positions.detach().clone().requires_grad_(True)
    _, _, E = d4_energy_components(numbers, pos_req, charges.to(pos_req), method, ref)
    grad, = torch.autograd.grad(E, pos_req, create_graph=False)
    return E.detach(), grad.detach()


def revd4_energy(
    numbers: Tensor,
    positions: Tensor,
    q: Tensor,
    C6: Dict[Tuple[int, int], float],
    C8: Dict[Tuple[int, int], float],
    method: D4Method,
    *,
    C9: Dict[Tuple[int, int, int], float] | None = None,
) -> Tensor:
    """Compute E^{revD4} per Eq. 161 with damping per Eq. 170–171.

    Requires precomputed C6/C8 pair coefficients and atomic charges q (for ζ). This function
    does not compute ζ_A (Eq. 167) nor CN-smoothed references; caller must pass consistent C_n.
    """
    device = positions.device
    dtype = positions.dtype
    nat = numbers.shape[0]
    # pair distances
    rij = positions.unsqueeze(0) - positions.unsqueeze(1)
    R = torch.linalg.norm(rij, dim=-1)
    E = torch.zeros((), dtype=dtype, device=device)
    for i in range(nat):
        Zi = int(numbers[i].item())
        for j in range(i):
            Zj = int(numbers[j].item())
            key = (min(Zi, Zj), max(Zi, Zj))
            c6 = C6.get(key)
            c8 = C8.get(key)
            if c6 is None or c8 is None:
                raise ValueError(f"Missing C6/C8 for pair {key}")
            # Eq. 170: R0 = sqrt(C8/C6); revD4 uses BJ with single a1 (no a2)
            R0 = (c8 / c6) ** 0.5
            r = R[i, j]
            f6 = r**6 / (r**6 + (method.a1 * R0) ** 6)
            f8 = r**8 / (r**8 + (method.a1 * R0) ** 8)
            E = E - (method.s6 * c6 * f6 / (r**6) + method.s8 * c8 * f8 / (r**8))
    # ATM three-body term (Eq. 161) when C9 provided, with zero-damping (Eq. 171)
    if C9 is not None and method.s9 != 0.0:
        # iterate triples A>B>C
        for a in range(nat):
            Za = int(numbers[a].item())
            for b in range(a):
                Zb = int(numbers[b].item())
                for c in range(b):
                    Zc = int(numbers[c].item())
                    key = tuple(sorted((Za, Zb, Zc)))  # (Zmin, Zmid, Zmax)
                    c9 = C9.get(key)
                    if c9 is None:
                        raise ValueError(f"Missing C9 for triple {key}")
                    Rab = R[a, b]; Rac = R[a, c]; Rbc = R[b, c]
                    # Cosines for angles at A, B, C
                    def cos_angle(i,j,k):  # angle at j with segments ji and jk
                        v1 = positions[i] - positions[j]
                        v2 = positions[k] - positions[j]
                        return torch.dot(v1, v2) / (torch.linalg.norm(v1) * torch.linalg.norm(v2))
                    cos_ABC = cos_angle(a, b, c)
                    cos_BCA = cos_angle(b, c, a)
                    cos_CAB = cos_angle(c, a, b)
                    angular = 3.0 * cos_ABC * cos_BCA * cos_CAB + 1.0
                    # Eq. 171: zero damping for ATM with R0 product
                    key_ab = (min(Za, Zb), max(Za, Zb))
                    key_ac = (min(Za, Zc), max(Za, Zc))
                    key_bc = (min(Zb, Zc), max(Zb, Zc))
                    c6ab = C6[key_ab]; c8ab = C8[key_ab]
                    c6ac = C6[key_ac]; c8ac = C8[key_ac]
                    c6bc = C6[key_bc]; c8bc = C8[key_bc]
                    R0ab = (c8ab / c6ab) ** 0.5
                    R0ac = (c8ac / c6ac) ** 0.5
                    R0bc = (c8bc / c6bc) ** 0.5
                    damp9 = 1.0 / (1.0 + 6.0 * ((method.a1**3) * (R0ab * R0bc * R0ac) / (Rab * Rbc * Rac)) ** 16)
                    E = E - method.s9 * c9 * angular / ((Rab * Rac * Rbc) ** 3) * damp9
    # Three-body ATM not included: requires C9 and angles; add when data available
    return E


def revd4_energy_from_mats(
    numbers: Tensor,
    positions: Tensor,
    C6_mat: Tensor,
    C8_mat: Tensor,
    method: D4Method,
    *,
    C9_mat: Tensor | None = None,
) -> Tensor:
    """revD4 energy using full pair matrices.

    - Two-body per Eq. 161 with BJ damping (Eq. 170) using R0 = sqrt(C8/C6).
    - Optional ATM (Eq. 161) with zero-damping (Eq. 171) using R0 products.

    All inputs are tensors on the same device/dtype.
    """
    device = positions.device
    dtype = positions.dtype
    nat = numbers.shape[0]
    rij = positions.unsqueeze(0) - positions.unsqueeze(1)
    R = torch.linalg.norm(rij, dim=-1)
    i = torch.arange(nat, device=device)
    j = torch.arange(nat, device=device)
    mask = (i[:, None] > j[None, :])
    # Avoid singular diag
    R = R + torch.eye(nat, device=device, dtype=dtype) * torch.finfo(dtype).eps
    R0 = torch.sqrt(torch.clamp(C8_mat / torch.clamp(C6_mat, min=torch.finfo(dtype).tiny), min=0.0))
    r = R
    f6 = r**6 / (r**6 + (method.a1 * R0) ** 6)
    f8 = r**8 / (r**8 + (method.a1 * R0) ** 8)
    term = -(
        method.s6 * torch.where(mask, C6_mat * f6 / r**6, torch.zeros_like(r))
        + method.s8 * torch.where(mask, C8_mat * f8 / r**8, torch.zeros_like(r))
    )
    E = term.sum()
    if C9_mat is not None and method.s9 != 0.0:
        # Triple loops vectorized are complex; keep simple for now (nat typically small)
        # Note: this term is still O(N^3) like literature.
        for a in range(nat):
            for b in range(a):
                for c in range(b):
                    Rab = R[a, b]; Rac = R[a, c]; Rbc = R[b, c]
                    if Rab <= 0 or Rac <= 0 or Rbc <= 0:
                        continue
                    vAB = positions[a] - positions[b]
                    vCB = positions[c] - positions[b]
                    vCA = positions[c] - positions[a]
                    vBA = positions[b] - positions[a]
                    vAC = positions[a] - positions[c]
                    cos_ABC = torch.dot(vAB, vCB) / (Rab * Rbc)
                    cos_BCA = torch.dot(vBC := -vCB, vAC) / (Rbc * Rac)
                    cos_CAB = torch.dot(vCA, vBA) / (Rac * Rab)
                    angular = 3.0 * cos_ABC * cos_BCA * cos_CAB + 1.0
                    R0ab = torch.sqrt(torch.clamp(C8_mat[a, b] / C6_mat[a, b], min=0.0))
                    R0ac = torch.sqrt(torch.clamp(C8_mat[a, c] / C6_mat[a, c], min=0.0))
                    R0bc = torch.sqrt(torch.clamp(C8_mat[b, c] / C6_mat[b, c], min=0.0))
                    damp9 = 1.0 / (
                        1.0
                        + 6.0
                        * ((method.a1**3) * (R0ab * R0bc * R0ac) / (Rab * Rbc * Rac)) ** 16
                    )
                    E = E - method.s9 * C9_mat[a, b, c] * angular / ((Rab * Rac * Rbc) ** 3) * damp9
    return E


# === revD4 Cn builder (Eqs. 162–169) ===


class RevD4DataMissing(Exception):
    pass


def _trapzd_noref(pol1: Tensor, pol2: Tensor | None = None) -> Tensor:
    """23‑point Casimir–Polder trapezoidal integration (consistent with D4 grid).

    - Input shapes: (nat,W) or (nat,nat,W). Returns (nat,nat).
    - Eq. 162: C6 = 3/pi ∫ α_A(iω) α_B(iω) dω; discretized by 23‑point weights.
    """
    thopi = 3.0 / 3.141592653589793238462643383279502884197
    weights = torch.tensor(
        [
            2.4999500000000000e-002,
            4.9999500000000000e-002,
            7.5000000000000010e-002,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1000000000000000,
            0.1500000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.2000000000000000,
            0.3500000000000000,
            0.5000000000000000,
            0.7500000000000000,
            1.0000000000000000,
            1.7500000000000000,
            2.5000000000000000,
            1.2500000000000000,
        ],
        device=pol1.device,
        dtype=pol1.dtype,
    )
    p2 = pol1 if pol2 is None else pol2
    if pol1.ndim == 2:
        return thopi * torch.einsum("w,iw,jw->ij", weights, pol1, p2)
    elif pol1.ndim == 3:
        return thopi * torch.einsum("w,ijw,ijw->ij", weights, pol1, p2)
    else:
        raise ValueError("trapzd_noref expects (nat,W) or (nat,nat,W) inputs")


def build_zeta_params(alpha0: Tensor, alpha_plus: Tensor, alpha_minus: Tensor, s: float = 1.0) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute (A,B,C,D) per Eq. 169 using α^0, α^+, α^- for each element and frequency.

    Shapes: all inputs (Zmax+1, W) for W frequency nodes. Returns 4 tensors of same shape.
    """
    # eq: 169a
    A = 0.5 * ((1 - s) * alpha_plus / alpha0 + (1 + s) * alpha_minus / alpha0)
    # eq: 169b
    B = 0.5 * ((1 - s) * alpha_plus / alpha0 - (1 + s) * alpha_minus / alpha0)
    # eq: 169d
    D = torch.atanh(torch.clamp((1.0 - A) / torch.clamp(B, min=torch.finfo(alpha0.dtype).eps), min=-1 + 1e-15, max=1 - 1e-15))
    # eq: 169c
    C = torch.atanh(torch.clamp(alpha_plus / (B * alpha0) - A / B, min=-1 + 1e-15, max=1 - 1e-15)) - D
    return A, B, C, D


def zeta_ratio(A: Tensor, B: Tensor, C: Tensor, D: Tensor, q: Tensor, qref: Tensor | float = 0.0) -> Tensor:
    """ζ_A(q)/ζ_A(qref) per Eq. 167 using tanh form.

    A,B,C,D: (nat, W) or (Zmax+1, W) mapped to atoms; q: (nat,); qref scalar or (nat,).
    Returns (nat, W).
    """
    # broadcast q to (nat, W)
    q = q.unsqueeze(-1)
    qref = torch.tensor(qref, device=q.device, dtype=q.dtype).expand_as(q) if not torch.is_tensor(qref) else qref.unsqueeze(-1)
    num = A + B * torch.tanh(C * q + D)
    den = A + B * torch.tanh(C * qref + D)
    return num / den


def build_revd4_c6_c8(
    numbers: Tensor,
    cn: Tensor,
    q: Tensor,
    beta2: Tensor,
    zeta_params: dict | None,
    *,
    ref: dict | None = None,
    zeta_mode: str = "revD4",
    d4s_data: dict | None = None,
) -> tuple[Tensor, Tensor]:
    """Construct pairwise C6 and C8 matrices for revD4.

    - numbers: (nat,) atomic numbers
    - cn: (nat,) CN^cov per Eq. 47 (caller supplies consistent model)
    - q: (nat,) Mulliken charges from SCF
    - beta2: (Zmax+1, Zmax+1) smoothing parameters for D4S weights (Eq. 165)
    - zeta_params: dict with tensors 'A','B','C','D' shaped (Zmax+1, W)

    Returns: (C6_mat (nat,nat), C8_mat (nat,nat)). Requires tad-dftd4 reference data.

    Notes:
    - Requires α^0, α^+, α^- derived A,B,C,D (Eq. 169). If not provided, raises.
    - Uses pair-specific weights W_{r,A}(CN_A; β2_AB) per Eq. 165 with N_s taken from refcounts per reference (refc ∈ {1,3}).
    - C8 is obtained via D4 relation C8 = C6 * 3 r4r2_A r4r2_B (as in tad-dftd4 twobody.py).
    """
    device = cn.device
    dtype = cn.dtype
    nat = numbers.shape[0]
    Zmax = int(numbers.max().item())

    if zeta_mode not in ("revD4", "d4s"):
        raise ValueError("zeta_mode must be 'revD4' or 'd4s'")
    if ref is None:
        raise RevD4DataMissing(
            "Missing revD4 reference dataset (refalpha/refcovcn/refc/r4r2). "
            "Do not import tad-dftd4 at runtime; provide these via parameters/ and pass through 'ref=' argument."
        )

    # Required references: Eq. 164 α^0_r,A(iω) and CN weights data
    refalpha = ref["refalpha"].to(device=device, dtype=dtype)  # (Z, R, W)
    refcovcn = ref["refcovcn"].to(device=device, dtype=dtype)  # (Z, R)
    refc = ref["refc"].to(device=device)  # (Z, R) integers in {0,1,3}
    r4r2 = ref["r4r2"].to(device=device, dtype=dtype)  # (Z,)
    trapzd = _trapzd_noref

    W = refalpha.shape[-1]
    maxr = refalpha.shape[1]

    # Gather per-atom reference arrays
    Z = numbers.long()
    a0 = refalpha[Z]  # (nat, R, W)
    cn_ref = refcovcn[Z]  # (nat, R)
    nref = refc[Z]  # (nat, R) Ns values (1 or 3)
    valid = (nref > 0)

    # Pair-specific β2
    beta2 = beta2.to(device=device, dtype=dtype)
    beta_ij = beta2[Z.unsqueeze(1), Z.unsqueeze(0)]  # (nat, nat)

    # Build Gaussian smoothing weights per Eq. 165: sum_{j=1..Ns} exp(-β2_ij j (ΔCN)^2)
    # ΔCN: (nat, R) per atom -> broadcast to (nat, nat, R)
    dcn_i = (cn.unsqueeze(-1) - cn_ref)  # (nat, R)
    dcn_i2 = dcn_i * dcn_i
    # Expand for pairs
    dcn_i2_ij = dcn_i2.unsqueeze(1).expand(nat, nat, maxr)  # (nat, nat, R)
    dcn_j = (cn.unsqueeze(-1) - cn_ref)  # reuse for j with swapped axis
    dcn_j2_ij = dcn_j.unsqueeze(0).expand(nat, nat, maxr)

    # Ns powers: We implement sum_{j=1..Ns} exp(-beta2 * j * x). Compute tmp = exp(-beta2*x); sum(tmp^j)
    tmp_i = torch.exp(-beta_ij.unsqueeze(-1) * dcn_i2_ij)  # (nat, nat, R)
    tmp_j = torch.exp(-beta_ij.unsqueeze(-1) * dcn_j2_ij)
    # Ns per atom-ref: (nat,R) -> pair shape
    Ns_i = nref.to(dtype=dtype).unsqueeze(1).expand(nat, nat, maxr)
    Ns_j = nref.to(dtype=dtype).unsqueeze(0).expand(nat, nat, maxr)

    def geom_sum(tmp: Tensor, Ns: Tensor) -> Tensor:
        # sum_{k=1..Ns} tmp^k = tmp * (1 - tmp^Ns) / (1 - tmp), handle tmp~1
        eps = torch.finfo(dtype).eps
        num = tmp * (1 - torch.pow(tmp, Ns))
        den = torch.clamp(1 - tmp, min=eps)
        return num / den

    w_i_raw = torch.where(valid.unsqueeze(1).expand_as(tmp_i), geom_sum(tmp_i, Ns_i), torch.zeros_like(tmp_i))
    w_j_raw = torch.where(valid.unsqueeze(0).expand_as(tmp_j), geom_sum(tmp_j, Ns_j), torch.zeros_like(tmp_j))

    # Normalize along reference dimension for each (i,j)
    w_i = w_i_raw / torch.clamp(w_i_raw.sum(dim=-1, keepdim=True), min=torch.finfo(dtype).tiny)
    w_j = w_j_raw / torch.clamp(w_j_raw.sum(dim=-1, keepdim=True), min=torch.finfo(dtype).tiny)

    # Weighted α^0 for each pair and frequency
    # a0: (nat,R,W); w_i: (nat,nat,R) -> (nat,nat,W)
    alpha0_i = torch.einsum("ijr,irw->ijw", w_i, a0)  # pair-specific α0 for atom i
    alpha0_j = torch.einsum("ijr,jrw->ijw", w_j, a0)  # for atom j

    if zeta_mode == "revD4":
        if zeta_params is None or not all(k in zeta_params for k in ("A", "B", "C", "D")):
            raise RevD4DataMissing(
                "Missing ζ parameters A/B/C/D (Eq. 169). Provide per-element frequency-resolved tensors."
            )
        # ζ parameters per atom from zeta_params (Zmax+1,W)
        A = zeta_params["A"].to(device=device, dtype=dtype)[Z]  # (nat, W)
        B = zeta_params["B"].to(device=device, dtype=dtype)[Z]
        C = zeta_params["C"].to(device=device, dtype=dtype)[Z]
        D = zeta_params["D"].to(device=device, dtype=dtype)[Z]
        zeta_i = zeta_ratio(A, B, C, D, q)  # (nat,W)
        zeta_j = zeta_ratio(A, B, C, D, q)
        # Broadcast to pairs
        zeta_i = zeta_i.unsqueeze(1).expand(nat, nat, W)
        zeta_j = zeta_j.unsqueeze(0).expand(nat, nat, W)
        alpha_i = alpha0_i * zeta_i
        alpha_j = alpha0_j * zeta_j
    else:
        # D4S hardness-based ζ (temporary alternative to Eq. 167)
        # Implements tad-dftd4 BaseModel._zeta with explicit parameters and no imports.
        # zeta(q) = exp(ga * (1 - exp(gam * (1 - q_ref/(q_mod - eps))))) for q_mod>0 else exp(ga)
        # where q_mod = q + ZEFF_Z and q_ref is the per-reference charge table (Z,R) + ZEFF_Z.
        if d4s_data is None or not all(k in d4s_data for k in ("refq", "gam", "zeff", "ga", "gc")):
            raise RevD4DataMissing(
                "Missing D4S data: require refq(Z,R), gam(Z), zeff(Z), and scalars ga,gc to build hardness-based ζ."
            )
        refq = d4s_data["refq"].to(device=device, dtype=dtype)  # (Z, R)
        gam_tbl = d4s_data["gam"].to(device=device, dtype=dtype)  # (Z,)
        zeff_tbl = d4s_data["zeff"].to(device=device, dtype=dtype)  # (Z,)
        ga = float(d4s_data["ga"])  # no hidden defaults: must be provided
        gc = float(d4s_data["gc"])  # no hidden defaults: must be provided
        # Per-atom constants
        zeff = zeff_tbl[Z]  # (nat,)
        gamZ = gam_tbl[Z] * gc  # (nat,)
        # Build per-atom per-ref ζ; then incorporate inside the reference sum
        qmod_i = q + zeff  # (nat,)
        qmod_j = q + zeff  # (nat,)
        qref_i = refq[Z] + zeff.unsqueeze(-1)  # (nat, R)
        qref_j = refq[Z] + zeff.unsqueeze(-1)  # (nat, R)
        eps = torch.tensor(torch.finfo(dtype).eps, device=device, dtype=dtype)
        # i-side ζ per reference
        scale_i = torch.exp(gamZ.unsqueeze(-1) * (1.0 - qref_i / (qmod_i.unsqueeze(-1) - eps)))
        zeta_i_ref = torch.where(
            qmod_i.unsqueeze(-1) > 0.0, torch.exp(torch.tensor(ga, dtype=dtype, device=device) * (1.0 - scale_i)), torch.exp(torch.tensor(ga, dtype=dtype, device=device))
        )  # (nat, R)
        # j-side ζ per reference
        scale_j = torch.exp(gamZ.unsqueeze(-1) * (1.0 - qref_j / (qmod_j.unsqueeze(-1) - eps)))
        zeta_j_ref = torch.where(
            qmod_j.unsqueeze(-1) > 0.0, torch.exp(torch.tensor(ga, dtype=dtype, device=device) * (1.0 - scale_j)), torch.exp(torch.tensor(ga, dtype=dtype, device=device))
        )  # (nat, R)
        # Combine ζ with CN weights inside the reference sum (per D4S design)
        wz_i = w_i * zeta_i_ref.unsqueeze(1).expand_as(w_i)
        wz_j = w_j * zeta_j_ref.unsqueeze(0).expand_as(w_j)
        # Weighted α0 per pair and frequency
        alpha_i = torch.einsum("ijr,irw->ijw", wz_i, a0)
        alpha_j = torch.einsum("ijr,jrw->ijw", wz_j, a0)

    # Casimir–Polder integration Eq. 162 using tad-dftd4 trapzd weights (consistent with 23-point grid)
    C6_mat = trapzd(alpha_i, alpha_j)  # (nat, nat)

    # C8 via D4 relation
    rA = r4r2[Z]
    rB = r4r2[Z]
    C8_mat = C6_mat * (3.0 * rA.unsqueeze(1) * rB.unsqueeze(0))

    return C6_mat, C8_mat


def build_revd4_c9_from_c6(C6_mat: Tensor) -> Tensor:
    """Construct ATM C9 coefficients via Eq. 163: C9_ABC ≈ sqrt(C6_AB C6_AC C6_BC).

    Returns tensor (nat, nat, nat), with zeros on invalid/diagonal combinations.
    """
    device = C6_mat.device
    dtype = C6_mat.dtype
    nat = C6_mat.shape[0]
    C9 = torch.zeros((nat, nat, nat), device=device, dtype=dtype)
    for a in range(nat):
        for b in range(nat):
            for c in range(nat):
                if a == b or a == c or b == c:
                    continue
                C9[a, b, c] = torch.sqrt(
                    torch.clamp(C6_mat[a, b] * C6_mat[a, c] * C6_mat[b, c], min=0.0)
                )
    return C9


def compute_d4s_atomic_potential(
    numbers: Tensor,
    positions: Tensor,
    cn: Tensor,
    q: Tensor,
    method: D4Method,
    beta2: Tensor,
    ref: dict,
    d4s_data: dict,
) -> Tensor:
    """Compute atomic potential V_A for D4S ζ path (temporary alternative), Eq. 174.

    Implements chain rule: V_A = (∂E/∂ζ_{A,r})·(∂ζ_{A,r}/∂q_A) summed over references r, using:
    - E (two‑body only): Eq. 161 with BJ damping (Eq. 170); ATM term independent of ζ (unity per Sec. 1.17), so no Fock.
    - C6: Eq. 162 via 23‑pt trapezoidal quadrature (embedded).
    - CN smoothing: Eq. 165 with pair β2_AB.

    Inputs
    - ref: {'refalpha':(Z,R,W),'refcovcn':(Z,R),'refc':(Z,R),'r4r2':(Z,)} tensors
    - d4s_data: {'refq':(Z,R),'gam':(Z,), 'zeff':(Z,), 'ga':float, 'gc':float}

    Returns
    - V_A: (nat,) tensor to be mapped onto AO via 0.5(V_A+V_B)S.
    """
    device = positions.device
    dtype = positions.dtype
    nat = numbers.shape[0]
    Z = numbers.long()
    # references
    refalpha = ref["refalpha"].to(device=device, dtype=dtype)  # (Z,R,W)
    refcovcn = ref["refcovcn"].to(device=device, dtype=dtype)  # (Z,R)
    refc = ref["refc"].to(device=device)  # (Z,R)
    r4r2 = ref["r4r2"].to(device=device, dtype=dtype)  # (Z,)
    # D4S data
    refq = d4s_data["refq"].to(device=device, dtype=dtype)  # (Z,R)
    gam_tbl = d4s_data["gam"].to(device=device, dtype=dtype)  # (Z,)
    zeff_tbl = d4s_data["zeff"].to(device=device, dtype=dtype)  # (Z,)
    ga = float(d4s_data["ga"])
    gc = float(d4s_data["gc"])
    a0 = refalpha[Z]  # (nat,R,W)
    cn_ref = refcovcn[Z]  # (nat,R)
    Ns = refc[Z]  # (nat,R)
    rA = r4r2[Z]  # (nat,)
    rB = rA
    # Pairwise CN smoothing weights (Eq. 165)
    maxr = a0.shape[1]
    beta2 = beta2.to(device=device, dtype=dtype)
    beta_ij = beta2[Z.unsqueeze(1), Z.unsqueeze(0)]  # (nat,nat)
    dcn = (cn.unsqueeze(-1) - cn_ref)  # (nat,R)
    dcn2_i = dcn * dcn
    dcn2_j = dcn2_i
    exp_i = torch.exp(-beta_ij.unsqueeze(-1) * dcn2_i.unsqueeze(1))  # (nat,nat,R)
    exp_j = torch.exp(-beta_ij.unsqueeze(-1) * dcn2_j.unsqueeze(0))  # (nat,nat,R)
    Ns_i = Ns.to(dtype=dtype).unsqueeze(1).expand(nat, nat, maxr)
    Ns_j = Ns.to(dtype=dtype).unsqueeze(0).expand(nat, nat, maxr)
    eps = torch.finfo(dtype).eps
    def geom_sum(tmp: Tensor, Ns: Tensor) -> Tensor:
        num = tmp * (1 - torch.pow(tmp, Ns))
        den = torch.clamp(1 - tmp, min=eps)
        return num / den
    w_i = geom_sum(exp_i, Ns_i)
    w_j = geom_sum(exp_j, Ns_j)
    # D4S ζ per reference (BaseModel._zeta): zeta(qmod) with qmod = q + ZEFF
    zeff = zeff_tbl[Z]
    gamZ = gam_tbl[Z] * gc
    qmod = q + zeff
    qref_A = refq[Z] + zeff.unsqueeze(-1)  # (nat,R)
    scale = torch.exp(gamZ.unsqueeze(-1) * (1.0 - qref_A / (qmod.unsqueeze(-1) - torch.tensor(eps, dtype=dtype, device=device))))
    zeta_ref = torch.where(qmod.unsqueeze(-1) > 0.0, torch.exp(torch.tensor(ga, dtype=dtype, device=device) * (1.0 - scale)), torch.exp(torch.tensor(ga, dtype=dtype, device=device)))  # (nat,R)
    # Derivative dζ/dq for D4S (BaseModel._dzeta)
    dzeta_ref = torch.where(
        qmod.unsqueeze(-1) > 0.0,
        -torch.tensor(ga, dtype=dtype, device=device) * gamZ.unsqueeze(-1) * scale * zeta_ref * (qref_A / (qmod.unsqueeze(-1) ** 2)),
        torch.zeros_like(zeta_ref),
    )  # (nat,R)
    # α^0 per pair/ref/frequency
    # a0_i: (nat,nat,R,W) where a0_i[ijrw] = a0[i,r,w]
    a0_i = a0.unsqueeze(1).expand(nat, nat, maxr, a0.shape[-1])
    a0_j = a0.unsqueeze(0).expand(nat, nat, maxr, a0.shape[-1])
    # Weighted ζ inside reference sum
    wz_i = w_i * zeta_ref.unsqueeze(1)  # (nat,nat,R)
    wz_j = w_j * zeta_ref.unsqueeze(0)  # (nat,nat,R)
    # α per pair/frequency: α_i = Σ_r wz_i·a0_i(r,ω)
    alpha_i = torch.einsum("ijr,ijrw->ijw", wz_i, a0_i)  # (nat,nat,W)
    alpha_j = torch.einsum("ijr,ijrw->ijw", wz_j, a0_j)
    # α^0 sums for derivatives per reference: a0w_i = w_i·a0_i; a0w_j = w_j·a0_j
    a0w_i = w_i.unsqueeze(-1) * a0_i  # (nat,nat,R,W)
    a0w_j = w_j.unsqueeze(-1) * a0_j
    # C6 and R0 for damping (Eq. 170)
    C6_mat = _trapzd_noref(alpha_i, alpha_j)
    C8_mat = C6_mat * (3.0 * rA.unsqueeze(1) * rB.unsqueeze(0))
    # Pair distances and BJ damping
    Rij = positions.unsqueeze(0) - positions.unsqueeze(1)
    R = torch.linalg.norm(Rij, dim=-1) + torch.eye(nat, device=device, dtype=dtype) * eps
    R0 = torch.sqrt(torch.clamp(C8_mat / torch.clamp(C6_mat, min=eps), min=0.0))
    f6 = R**6 / (R**6 + (method.a1 * R0) ** 6)
    f8 = R**8 / (R**8 + (method.a1 * R0) ** 8)
    W6 = method.s6 * f6 / (R**6)
    W8 = method.s8 * f8 / (R**8)
    # Derivative ∂C6/∂ζ_{A,r}: i-side and j-side (Eq. 162 linearity)
    # t_i_r = ∫ (a0w_i[ijr,w]) * alpha_j[ij,w] dω
    thopi = 3.0 / 3.141592653589793238462643383279502884197
    wgrid = torch.tensor([
        2.49995e-02, 4.99995e-02, 7.5e-02, 1e-01, 1e-01, 1e-01, 1e-01, 1e-01, 1e-01, 1e-01, 1e-01,
        1.5e-01, 2.0e-01, 2.0e-01, 2.0e-01, 2.0e-01, 3.5e-01, 5.0e-01, 7.5e-01, 1.0e+00, 1.75e+00, 2.5e+00, 1.25e+00
    ], device=device, dtype=dtype)
    t_i_r = thopi * torch.einsum("w,ijrw,ijw->ijr", wgrid, a0w_i, alpha_j)
    t_j_r = thopi * torch.einsum("w,ijw,ijrw->ijr", wgrid, alpha_i, a0w_j)
    # Assemble ∂E/∂ζ_{A,r} (two‑body only)
    rA_col = rA.unsqueeze(1)
    rB_row = rB.unsqueeze(0)
    F8fac = 3.0 * rA_col * rB_row
    # i‑side contributions for A at row A
    dE_i = -(W6.unsqueeze(-1) * t_i_r + W8.unsqueeze(-1) * (F8fac.unsqueeze(-1) * t_i_r))  # (nat,nat,R)
    # j‑side contributions for A at col A
    dE_j = -(W6.unsqueeze(-1) * t_j_r + W8.unsqueeze(-1) * (F8fac.T.unsqueeze(-1) * t_j_r))
    # Sum over partners
    dE_dzeta_ref = dE_i.sum(dim=1) + dE_j.sum(dim=0)  # (nat,R)
    # Multiply by dζ/dq per reference and sum over refs -> V_A
    V_A = torch.einsum("nr,nr->n", dE_dzeta_ref, dzeta_ref)
    return V_A
