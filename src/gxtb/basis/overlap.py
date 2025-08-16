from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

__all__ = [
    "rotation_matrices",
    "scale_diatomic_overlap",
    "rotation_derivatives",
]


def rotation_matrices(r_ab: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Build diatomic-frame rotation blocks O_ss, O_pp, O_dd, O_ff (doc/theory §1.3):
    - Eqs. (35) s
    - Eq.  (36) p (ordering: (p_y, p_z, p_x))
    - Eq.  (37) d (ordering: (d_xy, d_yz, d_z^2, d_xz, d_{x^2-y^2}))
    - Eq.  (38) f (unit)

    r_ab: (3,) vector from A to B
    returns dict with tensors on same device/dtype.
    """
    dtype = r_ab.dtype
    device = r_ab.device
    x, y, z = r_ab
    r = torch.linalg.norm(r_ab)
    # define angles robustly with eps to avoid singularities; continuous extension at r->0
    ct = z / torch.clamp(r, min=torch.finfo(dtype).eps)
    st = torch.sqrt(torch.clamp(1.0 - ct * ct, min=0.0))
    denom = torch.sqrt(torch.clamp(x * x + y * y, min=torch.finfo(dtype).eps))
    cp = x / denom
    sp = y / denom

    O_ss = torch.tensor([[1.0]], dtype=dtype, device=device)
    O_pp = torch.stack(
        [
            torch.stack([cp, torch.tensor(0.0, dtype=dtype, device=device), -sp]),
            torch.stack([st * sp, ct, st * cp]),
            torch.stack([ct * sp, -st, ct * cp]),
        ]
    )

    # d block per Eq. (37)
    O_dd = torch.empty((5, 5), dtype=dtype, device=device)
    sqrt3 = math.sqrt(3.0)
    # row 1
    O_dd[0, 0] = ct * (cp * cp - sp * sp)
    O_dd[0, 1] = st * (cp * cp - sp * sp)
    O_dd[0, 2] = sqrt3 * st * st * sp * cp
    O_dd[0, 3] = 2.0 * st * ct * sp * cp
    O_dd[0, 4] = (1.0 + ct * ct) * sp * cp
    # row 2
    O_dd[1, 0] = -st * cp
    O_dd[1, 1] = ct * cp
    O_dd[1, 2] = sqrt3 * st * ct * sp
    O_dd[1, 3] = (ct * ct - st * st) * sp
    O_dd[1, 4] = -st * ct * sp
    # row 3
    O_dd[2, 0] = 0.0
    O_dd[2, 1] = 0.0
    O_dd[2, 2] = 0.5 * (3.0 * ct * ct - 1.0)
    O_dd[2, 3] = -sqrt3 * st * ct
    O_dd[2, 4] = 0.5 * sqrt3 * st * st
    # row 4
    O_dd[3, 0] = sqrt3 * ct * st * (cp * cp - sp * sp)
    O_dd[3, 1] = (ct * ct - st * st) * (cp * cp - sp * sp)
    O_dd[3, 2] = -2.0 * ct * sp * cp
    O_dd[3, 3] = -ct * st * (cp * cp - sp * sp)
    O_dd[3, 4] = 2.0 * st * sp * cp
    # row 5
    O_dd[4, 0] = 2.0 * sqrt3 * ct * st * sp * cp
    O_dd[4, 1] = 2.0 * (ct * ct - st * st) * sp * cp
    O_dd[4, 2] = ct * (cp * cp - sp * sp)
    O_dd[4, 3] = -2.0 * ct * st * sp * cp
    O_dd[4, 4] = -st * (cp * cp - sp * sp)

    O_ff = torch.eye(7, dtype=dtype, device=device)
    return {"s": O_ss, "p": O_pp, "d": O_dd, "f": O_ff}


def rotation_derivatives(r_ab: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Derivatives of diatomic-frame rotation blocks with respect to Cartesian
    components of r_ab = R_B - R_A.

    Returns dict mapping l in {"s","p","d","f"} to dO/dR tensor with shape
    (3, n_l, n_l) where the first axis corresponds to (∂/∂x, ∂/∂y, ∂/∂z).

    Equations reference doc/theory/8_diatomic_frame_scaled_overlap.md:
    - O_pp: Eq. (36)
    - O_dd: Eq. (37)
    Chain rule applied via ct=z/r, st=sqrt(1-ct^2), cp=x/rho, sp=y/rho where rho=sqrt(x^2+y^2).
    """
    dtype = r_ab.dtype
    device = r_ab.device
    x, y, z = r_ab
    eps = torch.finfo(dtype).eps
    r = torch.linalg.norm(r_ab)
    if r.item() == 0.0:
        # Undefined angles; derivatives treated as zero by continuity
        zero = lambda n: torch.zeros((3, n, n), dtype=dtype, device=device)
        return {"s": zero(1), "p": zero(3), "d": zero(5), "f": zero(7)}
    rho = torch.sqrt(torch.clamp(x * x + y * y, min=eps))
    ct = z / r
    st = torch.sqrt(torch.clamp(1.0 - ct * ct, min=0.0))
    cp = x / rho
    sp = y / rho
    # Derivatives of ct, st, cp, sp wrt (x,y,z)
    r3 = r * r * r
    dct = torch.stack([
        -z * x / r3,
        -z * y / r3,
        (x * x + y * y) / r3,
    ])  # (3,)
    # dst = d/dR sqrt(1-ct^2) = (-ct/st) dct, guard st
    inv_st = 1.0 / torch.clamp(st, min=eps)
    dst = (-ct * inv_st) * dct
    rho3 = rho * rho * rho
    dcp = torch.stack([
        (y * y) / rho3,
        -x * y / rho3,
        torch.tensor(0.0, dtype=dtype, device=device),
    ])
    dsp = torch.stack([
        -x * y / rho3,
        (x * x) / rho3,
        torch.tensor(0.0, dtype=dtype, device=device),
    ])

    # s block derivative zeros
    dO_ss = torch.zeros((3, 1, 1), dtype=dtype, device=device)

    # p block Eq. (36): rows expressed via cp,sp,st,ct
    # O_pp = [[cp, 0, -sp], [st*sp, ct, st*cp], [ct*sp, -st, ct*cp]]
    O_pp = torch.empty((3, 3), dtype=dtype, device=device)
    O_pp[0, 0] = cp; O_pp[0, 1] = 0.0; O_pp[0, 2] = -sp
    O_pp[1, 0] = st * sp; O_pp[1, 1] = ct; O_pp[1, 2] = st * cp
    O_pp[2, 0] = ct * sp; O_pp[2, 1] = -st; O_pp[2, 2] = ct * cp
    dO_pp = torch.zeros((3, 3, 3), dtype=dtype, device=device)
    for k in range(3):  # derivative component (x,y,z)
        dcp_k = dcp[k]; dsp_k = dsp[k]; dct_k = dct[k]; dst_k = dst[k]
        # row 1
        dO_pp[k, 0, 0] = dcp_k
        dO_pp[k, 0, 1] = 0.0
        dO_pp[k, 0, 2] = -dsp_k
        # row 2
        dO_pp[k, 1, 0] = dst_k * sp + st * dsp_k
        dO_pp[k, 1, 1] = dct_k
        dO_pp[k, 1, 2] = dst_k * cp + st * dcp_k
        # row 3
        dO_pp[k, 2, 0] = dct_k * sp + ct * dsp_k
        dO_pp[k, 2, 1] = -dst_k
        dO_pp[k, 2, 2] = dct_k * cp + ct * dcp_k

    # d block Eq. (37): construct derivative via chain rule
    import math
    sqrt3 = math.sqrt(3.0)
    O_dd = torch.empty((5, 5), dtype=dtype, device=device)
    # row 1
    O_dd[0, 0] = ct * (cp * cp - sp * sp)
    O_dd[0, 1] = st * (cp * cp - sp * sp)
    O_dd[0, 2] = sqrt3 * st * st * sp * cp
    O_dd[0, 3] = 2.0 * st * ct * sp * cp
    O_dd[0, 4] = (1.0 + ct * ct) * sp * cp
    # row 2
    O_dd[1, 0] = -st * cp
    O_dd[1, 1] = ct * cp
    O_dd[1, 2] = sqrt3 * st * ct * sp
    O_dd[1, 3] = (ct * ct - st * st) * sp
    O_dd[1, 4] = -st * ct * sp
    # row 3
    O_dd[2, 0] = 0.0
    O_dd[2, 1] = 0.0
    O_dd[2, 2] = 0.5 * (3.0 * ct * ct - 1.0)
    O_dd[2, 3] = -sqrt3 * st * ct
    O_dd[2, 4] = 0.5 * sqrt3 * st * st
    # row 4
    O_dd[3, 0] = sqrt3 * ct * st * (cp * cp - sp * sp)
    O_dd[3, 1] = (ct * ct - st * st) * (cp * cp - sp * sp)
    O_dd[3, 2] = -2.0 * ct * sp * cp
    O_dd[3, 3] = -ct * st * (cp * cp - sp * sp)
    O_dd[3, 4] = 2.0 * st * sp * cp
    # row 5
    O_dd[4, 0] = 2.0 * sqrt3 * ct * st * sp * cp
    O_dd[4, 1] = 2.0 * (ct * ct - st * st) * sp * cp
    O_dd[4, 2] = ct * (cp * cp - sp * sp)
    O_dd[4, 3] = -2.0 * ct * st * sp * cp
    O_dd[4, 4] = -st * (cp * cp - sp * sp)

    dO_dd = torch.zeros((3, 5, 5), dtype=dtype, device=device)
    for k in range(3):
        dcp_k = dcp[k]; dsp_k = dsp[k]; dct_k = dct[k]; dst_k = dst[k]
        cpcp_minus_spsp = cp * cp - sp * sp
        dcpcp_minus_spsp = 2.0 * cp * dcp_k - 2.0 * sp * dsp_k
        # row 1
        dO_dd[k, 0, 0] = dct_k * cpcp_minus_spsp + ct * dcpcp_minus_spsp
        dO_dd[k, 0, 1] = dst_k * cpcp_minus_spsp + st * dcpcp_minus_spsp
        dO_dd[k, 0, 2] = sqrt3 * (2.0 * st * dst_k * sp * cp + st * st * (dsp_k * cp + sp * dcp_k))
        dO_dd[k, 0, 3] = 2.0 * (dst_k * ct * sp * cp + st * dct_k * sp * cp + st * ct * (dsp_k * cp + sp * dcp_k))
        dO_dd[k, 0, 4] = 2.0 * ct * dct_k * sp * cp + (1.0 + ct * ct) * (dsp_k * cp + sp * dcp_k)
        # row 2
        dO_dd[k, 1, 0] = -dst_k * cp - st * dcp_k
        dO_dd[k, 1, 1] = dct_k * cp + ct * dcp_k
        dO_dd[k, 1, 2] = sqrt3 * (dst_k * ct * sp + st * dct_k * sp + st * ct * dsp_k)
        dO_dd[k, 1, 3] = 2.0 * ct * dct_k * sp - 2.0 * st * dst_k * sp + (ct * ct - st * st) * dsp_k
        dO_dd[k, 1, 4] = -(dst_k * ct * sp + st * dct_k * sp + st * ct * dsp_k)
        # row 3
        dO_dd[k, 2, 0] = 0.0
        dO_dd[k, 2, 1] = 0.0
        dO_dd[k, 2, 2] = 3.0 * ct * dct_k
        dO_dd[k, 2, 3] = -sqrt3 * (dst_k * ct + st * dct_k)
        dO_dd[k, 2, 4] = 0.5 * sqrt3 * 2.0 * st * dst_k
        # row 4
        dO_dd[k, 3, 0] = sqrt3 * (dct_k * st * cpcp_minus_spsp + ct * dst_k * cpcp_minus_spsp + ct * st * dcpcp_minus_spsp)
        dO_dd[k, 3, 1] = (2.0 * ct * dct_k - 2.0 * st * dst_k) * cpcp_minus_spsp + (ct * ct - st * st) * dcpcp_minus_spsp
        dO_dd[k, 3, 2] = -2.0 * (dct_k * sp * cp + ct * (dsp_k * cp + sp * dcp_k))
        dO_dd[k, 3, 3] = -(dct_k * st + ct * dst_k) * cpcp_minus_spsp - ct * st * dcpcp_minus_spsp
        dO_dd[k, 3, 4] = 2.0 * (dst_k * sp * cp + st * (dsp_k * cp + sp * dcp_k))
        # row 5
        dO_dd[k, 4, 0] = 2.0 * sqrt3 * (dct_k * st * sp * cp + ct * dst_k * sp * cp + ct * st * (dsp_k * cp + sp * dcp_k))
        dO_dd[k, 4, 1] = 2.0 * ((2.0 * ct * dct_k - 2.0 * st * dst_k) * sp * cp + (ct * ct - st * st) * (dsp_k * cp + sp * dcp_k))
        dO_dd[k, 4, 2] = dct_k * cpcp_minus_spsp + ct * dcpcp_minus_spsp
        dO_dd[k, 4, 3] = -2.0 * (dct_k * st * sp * cp + ct * dst_k * sp * cp + ct * st * (dsp_k * cp + sp * dcp_k))
        dO_dd[k, 4, 4] = -(dst_k * cpcp_minus_spsp + st * dcpcp_minus_spsp)

    dO_ff = torch.zeros((3, 7, 7), dtype=dtype, device=device)
    return {"s": dO_ss, "p": dO_pp, "d": dO_dd, "f": dO_ff}


def _bond_type_indices(l: str) -> Dict[str, Tuple[slice, ...]]:
    """
    Return indices in diatomic frame to classify sigma/pi/delta channels
    for given angular momentum block (ordering as in rotation_matrices).
    """
    if l == "s":
        return {"sigma": (slice(0, 1),), "pi": (), "delta": ()}
    if l == "p":
        # (p_y, p_z, p_x); z is sigma, x/y are pi
        return {"sigma": (slice(1, 2),), "pi": (slice(0, 1), slice(2, 3)), "delta": ()}
    if l == "d":
        # (d_xy, d_yz, d_z^2, d_xz, d_x2-y2)
        return {
            "sigma": (slice(2, 3),),
            "pi": (slice(1, 2), slice(3, 4)),
            "delta": (slice(0, 1), slice(4, 5)),
        }
    if l == "f":
        # no specific scaling
        return {"sigma": (), "pi": (), "delta": ()}
    raise ValueError(f"Unknown l={l}")


def scale_diatomic_overlap(
    S_block: torch.Tensor,
    r_ab: torch.Tensor,
    lA: str,
    lB: str,
    kA: Dict[str, float],
    kB: Dict[str, float],
) -> torch.Tensor:
    """
    Apply diatomic frame scaling (doc/theory §1.3, Eqs. 31–32) to an AO block.

    - S_block: (nA, nB) overlap sub-block for shells lA (on A) and lB (on B)
    - r_ab: (3,) vector R_B - R_A
    - kA/kB: element-specific scaling params for interaction types L in {'sigma','pi','delta'}

    Returns scaled block in the lab frame.
    """
    O = rotation_matrices(r_ab)
    OA = O[lA]
    OB = O[lB]
    S_diat = OA @ S_block @ OB.T

    # Build scaling matrix K in diatomic frame
    nA, nB = S_diat.shape
    K = torch.ones_like(S_diat)
    typesA = _bond_type_indices(lA)
    typesB = _bond_type_indices(lB)
    # For each AO row/col type, apply factor 2 / (1/kA^L + 1/kB^L)
    for L in ("sigma", "pi", "delta"):
        if L not in kA or L not in kB or len(typesA[L]) == 0 or len(typesB[L]) == 0:
            continue
        kA_L = torch.tensor(kA[L], dtype=S_diat.dtype, device=S_diat.device)
        kB_L = torch.tensor(kB[L], dtype=S_diat.dtype, device=S_diat.device)
        if kA_L.item() == 0.0 or kB_L.item() == 0.0:
            continue
        scale = 2.0 / (1.0 / kA_L + 1.0 / kB_L)
        for sA in typesA[L]:
            for sB in typesB[L]:
                K[sA, sB] = scale

    S_scaled_diat = S_diat * K
    S_scaled = OA.T @ S_scaled_diat @ OB
    return S_scaled
