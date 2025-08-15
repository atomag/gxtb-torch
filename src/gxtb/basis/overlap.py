from __future__ import annotations

import math
from typing import Dict, Tuple

import torch

__all__ = [
    "rotation_matrices",
    "scale_diatomic_overlap",
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
    if r.item() == 0.0:
        # undefined; return identities
        eye = lambda n: torch.eye(n, dtype=dtype, device=device)
        return {"s": eye(1), "p": eye(3), "d": eye(5), "f": eye(7)}

    ct = z / r
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
