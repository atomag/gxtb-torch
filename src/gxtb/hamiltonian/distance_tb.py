from __future__ import annotations
"""Distance & electronegativity modulation Π_{l_A l_B}(R_{AB}) and X_{AB}.

Equations:
    - Π_{l_A l_B}(R_{AB}) = π_{l_A}(R_{AB}) π_{l_B}(R_{AB}) (Eq. 66) with π_l(R) = 1 + k^{shp}_A k^{shp,l}_{l} (R / R^{cov}_{AB}) (Eq. 67).
    - Here each element supplies polynomial coefficients (allowing higher than linear terms); we evaluate p_A(R), p_B(R) separately.
    - Combination rule: harmonic mean H(p_A, p_B) when both > 0 else arithmetic mean 0.5(p_A+p_B) (design extension for stability; original Eq. 66 is simple product of linear forms).
    - Optional electronegativity damping X_{AB} = max(0, 1 - k_en (ΔEN)^2) (design extension; not in Eq. 64) to attenuate hetero pairs with large ΔEN.

Notes:
    - Covalent radius normalization already baked into supplied polynomial variables (R scaled externally during parameter preparation) or embedded in coefficients; if explicit normalization needed, adapt before poly_eval.
    - Gradient dΠ/dR would require applying Eq. 70–71 chain rules; not yet implemented.
"""
import torch
from .utils_tb import poly_eval, poly_eval_derivative, harmonic_mean

Tensor = torch.Tensor
__all__ = ["distance_factor", "en_penalty"]

def distance_factor(lA: str, lB: str, ZA: int, ZB: int, R: Tensor, eht: dict) -> Tensor:
    """Evaluate Π_{l_A l_B}(R) (Eq. 66) using element polynomials (extended beyond linear Eq. 67)."""
    key = ''.join(sorted([lA, lB]))
    rkey = f"pi_r_{key}"
    if rkey not in eht:
        return torch.tensor(1.0, dtype=R.dtype, device=R.device)
    coeffA = eht[rkey][ZA]
    coeffB = eht[rkey][ZB]
    # Normalize distance by sum of covalent radii if provided (Eq.67)
    if 'r_cov' in eht:
        r_cov = eht['r_cov']  # tensor per element
        R_cov = float(r_cov[ZA].item() + r_cov[ZB].item())
        Rn = R / R_cov
    else:
        Rn = R
    # π_l(R) polynomial evaluation (linear or higher) on normalized distance
    pA = poly_eval(coeffA, Rn)
    pB = poly_eval(coeffB, Rn)
    # Theory Eq.66: Π_{lA lB}(R) = π_{lA}(R) * π_{lB}(R)
    return pA * pB

def distance_factor_with_grad(lA: str, lB: str, ZA: int, ZB: int, R: Tensor, eht: dict) -> tuple[Tensor, Tensor]:
    """Return Π_{l_A l_B}(R) and its radial derivative dΠ/dR (Eqs. 66–71).

    Assumes normalized distance Rn = R / R_cov, where R_cov = R_cov_A + R_cov_B if provided.
    dΠ/dR = (pA'(Rn) pB(Rn) + pA(Rn) pB'(Rn)) / R_cov.
    """
    key = ''.join(sorted([lA, lB]))
    rkey = f"pi_r_{key}"
    if rkey not in eht:
        return torch.tensor(1.0, dtype=R.dtype, device=R.device), torch.tensor(0.0, dtype=R.dtype, device=R.device)
    coeffA = eht[rkey][ZA]
    coeffB = eht[rkey][ZB]
    if 'r_cov' in eht:
        r_cov = eht['r_cov']
        R_cov = float(r_cov[ZA].item() + r_cov[ZB].item())
        Rn = R / R_cov
    else:
        R_cov = 1.0
        Rn = R
    pA = poly_eval(coeffA, Rn)
    pB = poly_eval(coeffB, Rn)
    dpA = poly_eval_derivative(coeffA, Rn)
    dpB = poly_eval_derivative(coeffB, Rn)
    Pi = pA * pB
    dPi_dR = (dpA * pB + pA * dpB) / float(R_cov)
    return Pi, dPi_dR

def en_penalty(ZA: int, ZB: int, en: Tensor | None, k_en: float | None) -> float:
    """Electronegativity damping X_{AB} = max(0, 1 - k_en (ΔEN)^2) (design extension; optional)."""
    if en is None or k_en is None:
        return 1.0
    dEN = abs(float(en[ZA].item() - en[ZB].item()))
    return max(0.0, 1.0 - k_en * dEN * dEN)
