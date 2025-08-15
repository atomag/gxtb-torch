from __future__ import annotations
"""Shared tight-binding utility functions.

Polynomial evaluation and mean combinations used in Extended Hückel assembly.

Equation references:
    - General polynomial evaluation used for:
            * CN onsite shift Π^{CN}_l(CN_A) (when higher than linear) extending linear Eq. 65.
            * Distance shell polynomials π_{l}(R) (Eq. 67) and their product Π_{l_A l_B}(R) (Eq. 66).
        We evaluate p(x) = Σ_{k=0}^{n-1} c_k x^k in Horner form (no explicit equation number; standard expansion feeding Eqs. 66–67).
    - Harmonic mean is applied when combining two positive shell distance polynomials (design choice noted near Eq. 66) and for σ/π/δ diatomic scaling factors (Eqs. 31–32 context) to ensure symmetry and penalize large disparities.
    - Geometric mean √(k_{l_A} k_{l_B}) replaces the arithmetic mean (k_{l_A}+k_{l_B})/2 of Eq. 64 for Wolfsberg factors (documented deviation in `eht.py`).
"""
import torch
Tensor = torch.Tensor

def poly_eval(coeff: Tensor, x: Tensor) -> Tensor:
    """Evaluate polynomial p(x)=Σ c_k x^k (Horner form) used in Eqs. 66–67 (distance) and CN extension of Eq. 65."""
    res = torch.zeros_like(x)
    for c in reversed(coeff.unbind(-1)):
        res = res * x + c
    return res

def poly_eval_derivative(coeff: Tensor, x: Tensor) -> Tensor:
    """Evaluate derivative p'(x) for p(x)=Σ c_k x^k.

    Uses reverse Horner accumulation on derivative coefficients:
        p'(x) = Σ_{k=1}^{n-1} k c_k x^{k-1}
    """
    if coeff.numel() <= 1:
        return torch.zeros_like(x)
    res = torch.zeros_like(x)
    n = coeff.shape[-1]
    for k in range(n-1, 0, -1):
        res = res * x + float(k) * coeff[..., k]
    return res

def harmonic_mean(a: Tensor, b: Tensor) -> Tensor:
    """Harmonic mean H=2/(1/a+1/b) applied when both operands >0 (distance poly & diatomic scaling, cf. Eqs. 31–32, 66–67)."""
    return 2.0 / (1.0 / torch.clamp(a, min=1e-12) + 1.0 / torch.clamp(b, min=1e-12))

def geometric_mean(a: Tensor, b: Tensor) -> Tensor:
    """Geometric mean √(ab) used for modified Wolfsberg factor (deviation from arithmetic form in Eq. 64)."""
    return torch.sqrt(torch.clamp(a, min=0.0) * torch.clamp(b, min=0.0))
