from __future__ import annotations
"""Electronegativity Equalization / Bond Capacity (EEQ / EEQBC) baseline charges.

Implements a generic electronegativity equalization model:
    Minimize  E(q) = 0.5 q^T J q + χ^T q   subject to  1^T q = Q_total

Stationary conditions give linear system with Lagrange multiplier λ:
    [ J   1 ] [ q ] = [ -χ ]
    [ 1^T 0 ] [ λ ]   [  Q ]

Solution yields atomic charges q (Σ q = Q_total). These become q^{EEQBC}_A
feeding the semi-classical repulsion (Eq. 53) and can seed SCF.

Model components:
  - χ_A (electronegativity) from selected parameter column (chi_index)
  - η_A (hardness) on diagonal: J_AA = η_A (eta_index)
  - Shielded Coulomb off-diagonal J_AB = 1 / sqrt(R_AB^2 + (r_A + r_B)^2)
    with atomic radius r_A (radius_index). This is a common, smooth attenuation.
  - Optional global scaling factors allow later calibration (not yet parameterized).

Notes:
  - EEQBC specific refinements (bond capacity factors, environment scaling) can
    be integrated later; interface kept extensible.
  - All indices default to (0,1,2) but can be overridden if schema differs.
  - Returns zeros if nat < 2 to avoid singular radius-based kernel degeneracy.
"""
import torch
from typing import Optional, Dict
from ..params.loader import EEQParameters

__all__ = ["compute_eeq_charges", "compute_eeq_charge_derivative"]


def _gather_params(numbers: torch.Tensor, eeq: EEQParameters) -> torch.Tensor:
  rows = []
  for z in numbers.tolist():
    elem = eeq.elements.get(int(z))
    if elem is None:
      raise ValueError(f"Missing EEQ parameters for Z={z}")
    rows.append(elem.values)
  return torch.stack(rows, dim=0)  # (nat, ncol)


def compute_eeq_charges(
  numbers: torch.Tensor,
  positions: torch.Tensor,
  eeq: EEQParameters,
  total_charge: float = 0.0,
  *,
  mapping: Optional[Dict[str, int]] = None,
  device: Optional[torch.device] = None,
  dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
  """Compute EEQ/EEQBC baseline charges.

  Parameters
  ----------
  numbers : (nat,) atomic numbers
  positions : (nat,3) Cartesian coordinates (Å)
  eeq : EEQParameters (raw per-element 10-column data)
  total_charge : net molecular charge Q
  mapping : optional dict specifying column indices {'chi':i,'eta':j,'radius':k}
  device, dtype : tensor placement / precision

  Returns
  -------
  q : (nat,) charges summing to total_charge
  """
  if device is None:
    device = positions.device
  numbers = numbers.to(device=device)
  positions = positions.to(device=device, dtype=dtype)
  nat = numbers.shape[0]
  if nat == 0:
    return torch.empty(0, dtype=dtype, device=device)
  if nat == 1:
    return torch.tensor([total_charge], dtype=dtype, device=device)

  mp = mapping or {"chi": 0, "eta": 1, "radius": 2}
  params = _gather_params(numbers, eeq).to(device=device, dtype=dtype)
  chi = params[:, mp["chi"]].clone()  # (nat,)
  eta = torch.clamp(params[:, mp["eta"]], min=1e-6)
  rad = torch.clamp(params[:, mp["radius"]], min=1e-6)

  # Build shielded distance matrix without materializing (nat,nat,3)
  dist = torch.cdist(positions, positions)  # (nat,nat)
  dist2 = dist * dist
  rsum = rad.unsqueeze(0) + rad.unsqueeze(1)
  # Off-diagonal kernel
  J = 1.0 / torch.sqrt(dist2 + rsum * rsum + torch.eye(nat, device=device, dtype=dtype) * 1e6)
  # Diagonal hardness
  J.fill_diagonal_(0.0)
  J = J + torch.diag(eta)

  # Assemble augmented system
  ones = torch.ones(nat, 1, dtype=dtype, device=device)
  top = torch.cat([J, ones], dim=1)
  bottom = torch.cat([ones.transpose(0, 1), torch.zeros(1, 1, dtype=dtype, device=device)], dim=1)
  A = torch.cat([top, bottom], dim=0)  # (nat+1, nat+1)
  b = torch.cat([-chi, torch.tensor([total_charge], dtype=dtype, device=device)])

  try:
    sol = torch.linalg.solve(A, b)
  except RuntimeError:
    # Fallback: add small ridge to diagonal and retry
    ridge = 1e-6 * torch.eye(nat + 1, dtype=dtype, device=device)
    sol = torch.linalg.solve(A + ridge, b)
  q = sol[:-1]
  # Numerical cleanup: enforce exact neutrality within tolerance
  dq = total_charge - q.sum()
  if abs(dq.item()) > 1e-10 and nat > 0:
    q = q + dq / nat
  return q


def compute_eeq_charge_derivative(
  numbers: torch.Tensor,
  positions: torch.Tensor,
  eeq: EEQParameters,
  total_charge: float = 0.0,
  *,
  mapping: Optional[Dict[str, int]] = None,
  device: Optional[torch.device] = None,
  dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
  """Analytic derivative dq^{EEQBC}/dR from the EEQ linear system.

  Theory mapping:
    - EEQ linear system (module header): A(positions) x = b with x = [q; λ].
    - Only the off-diagonal kernel J_AB depends on positions through R_AB.
    - Implicit differentiation: A ∂x = −(∂A) x, with ∂A having only the J block.
    - Returns tensor dq_dpos with shape (nat, nat, 3) where [A,X,k] = ∂q_A/∂R_Xk.

  Notes:
    - Uses the same kernel J_AB = 1/sqrt(R_AB^2 + (r_A + r_B)^2) as compute_eeq_charges.
    - Diagonal J_AA = η_A is position-independent; its derivative is zero.
    - This function recomputes q internally for consistency with the A used in differentiation.
  """
  if device is None:
    device = positions.device
  numbers = numbers.to(device=device)
  positions = positions.to(device=device, dtype=dtype)
  nat = int(numbers.shape[0])
  if nat == 0:
    return torch.empty((0, 0, 3), dtype=dtype, device=device)
  # Build parameters
  mp = mapping or {"chi": 0, "eta": 1, "radius": 2}
  params = _gather_params(numbers, eeq).to(device=device, dtype=dtype)
  chi = params[:, mp["chi"]].clone()  # (nat,)
  eta = torch.clamp(params[:, mp["eta"]], min=1e-6)
  rad = torch.clamp(params[:, mp["radius"]], min=1e-6)
  # Distance and kernel
  R = positions
  dist = torch.cdist(R, R)
  dist2 = dist * dist
  rsum = rad.unsqueeze(0) + rad.unsqueeze(1)
  g = dist2 + rsum * rsum  # (nat,nat)
  inv_s = torch.zeros_like(g)
  mask_off = ~torch.eye(nat, dtype=torch.bool, device=device)
  inv_s[mask_off] = torch.pow(g[mask_off], -0.5)
  J = inv_s
  J.fill_diagonal_(0.0)
  J = J + torch.diag(eta)
  # Augmented system A x = b
  ones = torch.ones(nat, 1, dtype=dtype, device=device)
  top = torch.cat([J, ones], dim=1)
  bottom = torch.cat([ones.transpose(0, 1), torch.zeros(1, 1, dtype=dtype, device=device)], dim=1)
  A = torch.cat([top, bottom], dim=0)  # (nat+1, nat+1)
  b = torch.cat([-chi, torch.tensor([total_charge], dtype=dtype, device=device)])
  # Solve for x = [q; λ]
  I = torch.eye(nat + 1, dtype=dtype, device=device)
  try:
    x = torch.linalg.solve(A, b)
    A_inv = torch.linalg.solve(A, I)
  except RuntimeError:
    ridge = 1e-6 * I
    x = torch.linalg.solve(A + ridge, b)
    A_inv = torch.linalg.solve(A + ridge, I)
  q = x[:-1]
  # Precompute g^(3/2)
  gp32 = torch.zeros_like(g)
  gp32[mask_off] = torch.pow(g[mask_off], 1.5)
  # Pairwise ΔR components
  dq_dpos = torch.zeros((nat, nat, 3), dtype=dtype, device=device)
  for k in range(3):
    dRk = R[:, k].unsqueeze(1) - R[:, k].unsqueeze(0)  # (nat,nat)
    # For a given moved atom X, build y = (∂J/∂R_Xk) q using the sparsity:
    for X in range(nat):
      y = torch.zeros(nat, dtype=dtype, device=device)
      # Row X: dJ[X,B] = - (R_Xk - R_Bk) / g^(3/2) for B != X
      rowX = torch.zeros(nat, dtype=dtype, device=device)
      rowX[mask_off[X]] = - dRk[X, mask_off[X]] / gp32[X, mask_off[X]]
      y[X] = (rowX * q).sum()
      # Column X: dJ[A,X] = + (R_Ak - R_Xk) / g^(3/2) for A != X
      colX = torch.zeros(nat, dtype=dtype, device=device)
      colmask = mask_off[:, X]
      colX[colmask] = dRk[colmask, X] / gp32[colmask, X]
      y[colmask] = y[colmask] + colX[colmask] * q[X]
      rhs = torch.cat([-y, torch.tensor([0.0], dtype=dtype, device=device)])
      dx = A_inv @ rhs
      dq = dx[:-1]
      dq_dpos[:, X, k] = dq
  return dq_dpos
