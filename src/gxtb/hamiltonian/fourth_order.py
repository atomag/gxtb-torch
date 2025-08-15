from __future__ import annotations

"""Fourth-order onsite atom-resolved tight-binding (doc/theory/19_fourth_order_tb.md).

Equations implemented:
 - Energy: E^(4) = 1/24 Σ_A q_A^4 Γ^(4)  (Eq. 140b)
 - Fock:  F^(4)_{μν} = (1/6) Σ_A S_{νμ}|_{ν∈A} q_A^3 Γ^(4)  (Eq. 143)

No placeholders: Γ^(4) must be provided explicitly. If absent, raise.
"""

from dataclasses import dataclass
import torch

Tensor = torch.Tensor

__doc_refs__ = {"file": "doc/theory/19_fourth_order_tb.md", "eqs": ["140b", "143"]}

__all__ = ["FourthOrderParams", "fourth_order_energy", "add_fourth_order_fock", "__doc_refs__"]


@dataclass(frozen=True)
class FourthOrderParams:
    gamma4: float  # Γ^(4) global parameter (Eq. 140b)


def fourth_order_energy(q: Tensor, params: FourthOrderParams) -> Tensor:
    """Compute E^(4) per Eq. 140b from atomic charges q.

    q: (nat,) atomic Δq (relative to reference)
    returns scalar tensor
    """
    g4 = torch.as_tensor(params.gamma4, dtype=q.dtype, device=q.device)
    return (g4 * (q ** 4).sum()) / 24.0


def add_fourth_order_fock(H: Tensor, S: Tensor, ao_atoms: Tensor, q: Tensor, params: FourthOrderParams) -> None:
    """Add F^(4) contribution to Hamiltonian per Eq. 143.

    H: (nao,nao) in-place updated
    S: (nao,nao) overlap
    ao_atoms: (nao,) AO→atom indices
    q: (nat,) atomic Δq
    """
    g4 = torch.as_tensor(params.gamma4, dtype=H.dtype, device=H.device)
    v = (q[ao_atoms] ** 3) * (g4 / 6.0)  # factor outside; applied symmetrically via S
    # Form rank-1 scaling over S entries per AO index
    H.add_(S * (v.unsqueeze(1) + v.unsqueeze(0)) * 0.5)
