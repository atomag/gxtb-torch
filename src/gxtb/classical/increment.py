from __future__ import annotations

import torch

__all__ = ["energy_increment"]


def energy_increment(numbers: torch.Tensor, delta_e_incr: torch.Tensor) -> torch.Tensor:
    """
    Atomic increment energy per doc/theory/10_atomic_energy_increment.md Eq. (50).
    E_incr = sum_A ΔE_A^incr

    numbers: (..., nat) integer Z
    delta_e_incr: (Zmax+1,) tensor mapping Z -> ΔE_incr (energy units consistent across code)
    returns scalar tensor energy (...,)
    """
    vals = delta_e_incr[numbers.long()]
    return vals.sum(-1)

