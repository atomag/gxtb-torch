from __future__ import annotations

import torch

__all__ = ["first_order_energy"]


def first_order_energy(H: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    """First-order (EHT) electronic energy E = Tr{ H^{EHT} P } (Eq. 63) alias of Eq. 18 in §1.9 context.

    Parameters
    ----------
    H : (nao, nao) EHT Hamiltonian (Eq. 64 variant)
    P : (nao, nao) density matrix
    """
    return torch.einsum("ij,ji->", H, P)

