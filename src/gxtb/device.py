from __future__ import annotations

import os
from typing import Optional

import torch

__all__ = ["get_device", "set_default_dtype"]


def get_device(prefer: Optional[str] = None) -> torch.device:
    """
    Choose a torch device with a simple preference policy.
    prefer: one of {"cuda", "mps", "cpu"} or None to auto.
    """
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def set_default_dtype(dtype: torch.dtype = torch.float64) -> None:
    torch.set_default_dtype(dtype)

