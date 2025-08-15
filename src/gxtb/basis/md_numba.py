"""Numba-accelerated McMurchie–Davidson helpers (CPU only).

Drop-in acceleration for Cartesian overlap sub-block evaluation used by
md_overlap._overlap_cart_block. Identical algebra; no theory changes.

Used only when running on CPU and numba is available.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception as _e:  # pragma: no cover - optional dependency
    njit = None  # type: ignore


if njit is not None:

    @njit(cache=True)
    def _cart_list_np(l: int) -> np.ndarray:
        n = (l + 1) * (l + 2) // 2
        arr = np.empty((n, 3), dtype=np.int64)
        idx = 0
        for lx in range(l, -1, -1):
            for ly in range(l - lx, -1, -1):
                lz = l - lx - ly
                arr[idx, 0] = lx
                arr[idx, 1] = ly
                arr[idx, 2] = lz
                idx += 1
        return arr


    @njit(cache=True)
    def _one_d_recur_np(l1: int, l2: int, PA: float, PB: float, gamma: float, K: float) -> np.ndarray:
        S = np.zeros((l1 + 1, l2 + 1), dtype=np.float64)
        S[0, 0] = K
        inv2g = 1.0 / (2.0 * gamma)
        for j in range(1, l2 + 1):
            S[0, j] = PB * S[0, j - 1] + (j - 1) * inv2g * (S[0, j - 2] if j > 1 else 0.0)
        for i in range(1, l1 + 1):
            S[i, 0] = PA * S[i - 1, 0] + (i - 1) * inv2g * (S[i - 2, 0] if i > 1 else 0.0)
            for j in range(1, l2 + 1):
                t = PA * S[i - 1, j] + PB * S[i, j - 1]
                if i > 1:
                    t += (i - 1) * inv2g * S[i - 2, j]
                if j > 1:
                    t += (j - 1) * inv2g * S[i, j - 2]
                S[i, j] = t
        return S


    @njit(cache=True)
    def overlap_cart_block_numba(
        li: int,
        lj: int,
        alpha_i: np.ndarray,
        c_i: np.ndarray,
        alpha_j: np.ndarray,
        c_j: np.ndarray,
        Rx: float,
        Ry: float,
        Rz: float,
    ) -> np.ndarray:
        cart_i = _cart_list_np(li)
        cart_j = _cart_list_np(lj)
        nci = cart_i.shape[0]
        ncj = cart_j.shape[0]
        out = np.zeros((nci, ncj), dtype=np.float64)
        # maxima for 1D recurrences
        max_ix = 0
        max_iy = 0
        max_iz = 0
        for ii in range(nci):
            if cart_i[ii, 0] > max_ix:
                max_ix = cart_i[ii, 0]
            if cart_i[ii, 1] > max_iy:
                max_iy = cart_i[ii, 1]
            if cart_i[ii, 2] > max_iz:
                max_iz = cart_i[ii, 2]
        max_jx = 0
        max_jy = 0
        max_jz = 0
        for jj in range(ncj):
            if cart_j[jj, 0] > max_jx:
                max_jx = cart_j[jj, 0]
            if cart_j[jj, 1] > max_jy:
                max_jy = cart_j[jj, 1]
            if cart_j[jj, 2] > max_jz:
                max_jz = cart_j[jj, 2]
        R2 = Rx * Rx + Ry * Ry + Rz * Rz
        # Note: RB = -R; Px = (b/gamma)*RB
        for ip in range(alpha_i.shape[0]):
            a = float(alpha_i[ip])
            ci_val = float(c_i[ip])
            for jp in range(alpha_j.shape[0]):
                b = float(alpha_j[jp])
                cj_val = float(c_j[jp])
                gamma = a + b
                mu = a * b / gamma
                Px = (b / gamma) * (-Rx)
                Py = (b / gamma) * (-Ry)
                Pz = (b / gamma) * (-Rz)
                PAx = Px
                PAy = Py
                PAz = Pz
                PBx = Px + Rx
                PBy = Py + Ry
                PBz = Pz + Rz
                K = (np.pi / gamma) ** 1.5 * np.exp(-mu * R2)
                Sx = _one_d_recur_np(max_ix, max_jx, PAx, PBx, gamma, K)
                Sy = _one_d_recur_np(max_iy, max_jy, PAy, PBy, gamma, 1.0)
                Sz = _one_d_recur_np(max_iz, max_jz, PAz, PBz, gamma, 1.0)
                w = ci_val * cj_val
                for ii in range(nci):
                    lix = cart_i[ii, 0]
                    liy = cart_i[ii, 1]
                    liz = cart_i[ii, 2]
                    for jj in range(ncj):
                        ljx = cart_j[jj, 0]
                        ljy = cart_j[jj, 1]
                        ljz = cart_j[jj, 2]
                        out[ii, jj] += w * Sx[lix, ljx] * Sy[liy, ljy] * Sz[liz, ljz]
        return out

else:
    # Fallback symbols to simplify conditional import
    overlap_cart_block_numba = None  # type: ignore

