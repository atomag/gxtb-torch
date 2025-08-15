import torch

from gxtb.hamiltonian.aes import _third_derivative_tensor, _fourth_derivative_tensor


def _analytic_T3(r: torch.Tensor) -> torch.Tensor:
    # Eq. 110c: T3_{abc} = - (15 r_a r_b r_c - 3 (r_a δ_bc + r_b δ_ac + r_c δ_ab) R^2) / R^7
    dtype = r.dtype
    device = r.device
    R = torch.linalg.norm(r)
    invR = 1.0 / R
    R2 = R * R
    g7 = invR ** 7
    eye = torch.eye(3, dtype=dtype, device=device)
    T = torch.zeros(3, 3, 3, dtype=dtype, device=device)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                term_rrr = 15.0 * r[a] * r[b] * r[c]
                term_mix = 3.0 * (r[a] * eye[b, c] + r[b] * eye[a, c] + r[c] * eye[a, b]) * R2
                T[a, b, c] = -(term_rrr - term_mix) * g7
    return T


def _analytic_T4(r: torch.Tensor) -> torch.Tensor:
    # Eq. 110d: see doc/theory; direct tensor construction
    dtype = r.dtype
    device = r.device
    R = torch.linalg.norm(r)
    invR = 1.0 / R
    R2 = R * R
    R4 = R2 * R2
    g9 = invR ** 9
    eye = torch.eye(3, dtype=dtype, device=device)
    T = torch.zeros(3, 3, 3, 3, dtype=dtype, device=device)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    term_r4 = 105.0 * r[a] * r[b] * r[c] * r[d]
                    term_r2 = 15.0 * (
                        r[a] * r[b] * eye[c, d]
                        + r[a] * r[c] * eye[b, d]
                        + r[b] * r[c] * eye[a, d]
                        + r[a] * r[d] * eye[b, c]
                        + r[b] * r[d] * eye[a, c]
                        + r[c] * r[d] * eye[a, b]
                    ) * R2
                    term_r0 = 3.0 * (
                        eye[a, b] * eye[c, d] + eye[a, c] * eye[b, d] + eye[a, d] * eye[b, c]
                    ) * R4
                    T[a, b, c, d] = (term_r4 - term_r2 + term_r0) * g9
    return T


def test_third_derivative_tensor_matches_eq_110c():
    dtype = torch.float64
    device = torch.device("cpu")
    r = torch.tensor([0.7, -0.2, 0.5], dtype=dtype, device=device)
    R = torch.linalg.norm(r)
    invR = torch.tensor([[1.0 / R]], dtype=dtype, device=device)
    rij = r.view(1, 1, 3)
    fdmp7 = torch.ones((1, 1), dtype=dtype, device=device)
    T3_ref = _analytic_T3(r)
    T3_num = _third_derivative_tensor(rij, invR, fdmp7)[0, 0]
    assert torch.allclose(T3_num, T3_ref, atol=1e-10, rtol=1e-10)


def test_fourth_derivative_tensor_matches_eq_110d():
    dtype = torch.float64
    device = torch.device("cpu")
    r = torch.tensor([0.4, 0.3, -0.6], dtype=dtype, device=device)
    R = torch.linalg.norm(r)
    invR = torch.tensor([[1.0 / R]], dtype=dtype, device=device)
    rij = r.view(1, 1, 3)
    fdmp9 = torch.ones((1, 1), dtype=dtype, device=device)
    T4_ref = _analytic_T4(r)
    T4_num = _fourth_derivative_tensor(rij, invR, fdmp9)[0, 0]
    assert torch.allclose(T4_num, T4_ref, atol=1e-10, rtol=1e-10)
