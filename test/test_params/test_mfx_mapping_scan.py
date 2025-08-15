import torch
import pytest

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.hamiltonian.mfx import MFXParams, build_gamma_ao, mfx_fock, mfx_energy


def _projector_like_P(S: torch.Tensor, nelec: int) -> torch.Tensor:
    evals, evecs = torch.linalg.eigh(S)
    X = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
    nocc = max(1, min(S.shape[0], nelec // 2))
    C = X
    return 2.0 * C[:, :nocc] @ C[:, :nocc].T


def test_mfx_mapping_scan_atomic_energies_across_elements():
    dtype = torch.float64
    device = torch.device("cpu")
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    bq = load_basisq('parameters/basisq')

    # Try schema-driven mapping first; if not present, skip test (mapping under evaluation)
    from gxtb.params.schema import map_mfx_element
    # Per-element shell mapping is required for this scan
    U_shell = map_mfx_element(gparams, schema)
    # For globals, fall back to theory-guided constants where schema incomplete
    try:
        from gxtb.params.schema import map_mfx_global
        gmap = map_mfx_global(gparams, schema)
        alpha = float(gmap.get('alpha', 0.6))
        omega = float(gmap.get('omega', 0.5))
        k1 = float(gmap.get('k1', 0.0))
        k2 = float(gmap.get('k2', 0.0))
        xi_l = gmap['xi_l'] if isinstance(gmap.get('xi_l', None), torch.Tensor) else torch.tensor([1.0,1.0,2.0,2.0], dtype=dtype)
    except Exception:
        alpha, omega, k1, k2 = 0.6, 0.5, 0.0, 0.0
        xi_l = torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=dtype)

    # Scan intersection of elements present in gxtb and basisq
    elements = sorted(set(gparams.elements.keys()) & set(bq.elements.keys()))
    # Limit to a representative subset to keep runtime modest if very large
    sample = elements  # full set; atoms are cheap

    for z in sample:
        numbers = torch.tensor([z], dtype=torch.long, device=device)
        positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=dtype, device=device)
        basis = build_atom_basis(numbers, bq)
        S, D, Q = build_moment_matrices(numbers, positions, basis)
        nao = S.shape[0]
        # Sanity: per-shell U should be finite (allow zeros for missing shells)
        assert torch.isfinite(U_shell[z]).all()
        # Build projector-like P
        nelec_guess = max(2, min(2*nao, 2))  # minimal even electrons
        P = _projector_like_P(S, nelec=nelec_guess)
        params = MFXParams(alpha=alpha, omega=omega, k1=k1, k2=k2, U_shell=U_shell.to(device=device, dtype=dtype), xi_l=xi_l.to(device=device, dtype=dtype))
        gamma = build_gamma_ao(numbers, positions, basis, params)
        F = mfx_fock(P, S, gamma)
        E = mfx_energy(P, F)
        # Finite and bounded energy (very loose bound)
        assert torch.isfinite(E)
        assert abs(float(E.item())) < 1e3


def test_mfx_mapping_scan_diatomics_XH():
    dtype = torch.float64
    device = torch.device("cpu")
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    bq = load_basisq('parameters/basisq')

    from gxtb.params.schema import map_mfx_element
    U_shell = map_mfx_element(gparams, schema)
    # Globals as in atomic scan
    try:
        from gxtb.params.schema import map_mfx_global
        gmap = map_mfx_global(gparams, schema)
        alpha = float(gmap.get('alpha', 0.6))
        omega = float(gmap.get('omega', 0.5))
        k1 = float(gmap.get('k1', 0.0))
        k2 = float(gmap.get('k2', 0.0))
        xi_l = gmap['xi_l'] if isinstance(gmap.get('xi_l', None), torch.Tensor) else torch.tensor([1.0,1.0,2.0,2.0], dtype=dtype)
    except Exception:
        alpha, omega, k1, k2 = 0.6, 0.5, 0.0, 0.0
        xi_l = torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=dtype)

    elements = sorted(set(gparams.elements.keys()) & set(bq.elements.keys()))
    # Exclude hydrogen from X to form X–H
    elements = [z for z in elements if z != 1]
    R = 1.1  # Å
    for z in elements:
        numbers = torch.tensor([z, 1], dtype=torch.long, device=device)
        positions = torch.tensor([[0.0,0.0,0.0],[R,0.0,0.0]], dtype=dtype, device=device)
        basis = build_atom_basis(numbers, bq)
        S, D, Q = build_moment_matrices(numbers, positions, basis)
        nao = S.shape[0]
        # Minimal occupancy to keep scan light
        P = _projector_like_P(S, nelec=2)
        params = MFXParams(alpha=alpha, omega=omega, k1=k1, k2=k2, U_shell=U_shell.to(device=device, dtype=dtype), xi_l=xi_l.to(device=device, dtype=dtype))
        gamma = build_gamma_ao(numbers, positions, basis, params)
        F = mfx_fock(P, S, gamma)
        E = mfx_energy(P, F)
        assert torch.isfinite(E)
        assert abs(float(E.item())) < 1e3
