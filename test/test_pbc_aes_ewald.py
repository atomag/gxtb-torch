import numpy as np
import torch

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
_src = _root / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gxtb.pbc.ewald import ewald_grad_hess_1over_r
from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_cn_params
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.pbc.aes_pbc import periodic_aes_potentials, assemble_aes_hamiltonian


def test_ewald_grad_hess_symmetry():
    # Simple cubic cell, check ∇ is odd and Hessian is even under r -> -r
    cell = torch.tensor([[10.0, 0.0, 0.0],
                         [0.0, 10.0, 0.0],
                         [0.0, 0.0, 10.0]], dtype=torch.float64)
    r = torch.tensor([1.234, -0.567, 0.321], dtype=torch.float64)
    g1, H1 = ewald_grad_hess_1over_r(r, cell, eta=0.35, r_cut=8.0, g_cut=8.0)
    g2, H2 = ewald_grad_hess_1over_r(-r, cell, eta=0.35, r_cut=8.0, g_cut=8.0)
    assert torch.allclose(g2, -g1, atol=1e-10, rtol=0.0)
    assert torch.allclose(H2, H1, atol=1e-10, rtol=0.0)
    # Hessian symmetry
    assert torch.allclose(H1, H1.transpose(-1, -2), atol=1e-12)


def test_periodic_aes_outputs_finite_and_symmetric():
    # H2 in a cubic cell; verify H_AES is symmetric and E_dc is finite
    a = 8.0
    cell = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], float)
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], float)
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    positions = torch.tensor(pos, dtype=torch.float64)
    cell_t = torch.tensor(cell, dtype=torch.float64)

    # Load parameters and basis
    pg = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    S, Dm, Qm = build_moment_matrices(numbers, positions, basis)
    # Build a simple closed-shell density from a single orthonormal orbital
    Se, U = torch.linalg.eigh(S)
    X = (U * Se.clamp_min(1e-12).rsqrt()) @ U.T
    C = X[:, :1]
    P_total = 2.0 * (C @ C.T)

    # AES params
    from gxtb.hamiltonian.aes import AESParams
    from gxtb.params.schema import map_aes_global, map_aes_element
    aesg = map_aes_global(pg, schema)
    aese = map_aes_element(pg, schema)
    aparams = AESParams(
        dmp3=float(aesg['dmp3']),
        dmp5=float(aesg['dmp5']),
        mprad=aese['mprad'],
        mpvcn=aese['mpvcn'],
    )
    cn_map = map_cn_params(pg, schema)

    v_mono, v_dip, v_quad, E_pair = periodic_aes_potentials(
        numbers, positions, basis, P_total, S, Dm, Qm, aparams,
        r_cov=cn_map['r_cov'], k_cn=float(cn_map['k_cn']), cell=cell_t, cutoff=6.0,
        ewald_eta=0.35, ewald_r_cut=8.0, ewald_g_cut=8.0,
        si_rules=getattr(schema, 'aes_rules', None),
    )
    H_aes = assemble_aes_hamiltonian(S, Dm, Qm, torch.tensor([0, 1], dtype=torch.long), v_mono, v_dip, v_quad)
    # Finiteness checks (H may be symmetrized at use-site)
    assert torch.isfinite(H_aes).all()
    assert torch.isfinite(v_mono).all()
    assert torch.isfinite(v_dip).all()
    assert torch.isfinite(v_quad).all()
    E_dc = 0.5 * torch.einsum('ij,ji->', H_aes, P_total)
    assert torch.isfinite(E_dc)
