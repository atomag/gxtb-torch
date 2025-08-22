import numpy as np
import torch

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
_src = _root / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_aes_global, map_aes_element, map_cn_params
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.moments_builder import build_moment_matrices
from gxtb.pbc.aes_pbc import periodic_aes_potentials, assemble_aes_hamiltonian
from gxtb.hamiltonian.aes import AESParams


def test_periodic_aes_high_order_terms_finite():
    # H2 cell; enable dmp7/dmp9 via schema mapping (mapped to same slots as dmp3/dmp5 in schema)
    a = 8.0
    cell = np.array([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], float)
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], float)
    numbers = torch.tensor([1, 1], dtype=torch.int64)
    positions = torch.tensor(pos, dtype=torch.float64)
    cell_t = torch.tensor(cell, dtype=torch.float64)

    pg = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    S, Dm, Qm = build_moment_matrices(numbers, positions, basis)
    Se, U = torch.linalg.eigh(S)
    X = (U * Se.clamp_min(1e-12).rsqrt()) @ U.T
    C = X[:, :1]
    P_total = 2.0 * (C @ C.T)
    aesg = map_aes_global(pg, schema)
    aese = map_aes_element(pg, schema)
    aparams = AESParams(
        dmp3=float(aesg['dmp3']), dmp5=float(aesg['dmp5']),
        mprad=aese['mprad'], mpvcn=aese['mpvcn'],
        dmp7=float(aesg.get('dmp7', aesg['dmp3'])), dmp9=float(aesg.get('dmp9', aesg['dmp5'])),
    )
    cn_map = map_cn_params(pg, schema)
    v_mono, v_dip, v_quad, E_pair = periodic_aes_potentials(
        numbers, positions, basis, P_total, S, Dm, Qm, aparams,
        r_cov=cn_map['r_cov'], k_cn=float(cn_map['k_cn']), cell=cell_t, cutoff=6.0,
        ewald_eta=0.35, ewald_r_cut=8.0, ewald_g_cut=8.0,
        si_rules=getattr(schema, 'aes_rules', None),
    )
    H_aes = assemble_aes_hamiltonian(S, Dm, Qm, torch.tensor([0, 1], dtype=torch.long), v_mono, v_dip, v_quad)
    assert torch.isfinite(H_aes).all()
    assert torch.isfinite(E_pair)

