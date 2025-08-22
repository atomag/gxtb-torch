import torch

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
_src = _root / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.pbc.cell import validate_cell
from gxtb.pbc.kpoints import validate_kpoints
from gxtb.pbc.bloch import eht_lattice_blocks, assemble_k_matrices


def test_spd_s_matrix_gamma_he():
    a = 5.0
    cell = torch.tensor([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]], dtype=torch.float64)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    numbers = torch.tensor([2], dtype=torch.int64)
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    cell_t = validate_cell(cell, (True, True, True))
    # Build lattice blocks and assemble S(k) at Gamma
    blocks = eht_lattice_blocks(numbers, positions, basis, gparams, schema, cell_t, cutoff=4.0, cn_cutoff=4.0)
    K, W = validate_kpoints([[0.0, 0.0, 0.0]], [1.0])
    mats = assemble_k_matrices(blocks['translations'], blocks['S_blocks_raw'], blocks['H_blocks'], K)
    Sg = mats['S_k'][0]
    # Hermitian and SPD check (floor > 1e-10)
    evals = torch.linalg.eigvalsh(Sg).real
    assert torch.allclose(Sg, Sg.conj().T, atol=1e-12)
    assert float(evals.min().item()) > 1e-10

