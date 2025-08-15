import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema, map_spin_kW
from gxtb.basis.qvszp import build_atom_basis
from gxtb.energy.total import compute_total_energy


def test_spin_uhf_smoke_runs():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    eeq = load_eeq_params('parameters/eeq')
    bq = load_basisq('parameters/basisq')
    numbers = torch.tensor([7], dtype=torch.long)  # N atom (unpaired)
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, bq)
    # W0: simple 4x4 symmetric matrix for (s,p,d,f) blocks
    W0 = torch.eye(4, dtype=torch.float64)
    # α/β electrons (odd electron) -> uhf=True
    res = compute_total_energy(numbers, positions, basis, gparams, schema, eeq,
                               total_charge=0.0, nelec=5, uhf=True, spin=True, spin_W0=W0)
    assert isinstance(res.scf.converged, bool)
    assert res.scf.P_alpha is not None and res.scf.P_beta is not None
