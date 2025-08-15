import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.energy.total import compute_total_energy
from gxtb.scf import _valence_electron_counts


def test_scf_converges_with_ofx_on_mno2():
    """SCF convergence check for MnO2 with small OFX alpha.

    Uses valence electrons to fit basis size and schema-mapped Λ^0.
    Asserts:
      - Baseline SCF converges
      - OFX-enabled SCF converges
      - Reported E_OFX is finite and non-negative (Eq. 155 structure)
    """
    numbers = torch.tensor([25, 8, 8], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0],
                              [0.0, 0.0, 1.80],
                              [0.0, 0.0,-1.80]], dtype=torch.float64)

    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    eeq = load_eeq_params('parameters/eeq')
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)

    nelec = int(_valence_electron_counts(numbers, basis).sum().item())
    common = dict(numbers=numbers, positions=positions, basis=basis,
                  gparams=gparams, schema=schema, eeq=eeq,
                  total_charge=0.0, nelec=nelec,
                  uhf=True, second_order=True, spin=False)

    # Baseline
    res0 = compute_total_energy(ofx=False, ofx_params=None, **common)
    assert res0.scf.converged is True

    # With OFX using schema-built Λ^0 and small alpha
    alpha = 0.05
    res1 = compute_total_energy(ofx=True, ofx_params={'alpha': alpha}, **common)
    assert res1.scf.converged is True
    assert res1.scf.E_OFX is not None
    # Energy must be finite
    e_ofx = float(res1.scf.E_OFX)
    assert torch.isfinite(res1.scf.E_OFX)
    # Given Eq. 155 structure and positive onsite exchange integrals, expect non-negative E_OFX
    assert e_ofx >= -1e-10

