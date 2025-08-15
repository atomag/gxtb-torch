import os
import numpy as np
import pytest
import torch

ase_ok = True
try:
    from ase import Atoms
except Exception:  # pragma: no cover
    ase_ok = False

@pytest.mark.skipif(not ase_ok, reason="ASE not available")
def test_calculator_adds_d4_energy_matches_direct_eval():
    from gxtb.ase_calc import GxTBCalculator, EH2EV
    from gxtb.classical.dispersion import load_d4_method, d4_energy
    from gxtb.params.loader import load_gxtb_params
    from gxtb.params.schema import load_schema, map_cn_params
    from gxtb.cn import coordination_number

    # Water geometry (Angstrom)
    positions = np.array(
        [
            [0.000000, 0.000000, 0.000000],
            [0.958, 0.000000, 0.000000],
            [-0.239, 0.927, 0.000000],
        ]
    )
    numbers = [8, 1, 1]
    atoms = Atoms(numbers=numbers, positions=positions)
    # Provide zero EEQ charges explicitly to avoid external dependency
    atoms.info['q_eeqbc'] = np.zeros(len(numbers))

    # Calculator without and with dispersion (TOML reference source)
    calc_no = GxTBCalculator(parameters_dir='parameters', enable_dispersion=False, device='cpu')
    calc_yes = GxTBCalculator(
        parameters_dir='parameters',
        device='cpu',
        enable_dispersion=True,
        d4_reference_path='parameters/d4_reference.toml',
        d4_variant='bj-eeq-atm',
        d4_functional=None,
    )

    atoms.calc = calc_no
    E_no = atoms.get_potential_energy()

    atoms.calc = calc_yes
    E_yes = atoms.get_potential_energy()

    dE_ev = E_yes - E_no

    # Direct D4 via our function using the same CN model as calculator
    device = torch.device('cpu')
    dtype = torch.float64
    tpos = torch.tensor(positions, dtype=dtype, device=device)
    tnum = torch.tensor(numbers, dtype=torch.int64, device=device)
    q = torch.zeros(len(numbers), dtype=dtype, device=device)

    # Build ref from local TOML subset
    from gxtb.params.loader import load_d4_reference_toml
    ref = load_d4_reference_toml('parameters/d4_reference.toml', device=device, dtype=dtype)

    # CN via our model to match calculator
    p_gxtb = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    cn_map = map_cn_params(p_gxtb, schema)
    cn = coordination_number(tpos, tnum, cn_map['r_cov'].to(dtype=dtype, device=device), float(cn_map['k_cn']))
    ref['cn'] = cn

    method = load_d4_method('parameters/dftd4parameters.toml', functional=None, variant='bj-eeq-atm')
    E_disp_h = d4_energy(tnum, tpos, q, method, ref)
    dE_theory_ev = float(E_disp_h.item() * EH2EV)

    assert np.isclose(dE_ev, dE_theory_ev, rtol=1e-6, atol=1e-7), f"ΔE(calc)={dE_ev} eV vs E_disp={dE_theory_ev} eV"
