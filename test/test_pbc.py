import pytest
import torch

from ase import Atoms

from gxtb.ase_calc import GxTBCalculator


def test_pbc_requires_explicit_mode_and_kpoints():
    # Simple H in a cubic box
    a = 5.0
    atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]], cell=[[a,0,0],[0,a,0],[0,0,a]], pbc=True)
    calc = GxTBCalculator(parameters_dir='parameters', device='cpu')
    atoms.calc = calc
    with pytest.raises(NotImplementedError):
        _ = atoms.get_potential_energy()

    # Mode provided but missing kpoints/weights
    calc = GxTBCalculator(parameters_dir='parameters', pbc_mode='eht-k', device='cpu')
    atoms.calc = calc
    with pytest.raises(ValueError):
        _ = atoms.get_potential_energy()


def test_pbc_disallows_unsupported_terms():
    a = 5.0
    atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]], cell=[[a,0,0],[0,a,0],[0,0,a]], pbc=True)
    # Provide required PBC inputs but enable a forbidden term
    calc = GxTBCalculator(
        parameters_dir='parameters',
        pbc_mode='eht-k',
        kpoints=[[0.0, 0.0, 0.0]],
        kpoint_weights=[1.0],
        pbc_cutoff=4.0,
        pbc_cn_cutoff=4.0,
        enable_second_order=True,
        device='cpu',
    )
    atoms.calc = calc
    with pytest.raises(NotImplementedError):
        _ = atoms.get_potential_energy()


def test_pbc_eht_gamma_energy_runs():
    a = 5.0
    # Use He (2 electrons) to ensure closed-shell occupancy under Γ-only EHT
    atoms = Atoms('He', positions=[[0.0, 0.0, 0.0]], cell=[[a,0,0],[0,a,0],[0,0,a]], pbc=True)
    # Γ-only mode: no kpoints required
    calc = GxTBCalculator(
        parameters_dir='parameters',
        pbc_mode='eht-gamma',
        pbc_cutoff=4.0,
        pbc_cn_cutoff=4.0,
        enable_second_order=False,
        enable_dispersion=False,
        enable_aes=False,
        device='cpu',
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    assert isinstance(e, float)


def test_pbc_gamma_alias_flag():
    a = 5.0
    atoms = Atoms('He', positions=[[0.0, 0.0, 0.0]], cell=[[a,0,0],[0,a,0],[0,0,a]], pbc=True)
    # Use gamma=True alias instead of pbc_mode
    calc = GxTBCalculator(
        parameters_dir='parameters',
        gamma=True,
        pbc_cutoff=4.0,
        pbc_cn_cutoff=4.0,
        enable_second_order=False,
        enable_dispersion=False,
        enable_aes=False,
        device='cpu',
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    assert isinstance(e, float)


def test_pbc_mp_grid_equivalent_to_gamma_for_111():
    a = 5.0
    atoms = Atoms('He', positions=[[0.0, 0.0, 0.0]], cell=[[a,0,0],[0,a,0],[0,0,a]], pbc=True)
    # Γ-only via alias
    calc_g = GxTBCalculator(parameters_dir='parameters', gamma=True, pbc_cutoff=4.0, pbc_cn_cutoff=4.0, device='cpu', enable_second_order=False)
    atoms.calc = calc_g
    e_gamma = atoms.get_potential_energy()
    # 1x1x1 MP grid is also Γ-only
    calc_mp = GxTBCalculator(parameters_dir='parameters', pbc_mode='eht-k', mp_grid=(1,1,1), mp_shift=(0.0,0.0,0.0), pbc_cutoff=4.0, pbc_cn_cutoff=4.0, device='cpu', enable_second_order=False)
    atoms.calc = calc_mp
    e_mp = atoms.get_potential_energy()
    assert abs(e_gamma - e_mp) < 1e-10


def test_pbc_gamma_with_d4_two_body_runs():
    import numpy as np
    a = 5.0
    atoms = Atoms('He', positions=[[0.0, 0.0, 0.0]], cell=[[a,0,0],[0,a,0],[0,0,a]], pbc=True)
    atoms.info['q_eeqbc'] = np.zeros((len(atoms),), dtype=float)
    # Use two-body-only D4 variant (no ATM)
    calc = GxTBCalculator(
        parameters_dir='parameters',
        gamma=True,
        pbc_cutoff=4.0,
        pbc_cn_cutoff=4.0,
        pbc_disp_cutoff=6.0,
        enable_second_order=False,
        enable_dispersion=True,
        d4_variant='bj-eeq-two',
        device='cpu',
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    assert isinstance(e, float)


def test_pbc_aes_requires_scf_k_and_cutoff():
    from ase import Atoms
    a = 5.0
    atoms = Atoms('He', positions=[[0.0, 0.0, 0.0]], cell=[[a,0,0],[0,a,0],[0,0,a]], pbc=True)
    # AES not allowed for eht-k
    from gxtb.ase_calc import GxTBCalculator
    calc = GxTBCalculator(parameters_dir='parameters', pbc_mode='eht-k', mp_grid=(1,1,1), mp_shift=(0,0,0), pbc_cutoff=4.0, pbc_cn_cutoff=4.0, enable_second_order=False, enable_aes=True, device='cpu')
    atoms.calc = calc
    import pytest
    with pytest.raises(NotImplementedError):
        _ = atoms.get_potential_energy()
    # scf-k requires pbc_aes_cutoff
    calc = GxTBCalculator(parameters_dir='parameters', pbc_mode='scf-k', mp_grid=(1,1,1), mp_shift=(0,0,0), pbc_cutoff=4.0, pbc_cn_cutoff=4.0, enable_second_order=True, ewald_eta=0.3, ewald_r_cut=6.0, ewald_g_cut=8.0, enable_aes=True, device='cpu')
    atoms.calc = calc
    with pytest.raises(ValueError):
        _ = atoms.get_potential_energy()
