import numpy as np
import pytest

from ase import Atoms

from gxtb.ase_calc import GxTBCalculator


def test_scf_k_aes_tio2_gamma():
    # Rutile TiO2 (synthetic), fractional positions with u ~ 0.305
    u = 0.305
    a = 4.5937
    c = 2.9587
    cell = np.array([[a, 0, 0], [0, a, 0], [0, 0, c]], float)
    fracs = [
        ('Ti', [0.0, 0.0, 0.0]),
        ('Ti', [0.5, 0.5, 0.5]),
        ('O', [u, u, 0.0]),
        ('O', [-u, -u, 0.0]),
        ('O', [0.5 + u, 0.5 - u, 0.5]),
        ('O', [0.5 - u, 0.5 + u, 0.5]),
    ]
    symbols = ''.join([s for s, _ in fracs])
    frac = np.array([p for _, p in fracs])
    pos = frac @ cell
    atoms = Atoms(symbols=symbols, positions=pos, cell=cell, pbc=True)

    # k-SCF with Ewald second-order and AES (Gamma-only grid)
    calc = GxTBCalculator(
        parameters_dir='parameters',
        pbc_mode='scf-k',
        mp_grid=(1, 1, 1), mp_shift=(0.0, 0.0, 0.0),
        pbc_cutoff=6.0, pbc_cn_cutoff=6.0, pbc_aes_cutoff=8.0,
        enable_second_order=True, ewald_eta=0.35, ewald_r_cut=10.0, ewald_g_cut=12.0,
        enable_aes=True,
        scf_mix=0.6, scf_tol=1e-6, scf_max_iter=50,
        s_psd_adapt=True, pbc_cutoff_max=9.0, pbc_cutoff_step=1.0, s_spd_floor=1e-3,
        device='cpu',
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    assert isinstance(e, float)
    assert np.isfinite(e)
