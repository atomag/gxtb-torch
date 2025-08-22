import numpy as np
import torch
import logging

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
from gxtb.pbc.aes_pbc import periodic_aes_potentials
from gxtb.hamiltonian.aes import AESParams


def _build_inputs():
    # Use a small lattice so that translations within 5.0 Å exist but none within 3.0 Å
    a = 4.0
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
    return numbers, positions, basis, P_total, S, Dm, Qm, aparams, cn_map, cell_t


def test_aes_high_order_cutoff_logging(caplog):
    numbers, positions, basis, P_total, S, Dm, Qm, aparams, cn_map, cell_t = _build_inputs()
    # Capture debug logs from module
    logger_name = 'gxtb.pbc.aes_pbc'
    caplog.set_level(logging.DEBUG, logger=logger_name)
    # Case 1: no explicit high_order_cutoff -> fallback log expected
    caplog.clear()
    _ = periodic_aes_potentials(
        numbers, positions, basis, P_total, S, Dm, Qm, aparams,
        r_cov=cn_map['r_cov'], k_cn=float(cn_map['k_cn']), cell=cell_t, cutoff=5.0,
        ewald_eta=0.35, ewald_r_cut=6.0, ewald_g_cut=6.0,
        high_order_cutoff=None,
        si_rules=None,
    )
    logs = "\n".join([rec.getMessage() for rec in caplog.records])
    assert "higher-order cutoff not provided" in logs
    assert "translations: nR=" in logs
    # Case 2: explicit high_order_cutoff -> no fallback log
    caplog.clear()
    _ = periodic_aes_potentials(
        numbers, positions, basis, P_total, S, Dm, Qm, aparams,
        r_cov=cn_map['r_cov'], k_cn=float(cn_map['k_cn']), cell=cell_t, cutoff=5.0,
        ewald_eta=0.35, ewald_r_cut=6.0, ewald_g_cut=6.0,
        high_order_cutoff=3.0,
        si_rules=None,
    )
    logs2 = "\n".join([rec.getMessage() for rec in caplog.records])
    assert "higher-order cutoff not provided" not in logs2
    assert "translations: nR=" in logs2
    # Case 3: compare nR counts for two explicit cutoffs (5.0 vs 3.0)
    import re
    caplog.clear()
    _ = periodic_aes_potentials(
        numbers, positions, basis, P_total, S, Dm, Qm, aparams,
        r_cov=cn_map['r_cov'], k_cn=float(cn_map['k_cn']), cell=cell_t, cutoff=5.0,
        ewald_eta=0.35, ewald_r_cut=6.0, ewald_g_cut=6.0,
        high_order_cutoff=5.0,
        si_rules=None,
    )
    logs5 = "\n".join([rec.getMessage() for rec in caplog.records])
    m5 = re.search(r"translations: nR=(\d+)", logs5)
    assert m5 is not None
    nR5 = int(m5.group(1))
    caplog.clear()
    _ = periodic_aes_potentials(
        numbers, positions, basis, P_total, S, Dm, Qm, aparams,
        r_cov=cn_map['r_cov'], k_cn=float(cn_map['k_cn']), cell=cell_t, cutoff=5.0,
        ewald_eta=0.35, ewald_r_cut=6.0, ewald_g_cut=6.0,
        high_order_cutoff=3.0,
        si_rules=None,
    )
    logs3 = "\n".join([rec.getMessage() for rec in caplog.records])
    m3 = re.search(r"translations: nR=(\d+)", logs3)
    assert m3 is not None
    nR3 = int(m3.group(1))
    assert nR5 >= nR3 and (nR5 > nR3)
