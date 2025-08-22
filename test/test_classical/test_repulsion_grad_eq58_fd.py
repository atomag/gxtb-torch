import numpy as np
import torch
import pytest

import sys
from pathlib import Path
_root = Path(__file__).resolve().parents[1]
_src = _root / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gxtb.params.loader import load_gxtb_params
from gxtb.params.schema import load_schema, map_repulsion_params, map_cn_params
from gxtb.classical.repulsion import RepulsionParams, repulsion_energy, repulsion_energy_and_gradient
from gxtb.params.loader import load_eeq_params
from gxtb.params.schema import load_schema
from gxtb.charges.eeq import compute_eeq_charges, compute_eeq_charge_derivative


def _build_rep_params(pg, schema):
    m = map_repulsion_params(pg, schema)
    cn = map_cn_params(pg, schema)
    return RepulsionParams(
        z_eff0=m['z_eff0'], alpha0=m['alpha0'], kq=m['kq'], kq2=m['kq2'], kcn_elem=m['kcn'], r0=m['r0'],
        kpen1_hhe=float(m['kpen1_hhe']), kpen1_rest=float(m['kpen1_rest']), kpen2=float(m['kpen2']),
        kpen3=float(m['kpen3']), kpen4=float(m['kpen4']), kexp=float(m['kexp']),
        r_cov=cn['r_cov'], k_cn=float(cn['k_cn'])
    )


def test_repulsion_gradient_eq58_matches_fd():
    # Choose atoms likely to have non-zero kq/kq2; fallback skip if both zero
    numbers = torch.tensor([8, 7], dtype=torch.long)  # O, N
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64)
    pg = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    rp = _build_rep_params(pg, schema)
    # EEQ charges and derivative
    eeq = load_eeq_params('parameters/eeq')
    eeq_map = getattr(schema, 'eeq', None)
    assert eeq_map is not None, "Missing [eeq] mapping in schema"
    mapping = {k: int(v) for k, v in eeq_map.items()}
    q = compute_eeq_charges(numbers, positions, eeq, 0.0, mapping=mapping, device=positions.device, dtype=positions.dtype)
    dq = compute_eeq_charge_derivative(numbers, positions, eeq, 0.0, mapping=mapping, device=positions.device, dtype=positions.dtype)
    # Check kq/kq2 non-zero for at least one atom; else skip
    kq = rp.kq[numbers.long()]
    kq2 = rp.kq2[numbers.long()]
    if not (bool((kq.abs() > 0).any()) or bool((kq2.abs() > 0).any())):
        pytest.skip("kq and kq2 are zero for chosen elements; Eq. 58 path inactive")
    # Analytic gradient with dq/dR (Eq. 58)
    E, g = repulsion_energy_and_gradient(positions, numbers, rp, q, dq_dpos=dq)
    assert torch.isfinite(E)
    assert torch.isfinite(g).all()
    # Finite-difference gradient by recomputing q at displaced geometries
    eps = 1.0e-4
    g_fd = torch.zeros_like(g)
    for A in range(positions.shape[0]):
        for k in range(3):
            d = torch.zeros_like(positions)
            d[A, k] = eps
            pos_p = positions + d
            pos_m = positions - d
            q_p = compute_eeq_charges(numbers, pos_p, eeq, 0.0, mapping=mapping, device=positions.device, dtype=positions.dtype)
            q_m = compute_eeq_charges(numbers, pos_m, eeq, 0.0, mapping=mapping, device=positions.device, dtype=positions.dtype)
            Ep = repulsion_energy(pos_p, numbers, rp, q_p)
            Em = repulsion_energy(pos_m, numbers, rp, q_m)
            g_fd[A, k] = (Ep - Em) / (2.0 * eps)
    # Compare
    assert torch.allclose(g, g_fd, atol=5e-4, rtol=5e-4)

