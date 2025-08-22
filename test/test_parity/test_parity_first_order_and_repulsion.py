import torch
import pytest

from pathlib import Path
import sys
_root = Path(__file__).resolve().parents[1]
_src = _root / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema, map_hubbard_params, map_cn_params, map_repulsion_params
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.scf_adapter import build_eht_core, make_core_builder
from gxtb.scf import scf
from gxtb.classical.repulsion import RepulsionParams, repulsion_energy_and_gradient
from gxtb.charges.eeq import compute_eeq_charges, compute_eeq_charge_derivative


cuda_available = torch.cuda.is_available()


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_first_order_tb_energy_parity_cpu_cuda():
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    hub = map_hubbard_params(gparams, schema)
    cn_map = map_cn_params(gparams, schema)
    basisq = load_basisq('parameters/basisq')
    numbers = torch.tensor([1, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=torch.float64)
    basis = build_atom_basis(numbers, basisq)
    core_cpu = build_eht_core(numbers, positions, basis, gparams, schema)
    builder = make_core_builder(basis, gparams, schema)
    # Run CPU
    res_cpu = scf(
        numbers, positions, basis, builder, core_cpu['S'], hub, core_cpu['ao_atoms'], nelec=2,
        max_iter=12, first_order=True, first_order_params=None,
        dynamic_overlap=False,
        qvszp_params={'r_cov': cn_map['r_cov'], 'k_cn': float(cn_map['k_cn'])},
        gparams=gparams, schema=schema,
    )
    # Run CUDA
    dev = torch.device('cuda')
    numbers_cu = numbers.to(dev)
    positions_cu = positions.to(dev)
    basis_cu = build_atom_basis(numbers_cu.cpu(), basisq)  # basis metadata is device-agnostic
    core_cu = build_eht_core(numbers_cu, positions_cu, basis_cu, gparams, schema)
    hub_cu = {k: v.to(dev) for k, v in hub.items()}
    cn_map_cu = {'r_cov': cn_map['r_cov'].to(dev), 'k_cn': cn_map['k_cn']}
    res_cu = scf(
        numbers_cu, positions_cu, basis_cu, builder, core_cu['S'], hub_cu, core_cu['ao_atoms'], nelec=2,
        max_iter=12, first_order=True, first_order_params=None,
        dynamic_overlap=False,
        qvszp_params={'r_cov': cn_map_cu['r_cov'], 'k_cn': float(cn_map_cu['k_cn'])},
        gparams=gparams, schema=schema,
    )
    e1_cpu = res_cpu.E_First if res_cpu.E_First is not None else torch.tensor(0.0)
    e1_cu = (res_cu.E_First if res_cu.E_First is not None else torch.tensor(0.0, device=dev)).to('cpu')
    assert torch.allclose(e1_cpu, e1_cu, atol=1e-8, rtol=1e-8)


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
def test_repulsion_gradient_parity_cpu_cuda():
    # Small dimer with likely non-zero kq/kq2; skip if zero
    numbers = torch.tensor([8, 7], dtype=torch.long)  # O, N
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64)
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    rep_map = map_repulsion_params(gparams, schema)
    cn_map = map_cn_params(gparams, schema)
    rp = RepulsionParams(
        z_eff0=rep_map['z_eff0'], alpha0=rep_map['alpha0'], kq=rep_map['kq'], kq2=rep_map['kq2'], kcn_elem=rep_map['kcn'], r0=rep_map['r0'],
        kpen1_hhe=float(rep_map['kpen1_hhe']), kpen1_rest=float(rep_map['kpen1_rest']), kpen2=float(rep_map['kpen2']),
        kpen3=float(rep_map['kpen3']), kpen4=float(rep_map['kpen4']), kexp=float(rep_map['kexp']),
        r_cov=cn_map['r_cov'], k_cn=float(cn_map['k_cn'])
    )
    eeq = load_eeq_params('parameters/eeq')
    eeq_map = getattr(schema, 'eeq', None)
    mapping = {k: int(v) for k, v in eeq_map.items()}
    q = compute_eeq_charges(numbers, positions, eeq, 0.0, mapping=mapping)
    dq = compute_eeq_charge_derivative(numbers, positions, eeq, 0.0, mapping=mapping)
    # Skip if both kq/kq2 zero
    if not (bool((rp.kq[numbers.long()].abs() > 0).any()) or bool((rp.kq2[numbers.long()].abs() > 0).any())):
        pytest.skip("kq/kq2 zero; parity test not informative")
    E_cpu, g_cpu = repulsion_energy_and_gradient(positions, numbers, rp, q, dq_dpos=dq)
    dev = torch.device('cuda')
    E_cu, g_cu = repulsion_energy_and_gradient(positions.to(dev), numbers.to(dev), RepulsionParams(
        z_eff0=rp.z_eff0.to(dev), alpha0=rp.alpha0.to(dev), kq=rp.kq.to(dev), kq2=rp.kq2.to(dev), kcn_elem=rp.kcn_elem.to(dev), r0=rp.r0.to(dev),
        kpen1_hhe=rp.kpen1_hhe, kpen1_rest=rp.kpen1_rest, kpen2=rp.kpen2, kpen3=rp.kpen3, kpen4=rp.kpen4, kexp=rp.kexp,
        r_cov=rp.r_cov.to(dev), k_cn=rp.k_cn
    ), q.to(dev), dq_dpos=dq.to(dev))
    assert torch.allclose(E_cpu, E_cu.to('cpu'), atol=1e-9, rtol=1e-9)
    assert torch.allclose(g_cpu, g_cu.to('cpu'), atol=1e-7, rtol=1e-7)

