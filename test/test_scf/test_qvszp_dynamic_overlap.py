import pytest
import torch

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_cn_params, map_hubbard_params
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.scf_adapter import build_eht_core, make_core_builder
from gxtb.scf import scf


def _simple_qvszp_pack(numbers, gparams, schema, device, dtype):
    # Build a deterministic qvszp pack: k0=1, others=0 so q_eff = q (Eq. 28)
    maxz = int(numbers.max().item())
    k0 = torch.zeros(maxz + 1, dtype=dtype, device=device); k0[:] = 1.0
    k1 = torch.zeros_like(k0)
    k2 = torch.zeros_like(k0)
    k3 = torch.zeros_like(k0)
    cn = map_cn_params(gparams, schema)
    return {
        'k0': k0,
        'k1': k1,
        'k2': k2,
        'k3': k3,
        'r_cov': cn['r_cov'].to(device=device, dtype=dtype),
        'k_cn': float(cn['k_cn']),
    }


def _scf_once(numbers, positions, basis, gparams, schema, eeq, nelec, eeq_charges, qvszp_pack, device):
    builder = make_core_builder(basis, gparams, schema)
    core = build_eht_core(numbers, positions, basis, gparams, schema)
    hub = map_hubbard_params(gparams, schema)
    res = scf(
        numbers.to(device=device),
        positions.to(device=device),
        basis,
        builder,
        core['S'].to(device=device),
        hub,
        core['ao_atoms'].to(device=device),
        nelec=nelec,
        max_iter=1,
        eeq_charges=eeq_charges.to(device=device),
        dynamic_overlap=True,
        qvszp_params=qvszp_pack,
    )
    return res


def test_dynamic_overlap_changes_with_q():
    # C–H diatomic
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]], dtype=torch.float64)
    g = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    pack = _simple_qvszp_pack(numbers, g, schema, positions.device, positions.dtype)
    # Two different initial q vectors via eeq_charges
    eeq0 = torch.zeros(numbers.shape[0], dtype=positions.dtype)
    eeq1 = torch.tensor([0.5, -0.5], dtype=positions.dtype)
    res0 = _scf_once(numbers, positions, basis, g, schema, None, nelec=7, eeq_charges=eeq0, qvszp_pack=pack, device=positions.device)
    res1 = _scf_once(numbers, positions, basis, g, schema, None, nelec=7, eeq_charges=eeq1, qvszp_pack=pack, device=positions.device)
    assert res0.S is not None and res1.S is not None
    assert not torch.allclose(res0.S, res1.S)


def test_dynamic_overlap_spd_cpu():
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]], dtype=torch.float64)
    g = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    pack = _simple_qvszp_pack(numbers, g, schema, positions.device, positions.dtype)
    eeq0 = torch.zeros(numbers.shape[0], dtype=positions.dtype)
    res = _scf_once(numbers, positions, basis, g, schema, None, nelec=7, eeq_charges=eeq0, qvszp_pack=pack, device=positions.device)
    S = res.S
    assert S is not None
    evals = torch.linalg.eigvalsh(0.5 * (S + S.T))
    assert torch.all(evals > 1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dynamic_overlap_determinism_cpu_gpu():
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]], dtype=torch.float64)
    g = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    pack_cpu = _simple_qvszp_pack(numbers, g, schema, positions.device, positions.dtype)
    pack_gpu = _simple_qvszp_pack(numbers, g, schema, torch.device('cuda'), positions.dtype)
    eeq0 = torch.zeros(numbers.shape[0], dtype=positions.dtype)
    res_cpu = _scf_once(numbers, positions, basis, g, schema, None, nelec=7, eeq_charges=eeq0, qvszp_pack=pack_cpu, device=torch.device('cpu'))
    res_gpu = _scf_once(numbers.to('cuda'), positions.to('cuda'), basis, g, schema, None, nelec=7, eeq_charges=eeq0.to('cuda'), qvszp_pack=pack_gpu, device=torch.device('cuda'))
    assert res_cpu.S is not None and res_gpu.S is not None
    # Compare within a reasonable tolerance for float64
    assert torch.allclose(res_cpu.S.cpu(), res_gpu.S.cpu(), atol=1e-10, rtol=1e-10)

