import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema, map_qvszp_prefactors, map_cn_params
from gxtb.basis.qvszp import build_atom_basis, AtomBasis, ShellDef
from gxtb.hamiltonian.scf_adapter import build_eht_core
from gxtb.hamiltonian.eht import eht_energy_gradient
from gxtb.scf import lowdin_orthogonalization, mulliken_charges
from gxtb.charges.eeq import compute_eeq_charges
from gxtb.grad.nuclear import total_gradient


def _basis_with_zero_c1(basis: AtomBasis) -> AtomBasis:
    shells = []
    for sh in basis.shells:
        prims = tuple((p[0], p[1], 0.0) for p in sh.primitives)
        shells.append(ShellDef(sh.atom_index, sh.element, sh.l, sh.nprims, prims))
    return AtomBasis(shells=shells, ao_counts=basis.ao_counts, ao_offsets=basis.ao_offsets, nao=basis.nao)


def test_total_gradient_cn_toggle_matches_eht_cn_component():
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]], dtype=torch.float64)
    g = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    basis0 = _basis_with_zero_c1(basis)
    core = build_eht_core(numbers, positions, basis, g, schema)
    S = core['S']
    ao_atoms = core['ao_atoms']
    # simple projector P
    X = lowdin_orthogonalization(S)
    nao = S.shape[0]
    nocc = max(1, nao // 2)
    C = X
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    # q_scf and q_eeqbc
    eeq = load_eeq_params('parameters/eeq')
    q_eeq = compute_eeq_charges(numbers, positions, eeq, total_charge=0.0, dtype=positions.dtype, device=positions.device)
    q_scf = numbers.to(positions.dtype) - mulliken_charges(P, S, ao_atoms)

    # EHT dynamic-overlap CN-only analytic component
    qv = map_qvszp_prefactors(g, schema)
    cnm = map_cn_params(g, schema)
    pack = {
        'k0': qv['k0'].to(positions),
        'k1': qv['k1'].to(positions),
        'k2': qv['k2'].to(positions),
        'k3': qv['k3'].to(positions),
        'r_cov': cnm['r_cov'].to(positions),
        'k_cn': float(cnm['k_cn']),
        'q_scf': q_scf,
        'q_eeqbc': q_eeq,
    }
    g_with = eht_energy_gradient(numbers, positions, basis, g, schema, P, dynamic_overlap_cn=pack)
    g_wo = eht_energy_gradient(numbers, positions, basis0, g, schema, P, dynamic_overlap_cn=pack)
    g_cn = g_with - g_wo

    # Aggregator with only CN-driven EHT path included
    g_total = total_gradient(
        numbers, positions, basis, g, schema,
        P=P,
        include_eht_stepA=False,
        include_dynamic_overlap_cn=True,
        q_scf=q_scf,
        q_eeqbc=q_eeq,
        include_second_order=False,
        include_aes=False,
    )
    assert torch.allclose(g_total, g_cn, atol=1e-10, rtol=1e-10)

