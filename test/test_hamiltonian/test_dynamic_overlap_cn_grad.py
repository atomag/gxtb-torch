import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema, map_qvszp_prefactors, map_cn_params
from gxtb.basis.qvszp import build_atom_basis, AtomBasis, ShellDef
from gxtb.hamiltonian.scf_adapter import build_eht_core
from gxtb.hamiltonian.eht import eht_energy_gradient
from gxtb.hamiltonian.scf_adapter import make_core_builder
from gxtb.hamiltonian.overlap_tb import build_scaled_overlap_dynamic, build_overlap_dynamic
from gxtb.scf import lowdin_orthogonalization, mulliken_charges
from gxtb.charges.eeq import compute_eeq_charges


def _basis_with_zero_c1(basis: AtomBasis) -> AtomBasis:
    shells = []
    for sh in basis.shells:
        prims = tuple((p[0], p[1], 0.0) for p in sh.primitives)
        shells.append(ShellDef(sh.atom_index, sh.element, sh.l, sh.nprims, prims))
    return AtomBasis(shells=shells, ao_counts=basis.ao_counts, ao_offsets=basis.ao_offsets, nao=basis.nao)


def test_dynamic_overlap_cn_grad_vanishes_when_c1_zero():
    # Simple C–H diatomic
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]], dtype=torch.float64)
    g = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    basis0 = _basis_with_zero_c1(basis)

    # Core and AO maps
    core = build_eht_core(numbers, positions, basis0, g, schema)
    S = core['S']
    ao_atoms = core['ao_atoms']

    # Construct a simple projector P using Löwdin X (not SCF)
    X = lowdin_orthogonalization(S)
    nao = S.shape[0]
    nocc = max(1, nao // 2)
    C = X
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T

    # Build dynamic-overlap CN pack (Eq. 28) with neutral charge
    qv = map_qvszp_prefactors(g, schema)
    cnm = map_cn_params(g, schema)
    eeq = load_eeq_params('parameters/eeq')
    q_eeq = compute_eeq_charges(numbers, positions, eeq, total_charge=0.0, dtype=positions.dtype, device=positions.device)
    q_scf = numbers.to(positions.dtype) - mulliken_charges(P, S, ao_atoms)
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

    # Compute gradient; dynamic CN path should be zero when c1==0
    dE = eht_energy_gradient(numbers, positions, basis0, g, schema, P, dynamic_overlap_cn=pack)
    assert torch.allclose(dE, torch.zeros_like(dE), atol=1e-12)


def test_dynamic_overlap_cn_grad_finite_difference_isolated():
    # Contrived finite-difference check: keep geometric S/R fixed in H builder and vary only coefficients via CN.
    numbers = torch.tensor([6, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]], dtype=torch.float64)
    g = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    bq = load_basisq('parameters/basisq')
    basis = build_atom_basis(numbers, bq)
    basis0 = _basis_with_zero_c1(basis)
    # Core and projector
    core = build_eht_core(numbers, positions, basis, g, schema)
    S = core['S']
    ao_atoms = core['ao_atoms']
    X = lowdin_orthogonalization(S)
    nao = S.shape[0]
    nocc = max(1, nao // 2)
    C = X
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T

    # qvSZP CN pack using baseline q_scf and q_eeq (held fixed across +/- to isolate CN-only effect)
    qv = map_qvszp_prefactors(g, schema)
    cnm = map_cn_params(g, schema)
    eeq = load_eeq_params('parameters/eeq')
    q_eeq = compute_eeq_charges(numbers, positions, eeq, total_charge=0.0, dtype=positions.dtype, device=positions.device)
    q_scf = numbers.to(positions.dtype) - mulliken_charges(P, S, ao_atoms)
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

    # Analytic CN-induced gradient contribution = grad(basis) - grad(basis with c1=0)
    g_with = eht_energy_gradient(numbers, positions, basis, g, schema, P, dynamic_overlap_cn=pack)
    g_wo = eht_energy_gradient(numbers, positions, basis0, g, schema, P, dynamic_overlap_cn=pack)
    g_cn = g_with - g_wo

    # Finite difference of contrived energy where only coefficients change with CN
    h = 1.0e-4
    pos_p = positions.clone(); pos_p[1, 0] += h
    pos_m = positions.clone(); pos_m[1, 0] -= h
    from gxtb.basis.qvszp import compute_effective_charge
    # q_eff at +/- using fixed q_scf and q_eeq; only CN changes with positions
    qeff_p = compute_effective_charge(numbers, pos_p, q_scf, q_eeq, r_cov=pack['r_cov'], k_cn=pack['k_cn'], k0=pack['k0'], k1=pack['k1'], k2=pack['k2'], k3=pack['k3'])
    qeff_m = compute_effective_charge(numbers, pos_m, q_scf, q_eeq, r_cov=pack['r_cov'], k_cn=pack['k_cn'], k0=pack['k0'], k1=pack['k1'], k2=pack['k2'], k3=pack['k3'])
    # Build coefficient maps at +/-
    coeffs_p = {}
    coeffs_m = {}
    for i, sh in enumerate(basis.shells):
        A = sh.atom_index
        c0 = torch.tensor([p[1] for p in sh.primitives], dtype=positions.dtype)
        c1 = torch.tensor([p[2] for p in sh.primitives], dtype=positions.dtype)
        coeffs_p[i] = c0 + c1 * qeff_p[A]
        coeffs_m[i] = c0 + c1 * qeff_m[A]
    # Build override overlaps at baseline geometry
    S_raw_p = build_overlap_dynamic(numbers, positions, basis, coeffs_p)
    S_sc_p = build_scaled_overlap_dynamic(numbers, positions, basis, coeffs_p, g, schema)
    S_raw_m = build_overlap_dynamic(numbers, positions, basis, coeffs_m)
    S_sc_m = build_scaled_overlap_dynamic(numbers, positions, basis, coeffs_m, g, schema)
    # Build H with overrides to freeze explicit S(R), Π(R), ε(R) at baseline geometry
    builder = make_core_builder(basis, g, schema)
    H_p = builder(numbers, positions, {'S_raw': S_raw_p, 'S_scaled': S_sc_p})['H0']
    H_m = builder(numbers, positions, {'S_raw': S_raw_m, 'S_scaled': S_sc_m})['H0']
    E_p = torch.einsum('ij,ji->', P, H_p)
    E_m = torch.einsum('ij,ji->', P, H_m)
    dE_num = (E_p - E_m) / (2*h)
    # Compare with analytic CN-only gradient component along displaced coordinate (atom 1, x)
    assert torch.isfinite(dE_num)
    assert torch.allclose(g_cn[1, 0], dE_num, atol=1e-6, rtol=1e-5)
