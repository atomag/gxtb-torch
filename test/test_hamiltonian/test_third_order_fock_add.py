import torch

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_hubbard_params, map_cn_params, map_third_order_params
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.scf_adapter import build_eht_core, make_core_builder
from gxtb.hamiltonian.second_order_tb import build_shell_second_order_params, compute_reference_shell_populations, compute_shell_charges
from gxtb.hamiltonian.third_order import compute_tau3_matrix, build_third_order_potentials, add_third_order_fock, ThirdOrderParams
from gxtb.scf import scf


def test_third_order_fock_adds_to_h():
    # Baseline SCF to get reasonable P and q
    numbers = torch.tensor([6, 1, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0], [-0.40, 0.90, 0.0]], dtype=torch.float64)
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    core = build_eht_core(numbers, positions, basis, gparams, schema)
    builder = make_core_builder(basis, gparams, schema)
    hub = map_hubbard_params(gparams, schema)
    nelec = 8
    res0 = scf(numbers, positions, basis, builder, core['S'], hubbard=hub, ao_atoms=core['ao_atoms'], nelec=nelec, max_iter=8, tol=1e-6)
    # Build third-order ingredients
    cnmap = map_cn_params(gparams, schema)
    from gxtb.cn import coordination_number
    cn_vec = coordination_number(positions, numbers, cnmap['r_cov'].to(dtype=positions.dtype), float(cnmap['k_cn']))
    sp = build_shell_second_order_params(int(numbers.max().item()), hub['gamma'])
    # U_shell from CN scaling
    lmap = {'s':0,'p':1,'d':2,'f':3}
    shells = basis.shells
    z_list = torch.tensor([sh.element for sh in shells], dtype=torch.long)
    l_idx = torch.tensor([lmap[sh.l] for sh in shells], dtype=torch.long)
    atom_idx = torch.tensor([sh.atom_index for sh in shells], dtype=torch.long)
    U0_shell = sp.U0[z_list, l_idx]
    kU_shell = sp.kU[z_list]
    U_shell = U0_shell * (1.0 + kU_shell * cn_vec[atom_idx])
    # Tau3
    top = map_third_order_params(gparams, schema)
    # Softened for stability
    tparams = ThirdOrderParams(
        gamma3_elem=top['gamma3_elem'] * 1e-2,
        kGamma_l=(float(top['kGamma'][0]*1e-1), float(top['kGamma'][1]*1e-1), float(top['kGamma'][2]*1e-1), float(top['kGamma'][3]*1e-1)),
        k3=float(top['k3'] * 1e-2),
        k3x=float(top['k3x']),
    )
    tau3 = compute_tau3_matrix(numbers, positions, basis, U_shell, tparams)
    # Shell charges from baseline density
    ref_pops = compute_reference_shell_populations(numbers, basis).to(dtype=positions.dtype)
    q_shell = compute_shell_charges(res0.P, core['S'], basis, ref_pops)
    q_atom = res0.q
    # Build F^(3) and add to a copy of H0
    V_shell3, V_atom3 = build_third_order_potentials(numbers, basis, q_shell, q_atom, tau3, tparams)
    H_before = core['H0'].clone()
    H_after = core['H0'].clone()
    add_third_order_fock(H_after, core['S'], basis, V_shell3, V_atom3)
    # Assert third-order Fock produces a non-trivial change
    assert not torch.allclose(H_before, H_after)

