import torch

from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_hubbard_params, map_cn_params, map_third_order_params
from gxtb.basis.qvszp import build_atom_basis
from gxtb.hamiltonian.scf_adapter import build_eht_core, make_core_builder
from gxtb.hamiltonian.second_order_tb import build_shell_second_order_params
from gxtb.scf import scf


def test_third_order_fock_changes_h():
    # Small asymmetric system to generate non-zero shell charges
    numbers = torch.tensor([6, 1, 1], dtype=torch.long)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.10, 0.0, 0.0], [-0.40, 0.90, 0.0]], dtype=torch.float64)
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    basis = build_atom_basis(numbers, load_basisq('parameters/basisq'))
    core = build_eht_core(numbers, positions, basis, gparams, schema)
    builder = make_core_builder(basis, gparams, schema)
    hub = map_hubbard_params(gparams, schema)
    # Electrons: simple closed-shell fill
    nelec = 2
    # Third-order parameter packs
    maxz = int(numbers.max().item())
    sp = build_shell_second_order_params(maxz, hub['gamma'])
    cnmap = map_cn_params(gparams, schema)
    from gxtb.cn import coordination_number
    cn_vec = coordination_number(positions, numbers, cnmap['r_cov'].to(dtype=positions.dtype), float(cnmap['k_cn']))
    top = map_third_order_params(gparams, schema)
    # Soften third-order parameters for numerical stability in this smoke test
    top = {
        'gamma3_elem': top['gamma3_elem'] * 1e-2,
        'kGamma': top['kGamma'] * 1e-1,
        'k3': top['k3'] * 1e-1,
        'k3x': top['k3x'],
    }

    # Baseline SCF without third-order
    res0 = scf(
        numbers, positions, basis, builder, core['S'], hubbard=hub, ao_atoms=core['ao_atoms'], nelec=nelec,
        max_iter=6, tol=1e-6, mixing={'scheme': 'linear', 'beta': 0.2}
    )
    # SCF with third-order Fock and energy enabled
    res1 = scf(
        numbers, positions, basis, builder, core['S'], hubbard=hub, ao_atoms=core['ao_atoms'], nelec=nelec,
        third_order=True,
        third_shell_params={'shell_params': sp, 'cn': cn_vec},
        third_params=top,
        max_iter=6, tol=1e-6, mixing={'scheme': 'linear', 'beta': 0.2}
    )
    # E3 should be computed in the third-order run
    assert res1.E3 is not None
    # Final Hamiltonians must differ due to F^(3) contribution
    assert not torch.allclose(res0.H, res1.H)
