import torch

from gxtb.params.loader import load_gxtb_params, load_basisq, load_eeq_params
from gxtb.params.schema import load_schema
from gxtb.basis.qvszp import build_atom_basis
from gxtb.energy.total import compute_total_energy


def test_ar_z_third_order_no_nan_small_set():
    g = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    bq = load_basisq('parameters/basisq')
    eeq = load_eeq_params('parameters/eeq')
    R = 3.0
    for Z in [1, 8, 14, 18, 20]:
        numbers = torch.tensor([18, Z], dtype=torch.long)
        pos = torch.tensor([[0.0,0.0,0.0],[R,0.0,0.0]], dtype=torch.float64)
        basis = build_atom_basis(numbers, bq)
        # estimate valence electrons like in scans
        from gxtb.hamiltonian.second_order_tb import _electron_configuration_valence_counts as _val
        elem_to_shells = {}
        for sh in basis.shells:
            elem_to_shells.setdefault(int(sh.element), set()).add(sh.l)
        nelec = 0
        for z in numbers.tolist():
            present = elem_to_shells.get(int(z), set())
            val = _val(int(z))
            nelec += int(round(sum(v for l, v in val.items() if l in present)))
        res = compute_total_energy(
            numbers, pos, basis, g, schema, eeq,
            total_charge=0.0, nelec=nelec,
            second_order=True, shell_second_order=True,
            third_order=True, fourth_order=True,
        )
        # E3 should be finite (or exactly zero if parameters null it), but not NaN
        if res.E3 is not None:
            assert torch.isfinite(res.E3)

