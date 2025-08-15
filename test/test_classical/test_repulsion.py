import pytest, torch
from gxtb.params.loader import load_gxtb_params
from gxtb.params.schema import load_schema, map_repulsion_params, map_cn_params
from gxtb.classical.repulsion import RepulsionParams, repulsion_energy
from gxtb.charges.eeq import compute_eeq_charges
from gxtb.params.loader import load_eeq_params

@pytest.mark.skipif(not (torch.cuda.is_available() or True), reason="Environment not restrictive")
def test_repulsion_symmetry_and_decay():
    # Simple H2 system at two distances
    gparams = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    rep_map = map_repulsion_params(gparams, schema)
    cn_map = map_cn_params(gparams, schema)
    eeq = load_eeq_params('parameters/eeq')

    numbers = torch.tensor([1,1], dtype=torch.long)
    def build_rep():
        return RepulsionParams(
            z_eff0=rep_map['z_eff0'], alpha0=rep_map['alpha0'], kq=rep_map['kq'], kq2=rep_map['kq2'],
            kcn_elem=rep_map['kcn'], r0=rep_map['r0'], kpen1_hhe=float(rep_map['kpen1_hhe']),
            kpen1_rest=float(rep_map['kpen1_rest']), kpen2=float(rep_map['kpen2']), kpen3=float(rep_map['kpen3']),
            kpen4=float(rep_map['kpen4']), kexp=float(rep_map['kexp']), r_cov=cn_map['r_cov'], k_cn=float(cn_map['k_cn'])
        )
    rep = build_rep()

    # Near distance
    r1 = 0.74
    pos1 = torch.tensor([[0.0,0.0,0.0],[0.0,0.0,r1]])
    q1 = compute_eeq_charges(numbers, pos1, eeq, total_charge=0.0)
    e1 = repulsion_energy(pos1, numbers, rep, q1)

    # Far distance (should decay strongly toward 0)
    r2 = 8.0
    pos2 = torch.tensor([[0.0,0.0,0.0],[0.0,0.0,r2]])
    q2 = compute_eeq_charges(numbers, pos2, eeq, total_charge=0.0)
    e2 = repulsion_energy(pos2, numbers, rep, q2)

    assert e1 > e2
    assert abs(e2.item()) < abs(e1.item())
    # Symmetry: swapping atoms leaves energy unchanged
    e1_swap = repulsion_energy(pos1.flip(0), numbers.flip(0), rep, q1.flip(0))
    assert pytest.approx(e1.item(), rel=1e-12) == e1_swap.item()
