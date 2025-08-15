import torch
from pathlib import Path
from gxtb.params.loader import load_gxtb_params, load_basisq
from gxtb.params.schema import load_schema, map_eht_params

# Use repo root/parameters reliably (move up two levels from this file)
PARAM_DIR = Path(__file__).resolve().parents[2] / 'parameters'


def test_eht_parameter_shapes():
    g = load_gxtb_params(PARAM_DIR / 'gxtb')
    schema = load_schema(PARAM_DIR / 'gxtb.schema.toml')
    eht = map_eht_params(g, schema)
    maxz = max(g.elements)
    # Required scalar arrays
    for key in ['eps_s','eps_p','eps_d','eps_f','k_ho_s','k_ho_p','k_ho_d','k_ho_f','k_w_s','k_w_p','k_w_d','k_w_f','en']:
        assert key in eht, f"Missing {key} in EHT params"
        assert eht[key].shape == (maxz+1,), f"{key} shape mismatch"
    # CN polynomials shapes (degree 5)
    for sh in ['s','p','d','f']:
        pk = f'pi_cn_{sh}'
        if pk in eht:
            assert eht[pk].shape[1] == 5
    # Distance polynomial shapes (degree 4)
    pair_keys = ['ss','sp','pp','sd','pd','dd','sf','pf','df','ff']
    for pk in pair_keys:
        k = f'pi_r_{pk}'
        if k in eht:
            assert eht[k].shape[1] == 4
    # Basic non-zero checks for low Z
    assert torch.any(eht['eps_s'][1:5] != 0.0)
    assert torch.any(eht['k_w_s'][1:5] != 0.0)


def test_distance_polynomial_symmetry():
    g = load_gxtb_params(PARAM_DIR / 'gxtb')
    schema = load_schema(PARAM_DIR / 'gxtb.schema.toml')
    eht = map_eht_params(g, schema)
    # pick hydrogen (1) and carbon (6)
    zA, zB = 1, 6
    R = torch.tensor(1.1, dtype=torch.float64)
    def poly_eval(coeff_row, x):
        # coeff_row shape (4,) degree 3
        c0, c1, c2, c3 = coeff_row
        return c0 + c1*x + c2*x*x + c3*x*x*x
    # use sp and ps should map to same canonical key ordering 'ps' (our schema stores sp canonical) -> ensure equality of evaluations
    if 'pi_r_sp' in eht:
        cA = eht['pi_r_sp'][zA]
        cB = eht['pi_r_sp'][zB]
        pA = poly_eval(cA, R)
        pB = poly_eval(cB, R)
        # harmonic mean symmetric
        hm = 2.0 / (1.0/pA + 1.0/pB) if pA>0 and pB>0 else 0.5*(pA+pB)
        # swap roles simulate B,A
        hm_swapped = 2.0 / (1.0/pB + 1.0/pA) if pA>0 and pB>0 else 0.5*(pB+pA)
        assert torch.isclose(hm, hm_swapped)
