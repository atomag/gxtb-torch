import torch
import pytest
from gxtb.hamiltonian.distance_tb import distance_factor
from gxtb.hamiltonian.onsite_tb import build_onsite
import torch
import pytest
from gxtb.hamiltonian.distance_tb import distance_factor
from gxtb.hamiltonian.onsite_tb import build_onsite
from gxtb.basis.qvszp import AtomBasis, ShellDef
from gxtb.params.loader import GxTBParameters
from gxtb.params.schema import GxTBSchema

# --- distance_factor tests ---
def test_distance_factor_linear_product():
    # Setup dummy eht with linear pi polynomials: pi_r_ss = coeffs for s-s
    # Here coeffs = [2.0] implies p(R) = 2.0 constant
    eht = {
        'pi_r_ss': torch.tensor([[2.0]], dtype=torch.float64),
    }
    # ZA and ZB indices = 0
    R = torch.tensor(5.0, dtype=torch.float64)
    val = distance_factor('s', 's', 0, 0, R, eht)
    # expected pA = 2, pB = 2, product = 4
    assert torch.isclose(val, torch.tensor(4.0, dtype=torch.float64))

def test_distance_factor_normalized_distance():
    # Test normalization: r_cov provided
    # coeffs = [1.0, 1.0] => p(Rn) = 1 + Rn
    eht = {
        'r_cov': torch.tensor([1.0, 2.0], dtype=torch.float64),
        # Use sorted key 'ps' for s,p combination
        'pi_r_ps': torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float64),
    }
    # ZA=0, ZB=1 => R_cov = 1+2=3
    R = torch.tensor(6.0, dtype=torch.float64)
    val = distance_factor('s', 'p', 0, 1, R, eht)
    # π_s = 1 + Rn = 1 + 6/3 = 3; same for π_p; product = 9
    assert torch.isclose(val, torch.tensor(9.0, dtype=torch.float64))

# --- build_onsite tests ---
class DummyBasis(AtomBasis):
    pass

def build_basis_shell(l, element=1):
    shells = [ShellDef(atom_index=0, element=element, l=l, nprims=1,
                       primitives=((1.0,1.0,0.0),))]
    return DummyBasis(shells=shells, ao_counts=[1], ao_offsets=[0], nao=1)

@pytest.fixture(autouse=True)
def patch_map_and_cn(monkeypatch):
    # Patch map_eht_params to provide h_s, k_ho_s, and pi_cn_s
    from gxtb.hamiltonian.onsite_tb import map_eht_params
    def fake_map(gparams, schema):
        # Provide arrays of length >=2 for element indices 0 and 1
        return {
            'h_s': torch.tensor([0.0, 5.0], dtype=torch.float64),
            'k_ho_s': torch.tensor([0.0, 2.0], dtype=torch.float64),
            'pi_cn_s': torch.tensor([
                [0.0, 0.0],  # dummy for element 0
                [1.0, 0.5],  # for element 1: eps shift = 1 + 0.5*CN
            ], dtype=torch.float64),
        }
    monkeypatch.setattr('gxtb.hamiltonian.onsite_tb.map_eht_params', fake_map)
    # Patch coordination_number to return sanity CN=4
    monkeypatch.setattr('gxtb.hamiltonian.onsite_tb.coordination_number',
                        lambda pos, nums, r_cov, k_cn: torch.tensor([4.0]))
    yield

def test_build_onsite_linear_and_poly(monkeypatch):
    numbers = torch.tensor([1], dtype=torch.int64)
    positions = torch.zeros((1,3), dtype=torch.float64)
    basis = build_basis_shell('s')  # default element=1
    # Provide r_cov and k_cn to trigger CN path
    r_cov = torch.tensor([1.0], dtype=torch.float64)
    k_cn = 1.0
    eps = build_onsite(numbers, positions, basis, None, None, r_cov, k_cn)
    # pi_cn_s: shift = 1 + 0.5*4 = 3; eps = h_s + shift = 5 + 3 = 8
    assert torch.isclose(eps[0], torch.tensor(8.0, dtype=torch.float64))
