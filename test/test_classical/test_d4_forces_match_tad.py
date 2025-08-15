import os
import pytest
import torch

# Try to import tad-dftd4 to compare gradients (forces) in tests only
try:
    import sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    TAD = os.path.join(ROOT, 'tad-dftd4', 'src')
    if TAD not in sys.path:
        sys.path.insert(0, TAD)
    from tad_dftd4.disp import dftd4 as tad_d4_energy  # type: ignore
    from tad_dftd4.damping import Param as TadParam  # type: ignore
    from tad_dftd4.cutoff import Cutoff  # type: ignore
    from tad_mctc.ncoord import cn_d4  # type: ignore
    TAD_OK = True
except Exception:
    TAD_OK = False


@pytest.mark.skipif(not TAD_OK, reason="tad-dftd4 not importable; skip parity test")
def test_d4_forces_match_tad_for_water_zero_charges():
    # System: water (in Angstrom)
    numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.000000, 0.000000, 0.000000],
            [0.958, 0.000000, 0.000000],
            [-0.239, 0.927, 0.000000],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    # Our method loader from TOML
    from gxtb.classical.dispersion import load_d4_method, d4_energy

    method = load_d4_method('parameters/dftd4parameters.toml', functional=None, variant='bj-eeq-atm')
    param = TadParam(
        s6=positions.new_tensor(method.s6),
        s8=positions.new_tensor(method.s8),
        s9=positions.new_tensor(method.s9),
        a1=positions.new_tensor(method.a1),
        a2=positions.new_tensor(method.a2),
    )

    # Reference dict from embedded TOML subset
    from gxtb.params.loader import load_d4_reference_toml
    device = positions.device
    dtype = positions.dtype
    ref = load_d4_reference_toml('parameters/d4_reference.toml', device=device, dtype=dtype)

    # Use tad-dftd4 CN for both implementations to ensure parity and differentiability
    cut = Cutoff(device=positions.device, dtype=positions.dtype)
    ref['cn'] = cn_d4(numbers, positions, cutoff=cut.cn)

    # Zero charges vector avoids charge response complications
    q = torch.zeros(numbers.shape[0], dtype=positions.dtype, device=positions.device)

    # Ours: energy and gradient
    E_ours = d4_energy(numbers, positions, q, method, ref)
    (g_ours,) = torch.autograd.grad(E_ours, positions, create_graph=False)

    # tad-dftd4: energy and gradient (sum over atoms)
    charge = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    E_atoms = tad_d4_energy(numbers, positions, charge, param, q=q)
    E_ref = E_atoms.sum()
    (g_ref,) = torch.autograd.grad(E_ref, positions, create_graph=False)

    # Forces are -gradients; compare components directly
    assert torch.allclose(
        -g_ours, -g_ref, rtol=1e-6, atol=1e-7
    ), f"force mismatch\nours=\n{(-g_ours).detach().cpu().numpy()}\nref=\n{(-g_ref).detach().cpu().numpy()}"
