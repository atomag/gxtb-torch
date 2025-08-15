import os
import pytest
import torch

# Try to import tad-dftd4 to compare against the original implementation
try:
    import sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    TAD = os.path.join(ROOT, 'tad-dftd4', 'src')
    if TAD not in sys.path:
        sys.path.insert(0, TAD)
    import tad_dftd4 as d4  # type: ignore
    from tad_dftd4.disp import dftd4 as tad_d4_energy  # type: ignore
    from tad_dftd4.damping import Param as TadParam  # type: ignore
    TAD_OK = True
except Exception:
    TAD_OK = False


@pytest.mark.skipif(not TAD_OK, reason="tad-dftd4 not importable; skip parity test")
def test_d4_energy_matches_tad_for_water_zero_charges():
    # System: water
    numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.000000, 0.000000, 0.000000],
            [0.958, 0.000000, 0.000000],
            [-0.239, 0.927, 0.000000],
        ],
        dtype=torch.float64,
    )

    # Our method loader from TOML
    from gxtb.classical.dispersion import load_d4_method, d4_energy

    # Use our TOML parameters but build tad-dftd4 Param dataclass
    method = load_d4_method('parameters/dftd4parameters.toml', functional=None, variant='bj-eeq-atm')
    param = TadParam(
        s6=positions.new_tensor(method.s6),
        s8=positions.new_tensor(method.s8),
        s9=positions.new_tensor(method.s9),
        a1=positions.new_tensor(method.a1),
        a2=positions.new_tensor(method.a2),
    )

    # Build reference dict from tad-dftd4 internals
    from tad_dftd4.reference.d4 import params as refp  # type: ignore
    from tad_dftd4.data.r4r2 import R4R2  # type: ignore
    from tad_dftd4.data.zeff import ZEFF  # type: ignore
    from tad_dftd4.data.hardness import GAM  # type: ignore
    device = positions.device
    dtype = positions.dtype
    ref = {
        'refsys': refp.refsys.to(device=device),
        'refascale': refp.refascale.to(device=device, dtype=dtype),
        'refscount': refp.refscount.to(device=device, dtype=dtype),
        'secscale': refp.secscale.to(device=device, dtype=dtype),
        'secalpha': refp.secalpha.to(device=device, dtype=dtype),
        'refalpha': refp.refalpha.to(device=device, dtype=dtype),
        'refcovcn': refp.refcovcn.to(device=device, dtype=dtype),
        'refc': refp.refc.to(device=device, dtype=dtype),
        'zeff': ZEFF(device=device).to(dtype),
        'gam': GAM(device=device, dtype=dtype),
        'r4r2': R4R2(device=device, dtype=dtype),
        'clsq': __import__('tad_dftd4.reference.d4.charge_eeq', fromlist=['clsq']).clsq.to(device=device, dtype=dtype),
        'clsh': __import__('tad_dftd4.reference.d4.charge_eeq', fromlist=['clsh']).clsh.to(device=device, dtype=dtype),
    }

    # Provide CN matching tad-dftd4 default
    from tad_dftd4.cutoff import Cutoff  # type: ignore
    from tad_mctc.ncoord import cn_d4  # type: ignore
    cut = Cutoff(device=positions.device, dtype=positions.dtype)
    ref['cn'] = cn_d4(numbers, positions, cutoff=cut.cn)

    # Our D4 method loaded above as `method`

    # Zero charges vector to avoid depending on external EEQ in parity test
    q = torch.zeros(numbers.shape[0], dtype=positions.dtype, device=positions.device)

    # Our implementation
    E_ours = d4_energy(numbers, positions, q, method, ref)

    # tad-dftd4 reference energy (atom-resolved sum)
    charge = torch.tensor(0.0, dtype=positions.dtype, device=positions.device)
    E_atoms = tad_d4_energy(numbers, positions, charge, param, q=q)
    E_ref = E_atoms.sum()

    # Compare with modest tolerance (integration and constants should match closely)
    assert torch.allclose(E_ours, E_ref, rtol=1e-6, atol=1e-8), f"ours={E_ours.item()} ref={E_ref.item()}"
