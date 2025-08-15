import os
import pytest
import torch


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), '..', '..', 'tad-dftd4', 'src')),
    reason="tad-dftd4 not present; skip full element parity test",
)
def test_d4_homonuclear_diatomics_parity_all_elements():
    # Import tad-dftd4 only inside the test
    import sys
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    TAD = os.path.join(ROOT, 'tad-dftd4', 'src')
    if TAD not in sys.path:
        sys.path.insert(0, TAD)
    from tad_dftd4.disp import dftd4 as tad_d4_energy  # type: ignore
    from tad_dftd4.damping import Param as TadParam  # type: ignore
    from tad_dftd4.cutoff import Cutoff  # type: ignore
    from tad_mctc.ncoord import cn_d4  # type: ignore

    from gxtb.classical.dispersion import load_d4_method, d4_energy
    from gxtb.params.loader import load_d4_reference_toml

    device = torch.device('cpu')
    dtype = torch.float64

    # Load our method and TOML reference (full table)
    method = load_d4_method('parameters/dftd4parameters.toml', functional=None, variant='bj-eeq-atm')
    ref = load_d4_reference_toml('parameters/d4_reference.toml', device=device, dtype=dtype)

    # Homonuclear diatomic for each Z = 1..103 at fixed R (Angstrom)
    R = 3.0
    Zmax = int(ref['r4r2'].shape[0] - 1)
    # Sanity: tad-dftd4 ships up to 103 typically
    N = min(Zmax, 103)

    # Build tad Param once
    param = TadParam(
        s6=torch.tensor(float(method.s6), dtype=dtype),
        s8=torch.tensor(float(method.s8), dtype=dtype),
        s9=torch.tensor(float(method.s9), dtype=dtype),
        a1=torch.tensor(float(method.a1), dtype=dtype),
        a2=torch.tensor(float(method.a2), dtype=dtype),
    )

    # Loop all elements
    for Z in range(1, N + 1):
        numbers = torch.tensor([Z, Z], dtype=torch.long, device=device)
        positions = torch.tensor([[0.0, 0.0, 0.0], [R, 0.0, 0.0]], dtype=dtype, device=device)
        # Use tad CN for both implementations
        cut = Cutoff(device=device, dtype=dtype)
        cn = cn_d4(numbers, positions, cutoff=cut.cn)
        # Our reference dict with CN per geometry
        ref_this = dict(ref)
        ref_this['cn'] = cn
        # Zero atomic charges vector (avoid charge response differences)
        q = torch.zeros(2, dtype=dtype, device=device)

        # Our implementation
        E_ours = d4_energy(numbers, positions, q, method, ref_this)
        # tad-dftd4 (atom-resolved sum)
        charge = torch.tensor(0.0, dtype=dtype, device=device)
        E_atoms = tad_d4_energy(numbers, positions, charge, param, q=q)
        E_ref = E_atoms.sum()

        assert torch.allclose(E_ours, E_ref, rtol=1e-6, atol=1e-7), (
            f"Z={Z}: ours={E_ours.item()} ref={E_ref.item()}"
        )

