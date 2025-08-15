import torch

from gxtb.params.loader import load_d4_reference_toml


def test_load_d4_reference_toml_subset_shapes():
    ref = load_d4_reference_toml('parameters/d4_reference.toml', device=torch.device('cpu'), dtype=torch.float64)
    # Required keys exist
    for k in ('secscale','secalpha','refsys','refascale','refscount','refalpha','refcovcn','refc','clsq','clsh','r4r2','zeff','gam','z_supported'):
        assert k in ref, f"missing {k}"
    # Sec arrays have consistent dimensions
    assert ref['secscale'].ndim == 1
    assert ref['secalpha'].ndim == 2
    # Element arrays padded (Z,R,*)
    assert ref['refsys'].ndim == 2 and ref['refalpha'].ndim == 3
    # Supported Z includes 1 and 8 for test subset
    zs = set(ref['z_supported'].tolist())
    assert 1 in zs and 8 in zs

