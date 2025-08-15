import torch
from gxtb.params.loader import load_gxtb_params
from gxtb.params.schema import load_schema, map_hubbard_params


def test_hubbard_mapping_exists():
    params = load_gxtb_params('parameters/gxtb')
    schema = load_schema('parameters/gxtb.schema.toml')
    hub = map_hubbard_params(params, schema)
    assert 'gamma' in hub and 'gamma3' in hub
    assert hub['gamma'].ndim == 1 and hub['gamma3'].ndim == 1
    # Heuristic range check (not all zeros, reasonable magnitude)
    nonzero = hub['gamma'][hub['gamma'] != 0.0]
    if nonzero.numel() > 0:
        assert torch.all(nonzero > -1.0) and torch.all(nonzero < 5.0)
