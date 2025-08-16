import torch

from gxtb.params.loader import load_gxtb_params
from gxtb.params.schema import load_schema, map_hubbard_params


def test_hubbard_gamma_positive_for_h_and_o():
    p = load_gxtb_params('parameters/gxtb')
    s = load_schema('parameters/gxtb.schema.toml')
    hub = map_hubbard_params(p, s)
    gamma = hub['gamma']
    assert torch.isfinite(gamma[1]) and gamma[1] > 0.0, "H (Z=1) gamma must be > 0 for second-order TB"
    assert torch.isfinite(gamma[8]) and gamma[8] > 0.0, "O (Z=8) gamma must be > 0 for second-order TB"

