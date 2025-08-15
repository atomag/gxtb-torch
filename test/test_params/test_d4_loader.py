from pathlib import Path

from gxtb.params.loader import load_d4_parameters, select_d4_params
from gxtb.classical.dispersion import load_d4_method


PARAM_DIR = Path(__file__).resolve().parents[2] / "parameters"


def test_select_d4_params_functional_variant_exact():
    d4 = load_d4_parameters(PARAM_DIR / "dftd4parameters.toml")
    rec = select_d4_params(d4, method="d4", functional="b3lyp", variant="bj-eeq-atm")
    # Spot-check known values from parameters file
    assert abs(float(rec["a1"]) - 0.40868035) < 1e-8
    assert abs(float(rec["a2"]) - 4.53807137) < 1e-8
    assert abs(float(rec["s8"]) - 2.02929367) < 1e-8
    # s6 defaults to 1.0 if absent for hybrid GGA
    assert float(rec.get("s6", 1.0)) == 1.0


def test_select_d4_params_default_section():
    d4 = load_d4_parameters(PARAM_DIR / "dftd4parameters.toml")
    rec = select_d4_params(d4, method="d4", functional=None, variant=None)
    # default.variant is bj-eeq-atm with s6=1.0, s9=1.0 in our file
    assert float(rec["s6"]) == 1.0
    assert float(rec["s9"]) == 1.0


def test_load_d4_method_wrapper():
    # Ensure wrapper produces the same fields as selector
    d4 = load_d4_parameters(PARAM_DIR / "dftd4parameters.toml")
    rec = select_d4_params(d4, method="d4", functional="b3lyp", variant="bj-eeq-atm")
    m = load_d4_method(str(PARAM_DIR / "dftd4parameters.toml"), variant="bj-eeq-atm", functional="b3lyp")
    assert abs(m.a1 - float(rec["a1"])) < 1e-12
    assert abs(m.a2 - float(rec["a2"])) < 1e-12
    assert abs(m.s8 - float(rec["s8"])) < 1e-12
