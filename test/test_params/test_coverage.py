import pathlib

from gxtb.params.loader import validate_parameter_coverage


def test_parameter_coverage_common_elements_exist():
    param_dir = pathlib.Path(__file__).resolve().parents[2] / 'parameters'
    info = validate_parameter_coverage(param_dir)
    assert len(info['gxtb']) > 0
    assert len(info['eeq']) > 0
    assert len(info['basisq']) > 0
    assert len(info['common']) > 0
