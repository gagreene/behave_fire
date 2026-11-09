"""Wheel and CI evidence use the current native float64 direction contract."""
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.rust


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1]/'tools'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direction_evidence_runs_without_retired_calibration_and_checks_widened_reference(capsys):
    _tool('probe_direction_oracle').main()
    report = json.loads(capsys.readouterr().out)
    assert report['native_physical_dtype'] == 'float64'
    for case in report['cases']:
        for orientation in ('RelativeToUpslope', 'RelativeToNorth'):
            assert case[orientation]['strict_pass']


@pytest.mark.parametrize('half', [False, True])
def test_wheel_direction_smoke_checks_native_against_same_widened_inputs(half):
    from behave_fire.components import surface as python
    from behave_fire.rust import surface as native

    smoke = _tool('smoke_test_wheel')
    smoke._direction(native, python, half)
    smoke._direction_reversed(native, python)
