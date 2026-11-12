"""Wheel and CI evidence use the current native float64 direction contract."""
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
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


def test_direction_layout_trace_uses_independent_reference_without_retired_normalizer():
    probe = _tool('probe_direction_layout')
    x = np.array([-1., 2., -3.], np.float32)[::-1]
    y = np.array([65534., -2., 4.], np.float32)[::-1]
    row = probe._record('reversed', x, y, math.atan2)
    assert row['native_strict_pass']
    assert row['native_physical_dtype'] == 'float64'
    assert row['iterator_chunks']


@pytest.mark.parametrize('half', [False, True])
def test_wheel_direction_smoke_checks_native_against_same_widened_inputs(half):
    from behave_fire.components import surface as python
    from behave_fire.rust import surface as native

    smoke = _tool('smoke_test_wheel')
    smoke._direction(native, python, half)
    smoke._direction_reversed(native, python)


def test_wheel_identity_report_requires_only_live_native_metadata():
    report = _tool('smoke_test_wheel')._native_report()
    assert report['native_physical_dtype'] == 'float64'
    assert report['binding_api_version'] == 4


def test_wheel_smoke_checks_normalization_missing_fields_stages_and_owned_buffers():
    from behave_fire import rust

    result = _tool('smoke_test_wheel')._native_contract(rust)
    assert result == {'float64_contract': True, 'staged_injection': True, 'owned_outputs': True}
