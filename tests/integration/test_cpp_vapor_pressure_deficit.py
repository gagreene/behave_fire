"""Verify the bounded vapor pressure deficit measurement contract."""

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from behave_fire.components.behave_units import FractionUnits, PressureUnits, TemperatureUnits

pytestmark = pytest.mark.cpp_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
vpd = importlib.import_module('cpp_vapor_pressure_deficit')


# ---- guarded measurements tests ----

def test_vpd_loads_both_pinned_observations():
    freezing, warm = vpd.load_observations(ROOT)

    assert freezing['id'] == 'freezing-dry'
    assert freezing['output'] == {'actual_vp': 0, 'saturated_vp': 0.611, 'vpd': 0.611}
    assert warm['id'] == 'warm-half-humidity'
    assert warm['output']['actual_vp'] == warm['output']['vpd'] == 1.1694678421549667


@pytest.mark.parametrize(('field', 'old', 'replacement'), [
    ('temperature_k', '273.15', '35.85'),
    ('temperature_k', '273.15', '"273.15"'),
    ('rh_pct', '0', '-0'),
    ('rh_pct', '0', '0.0'),
    ('rh_pct', '0', 'true'),
    ('rh_pct', '0', '101'),
])
def test_vpd_rejects_unapproved_inputs(tmp_path, field, old, replacement):
    directory = tmp_path / vpd.DATASET
    directory.mkdir(parents=True)
    source = (ROOT / vpd.DATASET / 'cases.jsonl').read_text(encoding='utf-8')
    (directory / 'cases.jsonl').write_text(
        source.replace(f'"{field}":{old}', f'"{field}":{replacement}', 1), encoding='utf-8')

    with pytest.raises(ValueError, match='approved cases'):
        vpd.load_cases(tmp_path)


# ---- measurement receipt tests ----

def test_vpd_requires_a_measurement_only_receipt():
    policy = vpd.MEASUREMENT_POLICY
    maxima = dict.fromkeys(vpd.FIELDS, 0.0)
    receipt = dict(artifacts=vpd.artifacts(ROOT), comparisons=96, nonmatching=0, max_abs=maxima)

    vpd.validate_measurement(ROOT, policy, None)
    vpd.validate_measurement(ROOT, policy, {vpd.OPERATION: receipt})
    with pytest.raises(ValueError, match='measurement-only'):
        vpd.validate_measurement(ROOT, dict(policy, approved=True), {vpd.OPERATION: receipt})
    with pytest.raises(ValueError, match='comparison statistics'):
        vpd.validate_measurement(ROOT, policy, {vpd.OPERATION: dict(receipt, nonmatching=17)})


def test_vpd_summarizes_warm_differences_without_approving_them(tmp_path):
    cpp_rows = vpd.load_observations(ROOT)
    exact = [dict(id=row['id'], output=dict(row['output'])) for row in cpp_rows]
    rounded = [dict(id=row['id'], output=dict(row['output'])) for row in cpp_rows]
    warm = rounded[1]['output']
    for field in vpd.FIELDS:
        warm[field] = math.nextafter(warm[field], math.inf)
    signature = vpd.artifacts(ROOT)
    reports = {}
    for compiler in ('GNU', 'MSVC'):
        reports[f'cpp-{compiler}'] = dict(kind='cpp', compiler={'id': compiler},
                                          protocol={vpd.OPERATION: vpd.GUARDS},
                                          **{vpd.OPERATION: dict(artifacts=signature, observations=cpp_rows)})
    for platform in ('linux-x86_64', 'linux-aarch64', 'linux-musl-x86_64', 'macos-x86_64', 'macos-arm64',
                     'windows-x86_64', 'windows-arm64'):
        reports[f'ports-{platform}'] = dict(kind='ports', platform_key=platform, **{
            vpd.OPERATION: dict(artifacts=signature, observations={'python': rounded, 'rust': exact}),
        })
    paths = []
    record = {'reports': {}}
    for name, report in reports.items():
        raw = json.dumps(report).encode()
        path = tmp_path / f'{name}.json'
        path.write_bytes(raw)
        paths.append(path)
        record['reports'][name] = {'sha256': hashlib.sha256(raw).hexdigest()}

    receipt = vpd.measurement_evidence(ROOT, paths, record)
    assert receipt['comparisons'] == 96
    assert receipt['nonmatching'] == 21
    assert receipt['max_abs'] == {field: warm[field] - cpp_rows[1]['output'][field] for field in vpd.FIELDS}
    vpd.validate_measurement(ROOT, vpd.MEASUREMENT_POLICY, {vpd.OPERATION: receipt})
    paths[-1].write_bytes(paths[-1].read_bytes() + b' ')
    with pytest.raises(ValueError, match='source-qualified evidence'):
        vpd.measurement_evidence(ROOT, paths, record)


# ---- port measurements tests ----

@pytest.mark.parametrize('backend', ['python', pytest.param('rust', marks=pytest.mark.rust)])
def test_vpd_measures_public_port_calls(backend):
    module = importlib.import_module(f'behave_fire.{backend}').vapor_pressure_deficit_calculator
    measured = vpd.measure_ports(ROOT, (backend,))[backend]
    cases = [('freezing-dry', 273.15, 0), ('warm-half-humidity', 293.15, 50)]
    for row, (name, temperature, humidity) in zip(measured, cases, strict=True):
        expected = module.calculate_vpd(temperature, TemperatureUnits.TemperatureUnitsEnum.Kelvin,
                                        humidity, FractionUnits.FractionUnitsEnum.Percent,
                                        PressureUnits.PressureUnitsEnum.KiloPascal)
        assert row == dict(id=name, output={field: np.asarray(expected[field]).item() for field in vpd.FIELDS})
