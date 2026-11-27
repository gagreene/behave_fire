"""Verify the guarded bidirectional Units measurement contract."""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpp_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
units = importlib.import_module('cpp_units')


# ---- guarded measurements tests ----

def test_units_loads_the_pinned_bidirectional_case():
    row, = units.load_observations(ROOT)

    assert row['id'] == 'temperature-minus10c-14f'
    assert row['output'] == {'celsius': -10, 'fahrenheit': 14}


@pytest.mark.parametrize(('field', 'original', 'replacement'), [
    ('input_celsius', '-10', '-10.0'),
    ('input_celsius', '-10', '"-10"'),
    ('input_celsius', '-10', '-40'),
    ('input_fahrenheit', '14', '14.0'),
    ('input_fahrenheit', '14', 'true'),
])
def test_units_rejects_unapproved_inputs(tmp_path, field, original, replacement):
    directory = tmp_path / units.DATASET
    directory.mkdir(parents=True)
    cases = (ROOT / units.DATASET / 'cases.jsonl').read_text(encoding='utf-8')
    (directory / 'cases.jsonl').write_text(
        cases.replace(f'"{field}":{original}', f'"{field}":{replacement}', 1), encoding='utf-8')

    with pytest.raises(ValueError, match='approved cases'):
        units.load_cases(tmp_path)


# ---- measurement receipt tests ----

def test_units_requires_nonnegative_measurement_only_maxima():
    policy = units.MEASUREMENT_POLICY
    receipt = dict(artifacts=units.artifacts(ROOT), comparisons=32, nonmatching=0, max_abs={
        'celsius': 0.0, 'fahrenheit': 0.0,
    })

    units.validate_measurement(ROOT, policy, None)
    units.validate_measurement(ROOT, policy, {units.OPERATION: receipt})
    with pytest.raises(ValueError, match='measurement-only'):
        units.validate_measurement(ROOT, dict(policy, approved=True), {units.OPERATION: receipt})
    with pytest.raises(ValueError, match='Invalid Units comparison maxima'):
        units.validate_measurement(ROOT, policy, {units.OPERATION: dict(
            receipt, max_abs={'celsius': -1.0, 'fahrenheit': 0.0})})


def test_units_summarizes_signed_temperatures(tmp_path):
    cpp_rows = units.load_observations(ROOT)
    port_rows = [dict(id=row['id'], output=row['output']) for row in cpp_rows]
    artifacts = units.artifacts(ROOT)
    reports = {}
    for compiler in ('GNU', 'MSVC'):
        reports[f'cpp-{compiler}'] = dict(kind='cpp', compiler={'id': compiler},
                                          protocol={units.OPERATION: units.GUARDS},
                                          **{units.OPERATION: dict(artifacts=artifacts, observations=cpp_rows)})
    for platform in ('linux-x86_64', 'linux-aarch64', 'linux-musl-x86_64', 'macos-x86_64', 'macos-arm64',
                     'windows-x86_64', 'windows-arm64'):
        reports[f'ports-{platform}'] = dict(kind='ports', platform_key=platform, **{
            units.OPERATION: dict(artifacts=artifacts, observations={'python': port_rows, 'rust': port_rows}),
        })
    paths = []
    record = {'reports': {}}
    for name, report in reports.items():
        raw = json.dumps(report).encode()
        path = tmp_path / f'{name}.json'
        path.write_bytes(raw)
        paths.append(path)
        record['reports'][name] = {'sha256': hashlib.sha256(raw).hexdigest()}

    receipt = units.measurement_evidence(ROOT, paths, record)
    assert receipt['comparisons'] == 32
    assert receipt['nonmatching'] == 0
    paths[-1].write_bytes(paths[-1].read_bytes() + b' ')
    with pytest.raises(ValueError, match='source-qualified evidence'):
        units.measurement_evidence(ROOT, paths, record)


# ---- port measurements tests ----

@pytest.mark.parametrize('backend', ['python', pytest.param('rust', marks=pytest.mark.rust)])
def test_units_measures_both_public_conversion_calls(backend):
    converter = importlib.import_module(f'behave_fire.{backend}').behave_units.TemperatureUnits
    celsius = converter.TemperatureUnitsEnum.Celsius
    measured, = units.measure_ports(ROOT, (backend,))[backend]

    assert measured == dict(id='temperature-minus10c-14f', output={
        'celsius': np.asarray(converter.fromBaseUnits(14, celsius)).item(),
        'fahrenheit': np.asarray(converter.toBaseUnits(-10, celsius)).item(),
    })
