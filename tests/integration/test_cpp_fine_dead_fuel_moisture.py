"""Verify the bounded fine dead fuel moisture measurement contract."""

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
moisture = importlib.import_module('cpp_fine_dead_fuel_moisture')


# ---- guarded measurements tests ----

def test_moisture_loads_the_three_pinned_results():
    rows = moisture.load_observations(ROOT)

    assert [row['output']['moisture_fraction'] for row in rows] == [0.14, 0.14 + 0.01, 0.12 + 0.06]


@pytest.mark.parametrize(('field', 'replacement'), [
    ('dry_bulb_i', '6'), ('rh_i', '21'), ('slope_i', '2'), ('aspect_i', '4'),
    ('shading_i', '2'), ('month_i', '3'), ('elev_i', '3'), ('time_i', '6'),
    ('month_i', '-0'), ('rh_i', 'true'),
])
def test_moisture_rejects_unapproved_indices(tmp_path, field, replacement):
    directory = tmp_path / moisture.DATASET
    directory.mkdir(parents=True)
    path = ROOT / moisture.DATASET / 'cases.jsonl'
    original = moisture.load_cases(ROOT)[0][field]
    mutated = path.read_text(encoding='utf-8').replace(f'"{field}":{original}', f'"{field}":{replacement}', 1)
    (directory / path.name).write_text(mutated, encoding='utf-8')

    with pytest.raises(ValueError, match='approved cases'):
        moisture.load_cases(tmp_path)


# ---- measurement receipt tests ----

def test_moisture_requires_a_measurement_only_receipt():
    policy = moisture.MEASUREMENT_POLICY
    receipt = dict(artifacts=moisture.artifacts(ROOT), comparisons=48, nonmatching=0, max_abs={
        'moisture_fraction': 0.0,
    })

    moisture.validate_measurement(ROOT, policy, None)
    moisture.validate_measurement(ROOT, policy, {'fine_dead_fuel_moisture': receipt})
    with pytest.raises(ValueError, match='measurement-only'):
        moisture.validate_measurement(ROOT, dict(policy, approved=True), {'fine_dead_fuel_moisture': receipt})
    with pytest.raises(ValueError, match='comparison statistics'):
        moisture.validate_measurement(ROOT, policy, {'fine_dead_fuel_moisture': dict(receipt, nonmatching=1)})


def test_moisture_summarizes_the_complete_report_matrix(tmp_path):
    cpp_rows = moisture.load_observations(ROOT)
    port_rows = [dict(id=row['id'], output=row['output']) for row in cpp_rows]
    signature = moisture.artifacts(ROOT)
    reports = {}
    for compiler in ('GNU', 'MSVC'):
        reports[f'cpp-{compiler}'] = dict(kind='cpp', compiler={'id': compiler},
                                          protocol={moisture.OPERATION: moisture.GUARDS},
                                          **{moisture.OPERATION: dict(artifacts=signature, observations=cpp_rows)})
    for platform in ('linux-x86_64', 'linux-aarch64', 'linux-musl-x86_64', 'macos-x86_64', 'macos-arm64',
                     'windows-x86_64', 'windows-arm64'):
        reports[f'ports-{platform}'] = dict(kind='ports', platform_key=platform, **{
            moisture.OPERATION: dict(artifacts=signature, observations={'python': port_rows, 'rust': port_rows}),
        })
    paths = []
    record = {'reports': {}}
    for name, report in reports.items():
        raw = json.dumps(report).encode()
        path = tmp_path / f'{name}.json'
        path.write_bytes(raw)
        paths.append(path)
        record['reports'][name] = {'sha256': hashlib.sha256(raw).hexdigest()}

    receipt = moisture.measurement_evidence(ROOT, paths, record)
    assert receipt['comparisons'] == 48
    assert receipt['nonmatching'] == 0
    paths[-1].write_bytes(paths[-1].read_bytes() + b' ')
    with pytest.raises(ValueError, match='source-qualified evidence'):
        moisture.measurement_evidence(ROOT, paths, record)


# ---- port measurements tests ----

@pytest.mark.parametrize('backend', ['python', pytest.param('rust', marks=pytest.mark.rust)])
def test_moisture_measures_public_port_calls(backend):
    module = importlib.import_module(f'behave_fire.{backend}').fine_dead_fuel_moisture_tool
    calculated = moisture.measure_ports(ROOT, (backend,))[backend]
    cases = [
        ('exposed-low-slope', (0, 20, 0, 0, 0, 0, 1, 3)),
        ('exposed-high-slope', (0, 20, 1, 0, 0, 0, 1, 3)),
        ('shaded-upper-indices', (5, 20, 1, 3, 1, 2, 2, 5)),
    ]
    expected = [dict(id=case_id, output={'moisture_fraction': np.asarray(
        module.calculate_fine_dead_fuel_moisture(*indices)).item()}) for case_id, indices in cases]

    assert calculated == expected
