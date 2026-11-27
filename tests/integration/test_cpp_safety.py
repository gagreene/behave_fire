"""Check the guarded standalone Safety measurements."""

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.cpp_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
safety = importlib.import_module('cpp_safety')


def _write_report(path, report, record):
    raw = json.dumps(report).encode('utf-8')
    path.write_bytes(raw)
    record['reports'][path.stem] = {'sha256': hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def _written_reports(tmp_path):
    cpp_rows = safety.load_observations(ROOT)
    port_rows = [dict(id=row['id'], output=row['output']) for row in cpp_rows]
    artifacts = safety.artifacts(ROOT)
    reports = {
        f'cpp-{compiler}': dict(kind='cpp', compiler={'id': compiler}, protocol={'safety': safety.GUARDS},
                                safety=dict(artifacts=artifacts, observations=cpp_rows))
        for compiler in ('GNU', 'MSVC')
    }
    for platform in ('linux-x86_64', 'linux-aarch64', 'linux-musl-x86_64', 'macos-x86_64', 'macos-arm64',
                     'windows-x86_64', 'windows-arm64'):
        reports[f'ports-{platform}'] = dict(kind='ports', platform_key=platform,
                                           safety=dict(artifacts=artifacts, observations={
                                               'python': port_rows, 'rust': port_rows}))
    record = {'reports': {}}
    paths = {name: tmp_path / f'{name}.json' for name in reports}
    for name, report in reports.items():
        _write_report(paths[name], report, record)
    return paths, record


# ---- guarded measurements tests ----

def test_safety_loads_all_eight_pinned_observations():
    observations = safety.load_observations(ROOT)

    assert [row['output']['radius_ft'] for row in observations] == [0, 3, 4, 5, 4, 7, 8, 9]
    assert all(row['upstream_entries'] == row['output_entries'] == 1 for row in observations)


@pytest.mark.parametrize('invalid_personnel', ['-0', '0.0', 'false', '0.5', '2'])
def test_safety_rejects_unapproved_occupancy(tmp_path, invalid_personnel):
    directory = tmp_path / safety.DATASET
    directory.mkdir(parents=True)
    source = (ROOT / safety.DATASET / 'cases.jsonl').read_text(encoding='utf-8')
    (directory / 'cases.jsonl').write_text(
        source.replace('"personnel":0', f'"personnel":{invalid_personnel}', 1), encoding='utf-8')

    with pytest.raises(ValueError, match='approved cases'):
        safety.load_cases(tmp_path)


# ---- measurement receipt tests ----

@pytest.mark.parametrize(('nonmatching', 'maxima'), [
    (1, dict(area_sqft=1.0, radius_ft=1.0, separation_ft=1.0)),
    (97, dict(area_sqft=0.0, radius_ft=0.0, separation_ft=0.0)),
])
def test_safety_rejects_inconsistent_receipt_statistics(nonmatching, maxima):
    receipt = dict(artifacts=safety.artifacts(ROOT), comparisons=384, nonmatching=nonmatching, max_abs=maxima)

    with pytest.raises(ValueError, match='Inconsistent Safety comparison statistics'):
        safety.validate_measurement(ROOT, safety.MEASUREMENT_POLICY, {'safety': receipt})


@pytest.mark.parametrize(('defect', 'message'), [
    ('missing-report', 'Incomplete Safety measurement evidence'),
    ('tampered-report', 'differs from the source-qualified evidence'),
    ('invalid-guard', 'Missing Safety entry-guard evidence'),
    ('reordered-rows', 'Safety case/result identity mismatch'),
    ('missing-row', 'Incomplete Safety report case identities'),
])
def test_safety_rejects_invalid_matrix_evidence(_written_reports, defect, message):
    paths, record = _written_reports
    cpp = paths['cpp-MSVC']
    ports = paths['ports-windows-x86_64']
    if defect == 'missing-report':
        paths.pop('cpp-MSVC')
    elif defect == 'tampered-report':
        cpp.write_bytes(cpp.read_bytes() + b' ')
    elif defect == 'invalid-guard':
        report = json.loads(cpp.read_bytes())
        report['protocol']['safety']['negative_cases'] = True
        _write_report(cpp, report, record)
    elif defect == 'reordered-rows':
        report = json.loads(cpp.read_bytes())
        report['safety']['observations'].reverse()
        _write_report(cpp, report, record)
    else:
        report = json.loads(ports.read_bytes())
        report['safety']['observations']['rust'].pop()
        _write_report(ports, report, record)

    with pytest.raises(ValueError, match=message):
        safety.measurement_evidence(ROOT, paths.values(), record)


def test_safety_requires_receipt_and_measurement_only_policy():
    policy = safety.MEASUREMENT_POLICY
    receipt = dict(artifacts=safety.artifacts(ROOT), comparisons=384, nonmatching=0,
                   max_abs={field: 0.0 for field in safety.FIELDS})

    safety.validate_measurement(ROOT, policy, None)
    safety.validate_measurement(ROOT, policy, {'safety': receipt})
    safety.validate_measurement(ROOT, policy, {'safety': dict(receipt, nonmatching=96)})
    with pytest.raises(ValueError, match='measurement receipt'):
        safety.validate_measurement(ROOT, policy, {})
    with pytest.raises(ValueError, match='measurement-only'):
        safety.validate_measurement(ROOT, dict(policy, approved=True), {'safety': receipt})


def test_safety_summarizes_signed_zero_and_field_differences(_written_reports):
    paths, record = _written_reports
    cpp = paths['cpp-MSVC']
    report = json.loads(cpp.read_bytes())
    report['safety']['observations'][0]['output'] = dict(area_sqft=-0.0, radius_ft=2.0, separation_ft=3.0)
    _write_report(cpp, report, record)

    receipt = safety.measurement_evidence(ROOT, paths.values(), record)

    assert receipt['comparisons'] == 384
    assert receipt['nonmatching'] == 3
    assert receipt['max_abs'] == dict(area_sqft=0.0, radius_ft=2.0, separation_ft=3.0)
    safety.validate_measurement(ROOT, safety.MEASUREMENT_POLICY, {'safety': receipt})


# ---- port measurements tests ----

@pytest.mark.parametrize('backend', ['python', pytest.param('rust', marks=pytest.mark.rust)])
def test_safety_measures_all_fields_in_case_order(backend):
    calculator = importlib.import_module(f'behave_fire.{backend}').safety.calculate_safety_zone
    result = safety.measure_ports(ROOT, (backend,))
    rows = result[backend]

    assert list(result) == [backend]
    for row, case in zip(rows, safety.load_cases(ROOT), strict=True):
        expected = calculator(case['flame_ft'], case['personnel'], 9 * math.pi, case['equipment'], 16 * math.pi)
        assert row['id'] == case['id']
        assert row['output'] == dict(area_sqft=np.asarray(expected['area']).item(),
                                     radius_ft=np.asarray(expected['radius']).item(),
                                     separation_ft=np.asarray(expected['separation_distance']).item())
        assert all(type(value) is float for value in row['output'].values())
