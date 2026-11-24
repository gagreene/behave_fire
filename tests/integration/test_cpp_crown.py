"""Validate the six guarded Scott and Reinhardt Crown reference cases."""

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpp_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))


# ---- guarded crown cases tests ----

def test_crown_loads_six_classifications():
    crown = importlib.import_module('cpp_crown')
    rows = crown.load_observations(ROOT)
    assert [(row['id'], row['output']['fire_type']) for row in rows] == [
        ('calm', 0), ('wind5', 0), ('wind15', 2), ('torching', 1), ('active', 3), ('zero-base', 1),
    ]
    assert rows[0]['output']['crown_length_to_width_ratio'] == 1.0
    assert 0 < rows[3]['output']['crown_fraction_burned'] < 1


@pytest.mark.parametrize(('old', 'new'), [
    ('"wind_fpm":0', '"wind_fpm":-0'),
    ('"wind_fpm":440', '"wind_fpm":"440"'),
    ('"wind_fpm":440', '"wind_fpm":440.0'),
    ('"wind_fpm":440', '"wind_fpm":441'),
    ('"canopy_base_ft":5', '"canopy_base_ft":15'),
    ('"crown_ratio":0.8333333333333334', '"crown_ratio":0.83'),
    ('"id":"active"', '"id":"unknown-crown"'),
    ('"id":"active"', '"id":"active","fuel_model":10'),
])
def test_crown_rejects_unreviewed_inputs(tmp_path, old, new):
    crown = importlib.import_module('cpp_crown')
    path = tmp_path / crown.DATASET / 'cases.jsonl'
    path.parent.mkdir(parents=True)
    source = (ROOT / crown.DATASET / 'cases.jsonl').read_text(encoding='utf-8')
    assert old in source
    path.write_text(source.replace(old, new, 1), encoding='utf-8')
    with pytest.raises(ValueError, match='approved cases'):
        crown.load_cases(tmp_path)


# ---- measurement receipt tests ----

def test_crown_requires_measurement_only_policy_and_complete_receipt():
    crown = importlib.import_module('cpp_crown')
    receipt = dict(artifacts=crown.artifacts(ROOT), comparisons=1056, nonmatching=0,
                   max_abs=dict.fromkeys(crown.FIELDS, 0.0))
    qualification = {crown.OPERATION: receipt}
    crown.validate_measurement(ROOT, crown.MEASUREMENT_POLICY, qualification)
    with pytest.raises(ValueError, match='measurement-only'):
        crown.validate_measurement(ROOT, dict(crown.MEASUREMENT_POLICY, approved=True), qualification)
    with pytest.raises(ValueError, match='receipt'):
        crown.validate_measurement(ROOT, crown.MEASUREMENT_POLICY, {crown.OPERATION: dict(receipt, comparisons=1055)})
    receipt['max_abs']['fire_type'] = 1
    with pytest.raises(ValueError, match='classification'):
        crown.validate_measurement(ROOT, crown.MEASUREMENT_POLICY, qualification)


def test_crown_summarizes_only_the_authenticated_observations(tmp_path):
    crown = importlib.import_module('cpp_crown')
    rows = crown.load_observations(ROOT)
    exact = [dict(id=row['id'], output=dict(row['output'])) for row in rows]
    changed = [dict(id=row['id'], output=dict(row['output'])) for row in rows]
    field = 'final_ros_ft_min'
    observed = changed[4]['output'][field] = math.nextafter(exact[4]['output'][field], math.inf)
    signature = crown.artifacts(ROOT)
    reports = {
        f'cpp-{compiler}': dict(kind='cpp', compiler={'id': compiler}, protocol={crown.OPERATION: crown.GUARDS},
                                crown=dict(artifacts=signature, observations=rows))
        for compiler in ('GNU', 'MSVC')
    }
    for platform in ('linux-x86_64', 'linux-aarch64', 'linux-musl-x86_64', 'macos-x86_64', 'macos-arm64',
                     'windows-x86_64', 'windows-arm64'):
        reports[f'ports-{platform}'] = dict(kind='ports', platform_key=platform,
                                            crown=dict(artifacts=signature, observations={
                                                'python': exact, 'rust': changed,
                                            }))
    paths, record = [], {'reports': {}}
    for name, report in reports.items():
        path = tmp_path / f'{name}.json'
        raw = json.dumps(report).encode()
        path.write_bytes(raw)
        paths.append(path)
        record['reports'][name] = {'sha256': hashlib.sha256(raw).hexdigest()}

    receipt = crown.measurement_evidence(ROOT, paths, record)
    maxima = dict.fromkeys(crown.FIELDS, 0.0)
    maxima[field] = observed - exact[4]['output'][field]
    assert receipt == dict(artifacts=signature, comparisons=1056, nonmatching=7, max_abs=maxima)
    crown.validate_measurement(ROOT, crown.MEASUREMENT_POLICY, {crown.OPERATION: receipt})

    paths[-1].write_bytes(paths[-1].read_bytes() + b' ')
    with pytest.raises(ValueError, match='source-qualified evidence'):
        crown.measurement_evidence(ROOT, paths, record)

# ---- port measurements tests ----

@pytest.mark.parametrize('backend', ['python', pytest.param('rust', marks=pytest.mark.rust)])
def test_crown_measures_public_ports(backend):
    crown = importlib.import_module('cpp_crown')
    measured = crown.measure_ports(ROOT, (backend,))[backend]
    assert [row['output']['fire_type'] for row in measured] == [0, 0, 2, 1, 3, 1]
    assert all(type(row['output']['fire_type']) is int for row in measured)
    assert all(set(row['output']) == set(crown.FIELDS) for row in measured)
    assert measured[0]['output']['crown_length_to_width_ratio'] == 1.0
    assert [row['id'] for row in measured] == [row['id'] for row in crown.load_cases(ROOT)]
