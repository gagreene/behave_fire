"""Guard the standalone Spot measurement scope and numeric decoding."""

import importlib
import json
import math
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpp_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
spot = importlib.import_module('cpp_spot')


@pytest.fixture
def _spot_cases(tmp_path):
    directory = tmp_path / 'tests/data/cpp_reference/spot'
    directory.mkdir(parents=True)
    cases = [
        dict(id=f'spot-flat-{wind_label}-{cover_label}-cover', operation='spot_surface',
             flame_ft=5, wind_mph=wind, cover_ft=cover)
        for wind_label, wind in [('calm', 0), ('wind', 5)]
        for cover_label, cover in [('zero', 0), ('ten', 10)]
    ]
    _write_jsonl(directory / 'cases.jsonl', cases)
    return cases, directory


def _write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


# ---- guarded measurements tests ----

def test_spot_preserves_signed_zero_in_observations(tmp_path, _spot_cases):
    cases, directory = _spot_cases
    path = directory / 'results.jsonl'
    _write_jsonl(path, [dict(id=case['id'], status='ok', upstream_entries=1, output_entries=1,
                            output={'flat_distance_ft': 0}) for case in cases])
    results = path.read_text(encoding='utf-8').replace('"flat_distance_ft": 0', '"flat_distance_ft": -0', 1)
    path.write_text(results, encoding='utf-8')

    distance = spot.read_cpp_results(tmp_path, path)[0]['output']['flat_distance_ft']
    assert math.copysign(1, distance) == -1


@pytest.mark.parametrize('invalid_wind', ['-1', '-0', '0.0', 'false'])
def test_spot_rejects_unapproved_input_values(tmp_path, _spot_cases, invalid_wind):
    _, directory = _spot_cases
    path = directory / 'cases.jsonl'
    cases = path.read_text(encoding='utf-8').replace('"wind_mph": 0', f'"wind_mph": {invalid_wind}', 1)
    path.write_text(cases, encoding='utf-8')

    with pytest.raises(ValueError, match='approved cases'):
        spot.load_cases(tmp_path)


def test_spot_requires_integer_output_entry_count(tmp_path, _spot_cases):
    cases, _ = _spot_cases
    rows = [dict(id=case['id'], status='ok', upstream_entries=1, output_entries=1,
                 output={'flat_distance_ft': 0}) for case in cases]
    rows[0]['output_entries'] = True

    with pytest.raises(ValueError, match='output action'):
        spot.validate_cpp_rows(tmp_path, rows)


# ---- measurement receipt tests ----

def test_spot_receipt_records_differences_without_approving_them(tmp_path, _spot_cases):
    _, directory = _spot_cases
    for name in ('manifest.json', 'results.jsonl', 'schema.json'):
        (directory / name).write_bytes(b'fixture')
    policy = dict(approved=False, fields={'flat_distance_ft': {'kind': 'measurement_only'}})
    receipt = dict(artifacts=spot.artifacts(tmp_path), comparisons=64, nonmatching=64, max_abs_ft=1e20)

    spot.validate_measurement(tmp_path, policy, None)
    spot.validate_measurement(tmp_path, policy, {'spot': receipt})
    spot.validate_measurement(tmp_path, policy, {'spot': dict(receipt, nonmatching=0, max_abs_ft=0.0)})
    spot.validate_measurement(tmp_path, policy, {'spot': dict(receipt, nonmatching=1, max_abs_ft=0.0)})
    with pytest.raises(ValueError, match='Inconsistent Spot comparison statistics'):
        spot.validate_measurement(tmp_path, policy, {'spot': dict(receipt, nonmatching=0)})
    with pytest.raises(ValueError, match='must remain measurement-only'):
        spot.validate_measurement(tmp_path, dict(policy, approved=True), {'spot': receipt})
    with pytest.raises(ValueError, match='must remain measurement-only'):
        spot.validate_measurement(tmp_path, dict(policy, approved=0), None)
    with pytest.raises(ValueError, match='measurement receipt'):
        spot.validate_measurement(tmp_path, policy, {'spot': dict(receipt, nonmatching=True)})
    with pytest.raises(ValueError, match='maximum absolute Spot difference'):
        spot.validate_measurement(tmp_path, policy, {'spot': dict(receipt, max_abs_ft=math.inf)})


# ---- port measurements tests ----

@pytest.mark.parametrize('backend', ['python', pytest.param('rust', marks=pytest.mark.rust)])
def test_spot_measures_selected_backend_in_case_order(tmp_path, _spot_cases, backend):
    cases, _ = _spot_cases
    result = spot.measure_ports(tmp_path, (backend,))
    assert set(result) == {backend}
    measured = result[backend]
    assert [row['id'] for row in measured] == [case['id'] for case in cases]
    for row in measured:
        assert set(row) == {'id', 'output'}
        assert set(row['output']) == {'flat_distance_ft'}
        distance = row['output']['flat_distance_ft']
        assert type(distance) is float and math.isfinite(distance) and distance >= 0
