"""Guard the standalone Mortality measurement scope and numeric decoding."""

import importlib
import math
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpp_reference

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
mortality = importlib.import_module('cpp_mortality')
reference = importlib.import_module('cpp_reference')


# ---- comparison tests ----

@pytest.mark.parametrize('backend', ['python', pytest.param('rust', marks=pytest.mark.rust)])
@pytest.mark.cpp_comparison
def test_approved_cases_match_exact_binary64(backend):
    policy = reference.load_reference()['policy']['mortality']
    expected = mortality.load_observations(ROOT)
    measured = mortality.measure_ports(ROOT, (backend,))[backend]
    for actual, original in zip(measured, expected, strict=True):
        assert actual['id'] == original['id']
        if policy['approved']:
            mortality.assert_matches_cpp(actual['output'], original['output'], policy)
        else:
            with pytest.raises(reference.UnapprovedCppPolicy):
                mortality.assert_matches_cpp(actual['output'], original['output'], policy)


def test_exact_policy_rejects_a_changed_zero_sign():
    expected = mortality.load_observations(ROOT)[0]['output']
    policy = dict(approved=True, fields={'scorch_height_ft': {'kind': 'exact_binary64'}})
    mortality.assert_matches_cpp(expected, expected, policy)
    with pytest.raises(AssertionError, match='scorch_height_ft'):
        mortality.assert_matches_cpp({'scorch_height_ft': -0.0}, expected, policy)


# ---- guarded measurements tests ----

def test_mortality_preserves_signed_zero_in_observations(tmp_path):
    directory = tmp_path / mortality.DATASET
    directory.mkdir(parents=True)
    (directory / 'cases.jsonl').write_bytes((ROOT / mortality.DATASET / 'cases.jsonl').read_bytes())
    rows = (ROOT / mortality.DATASET / 'results.jsonl').read_bytes()
    path = directory / 'results.jsonl'
    path.write_bytes(rows.replace(b'"scorch_height_ft":0', b'"scorch_height_ft":-0', 1))

    height = mortality.read_cpp_results(tmp_path, path)[0]['output']['scorch_height_ft']
    assert math.copysign(1, height) == -1


def test_mortality_rejects_signed_zero_in_inputs(tmp_path):
    directory = tmp_path / mortality.DATASET
    directory.mkdir(parents=True)
    rows = (ROOT / mortality.DATASET / 'cases.jsonl').read_bytes()
    (directory / 'cases.jsonl').write_bytes(rows.replace(b'"intensity_btu_ft_s":0', b'"intensity_btu_ft_s":-0', 1))

    with pytest.raises(ValueError, match='approved cases'):
        mortality.load_cases(tmp_path)


# ---- qualification tests ----

def test_exact_proof_rejects_a_boolean_count():
    policy = dict(approved=True, fields={'scorch_height_ft': {'kind': 'exact_binary64'}})
    qualification = {'mortality': dict(artifacts=mortality.artifacts(ROOT), comparisons=96, nonmatching=0)}
    mortality.validate_policy(ROOT, policy, qualification)
    qualification['mortality']['nonmatching'] = False
    with pytest.raises(ValueError, match='Incomplete or stale Mortality qualification'):
        mortality.validate_policy(ROOT, policy, qualification)
