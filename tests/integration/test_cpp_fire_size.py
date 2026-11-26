"""Compare the three guarded standalone FireSize cases with pinned C++."""

import sys
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
fire_size = import_module("cpp_fire_size")
reference = import_module("cpp_reference")

pytestmark = pytest.mark.cpp_reference


# ---- approved comparison tests ----

@pytest.mark.parametrize("backend", ["python", pytest.param("rust", marks=pytest.mark.rust)])
@pytest.mark.cpp_comparison
def test_approved_cases_match_exact_binary64(backend):
    policy = reference.load_reference()["policy"]["fire_size"]
    observations = fire_size.load_observations(ROOT)
    measured = fire_size.measure_ports(ROOT, (backend,))[backend]
    for actual, expected in zip(measured, observations, strict=True):
        assert actual["id"] == expected["id"]
        if policy["approved"]:
            fire_size.assert_matches_cpp(actual["output"], expected["output"], policy)
        else:
            with pytest.raises(reference.UnapprovedCppPolicy):
                fire_size.assert_matches_cpp(actual["output"], expected["output"], policy)


# ---- comparison rejection tests ----

def test_exact_policy_rejects_a_changed_zero_sign():
    record = next(row for row in fire_size.load_observations(ROOT) if row["id"] == "zero-elapsed")
    expected = record["output"]
    policy = {"approved": True, "fields": {field: {"kind": "exact_binary64"} for field in fire_size.FIELDS}}
    fire_size.assert_matches_cpp(expected, expected, policy)
    with pytest.raises(AssertionError, match="area_ft2"):
        fire_size.assert_matches_cpp(dict(expected, area_ft2=-0.0), expected, policy)


@pytest.mark.parametrize("malformed", ["boolean_count", "float_total"])
def test_exact_qualification_rejects_malformed_counts(malformed):
    qualification = {"source_commit": "1" * 40, "ci_run_url": "https://github.com/gagreene/behave_fire/actions/runs/1"}
    proof = dict(qualification, artifacts=fire_size.artifacts(ROOT), comparisons=144,
                 maxima={field: dict(absolute=0.0, relative=0.0, ulp=0, nonmatching=0) for field in fire_size.FIELDS})
    qualification["fire_size"] = proof
    policy = {"approved": True, "fields": fire_size.EXACT_FIELDS, "qualification": proof}
    fire_size.validate_policy(ROOT, policy, qualification)
    if malformed == "boolean_count":
        proof["maxima"]["area_ft2"]["nonmatching"] = False
    else:
        proof["comparisons"] = 144.0
    with pytest.raises(ValueError, match="Incomplete or stale FireSize qualification"):
        fire_size.validate_policy(ROOT, policy, qualification)
