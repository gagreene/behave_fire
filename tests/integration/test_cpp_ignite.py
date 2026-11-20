"""Verify exact standalone Ignite comparisons and the approved input boundary."""

import math
import sys
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
ignite = import_module("cpp_ignite")
reference = import_module("cpp_reference")

pytestmark = pytest.mark.cpp_reference


# ---- comparison tests ----

@pytest.mark.parametrize("backend", ["python", pytest.param("rust", marks=pytest.mark.rust)])
def test_approved_cases_match_exact_binary64(backend):
    policy = reference.load_reference()["policy"]["ignite"]
    expected = ignite.load_observations(ROOT)
    measured = ignite.measure_ports(ROOT, (backend,))[backend]
    for actual, original in zip(measured, expected, strict=True):
        assert actual["id"] == original["id"]
        if policy["approved"]:
            ignite.assert_matches_cpp(actual["output"], original["output"], policy)
        else:
            with pytest.raises(reference.UnapprovedCppPolicy):
                ignite.assert_matches_cpp(actual["output"], original["output"], policy)


def test_exact_policy_rejects_a_changed_zero_sign():
    wet = next(row["output"] for row in ignite.load_observations(ROOT) if row["id"] == "ignite-wet")
    policy = {"approved": True, "fields": {field: {"kind": "exact_binary64"} for field in ignite.FIELDS}}
    ignite.assert_matches_cpp(wet, wet, policy)
    with pytest.raises(AssertionError, match="firebrand_probability"):
        ignite.assert_matches_cpp(dict(wet, firebrand_probability=-0.0), wet, policy)


# ---- input validation tests ----

def test_ignite_case_reader_rejects_signed_zero(tmp_path):
    path = tmp_path / ignite.DATASET / "cases.jsonl"
    path.parent.mkdir(parents=True)
    original = (ROOT / ignite.DATASET / "cases.jsonl").read_bytes()
    assert original.count(b'"moisture_1h":0}') == 1
    path.write_bytes(original.replace(b'"moisture_1h":0}', b'"moisture_1h":-0}'))
    with pytest.raises(ValueError, match="Ignite inputs differ"):
        ignite.load_cases(tmp_path)


def test_ignite_result_reader_preserves_signed_zero(tmp_path):
    original = (ROOT / ignite.DATASET / "results.jsonl").read_bytes()
    assert original.count(b'"firebrand_probability":0,') == 1
    path = tmp_path / "results.jsonl"
    path.write_bytes(original.replace(b'"firebrand_probability":0,', b'"firebrand_probability":-0,'))
    wet = next(row for row in ignite.read_cpp_results(ROOT, path) if row["id"] == "ignite-wet")
    assert math.copysign(1.0, wet["output"]["firebrand_probability"]) == -1.0


# ---- qualification tests ----

def test_exact_proof_rejects_a_boolean_count():
    policy = {"approved": True, "fields": {field: {"kind": "exact_binary64"} for field in ignite.FIELDS}}
    qualification = {"ignite": {"artifacts": ignite.artifacts(ROOT), "comparisons": 128, "nonmatching": 0}}
    ignite.validate_policy(ROOT, policy, qualification)
    qualification["ignite"]["nonmatching"] = False
    with pytest.raises(ValueError, match="Incomplete or stale Ignite qualification"):
        ignite.validate_policy(ROOT, policy, qualification)
