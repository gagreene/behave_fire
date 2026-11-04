"""Regression tests for the exact non-finite int32 fixture portability boundary."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pytest.ini").is_file())
DRAFT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(DRAFT / "tools"), str(ROOT / "src")]
ORIGINAL_ASARRAY = np.asarray
SPECS = [("crown", "coerce_crown_fire_inputs/nan"),
         ("crown", "coerce_crown_fire_inputs/positive-infinity"),
         ("ignite", "calculate_lightning_ignition_probability/positive-infinity"),
         ("fine_dead_fuel_moisture_tool", "calculate_fine_dead_fuel_moisture/nan")]


def _arm_asarray(value, dtype=None, *args, **kwargs):
    if dtype is not None and np.dtype(dtype) == np.dtype("int32"):
        source = ORIGINAL_ASARRAY(value)
        if source.dtype.kind == "f" and np.any(~np.isfinite(source)):
            value = np.nan_to_num(source, nan=0.0, posinf=float(2**31 - 1), neginf=float(-(2**31)))
    return ORIGINAL_ASARRAY(value, dtype, *args, **kwargs)


def _case(domain, name):
    document = json.loads((ROOT / "tests/data/rust_port" / (domain + ".json")).read_text(encoding="utf-8"))
    return next(case for case in document["cases"] if case["id"] == name)


def _exporter():
    spec = importlib.util.spec_from_file_location("platform_test_exporter", DRAFT / "tools/export_rust_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policy():
    path = DRAFT / "tools/rust_reference_platform.py"
    assert path.is_file(), "Scoped platform reference policy is missing"
    spec = importlib.util.spec_from_file_location("platform_test_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _support():
    spec = importlib.util.spec_from_file_location("platform_test_support", DRAFT / "tests/parity_support.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _x86_asarray(value, dtype=None, *args, **kwargs):
    if dtype is not None and np.dtype(dtype) == np.dtype("int32"):
        source = ORIGINAL_ASARRAY(value)
        if source.dtype.kind == "f" and np.any(~np.isfinite(source)):
            value = np.where(np.isfinite(source), source, float(-(2**31)))
    return ORIGINAL_ASARRAY(value, dtype, *args, **kwargs)


@pytest.mark.parametrize("domain,name", SPECS)
def test_canonical_export_keeps_original_golden_after_arm_cast(monkeypatch, domain, name):
    policy, exporter = _policy(), _exporter()
    frozen = _case(domain, name)
    monkeypatch.setattr(np, "asarray", _arm_asarray)
    live, canonical = policy.reference_outputs(frozen, exporter.encode_value)
    assert live != frozen["output"]
    assert exporter.contracts_match(canonical, frozen["output"], True)
    current = dict(frozen, output=live)
    assert policy.canonical_case(current, exporter.encode_value, exporter.contracts_match) == frozen
    assert current["output"] == live


@pytest.mark.parametrize("domain,name", SPECS)
def test_compare_reports_frozen_diagnostic_and_retains_live_gate(monkeypatch, tmp_path, capsys, domain, name):
    from behave_fire import rust
    from importlib import import_module

    spec = importlib.util.spec_from_file_location("platform_test_compare", DRAFT / "tools/compare_backends.py")
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    frozen = _case(domain, name)
    fixtures = tmp_path / "tests/data/rust_port"
    fixtures.mkdir(parents=True)
    (fixtures / (domain + ".json")).write_text(json.dumps({"cases": [frozen]}), encoding="utf-8")
    monkeypatch.setattr(compare, "ROOT", tmp_path)
    monkeypatch.setattr(np, "asarray", _arm_asarray)
    module = getattr(rust, domain)
    function = name.split("/")[0]
    reference = getattr(import_module("behave_fire.components." + domain), function)
    monkeypatch.setattr(module, function, reference)
    assert compare.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    diagnostic, = report["platform_cast_diagnostics"]
    assert diagnostic["frozen_output"] == frozen["output"]
    assert diagnostic["original_cast_replay_matches_frozen"]
    assert not diagnostic["current_python_matches_frozen"]
    assert report["domains"][domain]["maximum_absolute_error"] == 0
    monkeypatch.setattr(module, function, lambda **arguments: compare.parity_support.decode(frozen["output"]))
    assert compare.main([]) == 1
    assert len(json.loads(capsys.readouterr().out)["failures"]) == 1


def test_export_default_and_check_keep_canonical_values(monkeypatch, tmp_path, capsys):
    exporter = _exporter()
    frozen = _case(*SPECS[0])
    monkeypatch.setattr(np, "asarray", _arm_asarray)
    original_documents = exporter.build_documents
    documents = original_documents()
    for path, document in documents.items():
        reviewed = json.loads((ROOT / path).read_text(encoding="utf-8"))
        assert exporter.contracts_match(document, reviewed), path
    name = "tests/data/rust_port/crown.json"
    exported = next(case for case in documents[name]["cases"] if case["id"] == frozen["id"])
    assert exported == frozen
    monkeypatch.setattr(exporter, "build_documents", lambda: {name: documents[name]})
    assert exporter.main(["--output-dir", str(tmp_path)]) == 0
    before = (tmp_path / name).read_bytes()
    assert exporter.main(["--check", "--output-dir", str(tmp_path)]) == 0
    assert (tmp_path / name).read_bytes() == before
    changed = json.loads(before)
    changed["cases"][0]["output"] = {"tampered": True}
    (tmp_path / name).write_text(json.dumps(changed), encoding="utf-8")
    assert exporter.main(["--check", "--output-dir", str(tmp_path)]) == 1
    assert "Reference fixture drift" in capsys.readouterr().out


def test_integer_array_comparison_rejects_single_unit_drift():
    support = _support()
    expected = {"dtype": "int32", "shape": [1], "values": [-(2**31)]}
    with pytest.raises(AssertionError):
        support.assert_result(np.array([-(2**31) + 1], dtype=np.int32), expected)


@pytest.mark.parametrize("domain,name", SPECS)
def test_live_reference_rejects_native_and_frozen_drift(monkeypatch, domain, name):
    policy, exporter = _policy(), _exporter()
    frozen = _case(domain, name)
    monkeypatch.setattr(np, "asarray", _arm_asarray)
    live = policy.acceptance_output(frozen, exporter.encode_value, exporter.contracts_match)
    assert live != frozen["output"]
    assert not exporter.contracts_match(frozen["output"], live, True)
    changed = deepcopy(frozen)
    if domain == "crown":
        changed["output"]["fm_grid"]["values"][0] += 1
    else:
        changed["output"]["values"][0] += .01
    with pytest.raises(AssertionError, match="Frozen cast diagnostic drift"):
        policy.acceptance_output(changed, exporter.encode_value, exporter.contracts_match)


@pytest.mark.parametrize("domain,name", SPECS)
def test_policy_excludes_other_inputs_and_cases(monkeypatch, domain, name):
    policy, exporter = _policy(), _exporter()
    frozen = _case(domain, name)
    monkeypatch.setattr(np, "asarray", _arm_asarray)
    changed = deepcopy(frozen)
    changed["id"] += "/unreviewed"
    assert policy.reference_outputs(changed, exporter.encode_value) is None
    changed = deepcopy(frozen)
    first = next(iter(changed["inputs"]))
    changed["inputs"][first] = 123
    assert policy.reference_outputs(changed, exporter.encode_value) is None
    assert policy.acceptance_output(changed, exporter.encode_value, exporter.contracts_match) == changed["output"]


@pytest.mark.parametrize("domain,name", SPECS)
def test_run_case_rejects_frozen_native_output_and_accepts_live_python(monkeypatch, domain, name):
    from behave_fire import rust
    from importlib import import_module

    support = _support()
    frozen = _case(domain, name)
    monkeypatch.setattr(np, "asarray", _arm_asarray)
    module = getattr(rust, domain)
    function = name.split("/")[0]
    reference = getattr(import_module("behave_fire.components." + domain), function)
    monkeypatch.setattr(module, function, lambda **arguments: support.decode(frozen["output"]))
    with pytest.raises(AssertionError):
        support.run_case(frozen)
    monkeypatch.setattr(module, function, reference)
    support.run_case(frozen)


@pytest.mark.parametrize("domain,name", SPECS)
def test_x86_cast_keeps_frozen_acceptance(monkeypatch, domain, name):
    policy, exporter = _policy(), _exporter()
    monkeypatch.setattr(np, "asarray", _x86_asarray)
    frozen = _case(domain, name)
    assert policy.reference_outputs(frozen, exporter.encode_value) is None
    assert policy.acceptance_output(frozen, exporter.encode_value, exporter.contracts_match) is frozen["output"]
