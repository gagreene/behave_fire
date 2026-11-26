"""Independent C++ observations fail closed before numerical approval."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from cpp_snapshot_support import write_snapshot

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import cpp_reference as reference

pytestmark = pytest.mark.cpp_reference
ROOT = Path(__file__).resolve().parents[2]


def _rewrite(root, relative, value):
    path = root / relative
    if path.suffix == ".jsonl":
        path.write_text("".join(json.dumps(row) + "\n" for row in value), encoding="utf-8")
    else:
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    hashes_path = root / "tests/data/cpp_reference/hashes.json"
    hashes = json.loads(hashes_path.read_text())
    hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    hashes_path.write_text(json.dumps(hashes), encoding="utf-8")


# ---- default offline comparison tests ----

def test_default_observations_are_offline_and_policy_gated():
    data = reference.load_reference()
    assert len(data["cases"]) == 634
    assert data["manifest"]["available"] == 583
    assert data["manifest"]["submodule_checkout_used"] is True
    result = next(row["output"] for row in data["results"] if row["status"] == "ok")
    if data['policy']['approved']:
        reference.assert_matches_cpp(result, result, data['policy'])
    else:
        with pytest.raises(reference.UnapprovedCppPolicy):
            reference.assert_matches_cpp(result, result, data["policy"])


def test_default_observations_need_neither_cpp_sources_nor_overlay(tmp_path, cpp_snapshot_template):
    root = write_snapshot(cpp_snapshot_template, tmp_path)
    assert not (root / "reference").exists()
    data = reference.load_reference(root)
    assert data["manifest"]["available"] == 583


@pytest.mark.parametrize('backend', ['python', pytest.param('rust', marks=pytest.mark.rust)])
def test_default_qualified_policy_checks_selected_backend_and_unit_controls(tmp_path, backend):
    import cpp_controls

    data = reference.load_reference()
    report = cpp_controls.measure_ports(ROOT, tmp_path / 'surface-check.json', backends=(backend,))
    expected_ids = {row['id'] for row in data['results'] if row['status'] == 'ok'}
    expected_ids.update(row['id'] for row in data['controls']['results'])
    assert {row['id'] for row in report['observations']} == expected_ids
    assert report['counts'] == {backend: dict(cases=722, fields=3610, paired_cases=139)}
    if data['policy']['approved']:
        assert data['policy']['qualification']['calibration_complete'] is True


# ---- reference provenance and comparison guards tests ----

@pytest.mark.parametrize(('model', 'loads'), [
    (152, {'dead_1h_load_lb_ft2': 'dead_1h'}),
    (153, {'dead_100h_load_lb_ft2': 'dead_100h', 'live_herb_load_lb_ft2': 'live_herb'}),
])
def test_explicit_load_controls_preserve_the_ports_model_definition(model, loads):
    from behave_fire.components.fuel_models import FuelModels

    cases = reference._rows(ROOT / 'tests/data/cpp_reference/surface/cases.jsonl')
    case = next(candidate for candidate in cases if candidate['fuel_model'] == model)
    control, = reference._custom_cases([case])
    model_record = FuelModels().fuel_models_[model]
    assert {name: control[name] for name in loads} == {name: model_record[key] for name, key in loads.items()}
    reference._validate_case(control)

    field = next(iter(loads))
    missing = {key: value for key, value in control.items() if key != field}
    with pytest.raises(ValueError, match='custom load control'):
        reference._validate_case(missing)
    with pytest.raises(ValueError, match='custom load control'):
        reference._validate_case(dict(control, **{field: control[field] + 0.001}))
    with pytest.raises(ValueError, match='custom load control'):
        reference._validate_case(dict(case, **{field: control[field]}))


@pytest.mark.parametrize("mutation", ("drop", "field", "duplicate", "unavailable", "status"))
def test_invalid_result_contracts_fail_even_when_the_hash_is_updated(tmp_path, mutation, cpp_snapshot_template):
    root = write_snapshot(cpp_snapshot_template, tmp_path)
    relative = "tests/data/cpp_reference/surface/results.jsonl"
    rows = reference._rows(root / relative)
    if mutation == "drop":
        rows.pop()
    elif mutation == "field":
        del rows[0]["output"][reference.FIELDS[0]]
    elif mutation == "duplicate":
        rows[-1]["id"] = rows[0]["id"]
    elif mutation == "unavailable":
        row = next(r for r in rows if r["status"] == "unavailable")
        row["output"] = {"length_to_width_ratio": 1}
    else:
        rows[0]["status"] = "waived"
    _rewrite(root, relative, rows)
    with pytest.raises(ValueError):
        reference.load_reference(root)


def test_live_lane_has_no_external_checkout_fallback(tmp_path):
    with pytest.raises(ValueError, match="no external fallback"):
        reference.require_live_checkout(tmp_path)


def test_live_source_filter_only_allows_untracked_ide_material():
    status = "?? .idea/workspace.xml\n M src/behave/surface.cpp\n?? injected.h\n M .idea/tracked.xml\n"
    assert reference._untrusted_changes(status) == [
        " M src/behave/surface.cpp", "?? injected.h", " M .idea/tracked.xml"]


def test_missing_artifact_hash_cannot_remove_verification(tmp_path, cpp_snapshot_template):
    root = write_snapshot(cpp_snapshot_template, tmp_path)
    path = root / "tests/data/cpp_reference/hashes.json"
    hashes = json.loads(path.read_text())
    hashes.pop("tests/data/cpp_reference/inventory.json")
    path.write_text(json.dumps(hashes), encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete expected hash list"):
        reference.load_reference(root)


def test_missing_supported_mode_measurement_is_rejected(tmp_path, cpp_snapshot_template):
    root = write_snapshot(cpp_snapshot_template, tmp_path)
    relative = "tests/data/cpp_reference/surface/cases.jsonl"
    cases = reference._rows(root / relative)
    row = next(row for row in cases if "-mode-" in row["id"] and row["wind_height"] == "TenMeter")
    row["wind_height"] = "TwentyFoot"
    _rewrite(root, relative, cases)
    with pytest.raises(ValueError, match="Surface mode matrix"):
        reference.load_reference(root)


def test_missing_tolerance_classification_is_rejected(tmp_path, cpp_snapshot_template):
    root = write_snapshot(cpp_snapshot_template, tmp_path)
    relative = "tests/data/cpp_reference/tolerance-policy.json"
    policy = json.loads((root / relative).read_text())
    policy["fields"].pop(reference.FIELDS[0])
    _rewrite(root, relative, policy)
    with pytest.raises(ValueError, match="Missing field tolerance"):
        reference.load_reference(root)


def test_numeric_comparator_requires_exact_boolean_approval_and_finite_values():
    actual = dict.fromkeys(reference.FIELDS, 1.0)
    policy = dict(approved="yes", fields={field: dict(kind="relative_absolute", rtol=0, atol=0)
        for field in reference.FIELDS})
    with pytest.raises(reference.UnapprovedCppPolicy):
        reference.assert_matches_cpp(actual, actual, policy)
    policy["approved"] = True
    reference.assert_matches_cpp(actual, actual, policy)
    changed = dict(actual)
    changed[reference.FIELDS[0]] = 2
    with pytest.raises(AssertionError):
        reference.assert_matches_cpp(changed, actual, policy)
    changed[reference.FIELDS[0]] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        reference.assert_matches_cpp(changed, changed, policy)


def test_reused_state_must_match_a_fresh_full_run(tmp_path, cpp_snapshot_template):
    root = write_snapshot(cpp_snapshot_template, tmp_path)
    relative = "tests/data/cpp_reference/surface/results.jsonl"
    results = reference._rows(root / relative)
    next(row for row in results if row["id"] == "fm124-state-A2")["output"]["spread_chains_per_hour"] += 1
    _rewrite(root, relative, results)
    with pytest.raises(ValueError, match="State transition"):
        reference.load_reference(root)


def test_signed_zero_comparison_is_explicit():
    actual = dict.fromkeys(reference.FIELDS, 0.0)
    expected = dict(actual)
    expected[reference.FIELDS[0]] = -0.0
    policy = dict(approved=True, fields={field: dict(kind="relative_absolute", rtol=0, atol=0)
        for field in reference.FIELDS})
    with pytest.raises(AssertionError, match="signed-zero"):
        reference.assert_matches_cpp(actual, expected, policy)


def test_source_identity_ignores_only_line_ending_translation(tmp_path):
    unix, windows = tmp_path / "unix.txt", tmp_path / "windows.txt"
    unix.write_bytes(b"alpha\nbeta\n")
    windows.write_bytes(b"alpha\r\nbeta\r\n")
    assert reference._digest(unix) != reference._digest(windows)
    assert reference._text_digest(unix) == reference._text_digest(windows)
    windows.write_bytes(b"alpha\r\nchanged\r\n")
    assert reference._text_digest(unix) != reference._text_digest(windows)


def test_unsafe_case_fields_are_rejected_independently_of_hashes(tmp_path, cpp_snapshot_template):
    root = write_snapshot(cpp_snapshot_template, tmp_path)
    relative = "tests/data/cpp_reference/surface/cases.jsonl"
    cases = reference._rows(root / relative)
    cases[0]["fuel_model"] = 256
    _rewrite(root, relative, cases)
    with pytest.raises(ValueError, match="Unsafe fuel"):
        reference.load_reference(root)
