"""Keep pinned C++ comparisons out of CI and inside the full local check, without drift."""

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CORE_TESTS = ROOT / "rust/crates/behave-fire/tests"
IGNORE = '#[ignore = "pinned C++ comparison: run by the full local check with --include-ignored"]'
PYTHON_COMPARISONS = {
    "tests/integration/test_cpp_calibration.py::test_revalidation_preserves_paired_bits_and_signed_zero",
    "tests/integration/test_cpp_crown.py::test_crown_measures_public_ports[python]",
    "tests/integration/test_cpp_crown.py::test_crown_measures_public_ports[rust]",
    "tests/integration/test_cpp_fire_size.py::test_approved_cases_match_exact_binary64[python]",
    "tests/integration/test_cpp_fire_size.py::test_approved_cases_match_exact_binary64[rust]",
    "tests/integration/test_cpp_ignite.py::test_approved_cases_match_exact_binary64[python]",
    "tests/integration/test_cpp_ignite.py::test_approved_cases_match_exact_binary64[rust]",
    *(f"tests/integration/test_cpp_local_enforcement.py::test_exact_mismatch_fails_while_qualification_is_withdrawn"
      f"[{backend}-{dataset}]" for backend in ("python", "rust") for dataset in ("fire_size", "ignite", "mortality")),
    "tests/integration/test_cpp_local_enforcement.py::"
    "test_surface_cap_violation_fails_while_qualification_is_withdrawn[python]",
    "tests/integration/test_cpp_local_enforcement.py::"
    "test_surface_cap_violation_fails_while_qualification_is_withdrawn[rust]",
    "tests/integration/test_cpp_measurement.py::test_python_only_comparison_never_imports_the_native_extension",
    "tests/integration/test_cpp_measurement.py::test_replacement_port_measurement_keeps_stale_acceptance_closed",
    "tests/integration/test_cpp_mortality.py::test_approved_cases_match_exact_binary64[python]",
    "tests/integration/test_cpp_mortality.py::test_approved_cases_match_exact_binary64[rust]",
    "tests/integration/test_cpp_receipt.py::test_archived_ports_report_reapplies_every_numerical_rule",
    *(f"tests/integration/test_cpp_receipt.py::test_archived_ports_report_rejects_tampering[{case}]"
      for case in ("surface", "exact", "backend", "platform", "measured", "measured-fields",
                   "measured-short", "measured-dataset", "origin")),
    *(f"tests/integration/test_cpp_receipt.py::test_receipt_verification_reads_the_archived_reports[{case}]"
      for case in ("valid", "report-bytes", "missing-report", "fixture-digest", "results", "extension",
                   "build-provenance", "toolchains", "archived-extension", "platform-name")),
    "tests/integration/test_cpp_reference.py::"
    "test_default_qualified_policy_checks_selected_backend_and_unit_controls[python]",
    "tests/integration/test_cpp_reference.py::"
    "test_default_qualified_policy_checks_selected_backend_and_unit_controls[rust]",
    "tests/regression/test_surface_cpp_boundaries.py::test_reaction_order_matches_pinned_cpp_near_extinction[python]",
    "tests/regression/test_surface_cpp_boundaries.py::test_reaction_order_matches_pinned_cpp_near_extinction[rust]",
}
RUST_COMPARISONS = {
    ("cpp_custom_controls.rs", "explicit_percent_controls_match_numbered_physics_and_consume_all_cpp_fields"),
    ("cpp_fire_size.rs", "cpp_fire_size_observations_follow_the_qualified_policy"),
    ("cpp_ignite.rs", "cpp_ignite_observations_follow_the_qualified_policy"),
    ("cpp_mortality.rs", "cpp_mortality_observations_follow_the_qualified_policy"),
    ("cpp_reference.rs", "cpp_surface_observations_follow_the_qualified_policy"),
    ("cpp_reference.rs", "perturbed_cpp_copy_fails_the_surface_caps"),
    *((name, "perturbed_cpp_copy_fails_the_exact_regression")
      for name in ("cpp_fire_size.rs", "cpp_ignite.rs", "cpp_mortality.rs")),
    ("cpp_surface_boundaries.rs", "reaction_order_matches_pinned_cpp_near_extinction"),
}


def _workflow(name):
    return yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


# ---- comparison selection tests ----

def test_ci_cargo_runs_never_include_ignored_comparisons():
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        text = path.read_text()
        assert "--include-ignored" not in text and "-- --ignored" not in text, path.name
    crate = _workflow("crate-validation.yml")
    commands = [step.get("run", "") for job in crate["jobs"].values() for step in job.get("steps", [])]
    assert any("cargo test" in command for command in commands)


def test_ci_pytest_runs_deselect_comparisons():
    jobs = _workflow("ci.yml")["jobs"]
    for name in ("python-only", "native-pytest"):
        runs = [step["run"] for step in jobs[name]["steps"] if "python -m pytest" in step.get("run", "")]
        assert runs and all('-m "not cpp_comparison"' in run for run in runs), name
    wheels = _workflow("wheel-validation.yml")["jobs"]
    suites = [step["run"] for job in wheels.values() for step in job["steps"]
              if "test_native_wheel_suite.py" in step.get("run", "")]
    assert len(suites) == 3 and all("--skip-cpp-comparisons" in run for run in suites)


def test_python_comparison_marker_selects_exactly_the_pinned_comparisons():
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run([sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "--collect-only",
                             "-q", "-o", "addopts=", "-m", "cpp_comparison"], cwd=ROOT, env=environment, text=True,
                            capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    collected = {line.strip() for line in result.stdout.splitlines() if "::" in line}
    assert collected == PYTHON_COMPARISONS


def test_rust_comparisons_are_ignored_by_default_and_named_exactly():
    ignored = set()
    for path in sorted(CORE_TESTS.glob("*.rs")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.strip().startswith("#[ignore"):
                assert line.strip() == IGNORE, f"{path.name}:{index + 1}"
                name = re.match(r"\s*fn (\w+)\(", lines[index + 1])
                assert name and lines[index - 1].strip() == "#[test]", f"{path.name}:{index + 1}"
                ignored.add((path.name, name.group(1)))
    assert ignored == RUST_COMPARISONS
