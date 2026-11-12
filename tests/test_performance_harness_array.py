"""Acceptance gates must reject incomplete or misleading performance evidence."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tools"
OPERATIONS = ("surface", "crown", "typed-crown", "typed-crown-b")


def _benchmark_worker(backend, validation=False):
    command = [sys.executable, "-B", str(TOOLS / "benchmark_backends.py"), "--worker",
               "--backend", backend, "--cells", "16", "--operation", "typed-crown-b",
               "--dataset", "heterogeneous", "--warm-calls", "1"]
    if validation:
        command.append("--validate-only")
    environment = dict(os.environ, PYTHONPATH=str(TOOLS.parent / "src"))
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, timeout=60)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def _evidence():
    requirements = {
        "schema_version": 1,
        "datasets": ["uniform", "heterogeneous"],
        "operations": list(OPERATIONS),
        "sizes": [10_000, 100_000, 1_000_000],
        "minimum_pairs": 7,
        "warm_calls": 5,
        "bootstrap_resamples": 200,
        "confidence": 0.95,
        "seed": 20261105,
        "targets": {"small_common": 1.25, "small_typed": 1.0, "large_serial": 2.0, "large_four": 4.0},
        "require_allocations": True,
        "require_scalar_baseline": True,
        "baseline_identity": {"source_hash": "pre-optimization-source", "extension_hash": "pre-optimization-native"},
    }
    report = {"schema_version": 1, "preset": "acceptance", "profile": False, "physical_cores": 4,
              "records": [], "validations": [], "allocations": [], "baseline": []}
    for dataset in requirements["datasets"]:
        for operation in OPERATIONS:
            for cells in [1, *requirements["sizes"]]:
                workers = [1] if cells < 100_000 else [1, 4]
                for block in range(7):
                    report["records"].append(_row(dataset, operation, cells, block, "python", 1, 10.0))
                    for threads in workers:
                        ratio = 4.0 if threads == 4 else 2.0
                        if cells == 10_000:
                            ratio = 1.0 if operation.startswith("typed-") else 1.25
                        seconds = 0.0005 if cells == 1 else 10.0 / ratio
                        row = _row(dataset, operation, cells, block, "rust", threads, seconds)
                        report["records"].append(row)
                        if cells == 1:
                            baseline = copy.deepcopy(row)
                            baseline.update(requirements["baseline_identity"])
                            report["baseline"].append(baseline)
                for threads in workers:
                    report["validations"].append({
                        "dataset": dataset, "operation": operation, "cells": cells, "threads": threads,
                        "passed": True, "numpy": "2.1.3", "numpy_build": "test-build",
                        "source_hash": "a" * 64, "extension_hash": "b" * 64, "seed": 20261105,
                    })
            report["allocations"].append({"dataset": dataset, "operation": operation,
                                          "kernel_per_cell": 0, "encoder_per_cell": 0,
                                          "source_hash": "a" * 64, "extension_hash": "b" * 64})
    from test_allocation_evidence_array import _evidence as allocation_fixture, _updated
    envelope = allocation_fixture()
    prototype = copy.deepcopy(envelope["raw"])
    envelope["raw"] = []
    for dataset in requirements["datasets"]:
        for operation in OPERATIONS:
            for sample in copy.deepcopy(prototype):
                sample.update(dataset=dataset, operation=operation, process_id=len(envelope["raw"]) + 1)
                envelope["raw"].append(sample)
    assembler = _tool("assemble_allocation_evidence")
    report["allocations"] = assembler.assemble(_updated(envelope), requirements)["allocations"]
    return report, requirements


def _row(dataset, operation, cells, block, backend, threads, seconds):
    return {"dataset": dataset, "operation": operation, "cells": cells, "block": block,
            "backend": backend, "threads": threads, "first_seconds": seconds,
            "warm_seconds": [seconds] * 5, "total_seconds": seconds,
            "numpy": "2.1.3", "numpy_build": "test-build", "python": "3.12.8",
            "platform": "test-host", "source_hash": "a" * 64, "extension_hash": "b" * 64,
            "seed": 20261105, "peak_bytes": 800 if backend == "rust" else 1000,
            "profile": False}


def _tool(name):
    path = TOOLS / (name + ".py")
    assert path.is_file(), f"Task 1 tool {name} has not been implemented"
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_any_missing_or_failed_route_fails_acceptance():
    checker = _tool("check_performance")
    report, requirements = _evidence()
    assert checker.check_report(report, requirements)["passed"]
    report["records"] = [row for row in report["records"] if row["operation"] != "typed-crown-b"]
    assert not checker.check_report(report, requirements)["passed"]


def test_benchmark_worker_keeps_legacy_fields_and_unprofiled_timings():
    result = _benchmark_worker("python")
    assert result["total_seconds"] == result["first_seconds"] > 0
    assert len(result["warm_seconds"]) == 1 and result["warm_seconds"][0] > 0
    assert result["numpy"] and result["numpy_build"] and result["source_hash"]
    assert result["peak_bytes"] > 0 and result["profile"] is False
    assert result["native_calls"] == []
    assert result["python_marshalling_and_output_seconds"] is None


@pytest.mark.parametrize("failed", [False, True])
def test_campaign_does_not_time_validation_only_or_failed_cases(monkeypatch, failed):
    driver = _tool("benchmark_backends")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if failed else 0, stderr="worker failed" if failed else "",
                               stdout=json.dumps({"kind": "validation", "passed": True}))

    monkeypatch.setattr(driver.subprocess, "run", run)
    args = SimpleNamespace(preset="screen", profile=False, physical_cores=4, baseline=None,
                           allocations=[], datasets=["uniform"], sizes=[16], operations=["surface"],
                           worker_threads=[1], repetitions=1, seed=20261105, warm_calls=1,
                           timeout=30, validate_only=not failed, output=None)
    report = driver._campaign(args)
    assert len(commands) == 1 and "--validate-only" in commands[0]
    assert report["records"] == []
    assert bool(report["errors"]) is failed


@pytest.mark.parametrize("identity", [None, "matching", "wrong"])
def test_checker_cli_requires_declared_preoptimization_identity(tmp_path, identity):
    report, requirements = _evidence()
    requirements.pop("baseline_identity")
    report_path = tmp_path / "report.json"
    requirements_path = tmp_path / "requirements.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    requirements_path.write_text(json.dumps(requirements), encoding="utf-8")
    command = [sys.executable, "-B", str(TOOLS / "check_performance.py"), "--results", str(report_path),
               "--requirements", str(requirements_path)]
    if identity is not None:
        command += ["--baseline-source-hash", "pre-optimization-source" if identity == "matching" else "other-source",
                    "--baseline-extension-hash", "pre-optimization-native"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert completed.returncode == (0 if identity == "matching" else 1), completed.stderr
    result = json.loads(completed.stdout)
    assert result["passed"] is (identity == "matching")
    if identity != "matching":
        assert any("baseline" in error.lower() and "identity" in error.lower() for error in result["errors"])


def test_each_dataset_must_pass():
    checker = _tool("check_performance")
    report, requirements = _evidence()
    for row in report["records"]:
        if row["dataset"] == "heterogeneous" and row["backend"] == "rust" and row["cells"] > 1:
            row["first_seconds"] = 20.0
    assert not checker.check_report(report, requirements)["passed"]


def test_four_worker_claim_requires_four_physical_cores():
    checker = _tool("check_performance")
    report, requirements = _evidence()
    report["physical_cores"] = 3
    assert not checker.check_report(report, requirements)["passed"]


@pytest.mark.parametrize("problem", ["duplicate", "missing_pair", "missing_warm", "nan", "zero", "interval"])
def test_incomplete_or_unstable_samples_cannot_pass(problem):
    checker = _tool("check_performance")
    report, requirements = _evidence()
    rows = [row for row in report["records"] if row["backend"] == "rust" and row["cells"] == 100_000
            and row["operation"] == "surface" and row["threads"] == 1 and row["dataset"] == "uniform"]
    if problem == "duplicate":
        report["records"].append(copy.deepcopy(rows[0]))
    elif problem == "missing_pair":
        report["records"].remove(rows[0])
    elif problem == "missing_warm":
        rows[0]["warm_seconds"].pop()
    elif problem in ("nan", "zero"):
        rows[0]["first_seconds"] = float("nan") if problem == "nan" else 0.0
    else:
        for row, seconds in zip(rows, [3.0, 3.0, 3.0, 5.0, 6.0, 6.0, 6.0]):
            row["first_seconds"] = seconds
    assert not checker.check_report(report, requirements)["passed"]


def test_mixed_case_layouts_are_distinct_and_reproducible():
    import numpy as np
    from behave_fire.python import BehaveRun, FuelModels

    cases = _tool("performance_cases")
    inputs = cases.build_inputs(16, "heterogeneous", 20261105)
    repeat = cases.build_inputs(16, "heterogeneous", 20261105)
    np.testing.assert_array_equal(inputs["fuel_model_grid"], [1, 4, 10, 101, 124, 145, 201, 0] * 2)
    for field in ("m1h", "m10h", "m100h", "mlh", "mlw", "wind_speed", "slope"):
        np.testing.assert_array_equal(inputs[field], repeat[field])
    model = BehaveRun(FuelModels())
    first = cases.supplied_surface(inputs, "typed-crown", model)
    second = cases.supplied_surface(inputs, "typed-crown-b", model)
    assert first["spread_rate"].dtype == np.float32
    assert first["fireline_intensity"].dtype == np.uint16
    np.testing.assert_array_equal(first["heat_per_unit_area"].mask, np.arange(16) % 7 == 0)
    assert second["heat_per_unit_area"].dtype == np.float32
    assert second["fireline_intensity"].dtype == np.int32
    np.testing.assert_array_equal(second["spread_rate"].mask, np.arange(16) % 5 == 0)


@pytest.mark.parametrize("problem", ["allocation", "baseline_duplicate", "baseline_profile", "baseline_version",
                                     "memory", "validation", "provenance", "scalar", "worker_error"])
def test_non_timing_requirements_are_binding(problem):
    checker = _tool("check_performance")
    report, requirements = _evidence()
    if problem == "allocation":
        report["allocations"][0]["encoder_per_cell"] = 679
    elif problem == "baseline_duplicate":
        report["baseline"].append(copy.deepcopy(report["baseline"][0]))
    elif problem == "baseline_profile":
        report["baseline"][0]["profile"] = True
    elif problem == "baseline_version":
        report["baseline"][0]["numpy"] = "2.5.4"
    elif problem == "worker_error":
        report["errors"] = [{"error": "A worker failed"}]
    elif problem == "validation":
        report["validations"][0]["passed"] = False
    else:
        for row in report["records"]:
            if row["backend"] == "rust":
                if problem == "memory":
                    row["peak_bytes"] = 2000
                elif problem == "provenance":
                    row["extension_hash"] = "different-binary"
                elif row["cells"] == 1:
                    row["first_seconds"] = 0.001
    assert not checker.check_report(report, requirements)["passed"]


def test_numpy_version_mismatch_fails_comparison():
    checker = _tool("check_performance")
    report, requirements = _evidence()
    report["records"][0]["numpy"] = "2.5.4"
    assert not checker.check_report(report, requirements)["passed"]


def test_output_comparator_keeps_masked_raw_data_and_zero_signs():
    import numpy as np

    cases = _tool("performance_cases")
    expected = np.ma.array([-0.0, 5.0], mask=[True, False])
    actual = expected.copy()
    actual.data[0] = 0.0
    with pytest.raises(AssertionError):
        cases.assert_outputs(actual, expected)
    with pytest.raises(AssertionError):
        cases.assert_outputs(np.array([2**60], dtype=np.uint64), np.array([2**60 + 1], dtype=np.uint64))


def test_pairing_and_seed_are_reproducible():
    cases = _tool("performance_cases")
    first = cases.paired_schedule(7, 20261105, [1, 4])
    assert first == cases.paired_schedule(7, 20261105, [1, 4])
    assert first != cases.paired_schedule(7, 20261106, [1, 4])
    for block in range(7):
        entries = [item for item in first if item[0] == block]
        assert sorted((backend, threads) for _, backend, threads in entries) == [
            ("python", 1), ("rust", 1), ("rust", 4)]


@pytest.mark.parametrize("problem", ["profile", "screen"])
def test_profiled_or_screen_runs_cannot_certify(problem):
    checker = _tool("check_performance")
    report, requirements = _evidence()
    report["profile" if problem == "profile" else "preset"] = True if problem == "profile" else "screen"
    assert not checker.check_report(report, requirements)["passed"]


def test_requirements_pin_owner_approved_thresholds():
    document = json.loads((TOOLS / "performance_requirements.json").read_text(encoding="utf-8"))
    assert document["targets"] == {"small_common": 1.25, "small_typed": 1.0,
                                    "large_serial": 2.0, "large_four": 4.0}
    assert document["minimum_pairs"] == 7 and document["warm_calls"] == 5
    assert document["confidence"] == 0.95
    assert document["datasets"] == ["uniform", "heterogeneous"]
    assert document["operations"] == list(OPERATIONS)


@pytest.mark.parametrize("problem", ["missing_declaration", "empty_declaration", "blank_declaration", "wrong_declaration",
                                     "missing_source", "missing_binary", "blank_source", "blank_binary",
                                     "mixed_source", "mixed_binary", "wrong_source", "wrong_binary"])
def test_scalar_baseline_rejects_missing_mixed_or_unrelated_builds(problem):
    checker = _tool("check_performance")
    report, requirements = _evidence()
    if problem == "missing_declaration":
        requirements.pop("baseline_identity")
    elif problem == "empty_declaration":
        requirements["baseline_identity"] = {}
    elif problem == "blank_declaration":
        requirements["baseline_identity"]["source_hash"] = " "
    elif problem == "wrong_declaration":
        requirements["baseline_identity"]["source_hash"] = "unrelated-source"
    else:
        key = "source_hash" if problem.endswith("source") else "extension_hash"
        rows = report["baseline"][:1] if problem.startswith("mixed") else report["baseline"]
        for row in rows:
            if problem.startswith("missing"):
                row.pop(key)
            else:
                row[key] = " " if problem.startswith("blank") else "unrelated-build"
    result = checker.check_report(report, requirements)
    assert not result["passed"]
    assert any("baseline" in error.lower() and "identity" in error.lower() for error in result["errors"])


def test_scalar_baseline_uses_independent_preoptimization_identity():
    checker = _tool("check_performance")
    report, requirements = _evidence()
    candidate = next(row for row in report["records"] if row["backend"] == "rust")
    assert report["baseline"][0]["source_hash"] != candidate["source_hash"]
    assert report["baseline"][0]["extension_hash"] != candidate["extension_hash"]
    assert checker.check_report(report, requirements)["passed"]


def test_typed_small_parity_target_is_not_common_speed_target():
    checker = _tool("check_performance")
    report, requirements = _evidence()
    assert checker.check_report(report, requirements)["passed"]
    for row in report["records"]:
        if row["cells"] == 10_000 and row["backend"] == "rust" and row["operation"].startswith("typed-"):
            row["first_seconds"] = 10.01
    assert not checker.check_report(report, requirements)["passed"]


@pytest.mark.rust
def test_validation_worker_checks_complete_mixed_outputs_outside_timing():
    result = _benchmark_worker("rust", validation=True)
    assert result["kind"] == "validation" and result["passed"] is True
    assert result["output_fields"] == 22
    assert result["source_hash"] and result["extension_hash"] and result["numpy_build"]
    assert "first_seconds" not in result and "peak_bytes" not in result
