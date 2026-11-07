"""Allocation certification checks raw scoped observations and dual-build identity."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _evidence():
    build = {"source_hash": "a" * 64, "extension_hash": "b" * 64,
             "source_files": {"rust/crates/behave-fire/src/lib.rs": "c" * 64},
             "compiler": "rustc test", "target": "test-target", "profile": "release", "features": [], "build_environment": {}}
    instrumented = copy.deepcopy(build)
    instrumented.update(extension_hash="d" * 64, features=["allocation-probe"])
    manifest = {"schema_version": 1, "timed": build, "instrumented": instrumented,
                "patch_sha256": hashlib.sha256(b"").hexdigest(), "source_differences": []}
    rows = []
    for workers in (1, 4):
        for cells in (1000, 10000):
            for repeat in range(3):
                rows.append({"dataset": "uniform", "operation": "surface", "cells": cells,
                             "workers": workers, "repeat": repeat, "process_id": len(rows) + 1,
                             "instrumented_extension_hash": "d" * 64,
                             "instrumented_source_hash": "a" * 64,
                             "manifest_sha256": _digest(manifest), "passed": True,
                             "scopes": {name: {"requests": 0, "bytes": 0, "calls": cells, "workers": 1 if cells == 1000 else workers,
                                               "definition": name + "-loop-v1"}
                                        for name in ("kernel", "encoder")},
                             "outside_scopes": {"requests": 30, "bytes": 10000}})
    return {"manifest": manifest, "manifest_sha256": _digest(manifest),
            "raw": rows, "raw_sha256": _digest(rows)}


def _tool():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("allocation_assembler", TOOLS / "assemble_allocation_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _updated(evidence):
    evidence["manifest_sha256"] = _digest(evidence["manifest"])
    for row in evidence["raw"]:
        row["manifest_sha256"] = evidence["manifest_sha256"]
    evidence["raw_sha256"] = _digest(evidence["raw"])
    return evidence


def test_distinct_instrumented_binary_with_identical_science_is_accepted():
    result = _tool().assemble(_evidence(), {"datasets": ["uniform"], "operations": ["surface"]})
    row, = result["allocations"]
    assert row["kernel_per_cell"] == row["encoder_per_cell"] == 0
    assert row["extension_hash"] == "b" * 64
    assert row["instrumented_extension_hash"] == "d" * 64


@pytest.mark.parametrize("fault", ["encoder", "sample", "worker", "repeat", "scope", "fixed",
                                   "identity", "manifest", "science", "duplicate", "parity", "hash",
                                   "calls", "bytes", "feature", "profile", "reused-process", "fake-workers"])
def test_invalid_evidence_cannot_certify_zero_allocations(fault):
    evidence = _evidence()
    row = evidence["raw"][0]
    if fault == "encoder":
        row["scopes"]["encoder"] = None
    elif fault == "sample":
        evidence["raw"].pop()
    elif fault == "worker":
        evidence["raw"] = [r for r in evidence["raw"] if r["workers"] == 1]
    elif fault == "repeat":
        evidence["raw"] = [r for r in evidence["raw"] if r["repeat"] != 2]
    elif fault == "scope":
        row["scopes"]["encoder"]["definition"] = "whole-batch"
    elif fault == "fixed":
        for r in evidence["raw"]:
            r["scopes"]["kernel"].update(requests=1, bytes=8)
    elif fault == "identity":
        row["instrumented_extension_hash"] = "f" * 64
    elif fault == "manifest":
        evidence["manifest"] = None
    elif fault == "science":
        evidence["manifest"]["instrumented"]["source_files"]["rust/crates/behave-fire/src/lib.rs"] = "f" * 64
    elif fault == "duplicate":
        evidence["raw"].append(copy.deepcopy(row))
    elif fault == "parity":
        row["passed"] = False
    elif fault == "calls":
        row["scopes"]["kernel"]["calls"] = 0
    elif fault == "bytes":
        row["scopes"]["kernel"]["bytes"] = -1
    elif fault == "feature":
        evidence["manifest"]["timed"]["features"] = ["allocation-probe"]
    elif fault == "profile":
        evidence["manifest"]["instrumented"]["profile"] = "debug"
    elif fault == "fake-workers":
        evidence["raw"][-1]["scopes"]["kernel"]["workers"] = 1
    elif fault == "reused-process":
        evidence["raw"][1]["process_id"] = row["process_id"]
    _updated(evidence)
    if fault == "hash":
        evidence["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        _tool().assemble(evidence, {"datasets": ["uniform"], "operations": ["surface"]})


def test_numeric_zeros_without_raw_evidence_fail_checker():
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    from check_performance import _allocation_errors
    report = {"allocations": [{"dataset": "uniform", "operation": "surface", "source_hash": "a" * 64,
                               "extension_hash": "b" * 64, "kernel_per_cell": 0, "encoder_per_cell": 0}]}
    assert _allocation_errors(report, {"datasets": ["uniform"], "operations": ["surface"]}, ("a" * 64, "b" * 64))