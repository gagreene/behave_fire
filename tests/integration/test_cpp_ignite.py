"""Verify the approved input boundary for the pinned Ignite corpus."""

import sys
from importlib import import_module
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
ignite = import_module("cpp_ignite")

pytestmark = pytest.mark.cpp_reference


# ---- input validation tests ----

def test_ignite_case_reader_rejects_signed_zero(tmp_path):
    path = tmp_path / ignite.DATASET / "cases.jsonl"
    path.parent.mkdir(parents=True)
    original = (ROOT / ignite.DATASET / "cases.jsonl").read_bytes()
    assert original.count(b'"moisture_1h":0}') == 1
    path.write_bytes(original.replace(b'"moisture_1h":0}', b'"moisture_1h":-0}'))
    with pytest.raises(ValueError, match="Ignite inputs differ"):
        ignite.load_cases(tmp_path)
