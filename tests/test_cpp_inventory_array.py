"""Check the offline public API inventory and its fail-closed coverage contracts."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpp_reference


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpp_inventory", ROOT / "tools/cpp_inventory.py")


def _inventory():
    return json.loads((ROOT / "tests/data/cpp_reference/inventory.json").read_text(encoding="utf-8"))


def _validator():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module.validate_inventory


def test_inventory_accepts_complete_offline_preparation():
    inventory = _inventory()
    summary = _validator()(inventory)
    assert summary["public_functions"] == 1394
    assert summary["scientific_candidates"] == 266
    assert summary["state_lifecycle_methods"] == 508
    assert summary["tested_functions"] == 0


@pytest.mark.parametrize("mutation", [
    "drop_row", "signature", "field", "route", "reverse", "source_hash", "universal_waiver",
])
def test_inventory_rejects_broken_coverage(mutation):
    inventory = deepcopy(_inventory())
    if mutation == "drop_row":
        inventory["public_functions"].pop()
    elif mutation == "signature":
        inventory["public_functions"][0]["lexically_qualified_signature"] += " corrupted"
    elif mutation == "field":
        del inventory["public_functions"][0]["function_status"]
    elif mutation == "route":
        inventory["public_functions"][0]["route_ids"] = ["missing"]
    elif mutation == "reverse":
        target = next(row for row in inventory["python_reverse_join"] if row["cpp_ids"])
        target["cpp_ids"].pop()
    elif mutation == "source_hash":
        inventory["source_inputs"][0]["sha256"] = "invalid"
    else:
        for row in inventory["public_functions"]:
            row["function_status"] = "waived"
    with pytest.raises(ValueError):
        _validator()(inventory)


def test_inventory_rejects_false_tested_claim():
    inventory = _inventory()
    row = next(row for row in inventory["public_functions"] if row["scientific_candidate"])
    row["function_status"] = "tested"
    with pytest.raises(ValueError, match="evidence"):
        _validator()(inventory)


def test_inventory_rejects_missing_execution_artifact():
    inventory = _inventory()
    route = next(route for route in inventory["routes"] if route["route_id"] == "surface_single")
    route["executable_now"] = True
    route["execution_evidence"] = [{"file": "surface/missing.jsonl", "sha256": "0" * 64}]
    with pytest.raises(ValueError, match="evidence"):
        _validator()(inventory)


def test_inventory_requires_executable_routes_for_readiness():
    with pytest.raises(ValueError, match="executable"):
        _validator()(_inventory(), require_ready=True)
