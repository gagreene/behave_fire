"""Check the offline public API inventory and its fail-closed coverage contracts."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpp_reference


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("cpp_inventory", ROOT / "tools/cpp_inventory.py")


def _inventory():
    return json.loads((ROOT / "tests/data/cpp_reference/inventory.json").read_text(encoding="utf-8"))


def _snapshot_inventory(tmp_path):
    inventory = _inventory()
    source = ROOT / 'tests/data/cpp_reference'
    evidence = [{'file': 'surface/' + name, 'sha256': hashlib.sha256((source / 'surface' / name).read_bytes()
    ).hexdigest()}
                for name in ('cases.jsonl', 'results.jsonl', 'manifest.json')]
    route = next(row for row in inventory['routes'] if row['route_id'] == 'surface_single')
    route.update(executable_now=True, dataset='surface', execution_evidence=evidence)
    function = next(row for row in inventory['public_functions']
                    if row['qualified_name'] == 'Surface::doSurfaceRunInDirectionOfMaxSpread')
    function.update(function_status='tested', dataset='surface', execution_evidence=evidence,
                    approval='Synthetic artifact-binding test, not a scientific approval')
    for row in evidence:
        path = tmp_path / row['file']
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / row['file'], path)
    path = tmp_path / 'inventory.json'
    path.write_text(json.dumps(inventory))
    return path


def _validator():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module.validate_inventory


def test_inventory_accepts_qualified_surface_coverage():
    inventory = _inventory()
    summary = _validator()(inventory)
    assert summary["public_functions"] == 1394
    assert summary["scientific_candidates"] == 266
    assert summary["state_lifecycle_methods"] == 508
    assert summary["tested_functions"] == 6
    assert summary['executable_scientific_bundles'] == 1


@pytest.mark.parametrize('mutation', ['missing', 'modified'])
def test_inventory_cli_uses_the_selected_snapshots_execution_artifacts(tmp_path, mutation):
    path = _snapshot_inventory(tmp_path)
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/cpp_inventory.py'),
                             '--check', '--inventory', str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    cases = tmp_path / 'surface/cases.jsonl'
    if mutation == 'missing':
        cases.unlink()
    else:
        cases.write_bytes(cases.read_bytes() + b'\n')
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/cpp_inventory.py'),
                             '--check', '--inventory', str(path)], capture_output=True, text=True)
    assert result.returncode != 0, 'The checker silently used the original checkout instead of the selected snapshot'
    assert 'execution evidence' in result.stderr


def test_inventory_readiness_is_independent_of_module_search_path():
    script = '''
import importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('inventory_isolated', root / 'tools/cpp_inventory.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
data = json.loads((root / 'tests/data/cpp_reference/inventory.json').read_text())
try:
    module.validate_inventory(data, require_ready=True)
except ValueError:
    print('QUALIFICATION_REJECTED')
else:
    print('READY')
'''
    result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(ROOT)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    policy = json.loads((ROOT / 'tests/data/cpp_reference/tolerance-policy.json').read_text())
    assert result.stdout.strip() == ('READY' if policy['approved'] else 'QUALIFICATION_REJECTED')


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
    row = next(row for row in inventory["public_functions"] if row["scientific_candidate"]
               and row['function_status'] != 'tested')
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
    inventory = _inventory()
    for route in inventory['routes']:
        route['executable_now'] = False
    with pytest.raises(ValueError, match="executable"):
        _validator()(inventory, require_ready=True)


def test_inventory_requires_per_function_witnesses_for_an_executable_scientific_route(tmp_path):
    path = _snapshot_inventory(tmp_path)
    inventory = json.loads(path.read_text())
    for row in inventory['public_functions']:
        if row['function_status'] == 'tested':
            row.update(function_status='pending_pilot', dataset=None, execution_evidence=[], approval=None)
    with pytest.raises(ValueError, match='witness'):
        _validator()(inventory, reference_root=tmp_path)


def test_inventory_requires_qualified_policy_for_readiness(tmp_path):
    path = _snapshot_inventory(tmp_path)
    inventory = json.loads(path.read_text())
    with pytest.raises(ValueError, match='qualification'):
        _validator()(inventory, require_ready=True, reference_root=tmp_path)
