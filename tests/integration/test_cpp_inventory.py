"""Check the offline public API inventory and its fail-closed coverage contracts."""

import importlib.util
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpp_reference


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("cpp_inventory", ROOT / "tools/cpp_inventory.py")


def _inventory():
    return json.loads((ROOT / "tests/data/cpp_reference/inventory.json").read_text(encoding="utf-8"))


def _snapshot_inventory(tmp_path):
    inventory = _inventory()
    source = ROOT / 'tests/data/cpp_reference'
    reference = tmp_path / 'tests/data/cpp_reference'
    executable_routes = [route for route in inventory['routes'] if route['executable_now']]
    tested_functions = [row for row in inventory['public_functions'] if row['function_status'] == 'tested']
    for filename in {item['file'] for row in executable_routes + tested_functions
                     for item in row['execution_evidence']}:
        path = reference / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / filename, path)
    path = reference / 'inventory.json'
    shutil.copyfile(source / path.name, path)
    return path


def _validator():
    module = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(module)
    return module.validate_inventory


# ---- inventory location and qualified coverage tests ----

def test_inventory_accepts_qualified_scientific_coverage():
    inventory = _inventory()
    summary = _validator()(inventory)
    assert summary["public_functions"] == 1394
    assert summary["scientific_candidates"] == 266
    assert summary["state_lifecycle_methods"] == 508
    assert summary["tested_functions"] == 12
    assert summary['executable_scientific_bundles'] == 3


def test_inventory_binds_only_direct_fire_size_witnesses():
    inventory = _inventory()
    expected = {
        'FireSize::calculateFireBasicDimensions': 'calculate_fire_basic_dimensions',
        'FireSize::getFireArea': 'get_fire_area',
        'FireSize::getFireLengthToWidthRatio': 'get_fire_length_to_width_ratio',
        'FireSize::getFirePerimeter': 'get_fire_perimeter',
    }
    tested = [row for row in inventory['public_functions']
              if row['function_status'] == 'tested' and (row['owner'] == 'FireSize' or row['dataset'] == 'fire_size')]
    route = next(row for row in inventory['routes'] if row['route_id'] == 'geometry')
    assert route['executable_now'] and route['dataset'] == 'fire_size'
    assert {row['qualified_name'] for row in tested} == expected.keys()
    assert set(route['method_ids']) == {row['id'] for row in tested}
    for row in tested:
        method = expected[row['qualified_name']]
        assert row['route_ids'] == ['geometry']
        assert row['dataset'] == 'fire_size' and row['execution_evidence'] == route['execution_evidence']
        assert [target['target'] for target in row['python_targets']] == [
            f'behave_fire.components.fire_size.FireSize.{method}']
        assert [target['target'] for target in row['rust_targets']] == [f'behave_fire::fire_size::<FireSize>::{method}']


def test_inventory_binds_only_direct_ignite_witnesses():
    inventory = _inventory()
    expected = {
        'Ignite::calculateFirebrandIgnitionProbability': 'calculate_firebrand_ignition_probability',
        'Ignite::getFuelTemperature': 'calculate_fuel_temperature',
    }
    tested = [row for row in inventory['public_functions']
              if row['function_status'] == 'tested' and (row['owner'] == 'Ignite' or row['dataset'] == 'ignite')]
    route = next(row for row in inventory['routes'] if row['route_id'] == 'ignite')
    assert route['executable_now'] and route['dataset'] == 'ignite'
    assert {row['qualified_name'] for row in tested} == expected.keys()
    assert set(route['method_ids']) == {row['id'] for row in tested}
    for row in tested:
        method = expected[row['qualified_name']]
        assert row['route_ids'] == ['ignite']
        assert row['dataset'] == 'ignite' and row['execution_evidence'] == route['execution_evidence']
        assert [target['target'] for target in row['python_targets']] == [f'behave_fire.components.ignite.{method}']
        assert [target['target'] for target in row['rust_targets']] == [f'behave_fire::ignite::{method}']


@pytest.mark.parametrize('dataset', ['fire_size', 'ignite', 'surface'])
@pytest.mark.parametrize('artifact', ['cases.jsonl', 'manifest.json', 'results.jsonl'])
@pytest.mark.parametrize('mutation', ['missing', 'modified'])
def test_inventory_cli_uses_the_selected_snapshots_execution_artifacts(tmp_path, dataset, artifact, mutation):
    path = _snapshot_inventory(tmp_path)
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/cpp_inventory.py'),
                             '--check', '--inventory', str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    evidence = path.parent / dataset / artifact
    assert evidence.is_file()
    if mutation == 'missing':
        evidence.unlink()
    else:
        evidence.write_bytes(evidence.read_bytes() + b'\n')
    result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/cpp_inventory.py'),
                             '--check', '--inventory', str(path)], capture_output=True, text=True)
    assert result.returncode != 0, 'The checker silently used the original checkout instead of the selected snapshot'
    assert 'execution evidence' in result.stderr
    assert f'{dataset}/{artifact}' in result.stderr


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
    approvals_active = policy['approved'] and policy['fire_size']['approved'] and policy['ignite']['approved']
    assert result.stdout.strip() == ('READY' if approvals_active else 'QUALIFICATION_REJECTED')


# ---- inventory validation and readiness tests ----

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
    inventory = json.loads(path.read_text(encoding='utf-8'))
    for row in inventory['public_functions']:
        if row['function_status'] == 'tested' and row['dataset'] == 'fire_size':
            row.update(function_status='pending_pilot', dataset=None, execution_evidence=[], approval=None)
    with pytest.raises(ValueError, match='witness'):
        _validator()(inventory, reference_root=path.parent)


@pytest.mark.parametrize('inactive', ['surface', 'fire_size', 'ignite'])
def test_inventory_requires_qualified_policy_for_readiness(tmp_path, inactive):
    path = _snapshot_inventory(tmp_path)
    inventory = json.loads(path.read_text(encoding='utf-8'))
    policy = {'approved': inactive != 'surface', 'fire_size': {'approved': inactive != 'fire_size'},
              'ignite': {'approved': inactive != 'ignite'}}
    (path.parent / 'tolerance-policy.json').write_text(json.dumps(policy), encoding='utf-8')
    _validator()(inventory, reference_root=path.parent)
    with pytest.raises(ValueError, match='Scientific qualification policy is inactive'):
        _validator()(inventory, require_ready=True, reference_root=path.parent)
