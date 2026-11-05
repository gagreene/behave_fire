"""Characterize the explicit Python oracle and its portable Rust fixtures."""

import importlib.util
import json
from pathlib import Path
import pickle
from copy import copy, deepcopy
import subprocess
import sys

import numpy as np
import pytest

from behave_fire.behave import BehaveRun
from behave_fire.components.contain import run_contain_sim_array
from behave_fire.components.fuel_models import FuelModels
from behave_fire.components.species_master_table import SpeciesMasterTable, SpeciesMasterTableRecord
from behave_fire.components.surface import calculate_eccentricity


ROOT = Path(__file__).resolve().parents[1]


def _decode_value(value):
    if isinstance(value, dict):
        if "dtype" in value and "shape" in value and "values" in value:
            return np.array([_decode_value(item) for item in value["values"]], dtype=value["dtype"]).reshape(value["shape"])
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def _exporter():
    path = ROOT / "tools" / "export_rust_reference.py"
    assert path.is_file(), "The explicit Python reference exporter is missing"
    spec = importlib.util.spec_from_file_location("rust_reference_exporter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_explicit_namespaces_have_source_modules():
    from behave_fire.python import BehaveRun as PythonBehaveRun

    assert importlib.util.find_spec("behave_fire.python") is not None
    assert importlib.util.find_spec("behave_fire.rust") is not None
    assert PythonBehaveRun is BehaveRun


def test_export_check_detects_changed_and_missing_fixtures(tmp_path, capsys):
    exporter = _exporter()
    assert exporter.main(["--output-dir", str(tmp_path)]) == 0
    assert exporter.main(["--output-dir", str(tmp_path), "--check"]) == 0
    path = tmp_path / "tests/data/rust_api_inventory.json"
    path.write_text("{}\n", encoding="utf-8")
    assert exporter.main(["--output-dir", str(tmp_path), "--check"]) == 1
    assert "tests/data/rust_api_inventory.json" in capsys.readouterr().out
    path.unlink()
    assert exporter.main(["--output-dir", str(tmp_path), "--check"]) == 1


def test_export_comparison_accepts_roundoff_but_rejects_science_and_metadata_drift():
    exporter = _exporter()
    expected = {"inputs": {"value": 1.0}, "output": {"dtype": "float64", "shape": [1], "values": [1.0]}}
    perturbed = deepcopy(expected)
    perturbed["output"]["values"][0] = np.nextafter(1., 2.).item()
    assert exporter.contracts_match(perturbed, expected)
    perturbed["output"]["values"][0] = 1.001
    assert not exporter.contracts_match(perturbed, expected)
    perturbed = deepcopy(expected)
    perturbed["output"]["dtype"] = "float32"
    assert not exporter.contracts_match(perturbed, expected)
    perturbed = deepcopy(expected)
    perturbed["inputs"]["value"] = np.nextafter(1., 2.).item()
    assert not exporter.contracts_match(perturbed, expected)


@pytest.mark.parametrize("location", ["inputs", "output"])
@pytest.mark.parametrize("actual,expected", [(-0.0, 0.0), (0.0, -0.0)])
def test_export_comparison_rejects_changed_zero_sign(location, actual, expected):
    exporter = _exporter()
    assert not exporter.contracts_match({location: {"value": actual}}, {location: {"value": expected}})
    assert exporter.contracts_match({location: {"value": actual}}, {location: {"value": actual}})


def test_export_crown_threshold_matrix_preserves_all_four_classifications():
    documents = _exporter().build_documents()
    root_cases = documents["tests/data/rust_port/boundaries.json"]["cases"]
    core_cases = documents["rust/crates/behave-fire/tests/data/crown.json"]["cases"]
    for cases in (root_cases, core_cases):
        matches = [case for case in cases if case["id"] == "crown/classification-threshold-matrix"]
        assert len(matches) == 1
        case = matches[0]
        below, above = np.nextafter(1., 0.), np.nextafter(1., 2.)
        assert case["inputs"]["transition_ratio"]["values"] == [below] * 3 + [1.] * 3 + [above] * 3
        assert case["inputs"]["active_ratio"]["values"] == [below, 1., above] * 3
        assert case["output"]["fire_type"]["values"] == [0, 2, 2, 1, 3, 3, 1, 3, 3]
        assert set(case["output"]["fire_type"]["values"]) == {0, 1, 2, 3}


def test_export_inventory_covers_facade_and_standalone_routes():
    document = _exporter().build_inventory()
    entries = {item["python_name"]: item for item in document["entries"]}
    methods = [name for name in entries if name.startswith("behave_fire.behave.BehaveRun.")
               and not name.rsplit(".", 1)[1].startswith("_")]
    assert len(methods) == 11
    for name in ("behave_fire.components.contain.run_contain_sim_array",
                 "behave_fire.components.fire_size.FireSize.get_fire_area",
                 "behave_fire.components.slope_tool.SlopeTool.calculate_horizontal_distance",
                 "behave_fire.components.ignite.calculate_fuel_temperature"):
        assert entries[name]["rust_core"]
        assert entries[name]["binding_route"]
    assert entries["behave_fire.behave.BehaveRun"]["aliases"] == ["behave_fire.BehaveRun"]
    assert entries["behave_fire.components.surface.calculate_eccentricity"]["rust_core"] == "surface::calculate_eccentricity"


def test_export_inventory_signatures_preserve_source_annotations_and_defaults():
    entries = {entry["python_name"]: entry for entry in _exporter().build_inventory()["entries"]}
    assert entries["behave_fire.components.behave_units.speed_to_base"]["signature"] == (
        "(value: Union[float, np.ndarray], units: int) -> np.ndarray")
    for name in ("CrownDamageEquationCode", "EquationType", "GACC"):
        assert entries[f"behave_fire.components.species_master_table.{name}"]["signature"] == "(value)"
    assert entries["behave_fire.components.fuel_models.FuelModels.get_fuelbed_depth"]["signature"] == (
        "(self, fuel_model_number, length_units=None)")
    assert "moisture_foliar: Union[float, np.ndarray]=100.0" in entries[
        "behave_fire.components.crown.coerce_crown_fire_inputs"]["signature"]
    assert "equation_type=EquationType.NotSet" in entries[
        "behave_fire.components.species_master_table.SpeciesMasterTableRecord"]["signature"]


def test_export_portable_values_preserve_nonfinite_dtype_shape_and_mask():
    values = np.ma.array([[np.nan, np.inf, -np.inf, -0.0]], mask=[[0, 1, 0, 0]])
    encoded = _exporter().encode_value(values)
    json.dumps(encoded, allow_nan=False)
    assert encoded["shape"] == [1, 4]
    assert encoded["dtype"] == "float64"
    assert encoded["values"] == [{"float": "nan"}, {"float": "+inf"},
                                  {"float": "-inf"}, -0.0]
    assert encoded["mask"] == [False, True, False, False]
    assert np.signbit(encoded["values"][3])


def test_export_provenance_tracks_reference_implementation_only(tmp_path, monkeypatch):
    exporter = _exporter()
    source = tmp_path / "src" / "behave_fire"
    (source / "components").mkdir(parents=True)
    (source / "rust").mkdir()
    (source / "behave.py").write_text("# explicit Python facade\n", encoding="utf-8")
    (source / "components" / "surface.py").write_text("# reference equations\n", encoding="utf-8")
    (source / "__init__.py").write_text("# automatic backend exports\n", encoding="utf-8")
    (source / "auto.py").write_text("# dispatch\n", encoding="utf-8")
    (source / "rust" / "surface.py").write_text("# native adapter\n", encoding="utf-8")
    monkeypatch.setattr(exporter, "ROOT", tmp_path)
    monkeypatch.setattr(exporter, "SOURCE", source)
    hashes = exporter._envelope({})["provenance"]["source_sha256"]
    assert set(hashes) == {
        "src/behave_fire/behave.py", "src/behave_fire/components/surface.py",
    }


def test_export_registry_receivers_replay_custom_clear_and_species_insert():
    exporter = _exporter()
    documents = exporter.build_documents()
    cases = {case["id"]: case for case in documents["tests/data/rust_port/registries.json"]["cases"]}
    clear = cases["FuelModels.clear_custom_fuel_model/baseline"]
    assert "receiver_before" in clear
    assert clear["receiver_before"]["base"] == {"reference": "fuels:standard_records"}
    fuels = FuelModels()
    fuels.fuel_models_ = {slot["number"]: deepcopy(slot["record"])
                         for slot in documents["tests/data/rust_port/fuels.json"]["slots"] if slot["record"] is not None}
    for change in clear["receiver_before"]["overrides"]:
        fuels.fuel_models_[change["number"]] = _decode_value(change["record"])
    for number in clear["receiver_before"]["removed"]:
        del fuels.fuel_models_[number]
    assert fuels.get_fuelbed_depth(250) == 2.0
    assert fuels.clear_custom_fuel_model(**clear["inputs"]) == clear["output"] is True
    assert fuels.get(250) is None
    assert clear["receiver_after"] == {"registry": "FuelModels", "base": {"reference": "fuels:standard_records"},
                                        "overrides": [], "removed": []}
    for name in ("insert_record", "add_species_record"):
        case = cases[f"SpeciesMasterTable.{name}/baseline"]
        standard = documents["tests/data/rust_port/species.json"]["records"]
        table = SpeciesMasterTable()
        table.records = [SpeciesMasterTableRecord(**deepcopy(standard[item["reference_index"]]))
                         if "reference_index" in item else SpeciesMasterTableRecord(**deepcopy(item["record"]))
                         for item in case["receiver_before"]["records"]]
        arguments = deepcopy(case["inputs"])
        if name == "add_species_record":
            arguments["record"] = SpeciesMasterTableRecord(**arguments["record"])
        getattr(table, name)(**arguments)
        assert len(table.records) == 25
        assert exporter.encode_value(vars(table.records[-1])) == case["receiver_after"]["records"][-1]["record"]
        assert table.records[-1].scientific_name == "Contract species"
        assert table.records[-1].species_code == ("pipo" if name == "insert_record" else "CUSTOM")
        expected = {"species_code": "pipo", "scientific_name": "Contract species", "common_name": "Contract species",
                    "mortality_equation_number": 2, "bark_equation_number": 31, "crown_coefficient_code": 12,
                    "alaska": False, "california": True, "eastern_area": False, "great_basin": True,
                    "northern_rockies": True, "northwest": True, "rocky_mountain": False,
                    "southern_area": False, "southwest": True, "equation_type": 0, "crown_damage_equation_code": -1}
        if name == "add_species_record":
            expected.update(species_code="CUSTOM", mortality_equation_number=-1, bark_equation_number=-1,
                            crown_coefficient_code=0, california=False, great_basin=False, northern_rockies=False,
                            northwest=False, southwest=False, equation_type=-1)
        assert exporter.encode_value(vars(table.records[-1])) == expected


def test_export_scenarios_cover_static_facade_methods_and_stateful_helpers():
    documents = _exporter().build_documents()
    records = documents["tests/data/rust_port/behave.json"]["cases"]
    assert len({case["entry_point"] for case in records}) == 11
    for domain in ("fire_size", "slope_tool", "unit_classes", "registries"):
        assert documents[f"tests/data/rust_port/{domain}.json"]["cases"]
    for record in records:
        if record["id"].endswith("/baseline"):
            assert "exception" not in record, record


def test_export_science_cases_cover_lightning_equations_and_terminal_states():
    documents = _exporter().build_documents()
    cases = documents["tests/data/rust_port/boundaries.json"]["cases"]
    ids = {case["id"] for case in cases}
    assert {f"lightning/charge-{charge}" for charge in range(3)} <= ids
    assert {"mortality/all-equations", "containment/mixed-terminal", "surface/all-fuel-slots"} <= ids
    mixed = next(case for case in cases if case["id"] == "containment/mixed-terminal")
    assert mixed["output"]["status"]["values"] == [3, 4, 8]


def test_first_contract_difference_names_signature_path_and_respects_output_tolerance():
    exporter = _exporter()
    assert hasattr(exporter, "first_contract_difference"), "Portable drift diagnostics are missing"
    expected = {"entries": [{"signature": "(value)"}]}
    actual = {"entries": [{"signature": "(*values)"}]}
    assert exporter.first_contract_difference(actual, expected) == (
        "$.entries[0].signature: actual='(*values)', expected='(value)'")
    assert exporter.first_contract_difference(
        {"output": 1. + 1e-12}, {"output": 1.}) is None


def test_float16_eccentricity_normalization_changes_overflow_contract():
    with np.errstate(all="ignore"):
        original = calculate_eccentricity(np.array([300], dtype=np.float16))
        normalized = calculate_eccentricity(np.array([300], dtype=np.float32))
    assert original.dtype == np.float16
    assert np.isposinf(original[0])
    assert normalized.dtype == np.float32
    assert normalized[0] == pytest.approx(0.9999944, abs=1e-7)


@pytest.mark.parametrize("restore", [copy, deepcopy, lambda model: pickle.loads(pickle.dumps(model))])
def test_model_copy_and_pickle_preserve_cached_fuel_snapshot(restore):
    fuels = FuelModels()
    runner = BehaveRun(fuels)
    original_depth = runner._lut["depth"][1]
    fuels.fuel_models_[1]["fuel_bed_depth"] = 91.0
    restored = restore(runner)
    assert restored._lut["depth"][1] == original_depth == 1.0
    assert restored._fuel_models.fuel_models_[1]["fuel_bed_depth"] == 91.0


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_protocols_retain_snapshots_and_custom_registries(protocol):
    fuels = FuelModels()
    fuels.set_custom_fuel_model(250, "CUSTOM", "Custom contract", 2, 0, .24, 0,
                               8000, 8000, 0, .01, .02, .03, .04, .05, 0,
                               1800, 1500, 1200, 0, True)
    runner = BehaveRun(fuels)
    fuels.fuel_models_[250]["fuel_bed_depth"] = 19.0
    restored = pickle.loads(pickle.dumps(runner, protocol=protocol))
    assert restored._lut["depth"][250] == 2.0
    assert restored._fuel_models.get_fuelbed_depth(250) == 19.0


def test_reference_documents_are_deterministic_and_match_reviewed_fixtures():
    exporter = _exporter()
    first = exporter.build_documents()
    second = exporter.build_documents()
    assert first == second
    for name, document in first.items():
        expected = json.loads((ROOT / name).read_text(encoding="utf-8"))
        assert exporter.contracts_match(document, expected), f"{name}: {exporter.first_contract_difference(document, expected)}"
        assert document["provenance"]["python_reference_sha"] == "7933789f00470c6bbfbe942dc22bdd7f74e192f9"


def test_root_class_identity_and_explicit_python_subclassing():
    from behave_fire import BehaveRun as RootBehaveRun

    class CustomRun(BehaveRun):
        pass

    assert RootBehaveRun is BehaveRun
    assert isinstance(CustomRun(FuelModels()), BehaveRun)


def test_spawn_round_trip_retains_snapshot_after_registry_mutation(tmp_path):
    fuels = FuelModels()
    runner = BehaveRun(fuels)
    fuels.fuel_models_[1]["fuel_bed_depth"] = 91.0
    payload = tmp_path / "snapshot.pkl"
    payload.write_bytes(pickle.dumps(runner))
    script = tmp_path / "spawn_probe.py"
    # A plain interpreter avoids pytest's faulthandler reporting a handled
    # Windows CreateProcess exception. Worker failures still fail this test.
    script.write_text('''from multiprocessing import get_context
from pathlib import Path
import json
import pickle
import sys

sys.path.insert(0, sys.argv[1])

def snapshot(queue, model):
    queue.put([float(model._lut["depth"][1]),
               float(model._fuel_models.fuel_models_[1]["fuel_bed_depth"])])

if __name__ == "__main__":
    model = pickle.loads(Path(sys.argv[2]).read_bytes())
    context = get_context("spawn")
    queue = context.Queue()
    worker = context.Process(target=snapshot, args=(queue, model))
    worker.start()
    worker.join(30)
    if worker.is_alive():
        worker.terminate()
        worker.join()
        raise AssertionError("spawn worker timed out")
    assert worker.exitcode == 0, worker.exitcode
    print(json.dumps(queue.get(timeout=5)))
''', encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), str(ROOT / "src"), str(payload)],
                            capture_output=True, text=True, timeout=40)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout) == [1.0, 91.0]


def test_terminal_containment_cells_freeze_while_active_cells_advance():
    arguments = dict(report_spread_rate=np.array([1.0, 1.0, 1.0]), lw_ratio=1.0,
                     report_size=np.array([1.0, 1e-8, 1.0]),
                     production_rate=np.array([1000.0, 0.0, 0.0]),
                     attack_time=np.array([0.0, 1000.0, 1000.0]))
    early = run_contain_sim_array(**arguments, max_steps=2)
    late = run_contain_sim_array(**arguments, max_steps=10)
    np.testing.assert_array_equal(late["status"], [3, 4, 8])
    assert late["status"].dtype == np.int32
    np.testing.assert_array_equal(late["final_area"][:2], early["final_area"][:2])
    assert late["final_area"][2] > early["final_area"][2]
