"""Native registry ownership and explicit facade contracts."""
import numpy as np
import pytest

pytestmark = pytest.mark.rust
from parity_support import cases, run_case


@pytest.mark.parametrize("height,base,expected", [(10,2,.8),(0,-1,1),(-1,2,0),(np.nan,1,np.nan),(10,np.nan,np.nan)])
def test_native_canopy_ratio(height, base, expected):
    from behave_fire.rust._bridge import _call
    result = _call("model.canopy_crown_ratio", {"height":height,"base":base})
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_native_facade_constructor():
    from behave_fire.rust import BehaveRun, FuelModels
    assert BehaveRun(FuelModels(), threads=1).threads == 1


def test_public_inventory_routes_exist():
    import importlib
    import json
    from parity_support import ROOT

    entries = json.loads((ROOT / "tests/data/rust_api_inventory.json").read_text())["entries"]
    import behave_fire.rust as native
    for entry in entries:
        value = native
        for part in entry["binding_route"].removeprefix("behave_fire.rust.").split("."):
            value = getattr(value, part)


@pytest.mark.parametrize("case", cases("registries", "fuel_models_array"))
def test_registry_fixture_parity(case):
    run_case(case)


def test_registry_round_trip_has_no_aliases():
    from behave_fire.components.fuel_models import FuelModels as PythonFuelModels
    from behave_fire.rust import FuelModels
    original = PythonFuelModels()
    native = FuelModels.from_python(original)
    restored = native.to_python()
    native.fuel_models_[1]["fuel_bed_depth"] = 99
    assert original.fuel_models_[1]["fuel_bed_depth"] == 1
    assert restored.fuel_models_[1]["fuel_bed_depth"] == 1


def test_registry_snapshots_all_fuel_slots():
    from behave_fire.python import BehaveRun as PythonRun, FuelModels as PythonFuels
    from behave_fire.rust import BehaveRun, FuelModels

    original = PythonFuels()
    native = FuelModels.from_python(original)
    expected_runner = PythonRun(original)
    actual_runner = BehaveRun(native)
    original.fuel_models_[1]["fuel_bed_depth"] = 90
    native.fuel_models_[1]["fuel_bed_depth"] = 80
    shape = (263,)
    inputs = dict(fuel_model_grid=np.arange(-1,262), m1h=np.full(shape,.06), m10h=np.full(shape,.07), m100h=np.full(shape,.08), mlh=np.full(shape,.6), mlw=np.full(shape,.9), wind_speed=5., wind_speed_units=5, wind_direction=0., wind_orientation_mode="RelativeToUpslope", slope=10.)
    expected = expected_runner.do_surface_run(**inputs)
    actual = actual_runner.do_surface_run(**inputs)
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-9,atol=1e-12)
