"""Real native calculations selected through the automatic facade."""

import inspect

import numpy as np
import pytest

pytestmark = pytest.mark.rust
native = pytest.importorskip("behave_fire._rust", exc_type=ImportError)


def _assert_result(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_result(actual[key], expected[key])
        return
    actual, expected = np.asanyarray(actual), np.asanyarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


@pytest.fixture
def _usable_native(monkeypatch):
    from behave_fire import auto
    from behave_fire import rust
    from behave_fire._backend_contract import REQUIRED_CAPABILITIES

    assert REQUIRED_CAPABILITIES.issubset(native.__capabilities__)
    monkeypatch.setattr(native, "__backend_complete__", True)
    monkeypatch.setattr(auto, "_PROBE_RESULT", None)
    monkeypatch.setattr(auto, "_WARNING_EMITTED", False)
    return rust


def test_all_eleven_instance_methods_enter_native_and_keep_signatures(_usable_native, monkeypatch):
    from behave_fire import BehaveRun, FuelModels
    from behave_fire.python import BehaveRun as PythonRun

    model = BehaveRun(FuelModels(), backend="auto", threads=3)
    reference = PythonRun(FuelModels())
    assert model.backend_in_use == "rust"
    original = native.component_batch
    calls = []

    def record(operation, *args, **kwargs):
        calls.append((operation, kwargs.get("threads", 1)))
        return original(operation, *args, **kwargs)

    monkeypatch.setattr(native, "component_batch", record)
    model_original = native.model_batch

    def record_model(kind, *args, **kwargs):
        calls.append(('model.run_surface' if kind == 'surface' else 'crown.run_crown_fire', kwargs.get('threads', 1)))
        return model_original(kind, *args, **kwargs)

    monkeypatch.setattr(native, 'model_batch', record_model)
    surface_inputs = dict(fuel_model_grid=np.array([1, 2, 10]), m1h=np.full(3, .06), m10h=np.full(3, .07),
                          m100h=np.full(3, .08), mlh=np.full(3, .6), mlw=np.full(3, .9), wind_speed=np.full(3, 8.), wind_speed_units=0,
                          wind_direction=np.full(3, 35.), wind_orientation_mode="RelativeToUpslope", slope=np.full(3, 12.))
    expected_surface = reference.do_surface_run(**surface_inputs)
    cases = {
        "calculate_crown_scorch_mortality": ((np.array([10., 20., 30.]), 50., .5, 12., 1), {}, "mortality.calculate_crown_scorch_mortality"),
        "calculate_fire_area": ((np.array([1., 2., 3.]), .5, 0, 2., 3., 0), {}, "surface.calculate_fire_area"),
        "calculate_fire_length": ((np.array([1., 2., 3.]), .5, 0, 3., 0), {}, "surface.calculate_fire_length"),
        "calculate_fire_perimeter": ((np.array([1., 2., 3.]), .5, 0, 2., 3., 0), {}, "surface.calculate_fire_perimeter"),
        "calculate_fire_width": ((np.array([1., 2., 3.]), .5, 0, 2., 3., 0), {}, "surface.calculate_fire_width"),
        "calculate_scorch_height": ((np.array([100., 200., 300.]), 0, 5., 5, 80., 0), {}, "mortality.calculate_scorch_height"),
        "calculate_spotting_from_burning_pile": ((np.array([2., 3., 4.]), 0, 10., 5, 20., 0), {}, "spot.calculate_spotting_from_burning_pile"),
        "calculate_spotting_from_surface_fire": ((np.array([2., 3., 4.]), 0, 10., 5, 20., 0), {}, "spot.calculate_spotting_from_surface_fire"),
        "calculate_spotting_from_torching_trees": ((np.array([10., 12., 14.]), 1, 50., 0, 2, 10., 5, 20., 0), {}, "spot.calculate_spotting_from_torching_trees"),
        "do_crown_run": ((), dict(surface_inputs, surface_results=expected_surface, canopy_base_height=4., canopy_height=30., canopy_bulk_density=.01), "crown.run_crown_fire"),
        "do_surface_run": ((), surface_inputs, "model.run_surface"),
    }
    assert len(cases) == 11
    for name, (args, kwargs, operation) in cases.items():
        calls.clear()
        expected = getattr(reference, name)(*args, **kwargs)
        actual = getattr(model, name)(*args, **kwargs)
        _assert_result(actual, expected)
        assert operation in [entry[0] for entry in calls], (name, calls)
        assert {entry[1] for entry in calls} == {3}, (name, calls)
        assert inspect.signature(getattr(model, name)) == inspect.signature(getattr(reference, name))
        assert inspect.signature(getattr(BehaveRun, name)) == inspect.signature(getattr(PythonRun, name))
        if isinstance(vars(PythonRun)[name], staticmethod):
            calls.clear()
            _assert_result(getattr(BehaveRun, name)(*args, **kwargs), expected)
            assert not calls, name
    assert model.backend_in_use == "rust"


def test_complete_extension_is_available_without_overrides():
    from behave_fire import BehaveRun, FuelModels, available_backends
    from behave_fire._backend_contract import REQUIRED_CAPABILITIES

    assert native.__backend_complete__ is True
    assert REQUIRED_CAPABILITIES.issubset(native.__capabilities__)
    assert available_backends() == ["python", "rust"]
    assert BehaveRun(FuelModels()).backend_in_use == "rust"


def test_incomplete_real_extension_remains_explicit_only(monkeypatch):
    from behave_fire import BehaveBackendWarning, BehaveRun, FuelModels, auto, available_backends, get_backend, rust

    monkeypatch.setattr(native, "__backend_complete__", False)
    monkeypatch.setattr(auto, "_PROBE_RESULT", None)
    monkeypatch.setattr(auto, "_WARNING_EMITTED", False)
    assert available_backends() == ["python"]
    with pytest.warns(BehaveBackendWarning, match="incomplete"):
        assert BehaveRun(FuelModels()).backend_in_use == "python"
    with pytest.raises(ImportError, match="incomplete"):
        get_backend("rust")
    _assert_result(rust.surface.calculate_fire_length(np.array([1.]), .5, 3.), np.array([4.5]))


def test_native_namespace_registry_routes_and_standalone_containment(_usable_native, monkeypatch):
    import importlib
    from behave_fire import BehaveRun, FuelModels, available_backends, get_backend
    from behave_fire.python import contain as python_contain

    rust = get_backend("rust")
    assert rust is _usable_native
    automatic = get_backend()
    assert automatic.preferred_backend == automatic.backend_in_use == 'rust'
    assert available_backends() == ["python", "rust"]
    names = ("behave_units", "contain", "crown", "fine_dead_fuel_moisture_tool", "fire_size", "fuel_models", "fuel_models_array", "ignite", "mortality", "safety", "slope_tool", "species_master_table", "spot", "surface", "vapor_pressure_deficit_calculator")
    for name in names:
        assert getattr(rust, name) is importlib.import_module("behave_fire.rust." + name)
    assert rust.RustBehaveRun is rust.BehaveRun
    assert rust.RustFuelModels is rust.FuelModels
    for name, module in (("FuelModels", "fuel_models"), ("SpeciesMasterTable", "species_master_table"), ("SpeciesMasterTableRecord", "species_master_table"), ("FireSize", "fire_size"), ("SlopeTool", "slope_tool")):
        assert getattr(rust, name) is getattr(getattr(rust, module), name)
    original_defaults, original_batch = native.defaults, native.component_batch
    calls = []

    def defaults(kind):
        calls.append(("defaults", kind))
        return original_defaults(kind)

    def batch(operation, *args, **kwargs):
        calls.append(("batch", operation))
        return original_batch(operation, *args, **kwargs)

    monkeypatch.setattr(native, "defaults", defaults)
    monkeypatch.setattr(native, "component_batch", batch)
    native_fuels = rust.FuelModels()
    species = rust.SpeciesMasterTable()
    assert species.get_number_of_records() > 0
    assert ("defaults", "fuels") in calls
    assert ("defaults", "species") in calls
    python_fuels = FuelModels()
    converted = rust.FuelModels.from_python(python_fuels)
    assert converted.fuel_models_ == python_fuels.fuel_models_
    assert converted.to_python().fuel_models_ == python_fuels.fuel_models_
    for registry in (native_fuels, python_fuels):
        assert BehaveRun(registry, backend="rust").backend_in_use == "rust"
        assert BehaveRun(registry, backend="python").backend_in_use == "python"
    model = BehaveRun(native_fuels, backend="python")
    native_fuels.fuel_models_[1]["fuel_bed_depth"] = 999.
    assert model._delegate._fuel_models.fuel_models_[1]["fuel_bed_depth"] != 999.
    args = (np.array([1., 2.]), np.array([2., 2.]), np.array([1., 2.]), np.array([30., 40.]), np.array([0., 0.]))
    actual = get_backend("rust").contain.run_contain_sim_array(*args, max_steps=10)
    _assert_result(actual, python_contain.run_contain_sim_array(*args, max_steps=10))
    assert ("batch", "contain.run_contain_sim_array") in calls
