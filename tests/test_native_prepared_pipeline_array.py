"""Public model pipelines retain one owned snapshot and one native compute call."""

import numpy as np
import pytest
from parity_support import ROOT
from float64_model_reference import model_reference

pytestmark = pytest.mark.rust


def _inputs(count=16):
    return dict(fuel_model_grid=np.full(count, 10, dtype=np.int32),
                m1h=np.full(count, .06), m10h=np.full(count, .07), m100h=np.full(count, .08),
                mlh=np.full(count, .6), mlw=np.full(count, .9), wind_speed=np.full(count, 8.),
                wind_speed_units=0, wind_direction=np.full(count, 35.), slope=np.full(count, 12.),
                wind_orientation_mode="RelativeToUpslope")


@pytest.mark.parametrize("route", ["surface", "crown"])
@pytest.mark.parametrize("storage", ["plain", "strided", "readonly"])
@pytest.mark.parametrize("varying_units", [False, True])
def test_fused_nonbase_units_preserve_reference_outputs(route, storage, varying_units):
    from behave_fire.python import BehaveRun as PythonRun, FuelModels
    from behave_fire.rust import BehaveRun
    from parity_support import assert_result, encode_value

    arguments = _inputs(2)
    arguments.update(moisture_units=1, slope_units=1,
                     wind_speed_units=[0, 1] if varying_units else 5,
                     canopy_height=10., canopy_height_units=[0, 4] if varying_units else 4,
                     canopy_base_height=2., canopy_base_height_units=4)
    for field, value in list(arguments.items()):
        if isinstance(value, np.ndarray):
            if storage == "strided":
                arguments[field] = value[::-1]
            elif storage == "readonly":
                value.flags.writeable = False
    python = PythonRun(FuelModels())
    rust = BehaveRun(FuelModels())
    if route == "surface":
        expected = python.do_surface_run(**arguments)
        actual = rust.do_surface_run(**arguments)
    else:
        supplied = dict(spread_rate=np.array([2., 3.], np.float32),
                        fireline_intensity=np.array([100, 200], np.uint16),
                        heat_per_unit_area=np.ma.array([1000., 2000.], mask=[False, True], fill_value=-17.))
        arguments.update(canopy_bulk_density=.1, canopy_bulk_density_units=1)
        expected = model_reference(python.do_crown_run, dict(surface_results=supplied, **arguments))
        actual = rust.do_crown_run(surface_results=supplied, **arguments)
    assert_result(actual, encode_value(expected))
@pytest.mark.parametrize("route", ["surface", "crown"])
def test_parallel_pipeline_profiles_direct_output_without_concatenation(monkeypatch, route):
    from behave_fire.python import BehaveRun as PythonRun, FuelModels
    from behave_fire.rust import BehaveRun, _bridge

    arguments = _inputs(10000)
    supplied = PythonRun(FuelModels()).do_surface_run(**arguments)
    original = _bridge._native.model_batch
    observed = []

    def profile(*args, **kwargs):
        kwargs["profile"] = True
        result = original(*args, **kwargs)
        if args[0] in ('surface', 'crown'):
            observed.append((float(result["__timing.appended_bytes"]),
                             float(result["__timing.concatenate_seconds"])))
        return result

    monkeypatch.setattr(_bridge._native, "model_batch", profile)
    model = BehaveRun(FuelModels(), threads=4)
    if route == "surface":
        model.do_surface_run(**arguments)
    else:
        model.do_crown_run(surface_results=supplied, **arguments)
    assert len(observed) == 1
    assert observed[0] == (0., 0.)

@pytest.mark.parametrize("route", ["surface", "crown"])
def test_public_model_uses_one_native_compute_call(monkeypatch, route):
    from behave_fire.python import BehaveRun as PythonRun, FuelModels
    from behave_fire.rust import BehaveRun, _bridge

    arguments = _inputs()
    supplied = PythonRun(FuelModels()).do_surface_run(**arguments)
    model = BehaveRun(FuelModels())
    calls = []
    original = _bridge._native.model_batch

    def observe(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(_bridge._native, "model_batch", observe)
    if route == "surface":
        model.do_surface_run(**arguments)
        assert calls == ['surface']
    else:
        model.do_crown_run(surface_results=supplied, canopy_base_height=4., canopy_height=30.,
                           canopy_bulk_density=.01, moisture_foliar=100., **arguments)
        assert calls == ['crown']
@pytest.mark.parametrize("route", ["surface", "crown"])
def test_registry_is_sent_once_as_numeric_rows(monkeypatch, route):
    from behave_fire.python import BehaveRun as PythonRun, FuelModels
    from behave_fire.rust import BehaveRun, _bridge

    arguments = _inputs()
    supplied = PythonRun(FuelModels()).do_surface_run(**arguments)
    original = _bridge._native.model_batch
    seen = []

    def observe(operation, arrays, controls, registry, *args, **kwargs):
        if operation in ('surface', 'crown'):
            assert registry is not None
            assert len(registry['depth']) == 257
            assert registry['is_defined'].dtype == np.bool_
            assert not any(key.startswith("model.") for key in arrays)
            seen.append(operation)
        return original(operation, arrays, controls, registry, *args, **kwargs)

    monkeypatch.setattr(_bridge._native, "model_batch", observe)
    model = BehaveRun(FuelModels())
    if route == "surface":
        model.do_surface_run(**arguments)
    else:
        model.do_crown_run(surface_results=supplied, **arguments)
    assert len(seen) == 1
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("explicit_ratio", [False, True])
def test_user_waf_discards_validated_canopy_selector_shapes(empty, explicit_ratio):
    from behave_fire.python import BehaveRun as PythonRun, FuelModels
    from behave_fire.rust import BehaveRun
    from parity_support import assert_result, encode_value

    arguments = _inputs(2)
    arguments.update(waf_method="UserInput", user_waf=.4,
                     canopy_height=[] if empty else [10., 20., 30.],
                     canopy_height_units=[] if empty else [0, 0, 0])
    if explicit_ratio:
        arguments.update(crown_ratio=[.1, .2, .3], crown_ratio_units=0)
    expected = PythonRun(FuelModels()).do_surface_run(**arguments)
    actual = BehaveRun(FuelModels()).do_surface_run(**arguments)
    assert_result(actual, encode_value(expected))
