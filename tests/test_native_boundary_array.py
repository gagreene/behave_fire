"""Primitive f64 ownership, coercion, layouts, and validation boundaries."""
import numpy as np
import pytest
pytestmark = pytest.mark.rust

def test_model_calculations_never_call_reference(monkeypatch):
    import inspect
    from behave_fire import components
    from behave_fire.rust import BehaveRun, FuelModels

    def forbidden(*args, **kwargs):
        raise AssertionError("Python science function reached from explicit native model")

    for module_name in ("surface", "crown", "mortality", "spot", "behave_units"):
        module = __import__("behave_fire.components." + module_name, fromlist=[module_name])
        for name, value in vars(module).copy().items():
            if inspect.isfunction(value) and not name.startswith("_"):
                monkeypatch.setattr(module,name,forbidden)
    model = BehaveRun(FuelModels())
    inputs = dict(fuel_model_grid=np.array([124]),m1h=np.array([.06]),m10h=np.array([.07]),m100h=np.array([.08]),mlh=np.array([.6]),mlw=np.array([.9]),wind_speed=5.,wind_speed_units=5,wind_direction=0.,wind_orientation_mode="RelativeToUpslope",slope=10.)
    result = model.do_surface_run(**inputs)
    crown = model.do_crown_run(result, **inputs)
    assert result["spread_rate"][0] > 0
    assert "fire_type" in crown



def test_native_malformed_input_never_panics():
    from behave_fire.rust import surface

    for values in (np.array(["bad"], dtype=object), np.ones((2, 3))):
        try:
            surface.calculate_backing_spread_rate(values, np.ones(4))
        except (ValueError, TypeError):
            pass
        except BaseException as error:
            pytest.fail(f"unexpected native exception: {type(error).__name__}: {error}")
        else:
            pytest.fail("malformed input was accepted")



def test_native_never_calls_reference(monkeypatch):
    from behave_fire.components import safety as reference
    from behave_fire.rust import safety

    def forbidden(*args, **kwargs):
        raise AssertionError("native implementation called the Python reference")

    monkeypatch.setattr(reference, "calculate_safety_zone", forbidden)
    result = safety.calculate_safety_zone(5.0, 0, 0, 0, 0)
    assert result["separation_distance"] == 20.0
    np.testing.assert_allclose(result["area"], np.pi * 400)



@pytest.mark.parametrize("shape", [(), (0,), (3,), (2, 3), (2, 2, 3)])
def test_native_owned_layouts(shape):
    from behave_fire.rust import surface

    values = np.full(shape, 2.0)
    values.setflags(write=False)
    result = surface.calculate_eccentricity(values)
    assert result.shape == shape
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.sqrt(3) / 2)
    assert not np.shares_memory(result, values)



@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_native_precision(dtype):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    values = np.array([1, 2, 300], dtype=dtype)
    expected = reference.calculate_eccentricity(values.astype(np.float64))
    actual = surface.calculate_eccentricity(values)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-6 if expected.dtype == np.float32 else 1e-9)



@pytest.mark.parametrize("layout", ["fortran", "reverse", "slice", "broadcast", "readonly"])
def test_native_strided_array_ownership(layout):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    original = np.arange(24.0).reshape(4, 6) + 1
    values = {"fortran": np.asfortranarray(original), "reverse": original[::-1, ::-1], "slice": original[::2, 1::2], "broadcast": np.broadcast_to(original[0], (4, 6)), "readonly": original.copy()}[layout]
    if layout == "readonly":
        values.setflags(write=False)
    before = values.copy()
    actual = surface.calculate_eccentricity(values)
    expected = reference.calculate_eccentricity(values)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(values, before)
    assert not np.shares_memory(actual, values)



@pytest.mark.parametrize("kind", ["SpeedUnits", "FractionUnits", "TemperatureUnits", "SurfaceAreaToVolumeUnits"])
@pytest.mark.parametrize("unit", [-100, -1, 0.0, 1.0, 99, True, False])
def test_unit_code_behavior_matches_reference(kind, unit):
    from behave_fire.components import behave_units as reference
    from behave_fire.rust import behave_units as native

    ref = getattr(reference, kind).toBaseUnits
    actual = getattr(native, kind).toBaseUnits
    try:
        expected = ref(np.array([1.]), unit)
    except Exception as error:
        with pytest.raises(type(error)):
            actual(np.array([1.]), unit)
    else:
        result = actual(np.array([1.]), unit)
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-12)
