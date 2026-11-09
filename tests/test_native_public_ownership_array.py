"""Public calculations must use one snapshot across all detached stages."""

import numpy as np
import pytest

from parity_support import assert_result, encode_value

pytestmark = pytest.mark.rust
native = pytest.importorskip("behave_fire.rust", exc_type=ImportError)
from behave_fire.python import BehaveRun as PythonRun, FuelModels as PythonFuels
from behave_fire.rust import _bridge


def _mutate_after_first_copy(monkeypatch, mutate):
    calls = []

    def instrument(original):
        def batch(*args, **kwargs):
            if not calls:
                kwargs["_owned_callback"] = mutate
            calls.append(args[0])
            return original(*args, **kwargs)

        return batch

    for name in ('component_batch', 'model_batch', 'model_groups'):
        monkeypatch.setattr(_bridge._native, name, instrument(getattr(_bridge._native, name)))
    return calls


def _surface_inputs():
    return dict(fuel_model_grid=np.array([124]), m1h=np.array([.06]), m10h=np.array([.07]),
                m100h=np.array([.08]), mlh=np.array([.6]), mlw=np.array([.9]),
                wind_speed=np.array([5.]), wind_speed_units=5, wind_direction=0.,
                wind_orientation_mode="RelativeToUpslope", slope=30.,
                canopy_cover=.5, canopy_height=30., crown_ratio=.5)


def test_crown_partition_owns_later_masked_field(monkeypatch):
    crown_ros = np.ma.array([4., 6.], mask=[False, True], fill_value=-99.)
    inputs = dict(surface_fli=np.array([2.]), crown_ros=crown_ros,
                  critical_surface_fli=np.array([4.]), critical_crown_ros=np.array([8.]))
    from behave_fire.python import crown
    from component_f64_reference import reference
    expected = reference(crown.calculate_crown_fire_ratios, inputs)

    def mutate():
        crown_ros.data[:] = 40.
        crown_ros.mask[:] = False
        crown_ros.set_fill_value(777.)

    _mutate_after_first_copy(monkeypatch, mutate)
    actual = native.crown.calculate_crown_fire_ratios(**inputs)
    assert_result(actual, encode_value(expected))


def test_empty_vpd_owns_independent_temperature(monkeypatch):
    temperature = np.array([25.])
    inputs = dict(temperature=temperature, temp_units=1, relative_humidity=np.empty(0), rh_units=0)
    from behave_fire.python import vapor_pressure_deficit_calculator as reference
    expected = reference.calculate_vpd(**inputs)
    _mutate_after_first_copy(monkeypatch, lambda: temperature.fill(50.))
    assert_result(native.vapor_pressure_deficit_calculator.calculate_vpd(**inputs), encode_value(expected))


def test_facade_crown_owns_nested_surface_and_output_units(monkeypatch):
    inputs = _surface_inputs()
    reference = PythonRun(PythonFuels())
    surface = reference.do_surface_run(**inputs)
    inputs = {key: value for key, value in inputs.items() if key not in ("canopy_cover", "crown_ratio")}
    inputs.update(surface_results=surface, canopy_base_height=4., canopy_bulk_density=.01,
                  out_units={"final_spread_rate": 5})
    expected = reference.do_crown_run(**inputs)
    model = native.BehaveRun(native.FuelModels())

    def mutate():
        surface["fireline_intensity"].fill(99999.)
        surface["spread_rate"].fill(777.)
        inputs["out_units"]["final_spread_rate"] = 0

    _mutate_after_first_copy(monkeypatch, mutate)
    assert_result(model.do_crown_run(**inputs), encode_value(expected))


@pytest.mark.parametrize("method, inputs, mutated", [
    ("calculate_scorch_height", dict(fireline_intensity=np.array([100.]), fireline_intensity_units=0,
       midflame_wind_speed=np.array([5.]), wind_speed_units=5, air_temperature=np.array([80.]), temperature_units=0), "air_temperature"),
    ("calculate_spotting_from_surface_fire", dict(flame_length=np.array([4.]), flame_length_units=0,
       wind_speed=np.array([5.]), wind_speed_units=5, cover_height=np.array([20.]), cover_height_units=0), "cover_height"),
    ("calculate_fire_width", dict(forward_ros=np.array([4.]), backing_ros=np.array([2.]), ros_units=0,
       lwr=np.array([2.]), elapsed_time=1., elapsed_time_units=0), "lwr"),
])
def test_facade_static_calls_own_later_inputs(monkeypatch, method, inputs, mutated):
    expected = getattr(PythonRun, method)(**inputs)
    _mutate_after_first_copy(monkeypatch, lambda: inputs[mutated].fill(40.))
    assert_result(getattr(native.BehaveRun, method)(**inputs), encode_value(expected))


def test_facade_surface_owns_moisture_before_first_conversion(monkeypatch):
    inputs = _surface_inputs()
    expected = PythonRun(PythonFuels()).do_surface_run(**inputs)
    model = native.BehaveRun(native.FuelModels())
    _mutate_after_first_copy(monkeypatch, lambda: inputs["m10h"].fill(.7))
    assert_result(model.do_surface_run(**inputs), encode_value(expected))


@pytest.mark.parametrize("value", [np.float64(2.), np.array(2.), np.array([2.]),
                                     np.array([np.int64(2**53 + 1), np.float32(-0.)], dtype=object)])
def test_live_scalar_array_and_boxed_payload_types_match_reference(value):
    from behave_fire.python import surface

    if np.asanyarray(value).dtype.kind == 'O':
        with pytest.raises(TypeError):
            native.surface.calculate_fire_length(value, 0, 1)
        return
    expected = surface.calculate_fire_length(np.asarray(value, dtype=np.float64), 0., 1.)
    actual = native.surface.calculate_fire_length(value, 0, 1)
    if np.ndim(expected) == 0:
        assert isinstance(actual, np.ndarray)
        assert actual.shape == ()
        assert actual.dtype == np.asanyarray(expected).dtype
    else:
        assert type(actual) is type(expected)
    assert_result(actual, encode_value(expected))
    if np.asanyarray(expected).dtype == object:
        assert [type(item) for item in actual.flat] == [type(item) for item in expected.flat]


def test_owned_snapshot_reuses_views_and_preserves_nested_payload_metadata():
    from behave_fire._ownership import _owned_inputs

    values = np.ma.array([np.int64(2**53 + 1), np.float32(-0.)], dtype=object,
                         mask=[False, True], fill_value=17)
    with _owned_inputs() as owner:
        owned = owner.snapshot({"nested": [values, values], "scalar": np.float32(2.)})
        array = owned["nested"][0]
        values.data[0] = 0
        values.mask[:] = False
        values.set_fill_value(99)
        assert array is owned["nested"][1]
        assert type(array.data[0]) is np.int64
        assert array.data[0] == 2**53 + 1
        assert type(array.data[1]) is np.float32
        assert np.signbit(array.data[1])
        np.testing.assert_array_equal(array.mask, [False, True])
        assert array.fill_value == 17
        assert type(owned["scalar"]) is np.float32
        with _owned_inputs() as nested:
            assert nested.snapshot(array) is array
            view = array.data[:1]
            assert np.shares_memory(nested.snapshot(view), array.data)
            fresh = nested.remember(np.array([3.]))
            assert nested.snapshot(fresh) is fresh
    # A new public call must observe fresh caller values, never a stale memo.
    with _owned_inputs() as owner:
        assert owner.snapshot(values).data[0] == 0
