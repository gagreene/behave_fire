"""Crown fused routing preserves normalized controls and typed supplied inputs."""
import numpy as np
import pytest

pytestmark = pytest.mark.rust
_native = pytest.importorskip("behave_fire.rust", exc_type=ImportError)
from behave_fire.python import BehaveRun as PythonRun, FuelModels
from behave_fire.rust import BehaveRun, _bridge


def _inputs():
    return dict(fuel_model_grid=np.full(3, 10, dtype=np.int32),
                m1h=np.full(3, .06), m10h=np.full(3, .07), m100h=np.full(3, .08),
                mlh=np.full(3, .6), mlw=np.full(3, .9), wind_speed=np.full(3, 8.),
                wind_speed_units=0, wind_direction=np.full(3, 35.), slope=np.full(3, 12.),
                wind_orientation_mode="RelativeToUpslope")


@pytest.mark.parametrize("field,value", [("aspect", 0.), ("aspect", -90.), ("aspect", np.nan),
    ("aspect", np.inf), ("moisture_foliar", 100.), ("moisture_foliar", np.nan),
    ("moisture_foliar", np.inf), ("wind_speed_units", 1), ("slope_units", 1)])
@pytest.mark.parametrize("surface_kind", ["f64", "weak", "object", "f32", "masked", "integer"])
@pytest.mark.parametrize("orientation", ["RelativeToUpslope", "RelativeToNorth"])
def test_crown_normalized_controls_preserve_routing_and_reference(
        monkeypatch,
        field,
        value,
        surface_kind,
        orientation
):
    inputs = _inputs()
    inputs["wind_orientation_mode"] = orientation
    inputs[field] = value
    reference = PythonRun(FuelModels())
    surface = dict(spread_rate=np.full(3, 2.), fireline_intensity=np.full(3, 200.),
                   heat_per_unit_area=np.full(3, 1000.))
    if surface_kind == "weak":
        surface["spread_rate"] = 2.
    elif surface_kind == "object":
        surface["spread_rate"] = np.full(3, 2., dtype=object)
    elif surface_kind == "f32":
        surface["spread_rate"] = np.full(3, 2., dtype=np.float32)
    elif surface_kind == "masked":
        surface["spread_rate"] = np.ma.array([2., 2., 2.], mask=[False, True, False])
    elif surface_kind == "integer":
        surface["spread_rate"] = np.full(3, 2, dtype=np.uint16)
    inputs.update(canopy_base_height=4., canopy_height=30., canopy_bulk_density=.01)
    metadata = []
    original = _bridge._native.batch

    def profile(*args, **kwargs):
        kwargs["profile"] = True
        result = original(*args, **kwargs)
        if args[0] == "crown.run_crown_fire":
            metadata.append(bool(result["__schema"]))
        return result

    monkeypatch.setattr(_bridge._native, "batch", profile)
    with np.errstate(all="ignore"):
        expected = reference.do_crown_run(surface_results=surface, **inputs)
        actual = BehaveRun(FuelModels()).do_crown_run(surface_results=surface, **inputs)
    assert metadata and all(present == (surface_kind != "f64") for present in metadata)
    assert actual.keys() == expected.keys()
    for key in expected:
        assert actual[key].dtype == expected[key].dtype, key
        assert actual[key].shape == expected[key].shape, key
        assert np.ma.isMaskedArray(actual[key]) == np.ma.isMaskedArray(expected[key]), key
        np.testing.assert_array_equal(np.ma.getmaskarray(actual[key]), np.ma.getmaskarray(expected[key]))
        a = np.asarray(np.ma.getdata(actual[key]), dtype=float)
        e = np.asarray(np.ma.getdata(expected[key]), dtype=float)
        np.testing.assert_allclose(a, e, rtol=1e-9, atol=1e-12, err_msg=key)
        np.testing.assert_array_equal(np.isposinf(a), np.isposinf(e))
        np.testing.assert_array_equal(np.isneginf(a), np.isneginf(e))
