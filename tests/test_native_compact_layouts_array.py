"""Ordinary mixed Crown layouts preserve complete output contracts after fusion."""

import numpy as np
import pytest

from parity_support import assert_result, encode_value

pytestmark = pytest.mark.rust


def _inputs():
    return dict(fuel_model_grid=np.full(8, 10, dtype=np.int32),
                m1h=np.full(8, .06), m10h=np.full(8, .07), m100h=np.full(8, .08),
                mlh=np.full(8, .6), mlw=np.full(8, .9), wind_speed=np.full(8, 8.),
                wind_speed_units=0, wind_direction=np.full(8, 35.), slope=np.full(8, 12.),
                wind_orientation_mode="RelativeToUpslope", canopy_base_height=4.,
                canopy_height=30., canopy_bulk_density=.01)


@pytest.mark.parametrize("layout", ["a", "b", "third", "exact-integer"])
@pytest.mark.parametrize("storage", ["plain", "strided", "readonly", "endian"])
def test_compact_layouts_preserve_all_values_masks_and_types(layout, storage):
    from behave_fire.python import BehaveRun as PythonRun, FuelModels
    from behave_fire.rust import BehaveRun

    near = np.float32(1e-7)
    values = np.array([-0., 0., np.nextafter(near, np.float32(0)), near,
                       np.nextafter(near, np.float32(np.inf)), -10., np.nan, np.inf])
    types = {"a": (np.float32, np.uint16, np.float64),
             "b": (np.float64, np.float32, np.float32),
             "third": (np.int16, np.float64, np.float32),
             "exact-integer": (np.uint64, np.float32, np.float64)}[layout]
    surface = {}
    for index, (field, dtype) in enumerate(zip(
            ("spread_rate", "fireline_intensity", "heat_per_unit_area"), types)):
        data = values.copy() if np.issubdtype(dtype, np.floating) else np.arange(8)
        if layout == "exact-integer" and index == 0:
            data = np.array([0, 1, (1 << 53) + 1, (1 << 63) + 7, (1 << 64) - 1, 2, 3, 4], np.uint64)
        data = data.astype(dtype)
        if storage == "strided":
            data = data[::-1]
        elif storage == "readonly":
            data.flags.writeable = False
        elif storage == "endian":
            data = data.astype(data.dtype.newbyteorder(">"))
        if index == 2:
            data = np.ma.array(data, mask=[False, True, False, True, False, False, True, False], fill_value=-321.)
        surface[field] = data
    with np.errstate(all="ignore"):
        expected = PythonRun(FuelModels()).do_crown_run(surface_results=surface, **_inputs())
        actual = BehaveRun(FuelModels()).do_crown_run(surface_results=surface, **_inputs())
    assert_result(actual, encode_value(expected))