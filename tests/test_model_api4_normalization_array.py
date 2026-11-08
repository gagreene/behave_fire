"""The new model adapter widens once and sends primitive typed columns."""
import numpy as np
import pytest

from behave_fire import _rust as native
from behave_fire.components.fuel_models import FuelModels
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays

pytestmark = pytest.mark.rust


def _arguments(dtype):
    physical = {name: np.full(5, value, dtype=dtype) for name, value in dict(
        m1h=.06, m10h=.07, m100h=.08, mlh=.6, mlw=.9, wind_speed=88.,
        wind_direction=35., slope=12., aspect=0., canopy_cover=0., canopy_height=30.,
        crown_ratio=0., canopy_base_height=4., canopy_bulk_density=.01, moisture_foliar=100.,
        surface_ros=2., surface_fli=200., surface_hpua=1000.).items()}
    return dict(physical, fuel_model_grid=np.full(5, 10, np.int32),
                lut=build_fuel_lookup_arrays(FuelModels()))


def test_common_model_mask_is_nan_and_ids_never_coerce_through_float():
    from behave_fire.rust._model_api4 import execute_common

    arguments = _arguments(np.float32)
    arguments['m1h'] = np.ma.array(arguments['m1h'], mask=[False, True, False, False, False])
    actual = execute_common('surface', arguments)
    assert np.isnan(actual['reaction_intensity'][1])
    arguments['fuel_model_grid'] = np.array([2**53+1], dtype=np.int64)
    with pytest.raises(OverflowError):
        execute_common('surface', arguments)


@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64, np.int16])
@pytest.mark.parametrize('kind', ['surface', 'crown'])
def test_common_model_normalization_returns_float64_and_one_native_call(dtype, kind, monkeypatch):
    from behave_fire.rust._model_api4 import execute_common

    arguments = _arguments(dtype)
    calls = []
    original = native.model_batch

    def compute(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(native, 'model_batch', compute)
    actual = execute_common(kind, arguments)
    assert calls == [True]
    for field, value in actual.items():
        assert value.dtype == (np.int32 if field == 'fire_type' else np.float64)
        assert value.shape == (5,)
