"""Migrated public models use one primitive float64 compute boundary."""
import numpy as np
import pytest

from behave_fire import _rust as native
from behave_fire.rust import BehaveRun, FuelModels

pytestmark = pytest.mark.rust


def _inputs():
    return dict(fuel_model_grid=np.array([10, 10], np.int32),
                m1h=np.full(2,.06), m10h=np.full(2,.07), m100h=np.full(2,.08),
                mlh=np.full(2,.6), mlw=np.full(2,.9), wind_speed=5., wind_speed_units=5,
                wind_direction=0., wind_orientation_mode='RelativeToUpslope', slope=10.,
                canopy_height=30.)


def test_crown_model_widens_typed_surface_fields_without_mask_emulation(monkeypatch):
    model = BehaveRun(FuelModels())
    arguments = _inputs()
    surface = model.do_surface_run(**arguments)
    surface['spread_rate'] = surface['spread_rate'].astype(np.float32)
    surface['fireline_intensity'] = surface['fireline_intensity'].astype(np.uint16)
    surface['heat_per_unit_area'] = np.ma.array(surface['heat_per_unit_area'], mask=[True, False])
    calls = []
    original = native.model_batch

    def compute(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(native, 'model_batch', compute)
    actual = model.do_crown_run(**arguments, surface_results=surface,
                                canopy_base_height=4., canopy_bulk_density=.01)
    assert calls == ['crown']
    assert not any(np.ma.isMaskedArray(value) for value in actual.values())
    assert all(value.dtype == (np.int32 if name == 'fire_type' else np.float64)
               for name, value in actual.items())
    assert np.isnan(actual['crown_fire_heat_per_unit_area'][0])
    assert np.isfinite(actual['crown_fire_spread_rate'][0])


def test_surface_model_normalizes_each_physical_operand_only_once(monkeypatch):
    from behave_fire.rust import core, _model_api4

    model = BehaveRun(FuelModels())
    calls = []
    original = _model_api4.normalize_physical

    def normalize(value):
        calls.append(True)
        return original(value)

    monkeypatch.setattr(core, 'normalize_physical', normalize)
    monkeypatch.setattr(_model_api4, 'normalize_physical', normalize)
    model.do_surface_run(**_inputs())
    # Twelve actual physical inputs (including the explicit crown ratio),
    # plus twelve immutable physical lookup properties; no repeated widening.
    assert len(calls) == 24


def test_surface_model_preserves_mask_before_unit_conversion(monkeypatch):
    model = BehaveRun(FuelModels())
    arguments = _inputs()
    arguments['m1h'] = np.ma.array([.06, .06], mask=[True, False])
    calls = []
    original = native.model_batch

    def compute(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(native, 'model_batch', compute)
    result = model.do_surface_run(**arguments)
    assert calls == ['surface']
    assert np.isnan(result['reaction_intensity'][0])
    assert np.isfinite(result['reaction_intensity'][1])


def test_surface_model_rejects_identifier_overflow_before_integer_conversion():
    arguments = _inputs()
    arguments['fuel_model_grid'] = np.full(2, 2**32 + 10, np.int64)
    with pytest.raises(OverflowError):
        BehaveRun(FuelModels()).do_surface_run(**arguments)
