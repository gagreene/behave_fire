"""Dimensional SAVR checks and upstream heat-convention regression tests."""

import numpy as np
import pytest

from behave_fire.components.behave_units import HeatOfCombustionUnits, SurfaceAreaToVolumeUnits
from behave_fire.components.fuel_models import FuelModels
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays


def test_custom_metric_fuel_converts_heat_and_savr_once():
    fuels = FuelModels()
    assert fuels.set_custom_fuel_model(
        250, "MET", "Metric custom fuel", 0.3048, 4, 24.0, 1,
        19000.0, 20000.0, 1, 0.01, 0.02, 0.03, 0.04, 0.05, 0,
        60.0, 50.0, 40.0, 3, True,
    )
    assert fuels.set_custom_fuel_model(
        251, "IMP", "Imperial reference fuel", 1.0, 0, 0.24, 0,
        8162.248, 8591.84, 0, 0.01, 0.02, 0.03, 0.04, 0.05, 0,
        1828.8, 1524.0, 1219.2, 0, True,
    )
    lut = build_fuel_lookup_arrays(fuels)
    for key in ("depth", "moe_dead", "hoc_dead", "hoc_live", "dead_1h",
                "dead_10h", "dead_100h", "live_herb", "live_woody",
                "savr_1h", "savr_lh", "savr_lw"):
        np.testing.assert_allclose(lut[key][250], lut[key][251], rtol=1e-6, atol=1e-10)
    assert fuels.get_savr_one_hour(250, 3) == pytest.approx(60.0)
    assert fuels.get_savr_live_herbaceous(250, 3) == pytest.approx(50.0)
    assert fuels.get_savr_live_woody(250, 3) == pytest.approx(40.0)
    assert fuels.get_heat_of_combustion_dead(250, 1) == pytest.approx(19000.0, rel=1e-5)
    assert fuels.get_heat_of_combustion_live(250, 1) == pytest.approx(20000.0, rel=1e-5)


@pytest.mark.parametrize("shape", [(), (2, 2)])
@pytest.mark.parametrize("units,to_factor,from_factor", [(0, 1.0, 1.0), (1, 0.429592, 2.32779)])
def test_heat_matches_upstream_in_both_directions(shape, units, to_factor, from_factor):
    value = np.asarray(19000.0) if not shape else np.array([[0.0, 1.0], [19000.0, 20000.0]])
    converted = HeatOfCombustionUnits.toBaseUnits(value, units)
    restored = HeatOfCombustionUnits.fromBaseUnits(value, units)
    np.testing.assert_allclose(converted, value * to_factor, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(restored, value * from_factor, rtol=1e-6, atol=1e-10)
    assert np.shape(converted) == shape
    np.testing.assert_allclose(
        HeatOfCombustionUnits.fromBaseUnits(converted, units), value, rtol=1e-5, atol=1e-10,
    )


@pytest.mark.parametrize("shape", [(), (2, 2)])
@pytest.mark.parametrize("units,base", [(0, 1.0), (1, 0.3048), (2, 12.0), (3, 30.48)])
def test_savr_dimensional_anchors(shape, units, base):
    value = np.asarray(1.0) if not shape else np.array([[0.0, 1.0], [2.0, 60.0]])
    converted = SurfaceAreaToVolumeUnits.toBaseUnits(value, units)
    np.testing.assert_allclose(converted, value * base, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(
        SurfaceAreaToVolumeUnits.fromBaseUnits(value * base, units), value, rtol=1e-6, atol=1e-10,
    )
    assert np.shape(converted) == shape
