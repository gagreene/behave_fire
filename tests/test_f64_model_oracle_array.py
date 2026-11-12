"""The independent API4 oracle widens inputs and applies reviewed missing rules."""
import numpy as np
import pytest
from parity_support import ROOT


def test_missing_density_cannot_classify_as_valid_surface_fire():
    from float64_model_reference import apply_crown_rules
    from behave_fire.python import BehaveRun, FuelModels
    from performance_cases import build_inputs

    inputs = build_inputs(2, 'uniform')
    model = BehaveRun(FuelModels())
    supplied = dict(spread_rate=np.full(2, 2.), fireline_intensity=np.full(2, 200.),
                    heat_per_unit_area=np.full(2, 1000.))
    expected = model.do_crown_run(surface_results=supplied, **inputs,
        canopy_base_height=4., canopy_height=30., canopy_bulk_density=np.nan)
    actual = apply_crown_rules(expected, supplied, density=np.nan)
    np.testing.assert_array_equal(actual['fire_type'], [-1, -1])
    assert np.isnan(actual['final_spread_rate']).all()


def test_missing_heat_preserves_crown_spread_and_classification():
    from float64_model_reference import apply_crown_rules
    from behave_fire.python import BehaveRun, FuelModels
    from performance_cases import build_inputs

    inputs = build_inputs(3, 'uniform')
    model = BehaveRun(FuelModels())
    supplied = dict(spread_rate=np.full(3, 2.), fireline_intensity=np.full(3, 200.),
                    heat_per_unit_area=np.array([np.nan, 1000., 1000.]))
    expected = model.do_crown_run(surface_results=supplied, **inputs,
        canopy_base_height=4., canopy_height=30., canopy_bulk_density=.01)
    actual = apply_crown_rules(expected, supplied)
    assert np.isnan(actual['surface_fire_critical_spread_rate'][0])
    assert np.isfinite(actual['crown_fire_spread_rate'][0])
    assert actual['fire_type'][0] >= 0
