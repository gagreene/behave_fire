"""Public primitive callers validate controls before private transport casts."""
import numpy as np
import pytest

pytest.importorskip('behave_fire._rust')
from behave_fire import FuelModels
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays
from behave_fire.rust import crown, surface




@pytest.mark.parametrize('value', [np.nan, np.ma.array([1], mask=[True])])
def test_public_final_selection_rejects_missing_flags(value):
    with pytest.raises((TypeError, ValueError)):
        crown.assign_final_crown_fire_behavior(value, False, *([1.]*11))


@pytest.mark.parametrize('value', [1.5, np.nan])
def test_public_particle_builder_uses_fuel_specific_identifier_rules(value):
    lookup = build_fuel_lookup_arrays(FuelModels())
    arguments = {name: np.array([.06]) for name in ('m1h','m10h','m100h','mlh','mlw')}
    if np.isfinite(value):
        with pytest.raises((TypeError, ValueError)):
            surface.build_particle_arrays(lookup, np.array([value]), **arguments)
    else:
        result = surface.build_particle_arrays(lookup, np.array([value]), **arguments)
        assert not result['is_defined'].any()
        np.testing.assert_array_equal(result['depth'], [0.])
