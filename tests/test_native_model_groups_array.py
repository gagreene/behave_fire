"""One owned native call executes genuinely independent output groups."""
import numpy as np
import pytest

from behave_fire import _rust as native
from behave_fire.components.fuel_models import FuelModels
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays

pytestmark = pytest.mark.rust


def _groups():
    return [dict(outputs=['crown_critical_fire_spread_rate'], shape=[2],
                 inputs=dict(canopy_bulk_density=np.array([.01,.02])), controls={}),
            dict(outputs=['canopy_heat_per_unit_area'], shape=[3], inputs=dict(
                canopy_base_height=np.array([4.]),canopy_height=np.array([30.,40.,50.]),
                canopy_bulk_density=np.array([.01])),controls={})]


def test_all_group_snapshots_precede_the_single_owned_callback():
    groups = _groups()
    called = []

    def mutate():
        called.append(True)
        groups[0]['inputs']['canopy_bulk_density'][:] = 0.
        groups[1]['inputs']['canopy_height'][:] = 0.

    actual = native.model_groups('crown',groups,build_fuel_lookup_arrays(FuelModels()),{},_owned_callback=mutate)
    assert called == [True]
    assert np.all(actual['crown_critical_fire_spread_rate'] > 0.)
    np.testing.assert_allclose(actual['canopy_heat_per_unit_area'],[2080.,2880.,3680.])


def test_independent_groups_keep_distinct_shapes_inside_one_native_call():
    result = native.model_groups('crown',_groups(),build_fuel_lookup_arrays(FuelModels()),{})
    assert result['crown_critical_fire_spread_rate'].shape == (2,)
    assert result['canopy_heat_per_unit_area'].shape == (3,)
    np.testing.assert_allclose(result['canopy_heat_per_unit_area'],[2080.,2880.,3680.])


def test_invalid_later_group_prevents_any_owned_callback_or_execution():
    groups = _groups()
    del groups[1]['inputs']['canopy_base_height']
    called = []
    with pytest.raises(ValueError,match='missing|unbound'):
        native.model_groups('crown',groups,build_fuel_lookup_arrays(FuelModels()),{},
                            _owned_callback=lambda: called.append(True))
    assert not called
