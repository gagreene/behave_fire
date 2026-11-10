"""Public primitive callers validate controls before private transport casts."""
import numpy as np
import pytest

pytest.importorskip('behave_fire._rust')
from behave_fire import FuelModels
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays
from behave_fire.rust import contain, crown, surface


@pytest.mark.parametrize('value', [1.5, np.nan])
def test_public_containment_rejects_noninteger_iteration_controls(value):
    with pytest.raises((TypeError, ValueError)):
        contain.run_contain_sim_array(report_spread_rate=1., lw_ratio=2., report_size=1.,
                                     production_rate=30., attack_time=1., max_steps=value)


@pytest.mark.parametrize('value', [np.nan, np.ma.array([1], mask=[True])])
def test_public_final_selection_rejects_missing_flags(value):
    with pytest.raises((TypeError, ValueError)):
        crown.assign_final_crown_fire_behavior(value, False, *([1.]*11))


@pytest.mark.parametrize('value', [1.5, np.nan])
def test_public_particle_builder_rejects_invalid_identifiers(value):
    lookup = build_fuel_lookup_arrays(FuelModels())
    arguments = {name: np.array([.06]) for name in ('m1h','m10h','m100h','mlh','mlw')}
    with pytest.raises((TypeError, ValueError)):
        surface.build_particle_arrays(lookup, np.array([value]), **arguments)
