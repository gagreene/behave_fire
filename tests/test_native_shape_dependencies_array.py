"""Independent output dependencies must not acquire unrelated input axes."""

import numpy as np
import pytest

pytestmark = pytest.mark.rust


def test_crown_ratio_fields_allow_independent_shapes():
    from behave_fire.components import crown as reference
    from behave_fire.rust import crown

    inputs = dict(surface_fli=np.ones(2), crown_ros=np.ones(3), critical_surface_fli=2., critical_crown_ros=4.)
    expected=reference.calculate_crown_fire_ratios(**inputs)
    actual=crown.calculate_crown_fire_ratios(**inputs)
    for key in actual:
        assert actual[key].shape==expected[key].shape
        np.testing.assert_allclose(actual[key],expected[key])


def test_pressure_unit_boolean_index_keeps_its_reference_axes():
    from behave_fire.components import vapor_pressure_deficit_calculator as reference
    from behave_fire.rust import vapor_pressure_deficit_calculator as native

    expected=reference.calculate_vpd(25.,1,.5,0,True)
    actual=native.calculate_vpd(25.,1,.5,0,True)
    for key in expected:
        assert actual[key].shape==expected[key].shape,key
        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-9,atol=1e-12)


def test_reaction_accepts_only_the_fields_it_reads():
    import json
    from parity_support import ROOT,decode
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    cases=json.loads((ROOT/'tests/data/rust_port/surface.json').read_text())['cases']
    values=decode(next(case for case in cases if case['id']=='calculate_reaction_intensity/baseline')['inputs'])['ib']
    names=('sigma','relative_packing_ratio','w_moist_dead','w_moist_live','moe_dead','moe_live','w_silica_dead','w_silica_live','wfl_dead','wfl_live','w_heat_dead','w_heat_live')
    partial={name:values[name] for name in names}
    expected=reference.calculate_reaction_intensity(partial)
    actual=surface.calculate_reaction_intensity(partial)
    np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-12)


def test_reaction_ignores_unrelated_dictionary_arrays():
    import json
    from parity_support import ROOT,decode
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    cases=json.loads((ROOT/'tests/data/rust_port/surface.json').read_text())['cases']
    values=decode(next(case for case in cases if case['id']=='calculate_reaction_intensity/baseline')['inputs'])['ib']
    values['unrelated_metadata']=np.ones((3,4))
    expected=reference.calculate_reaction_intensity(values)
    actual=surface.calculate_reaction_intensity(values)
    assert actual.shape==expected.shape
    np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-12)


def test_safety_separation_survives_empty_personnel_array():
    from behave_fire.components import safety as reference
    from behave_fire.rust import safety

    arguments=(5.,np.empty(0),50.,1.,300.)
    expected=reference.calculate_safety_zone(*arguments)
    actual=safety.calculate_safety_zone(*arguments)
    assert actual['separation_distance'].shape==expected['separation_distance'].shape
    np.testing.assert_allclose(actual['separation_distance'],expected['separation_distance'])
    assert actual['area'].shape==(0,)


def test_surface_intermediates_keep_their_own_grid_shape():
    from behave_fire.python import BehaveRun as PythonRun,FuelModels as PythonFuels
    from behave_fire.rust import BehaveRun,FuelModels

    inputs=dict(fuel_model_grid=np.array([124,124]),m1h=np.full(2,.06),m10h=np.full(2,.07),m100h=np.full(2,.08),mlh=np.full(2,.6),mlw=np.full(2,.9),wind_speed=np.ones((3,1)),wind_speed_units=5,wind_direction=0.,wind_orientation_mode='RelativeToUpslope',slope=10.)
    expected=PythonRun(PythonFuels()).do_surface_run(**inputs)
    actual=BehaveRun(FuelModels()).do_surface_run(**inputs)
    for key in expected:
        assert actual[key].shape==expected[key].shape,key
        np.testing.assert_allclose(actual[key],expected[key],rtol=1e-9,atol=1e-12,err_msg=key)


def test_upslope_direction_ignores_aspect_shape():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    inputs=dict(x_component=np.ones(2), y_component=np.ones(2), rate_vector=np.ones(2),wind_orientation_mode="RelativeToUpslope",aspect=np.ones(3))
    expected=reference.calculate_direction_of_max_spread(**inputs)
    actual=surface.calculate_direction_of_max_spread(**inputs)
    assert actual.shape==expected.shape
    np.testing.assert_allclose(actual,expected)


def test_user_waf_ignores_unrelated_canopy_shape():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    inputs=dict(canopy_cover=np.ones(2),canopy_height=np.ones(3),crown_ratio=np.ones(4),depth=np.ones(5),waf_method="UserInput",user_waf=np.array([.2]))
    expected=reference.calculate_surface_wind_adjustment_factor(**inputs)
    actual=surface.calculate_surface_wind_adjustment_factor(**inputs)
    assert actual.shape==expected.shape
    np.testing.assert_allclose(actual,expected)
