"""Full crown entry points retain supplied surface-field dtype and mask semantics."""

import numpy as np
import pytest

pytestmark = pytest.mark.rust


@pytest.mark.parametrize('masked',[False,True])
def test_full_crown_preserves_supplied_surface_field_precision(masked):
    from behave_fire.components import crown as reference
    from behave_fire.rust import FuelModels,crown,fuel_models_array

    surface={name:np.array([value],dtype=np.float32) for name,value in [('spread_rate',1.),('fireline_intensity',100.),('heat_per_unit_area',1000.)]}
    if masked:
        surface={name:np.ma.array(value,mask=[True],fill_value=17.) for name,value in surface.items()}
    inputs=dict(surface_results=surface,lut=fuel_models_array.build_fuel_lookup_arrays(FuelModels()),fuel_model_grid=np.array([1]),m1h=np.array([.06]),m10h=np.array([.07]),m100h=np.array([.08]),mlh=np.array([.6]),mlw=np.array([.9]),wind_speed=np.array([5.]),wind_speed_units=5,wind_direction=np.array([0.]),wind_orientation_mode='RelativeToUpslope',slope=np.array([0.]))
    inputs.update(canopy_base_height=50.,canopy_height=60.)
    expected=reference.run_crown_fire(**inputs)
    actual=crown.run_crown_fire(**inputs)
    for key in expected:
        assert actual[key].dtype==expected[key].dtype,key
        assert np.ma.isMaskedArray(actual[key])==np.ma.isMaskedArray(expected[key]),key
        np.testing.assert_array_equal(np.ma.getmaskarray(actual[key]),np.ma.getmaskarray(expected[key]))
        np.testing.assert_allclose(np.ma.getdata(actual[key]),np.ma.getdata(expected[key]),rtol=1e-9,atol=1e-12,err_msg=key)
