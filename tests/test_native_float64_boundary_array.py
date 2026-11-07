"""The new numerical contract normalizes once and retains only structural shape data."""
import numpy as np
import pytest

pytestmark=pytest.mark.rust


def test_float64_snapshot_preserves_missing_values_and_layouts():
    from behave_fire.rust._numeric import normalize_physical

    source=np.array([-0.,1.,2.,3.],dtype='>f4')[::-1]
    source.flags.writeable=False
    masked=np.ma.array(source,mask=[False,True,False,False],fill_value=-17.)
    actual=normalize_physical(masked)
    assert type(actual) is np.ndarray
    assert actual.dtype==np.float64
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual,[3.,np.nan,1.,-0.])
    assert np.signbit(actual[-1])
    assert not np.shares_memory(actual,source)
    assert not np.shares_memory(actual,np.ma.getmaskarray(masked))
    assert normalize_physical(np.float32(.1)).shape==()
    assert normalize_physical(np.empty((2,0))).shape==(2,0)
    assert normalize_physical(np.array([(1<<63)+7],np.uint64))[0]==float((1<<63)+7)


def test_normalization_rejects_opaque_and_complex_values_before_float_cast():
    from behave_fire.rust._numeric import normalize_physical

    class CustomFloat(float):
        pass

    for value in ([1+2j],np.array([1+0j]),np.ma.array([1+2j]),np.array([1.],dtype=object),CustomFloat(1.)):
        with pytest.raises(TypeError):normalize_physical(value)


def test_shape_groups_compute_independent_shapes_including_empty_groups():
    from behave_fire.rust._numeric import plan_shapes

    inputs=dict(surface=np.ones(2),crown=np.ones(3),critical_surface=np.asarray(2.),critical_crown=np.asarray(4.))
    groups=plan_shapes(inputs,{'transition':('surface','critical_surface'),'active':('crown','critical_crown')})
    assert {group.outputs:group.shape for group in groups}=={('transition',):(2,),('active',):(3,)}
    groups=plan_shapes({'flame':np.asarray(5.),'people':np.empty(0)}, {'separation':('flame',),'area':('flame','people')})
    assert {group.outputs:group.shape for group in groups}=={('separation',):(),('area',):(0,)}
    groups=plan_shapes({'fuel':np.ones(2),'wind':np.ones((3,1))},{'reaction':('fuel',),'spread':('fuel','wind')})
    assert {group.outputs:group.shape for group in groups}=={('reaction',):(2,),('spread',):(3,2)}
    with pytest.raises(KeyError):plan_shapes(inputs,{'bad':('undeclared',)})


def test_typed_controls_do_not_round_through_float64():
    from behave_fire.rust._numeric import normalize_control, normalize_selector
    from behave_fire.rust._bridge import _unit_code

    source=np.array([(1<<53)+1],dtype=np.int64)
    result=normalize_control(source,np.int64)
    assert result.dtype==np.int64 and int(result[0])==(1<<53)+1
    assert not np.shares_memory(source,result)
    with pytest.raises(OverflowError):normalize_control(source,np.int32)
    with pytest.raises(ValueError):normalize_control(np.ma.array([1],mask=[True]),np.int32)
    with pytest.raises(TypeError):normalize_control(np.array([1.5]),np.int64)
    expected=np.asarray(_unit_code('Pressure',True),dtype=np.int32)
    actual=normalize_selector('Pressure',True)
    np.testing.assert_array_equal(actual,expected)
    assert actual.shape==expected.shape