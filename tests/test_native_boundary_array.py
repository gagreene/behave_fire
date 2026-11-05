"""Explicit native boundary contracts, independent of automatic selection."""

import importlib.util

import numpy as np
import pytest

pytestmark = pytest.mark.rust


@pytest.mark.parametrize('dtype',[np.bool_,np.int8,np.uint8])
def test_direction_generated_half_loop_is_widened(dtype):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    x=np.array([1,3,11],dtype=dtype)
    y=np.array([1,5,17],dtype=dtype)
    rate=np.ones(3)
    expected=reference.calculate_direction_of_max_spread(x.astype(np.float32),y.astype(np.float32),rate,'RelativeToUpslope')
    actual=surface.calculate_direction_of_max_spread(x,y,rate,'RelativeToUpslope')
    assert actual.dtype==expected.dtype==np.float32
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)


def test_empty_particle_dictionary_preserves_each_field_dtype():
    from behave_fire.components import surface as reference
    from behave_fire.rust import FuelModels,fuel_models_array,surface

    lookup=fuel_models_array.build_fuel_lookup_arrays(FuelModels())
    empty=np.empty(0)
    particles=reference.build_particle_arrays(lookup,np.empty(0,dtype=int),empty,empty,empty,empty,empty)
    particles={key:value if value.dtype==bool else value.astype(np.float32) for key,value in particles.items()}
    expected=reference.calculate_fuelbed_intermediates(particles)
    actual=surface.calculate_fuelbed_intermediates(particles)
    for key in expected:
        assert actual[key].shape==expected[key].shape==(0,)
        assert actual[key].dtype==expected[key].dtype,key


@pytest.mark.parametrize('target',['X86_V3','X86_V4'])
def test_grouped_numpy_cpu_target_selects_native_simd_profile(monkeypatch,target):
    from numpy._core import _multiarray_umath
    from behave_fire.rust._bridge import _math_profile
    monkeypatch.setattr(_multiarray_umath,'__cpu_targets_info__',{'exp':{'ff':{'current':target,'available':target+' baseline(X86_V2)'}}})
    assert _math_profile()==1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(), (1,), (2, 1)])
@pytest.mark.parametrize("mask", [False, True])
def test_masked_boundary_preserves_array_kind(dtype, shape, mask):
    import warnings

    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    value = np.ma.array(np.full(shape, 2, dtype=dtype), mask=mask, fill_value=17)
    for name in ("calculate_eccentricity", "calculate_fire_length_to_width_ratio"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'^__array_wrap__ must accept context and return_scalar arguments \(positionally\) in the future\. \(Deprecated NumPy 2\.0\)$',
                category=DeprecationWarning,
                module=r'^behave_fire\.components\.surface$',
            )
            expected = getattr(reference, name)(value)
        actual = getattr(surface, name)(value)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert np.ma.isMaskedArray(actual) == np.ma.isMaskedArray(expected)
        np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(expected))
        np.testing.assert_allclose(np.ma.getdata(actual),np.ma.getdata(expected),rtol=1e-6 if expected.dtype == np.float32 else 1e-9,atol=1e-7 if expected.dtype == np.float32 else 1e-12)


def test_masked_output_retains_fill_metadata_for_followup_operations():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    r0=np.ma.array([1.,2.],mask=[False,True],fill_value=17.)
    arguments=(r0,1.,1.,0.,'RelativeToUpslope')
    expected=reference.calculate_surface_rate_vector(*arguments)
    actual=surface.calculate_surface_rate_vector(*arguments)
    for key in expected:
        assert actual[key].fill_value==expected[key].fill_value,key
        np.testing.assert_array_equal(actual[key].filled(),expected[key].filled())


def test_mixed_precision_preserves_each_input_rounding():
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    boundary = np.nextafter(np.float32(.5), np.float32(1))
    values = [(np.array([10.],dtype=np.float64),np.array([boundary],dtype=np.float32)),(np.array([10.],dtype=np.float32),np.array([boundary],dtype=np.float64))]
    for rate, eccentricity in values:
        expected=reference.calculate_backing_spread_rate(rate,eccentricity)
        actual=surface.calculate_backing_spread_rate(rate,eccentricity)
        assert actual.dtype==expected.dtype
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-12)


def test_model_calculations_never_call_reference(monkeypatch):
    import inspect
    from behave_fire import components
    from behave_fire.rust import BehaveRun, FuelModels

    def forbidden(*args, **kwargs):
        raise AssertionError("Python science function reached from explicit native model")

    for module_name in ("surface", "crown", "mortality", "spot", "behave_units"):
        module = __import__("behave_fire.components." + module_name, fromlist=[module_name])
        for name, value in vars(module).copy().items():
            if inspect.isfunction(value) and not name.startswith("_"):
                monkeypatch.setattr(module,name,forbidden)
    model = BehaveRun(FuelModels())
    inputs = dict(fuel_model_grid=np.array([124]),m1h=np.array([.06]),m10h=np.array([.07]),m100h=np.array([.08]),mlh=np.array([.6]),mlw=np.array([.9]),wind_speed=5.,wind_speed_units=5,wind_direction=0.,wind_orientation_mode="RelativeToUpslope",slope=10.)
    result = model.do_surface_run(**inputs)
    crown = model.do_crown_run(result, **inputs)
    assert result["spread_rate"][0] > 0
    assert "fire_type" in crown


def test_native_malformed_input_never_panics():
    from behave_fire.rust import surface

    for values in (np.array(["bad"], dtype=object), np.ones((2, 3))):
        try:
            surface.calculate_backing_spread_rate(values, np.ones(4))
        except (ValueError, TypeError):
            pass
        except BaseException as error:
            pytest.fail(f"unexpected native exception: {type(error).__name__}: {error}")
        else:
            pytest.fail("malformed input was accepted")


def test_native_never_calls_reference(monkeypatch):
    from behave_fire.components import safety as reference
    from behave_fire.rust import safety

    def forbidden(*args, **kwargs):
        raise AssertionError("native implementation called the Python reference")

    monkeypatch.setattr(reference, "calculate_safety_zone", forbidden)
    result = safety.calculate_safety_zone(5.0, 0, 0, 0, 0)
    assert result["separation_distance"] == 20.0
    np.testing.assert_allclose(result["area"], np.pi * 400)


@pytest.mark.parametrize("shape", [(), (0,), (3,), (2, 3), (2, 2, 3)])
def test_native_owned_layouts(shape):
    from behave_fire.rust import surface

    values = np.full(shape, 2.0)
    values.setflags(write=False)
    result = surface.calculate_eccentricity(values)
    assert result.shape == shape
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, np.sqrt(3) / 2)
    assert not np.shares_memory(result, values)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_native_precision(dtype):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    values = np.array([1, 2, 300], dtype=dtype)
    expected = reference.calculate_eccentricity(values.astype(np.float32) if dtype == np.float16 else values)
    actual = surface.calculate_eccentricity(values)
    assert actual.dtype == expected.dtype
    np.testing.assert_allclose(actual, expected, rtol=1e-6 if expected.dtype == np.float32 else 1e-9)


@pytest.mark.parametrize("layout", ["fortran", "reverse", "slice", "broadcast", "readonly"])
def test_native_strided_array_ownership(layout):
    from behave_fire.components import surface as reference
    from behave_fire.rust import surface

    original = np.arange(24.0).reshape(4, 6) + 1
    values = {"fortran": np.asfortranarray(original), "reverse": original[::-1, ::-1], "slice": original[::2, 1::2], "broadcast": np.broadcast_to(original[0], (4, 6)), "readonly": original.copy()}[layout]
    if layout == "readonly":
        values.setflags(write=False)
    before = values.copy()
    actual = surface.calculate_eccentricity(values)
    expected = reference.calculate_eccentricity(values)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(values, before)
    assert not np.shares_memory(actual, values)


@pytest.mark.parametrize('disable_simd',[False,True])
def test_native_subprocess_shape_safety_and_numpy_math_profile(disable_simd):
    import os
    from pathlib import Path
    import subprocess
    import sys
    from behave_fire import _rust
    from behave_fire import rust

    script = '''
import importlib.util
from importlib.machinery import ExtensionFileLoader
import sys
import behave_fire
behave_fire.__path__.insert(0,sys.argv[2])
spec=importlib.util.spec_from_file_location('behave_fire._rust',sys.argv[1],loader=ExtensionFileLoader('behave_fire._rust',sys.argv[1]))
native=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=native
spec.loader.exec_module(native)
import numpy as np
from behave_fire.components import surface as reference
from behave_fire.rust import surface,FuelModels,fuel_models_array
from behave_fire.rust._bridge import _math_profile
if sys.argv[3]=='True': assert _math_profile()==0
for shape in ([-1],[3],[2**64],[2**62]):
    try: native.batch('surface.calculate_eccentricity',{'fire_length_to_width_ratio':np.ones(2 if shape==[3] else 1)}, {},shape,False)
    except (ValueError,TypeError): pass
    except BaseException as error: raise AssertionError(type(error).__name__) from error
    else: raise AssertionError('invalid shape accepted')
lookup=fuel_models_array.build_fuel_lookup_arrays(FuelModels())
particles=reference.build_particle_arrays(lookup,np.array([124]),*[np.array([v]) for v in (.06,.07,.08,.6,.9)])
ib=reference.calculate_fuelbed_intermediates(particles)
ib['sigma']=np.array([1.],dtype=np.float32)
args=dict(ri=np.array([1000.]),ib=ib,wind_speed=np.array([5.]),wind_speed_units=5,wind_direction=np.array([0.]),wind_orientation_mode='RelativeToUpslope',slope=np.array([0.]))
expected=reference.run_surface_fire(**args)
actual=surface.run_surface_fire(**args)
for key in expected: np.testing.assert_allclose(actual[key],expected[key],rtol=1e-9,atol=1e-12,err_msg=key)
'''
    environment=os.environ.copy()
    if disable_simd:
        from numpy._core import _multiarray_umath
        targets=getattr(_multiarray_umath,'__cpu_dispatch__',())
        disabled=[name for name in targets if name.startswith(('AVX2','AVX512','FMA3','X86_V3','X86_V4'))]
        if disabled:
            environment['NPY_DISABLE_CPU_FEATURES']=','.join(disabled)
    environment['PYTHONPATH']=str(next(parent for parent in Path(__file__).resolve().parents if (parent/'pytest.ini').exists())/'src')
    completed=subprocess.run([sys.executable,'-c',script,_rust.__file__,str(Path(rust.__file__).parent.parent),str(disable_simd)],capture_output=True,text=True,env=environment)
    assert completed.returncode==0,completed.stdout+completed.stderr


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("masked", [False, True])
def test_particle_dictionary_dtype_and_mask(dtype, masked):
    from behave_fire.components import surface as reference
    from behave_fire.rust import FuelModels, fuel_models_array, surface

    lookup = fuel_models_array.build_fuel_lookup_arrays(FuelModels())
    particles = reference.build_particle_arrays(lookup, np.array([124, 124]), *[np.array([value,value]) for value in (.06,.07,.08,.6,.9)])
    for key,value in particles.items():
        if value.dtype != bool:
            particles[key] = value.astype(dtype)
            if masked:
                particles[key] = np.ma.array(particles[key],mask=np.broadcast_to([False,True],value.shape))
    expected=reference.calculate_fuelbed_intermediates(particles)
    actual=surface.calculate_fuelbed_intermediates(particles)
    assert actual.keys()==expected.keys()
    for key in actual:
        assert actual[key].dtype==expected[key].dtype,key
        assert np.ma.isMaskedArray(actual[key])==np.ma.isMaskedArray(expected[key]),key
        np.testing.assert_array_equal(np.ma.getmaskarray(actual[key]),np.ma.getmaskarray(expected[key]))
        np.testing.assert_allclose(np.ma.getdata(actual[key]),np.ma.getdata(expected[key]),rtol=1e-6 if expected[key].dtype==np.float32 else 1e-9,atol=1e-7 if expected[key].dtype==np.float32 else 1e-12,err_msg=key)


@pytest.mark.parametrize("kind", ["SpeedUnits", "FractionUnits", "TemperatureUnits", "SurfaceAreaToVolumeUnits"])
@pytest.mark.parametrize("unit", [-100, -1, 0.0, 1.0, 99, True, False])
def test_unit_code_behavior_matches_reference(kind, unit):
    from behave_fire.components import behave_units as reference
    from behave_fire.rust import behave_units as native

    ref = getattr(reference, kind).toBaseUnits
    actual = getattr(native, kind).toBaseUnits
    try:
        expected = ref(np.array([1.]), unit)
    except Exception as error:
        with pytest.raises(type(error)):
            actual(np.array([1.]), unit)
    else:
        result = actual(np.array([1.]), unit)
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=1e-9, atol=1e-12)
