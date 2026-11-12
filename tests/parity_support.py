"""Frozen explicit-reference oracle decoding and strict comparison."""

from importlib import import_module
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pytest

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pytest.ini").is_file())
TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from export_rust_reference import contracts_match, encode_value
from rust_reference_platform import acceptance_output




def _assert_floating(actual, expected, single=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if np.iscomplexobj(expected) or np.iscomplexobj(actual):
        _assert_floating(actual.real, expected.real, single)
        _assert_floating(actual.imag, expected.imag, single)
        return
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(expected))
    np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(expected))
    zeros = (actual == 0) & (expected == 0)
    np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(expected[zeros]))
    np.testing.assert_allclose(actual, expected, rtol=1e-6 if single else 1e-9,
                               atol=1e-7 if single else 1e-12, equal_nan=True)


def _assert_native_error(operation, arguments, category):
    """Require the declared rejection while retaining ordinary comparator errors."""
    try:
        operation(**arguments)
    except category:
        return
    raise AssertionError(f'Native contract requires {category.__name__} before calculation')


def _float64_case(case, operation, arguments):
    """Keep historical Python evidence separate from the new native model contract."""
    import builtins
    from float64_model_reference import model_reference
    from performance_cases import assert_outputs
    from behave_fire import python as python_backend
    from behave_fire.python import BehaveRun, FuelModels

    name = case['entry_point'].rsplit('.', 1)[1]
    if '.BehaveRun.' in case['entry_point']:
        reference = getattr(BehaveRun(FuelModels()), name)
    else:
        reference = python_backend
        for part in case['entry_point'].removeprefix('behave_fire.components.').split('.'):
            reference = getattr(reference, part)
    with warnings.catch_warnings(), np.errstate(all='ignore'):
        warnings.simplefilter('ignore')
        if 'exception' in case:
            error = case['exception']
            category = error['type'] if isinstance(error, dict) else error
            with pytest.raises(getattr(builtins, category)):
                reference(**arguments)
        else:
            assert_result(reference(**arguments), acceptance_output(case, encode_value, contracts_match))
        physical = {key: value for key, value in arguments.items()
                    if key not in ('lut', 'out_units') and not key.endswith('units') and not isinstance(value, str)}

        def unsupported(value):
            if isinstance(value, dict):
                return any(unsupported(item) for item in value.values())
            return value is not None and np.asanyarray(value).dtype.kind in 'Oc'

        if any(unsupported(value) for value in physical.values()):
            _assert_native_error(operation, arguments, TypeError)
            return
        for key, value in arguments.items():
            if key in ('is_surface_fire', 'is_passive_crown', 'is_active_crown'):
                flags = np.asanyarray(value)
                if np.ma.getmaskarray(flags).any() or np.isnan(flags).any():
                    _assert_native_error(operation, arguments, ValueError)
                    return
            elif key.endswith(('_grid', '_i')):
                ids = np.asanyarray(value)
                fuel_ids = key == 'fuel_model_grid' and name != 'coerce_crown_fire_inputs'
                fractional = np.any(np.isfinite(ids) & (ids != np.trunc(ids)))
                if (np.ma.getmaskarray(ids).any() or fractional or
                        (not fuel_ids and not np.isfinite(ids).all())):
                    _assert_native_error(operation, arguments, ValueError)
                    return
        try:
            if '.BehaveRun.' in case['entry_point'] or name in ('run_crown_fire', 'coerce_crown_fire_inputs'):
                expected = model_reference(reference, arguments)
                if name.startswith('calculate_spotting_from_'):
                    from component_f64_reference import reference as component_reference
                    expected = component_reference(reference, arguments)
                if name == 'calculate_scorch_height':
                    value = np.asarray(np.ma.asarray(arguments['fireline_intensity'], dtype=np.float64).filled(np.nan))
                    expected = np.where(np.isnan(value), np.nan, expected)
            else:
                from component_f64_reference import reference as component_reference
                expected = component_reference(reference, arguments)
        except (TypeError, ValueError, IndexError) as error:
            _assert_native_error(operation, arguments, type(error))
            return
        assert_outputs(operation(**arguments), expected)


def _numeric_family(value):
    families = ((bool, np.bool_), (int, np.integer), (float, np.floating),
                (complex, np.complexfloating))
    for index, family in enumerate(families):
        if isinstance(value, family):
            return index
    raise AssertionError(f"unsupported boxed numeric payload: {type(value).__name__}")


def assert_result(actual, expected):
    if hasattr(actual, "__dict__") and not isinstance(actual, (np.ndarray, int, float, str)):
        actual = vars(actual)
    if isinstance(expected, dict) and 'float' in expected:
        assert not isinstance(actual, np.ndarray)
        _assert_floating(actual, decode(expected))
    elif isinstance(expected, (float, complex)):
        assert not isinstance(actual, np.ndarray)
        _assert_floating(actual, expected)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_result(left, right)
    elif isinstance(expected, dict) and "dtype" not in expected:
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_result(actual[key], expected[key])
    elif isinstance(expected, dict) and "dtype" in expected:
        wanted = decode(expected)
        # Historical shape=[] records conflate NumPy scalars and zero-rank
        # arrays. Enforce array structure only when the record distinguishes it.
        if expected["shape"]:
            assert isinstance(actual, np.ndarray)
        actual = np.asanyarray(actual)
        assert actual.shape == wanted.shape
        assert actual.dtype == wanted.dtype
        assert np.ma.isMaskedArray(actual) == np.ma.isMaskedArray(wanted)
        np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(wanted))
        actual_data, wanted_data = np.ma.getdata(actual), np.ma.getdata(wanted)
        if wanted.dtype == object:
            # Read the encoded entries directly: float coercion loses integers
            # above 2**53 and obscures the boxed bool/integer distinction.
            for left, right in zip(actual_data.ravel(), expected["values"]):
                payload = decode(right)
                if isinstance(payload, np.ndarray) and payload.ndim == 0:
                    payload = payload[()]
                assert _numeric_family(left) == _numeric_family(payload)
                assert_result(left, right)
        elif wanted.dtype.kind in "biu":
            np.testing.assert_array_equal(actual_data, wanted_data)
        else:
            _assert_floating(actual_data, wanted_data, wanted.dtype in (np.dtype("float32"), np.dtype("complex64")))
    elif isinstance(expected, bool):
        assert isinstance(actual, (bool, np.bool_))
        assert actual == expected
    elif isinstance(expected, int):
        assert isinstance(actual, (int, np.integer)) and not isinstance(actual, (bool, np.bool_))
        assert actual == expected
    else:
        assert actual == expected


def cases(*domains):
    result = []
    for domain in domains:
        document = json.loads((ROOT / "tests/data/rust_port" / (domain + ".json")).read_text(encoding="utf-8"))
        result.extend(pytest.param(case, id=domain + ":" + case["id"]) for case in document["cases"] if case.get("policy", "acceptance") == "acceptance")
    return result


def decode(value):
    if isinstance(value, list):
        return [decode(item) for item in value]
    if not isinstance(value, dict):
        return value
    if "float" in value:
        return float(value["float"])
    if "dtype" in value:
        result = np.array([decode(item) for item in value["values"]], dtype=value["dtype"]).reshape(value["shape"])
        if "mask" in value:
            result = np.ma.array(result, mask=np.array(value["mask"]).reshape(value["shape"]))
        if value.get("read_only"):
            result.setflags(write=False)
        return result
    if "reference" in value:
        from behave_fire.rust import fuel_models, fuel_models_array, mortality
        if value["reference"] == "fuels:standard_lookup":
            return fuel_models_array.build_fuel_lookup_arrays(fuel_models.FuelModels())
        if value["reference"] in ("mortality:standard_coefficients", "species:mortality_coefficients"):
            return mortality.build_mortality_lookup()
        raise AssertionError(value)
    if value.get("registry") == "standard FuelModels":
        from behave_fire.rust import FuelModels
        return FuelModels()
    if value.get("registry") == "standard SpeciesMasterTable":
        from behave_fire.rust import SpeciesMasterTable
        return SpeciesMasterTable()
    return {key: decode(item) for key, item in value.items()}


def registry_state(state):
    from behave_fire.rust import FuelModels, SpeciesMasterTable, SpeciesMasterTableRecord
    if state["registry"] == "FuelModels":
        instance = FuelModels()
        for number in state["removed"]:
            del instance.fuel_models_[number]
        for item in state["overrides"]:
            instance.fuel_models_[item["number"]] = decode(item["record"])
        return instance
    if state["registry"] == "SpeciesMasterTable":
        instance = SpeciesMasterTable()
        standard = instance.records
        instance.records = [standard[item["reference_index"]] if "reference_index" in item else SpeciesMasterTableRecord(**decode(item["record"])) for item in state["records"]]
        return instance
    return SpeciesMasterTableRecord(**decode(state["record"]))




def run_case(case):
    route = case["entry_point"].replace("behave_fire.components", "behave_fire.rust").replace("behave_fire.behave", "behave_fire.rust.core")
    module, function = route.rsplit(".", 1)
    instance = None
    if "receiver_before" in case:
        instance = registry_state(case["receiver_before"])
        operation = getattr(instance, function)
    elif "state_before" in case:
        module, cls = module.rsplit(".", 1)
        instance = getattr(import_module(module), cls)()
        vars(instance).update(decode(case["state_before"]))
        operation = getattr(instance, function)
    elif ".BehaveRun." in route:
        from behave_fire.rust import BehaveRun, FuelModels
        operation = getattr(BehaveRun(FuelModels()), function)
    else:
        import behave_fire.rust as native
        operation = native
        for part in route.removeprefix("behave_fire.rust.").split("."):
            operation = getattr(operation, part)
    arguments = decode(case["inputs"])
    entry = case['entry_point']
    stateless = any(f'.{family}.' in entry for family in ('behave_units', 'ignite', 'safety', 'spot', 'vapor_pressure_deficit_calculator', 'surface', 'crown', 'fine_dead_fuel_moisture_tool'))
    stateless = stateless or ('.mortality.calculate_' in entry)
    if stateless or '.BehaveRun.' in entry or entry.rsplit('.', 1)[1] in ('run_crown_fire', 'coerce_crown_fire_inputs'):
        _float64_case(case, operation, arguments)
        return
    if case['entry_point']=='behave_fire.components.surface.calculate_direction_of_max_spread':
        x,y=arguments['x_component'],arguments['y_component']
        loop=np.arctan2.resolve_dtypes((np.asanyarray(y).dtype,np.asanyarray(x).dtype,None))[2]
        if loop==np.dtype('float16'):
            from behave_fire.components import surface as reference
            normalized=dict(arguments,x_component=np.asanyarray(x).astype(np.float32),y_component=np.asanyarray(y).astype(np.float32))
            expected=np.asanyarray(reference.calculate_direction_of_max_spread(**normalized))
            case=dict(case,output={'dtype':str(expected.dtype),'shape':list(expected.shape),'values':expected.ravel().tolist()})
    if "record" in arguments and isinstance(arguments["record"], dict):
        from behave_fire.rust import SpeciesMasterTableRecord
        arguments["record"] = SpeciesMasterTableRecord(**arguments["record"])
    with warnings.catch_warnings(record=True) as caught,np.errstate(all='ignore'):
        warnings.simplefilter('always')
        if "exception" in case:
            import builtins
            error = case["exception"]
            category = error["type"] if isinstance(error, dict) else error
            with pytest.raises(getattr(builtins, category)):
                operation(**arguments)
        else:
            result = operation(**arguments)
            assert_result(result, acceptance_output(case, encode_value, contracts_match))
    if 'warnings' in case:
        assert sorted(set(type(item.message).__name__ for item in caught))==case['warnings']
    if "receiver_after" in case:
        expected = registry_state(case["receiver_after"])
        if hasattr(instance, "fuel_models_"):
            assert_result(instance.fuel_models_, expected.fuel_models_)
        elif hasattr(instance, "records"):
            assert_result(instance.records, [vars(record) for record in expected.records])
        else:
            assert_result(instance, vars(expected))
    elif instance is not None:
        assert_result(vars(instance), case["state_after"])
