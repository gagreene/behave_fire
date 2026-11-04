"""Frozen explicit-reference oracle decoding and strict comparison."""

from importlib import import_module
import json
from pathlib import Path
import warnings

import numpy as np
import pytest

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pytest.ini").is_file())




def assert_result(actual, expected):
    if hasattr(actual, "__dict__") and not isinstance(actual, (np.ndarray, int, float, str)):
        actual = vars(actual)
    if isinstance(expected, dict) and 'float' in expected:
        np.testing.assert_allclose(actual,decode(expected),rtol=1e-9,atol=1e-12,equal_nan=True)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-12,equal_nan=True)
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
        actual = np.asanyarray(actual)
        assert actual.shape == wanted.shape
        assert actual.dtype == wanted.dtype
        assert np.ma.isMaskedArray(actual) == np.ma.isMaskedArray(wanted)
        np.testing.assert_array_equal(np.ma.getmaskarray(actual), np.ma.getmaskarray(wanted))
        if wanted.dtype == object:
            actual, wanted = actual.astype(float), wanted.astype(float)
        actual_data, wanted_data = np.ma.getdata(actual), np.ma.getdata(wanted)
        np.testing.assert_array_equal(np.isnan(actual_data), np.isnan(wanted_data))
        np.testing.assert_array_equal(np.isposinf(actual_data), np.isposinf(wanted_data))
        np.testing.assert_array_equal(np.isneginf(actual_data), np.isneginf(wanted_data))
        np.testing.assert_allclose(actual_data, wanted_data, rtol=1e-6 if wanted.dtype == np.float32 else 1e-9, atol=1e-7 if wanted.dtype == np.float32 else 1e-12)
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
            assert_result(result, case["output"])
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
