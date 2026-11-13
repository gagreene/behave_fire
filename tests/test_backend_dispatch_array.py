"""Fresh-process contracts for automatic backend selection."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def _run(source):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("outcome", ["absent", "loader", "version", "incomplete", "usable"])
def test_cached_selection_and_warnings(outcome):
    _run(f'''
        import importlib, importlib.util, types, warnings
        import behave_fire as bf
        import behave_fire.auto as auto
        outcome = {outcome!r}
        calls = []
        extension = types.SimpleNamespace(__binding_api_version__=4, __backend_complete__=True, __capabilities__=auto.REQUIRED_CAPABILITIES)
        if outcome == "version": extension.__binding_api_version__ = 999
        if outcome == "incomplete": extension.__backend_complete__ = False
        original_import = importlib.import_module
        def discover(name):
            assert name == "behave_fire._rust"
            calls.append(name)
            return None if outcome == "absent" else object()
        def load(name):
            if name == "behave_fire._rust":
                if outcome == "loader": raise OSError("broken native loader")
                return extension
            if name == "behave_fire.rust":
                return types.SimpleNamespace(BehaveRun=lambda fuels, threads=1: object())
            return original_import(name)
        importlib.util.find_spec = discover
        importlib.import_module = load
        expected = ["python", "rust"] if outcome == "usable" else ["python"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert bf.available_backends() == expected
            assert not caught
            for _ in range(3):
                model = bf.BehaveRun(bf.FuelModels())
                assert model.backend_in_use == ("rust" if outcome == "usable" else "python")
            assert len(caught) == (0 if outcome == "usable" else 1)
            if caught:
                assert caught[0].category is bf.BehaveBackendWarning
                assert caught[0].filename == "<string>"
                assert outcome in str(caught[0].message) or (outcome == "version" and "API" in str(caught[0].message))
        assert len(calls) == 1
        if outcome != "usable":
            for construct in (lambda: bf.BehaveRun(bf.FuelModels(), backend="rust"), lambda: bf.get_backend("rust")):
                try: construct()
                except ImportError as error:
                    assert error.__cause__ is not None
                    assert "install" in str(error).lower()
                else: raise AssertionError("forced Rust did not fail")
    ''')


def test_error_filter_does_not_consume_warning():
    _run('''
        import importlib.util, warnings
        import behave_fire as bf
        importlib.util.find_spec = lambda name: object()
        original = importlib.import_module
        def load(name):
            if name == "behave_fire._rust": raise OSError("loader broken")
            return original(name)
        importlib.import_module = load
        warnings.simplefilter("error", bf.BehaveBackendWarning)
        for _ in range(3):
            try: bf.BehaveRun(bf.FuelModels())
            except bf.BehaveBackendWarning: pass
            else: raise AssertionError("strict warning consumed")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", bf.BehaveBackendWarning)
            bf.BehaveRun(bf.FuelModels())
            bf.BehaveRun(bf.FuelModels())
            assert len(caught) == 1
        warnings.simplefilter("error", bf.BehaveBackendWarning)
        bf.BehaveRun(bf.FuelModels())
    ''')


def test_facade_and_calculation_errors_do_not_fallback():
    _run('''
        import importlib, importlib.util, types
        import behave_fire as bf
        from behave_fire._backend_contract import REQUIRED_CAPABILITIES
        extension = types.SimpleNamespace(__binding_api_version__=4, __backend_complete__=True, __capabilities__=REQUIRED_CAPABILITIES)
        importlib.util.find_spec = lambda name: object()
        original = importlib.import_module
        def load(name):
            if name == "behave_fire._rust": return extension
            if name == "behave_fire.rust": raise ImportError("facade broken")
            return original(name)
        importlib.import_module = load
        try: bf.BehaveRun(bf.FuelModels())
        except ImportError as error: assert str(error) == "facade broken"
        else: raise AssertionError("facade error swallowed")
        class Delegate:
            def __init__(self, fuels, threads=1): pass
            def calculate_fire_area(self, *args, **kwargs): raise ArithmeticError("calculation broken")
        def load2(name):
            if name == "behave_fire.rust": return types.SimpleNamespace(BehaveRun=Delegate)
            return load(name)
        importlib.import_module = load2
        model = bf.BehaveRun(bf.FuelModels())
        try: model.calculate_fire_area(1, 0, 1, 1, 1, 0)
        except ArithmeticError: pass
        else: raise AssertionError("calculation error swallowed")
        assert model.backend_in_use == "rust"
    ''')


def test_imports_and_explicit_python_never_probe():
    _run('''
        import importlib.abc, importlib.util, sys
        class Poison(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "behave_fire._rust": raise AssertionError("native imported")
        sys.meta_path.insert(0, Poison())
        original = importlib.util.find_spec
        def discover(name, *args):
            if name == "behave_fire._rust": raise AssertionError("native probed")
            return original(name, *args)
        importlib.util.find_spec = discover
        import behave_fire as bf
        import behave_fire.python as py
        from behave_fire.behave import BehaveRun as Legacy
        assert py.BehaveRun is Legacy
        assert bf.get_backend("python") is py
        model = bf.BehaveRun(bf.FuelModels(), backend="python", threads=object())
        assert model.backend_in_use == "python"
        assert bf.BehaveRun.calculate_fire_area(1, 0, 1, 1, 1, 0) is not None
        for value in (None, "bad", "RUST", 1):
            try: bf.get_backend(value)
            except ValueError: pass
            else: raise AssertionError("invalid selector accepted")
            try: bf.BehaveRun(bf.FuelModels(), backend=value)
            except ValueError: pass
            else: raise AssertionError("invalid constructor selector accepted")
        modules = ("behave_units", "crown", "fine_dead_fuel_moisture_tool", "fire_size", "fuel_models", "fuel_models_array", "ignite", "mortality", "safety", "slope_tool", "species_master_table", "spot", "surface", "vapor_pressure_deficit_calculator")
        import importlib
        for name in modules:
            assert getattr(py, name) is importlib.import_module("behave_fire.components." + name)
        assert "behave_fire._rust" not in sys.modules
    ''')


def test_instance_backend_identity_immutability_and_signatures():
    _run('''
        import inspect
        import behave_fire as bf
        from behave_fire.behave import BehaveRun as Legacy
        from behave_fire.python import BehaveRun as Python
        from behave_fire.components.fuel_models import FuelModels
        assert Python is Legacy
        assert bf.FuelModels is FuelModels
        class Subclass(Python): pass
        assert isinstance(Subclass(FuelModels()), Legacy)
        model = bf.BehaveRun(FuelModels(), backend="python")
        assert not isinstance(model, Legacy)
        assert repr(model).startswith("<behave_fire.auto.BehaveRun object")
        try: model.backend_in_use = "rust"
        except AttributeError: pass
        else: raise AssertionError("backend mutable")
        names = [name for name in vars(Legacy) if not name.startswith("_")]
        assert len(names) == 11
        for name in names:
            assert inspect.signature(getattr(bf.BehaveRun, name)) == inspect.signature(getattr(Legacy, name)), name
            assert inspect.signature(getattr(model, name)) == inspect.signature(getattr(Legacy(FuelModels()), name)), name
    ''')


@pytest.mark.parametrize("capabilities", [None, (), ("unrelated.operation",), "not-an-operation-list", (1, 2), "one-missing"])
def test_missing_capability_inventory_rejects_claimed_complete_backend(capabilities):
    _run(f'''
        import importlib, importlib.util, types, warnings
        import behave_fire as bf
        extension = types.SimpleNamespace(__binding_api_version__=4, __backend_complete__=True)
        capabilities = {capabilities!r}
        if capabilities == "one-missing":
            from behave_fire._backend_contract import REQUIRED_CAPABILITIES
            capabilities = tuple(sorted(REQUIRED_CAPABILITIES)[1:])
        if capabilities is not None:
            extension.__capabilities__ = capabilities
        original = importlib.import_module
        importlib.util.find_spec = lambda name: object()
        def load(name):
            if name == "behave_fire._rust": return extension
            return original(name)
        importlib.import_module = load
        assert bf.available_backends() == ["python"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert bf.BehaveRun(bf.FuelModels()).backend_in_use == "python"
            assert len(caught) == 1
            assert "capabilit" in str(caught[0].message)
        try: bf.get_backend("rust")
        except ImportError as error: assert error.__cause__ is not None
        else: raise AssertionError("incomplete operation inventory accepted")
    ''')


def test_stale_api_two_extension_is_rejected():
    _run('''
        import importlib, importlib.util, types
        import behave_fire as bf
        from behave_fire._backend_contract import REQUIRED_CAPABILITIES
        extension = types.SimpleNamespace(__binding_api_version__=2, __backend_complete__=True,
                                          __capabilities__=REQUIRED_CAPABILITIES)
        original = importlib.import_module
        importlib.util.find_spec = lambda name: object()
        def load(name):
            if name == "behave_fire._rust": return extension
            return original(name)
        importlib.import_module = load
        assert bf.available_backends() == ["python"]
        try: bf.get_backend("rust")
        except ImportError: pass
        else: raise AssertionError("stale API-2 extension was accepted")
    ''')

def test_static_capability_contract_matches_tracked_inventory():
    import json
    from behave_fire._backend_contract import BINDING_API_VERSION, REQUIRED_CAPABILITIES

    inventory = json.loads((Path(__file__).parent / "data/rust_api_inventory.json").read_text(encoding="utf-8"))
    expected = {entry["binding_route"] for entry in inventory["entries"] if entry["kind"] in ("function", "method")}
    assert BINDING_API_VERSION == 4
    assert REQUIRED_CAPABILITIES == expected


def test_unexpected_discovery_error_propagates():
    _run('''
        import importlib.util
        import behave_fire as bf
        def discover(name): raise RuntimeError("discovery broken")
        importlib.util.find_spec = discover
        try: bf.available_backends()
        except RuntimeError as error: assert str(error) == "discovery broken"
        else: raise AssertionError("discovery error swallowed")
    ''')
