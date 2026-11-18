"""Fresh-process contracts for automatic backend selection."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _run(source):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)], capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("outcome", ["absent", "loader", "version", "incomplete", "usable"])
def test_cached_selection_and_warnings(outcome):
    _run(
        '\n'
        '        import importlib, importlib.util, types, warnings\n'
        '        import behave_fire as bf\n'
        '        import behave_fire.auto as auto\n'
        '        outcome = '
        f'{outcome!r}'
        '\n'
        '        calls = []\n'
        '        extension = types.SimpleNamespace(__binding_api_version__=4, __backend_complete__=True, '
        '__capabilities__=auto.REQUIRED_CAPABILITIES)\n'
        '        if outcome == "version": extension.__binding_api_version__ = 999\n'
        '        if outcome == "incomplete": extension.__backend_complete__ = False\n'
        '        original_import = importlib.import_module\n'
        '        def discover(name):\n'
        '            assert name == "behave_fire._rust"\n'
        '            calls.append(name)\n'
        '            return None if outcome == "absent" else object()\n'
        '        def load(name):\n'
        '            if name == "behave_fire._rust":\n'
        '                if outcome == "loader": raise OSError("broken native loader")\n'
        '                return extension\n'
        '            if name == "behave_fire.rust":\n'
        '                return types.SimpleNamespace(BehaveRun=lambda fuels, threads=1: object())\n'
        '            return original_import(name)\n'
        '        importlib.util.find_spec = discover\n'
        '        importlib.import_module = load\n'
        '        expected = ["python", "rust"] if outcome == "usable" else ["python"]\n'
        '        with warnings.catch_warnings(record=True) as caught:\n'
        '            warnings.simplefilter("always")\n'
        '            assert bf.available_backends() == expected\n'
        '            assert not caught\n'
        '            for _ in range(3):\n'
        '                model = bf.BehaveRun(bf.FuelModels())\n'
        '                assert model.backend_in_use == ("rust" if outcome == "usable" else "python")\n'
        '            assert len(caught) == (0 if outcome == "usable" else 1)\n'
        '            if caught:\n'
        '                assert caught[0].category is bf.BehaveBackendWarning\n'
        '                assert caught[0].filename == "<string>"\n'
        '                assert outcome in str(caught[0].message) or (outcome == "version" and "API" in '
        'str(caught[0].message))\n'
        '        assert len(calls) == 1\n'
        '        if outcome != "usable":\n'
        '            for construct in (lambda: bf.BehaveRun(bf.FuelModels(), backend="rust"), lambda: '
        'bf.get_backend("rust")):\n'
        '                try: construct()\n'
        '                except ImportError as error:\n'
        '                    assert error.__cause__ is not None\n'
        '                    assert "install" in str(error).lower()\n'
        '                else: raise AssertionError("forced Rust did not fail")\n'
        '    '
    )


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
    _run(
        '\n'
        '        import importlib, importlib.util, types\n'
        '        import behave_fire as bf\n'
        '        from behave_fire._backend_contract import REQUIRED_CAPABILITIES\n'
        '        extension = types.SimpleNamespace(__binding_api_version__=4, __backend_complete__=True, '
        '__capabilities__=REQUIRED_CAPABILITIES)\n'
        '        importlib.util.find_spec = lambda name: object()\n'
        '        original = importlib.import_module\n'
        '        def load(name):\n'
        '            if name == "behave_fire._rust": return extension\n'
        '            if name == "behave_fire.rust": raise ImportError("facade broken")\n'
        '            return original(name)\n'
        '        importlib.import_module = load\n'
        '        try: bf.BehaveRun(bf.FuelModels())\n'
        '        except ImportError as error: assert str(error) == "facade broken"\n'
        '        else: raise AssertionError("facade error swallowed")\n'
        '        class Delegate:\n'
        '            def __init__(self, fuels, threads=1): pass\n'
        '            def calculate_fire_area(self, *args, **kwargs): raise ArithmeticError("calculation '
        'broken")\n'
        '        def load2(name):\n'
        '            if name == "behave_fire.rust": return types.SimpleNamespace(BehaveRun=Delegate)\n'
        '            return load(name)\n'
        '        importlib.import_module = load2\n'
        '        model = bf.BehaveRun(bf.FuelModels())\n'
        '        try: model.calculate_fire_area(1, 0, 1, 1, 1, 0)\n'
        '        except ArithmeticError: pass\n'
        '        else: raise AssertionError("calculation error swallowed")\n'
        '        assert model.backend_in_use == "rust"\n'
        '    '
    )


def test_imports_and_explicit_python_never_probe():
    _run(
        '\n'
        '        import importlib.abc, importlib.util, sys\n'
        '        class Poison(importlib.abc.MetaPathFinder):\n'
        '            def find_spec(self, fullname, path=None, target=None):\n'
        '                if fullname == "behave_fire._rust": raise AssertionError("native imported")\n'
        '        sys.meta_path.insert(0, Poison())\n'
        '        original = importlib.util.find_spec\n'
        '        def discover(name, *args):\n'
        '            if name == "behave_fire._rust": raise AssertionError("native probed")\n'
        '            return original(name, *args)\n'
        '        importlib.util.find_spec = discover\n'
        '        import behave_fire as bf\n'
        '        import behave_fire.python as py\n'
        '        from behave_fire.behave import BehaveRun as Legacy\n'
        '        assert py.BehaveRun is Legacy\n'
        '        assert bf.get_backend("python") is py\n'
        '        model = bf.BehaveRun(bf.FuelModels(), backend="python", threads=object())\n'
        '        assert model.backend_in_use == "python"\n'
        '        assert bf.BehaveRun.calculate_fire_area(1, 0, 1, 1, 1, 0) is not None\n'
        '        for value in (None, "bad", "RUST", 1):\n'
        '            try: bf.get_backend(value)\n'
        '            except ValueError: pass\n'
        '            else: raise AssertionError("invalid selector accepted")\n'
        '            try: bf.BehaveRun(bf.FuelModels(), backend=value)\n'
        '            except ValueError: pass\n'
        '            else: raise AssertionError("invalid constructor selector accepted")\n'
        '        modules = ("behave_units", "crown", "fine_dead_fuel_moisture_tool", "fire_size", '
        '"fuel_models", "fuel_models_array", "ignite", "mortality", "safety", "slope_tool", '
        '"species_master_table", "spot", "surface", "vapor_pressure_deficit_calculator")\n'
        '        import importlib\n'
        '        for name in modules:\n'
        '            assert getattr(py, name) is importlib.import_module("behave_fire.components." + name)\n'
        '        assert "behave_fire._rust" not in sys.modules\n'
        '    '
    )


def test_instance_backend_identity_immutability_and_signatures():
    _run(
        '\n'
        '        import inspect\n'
        '        import behave_fire as bf\n'
        '        from behave_fire.behave import BehaveRun as Legacy\n'
        '        from behave_fire.python import BehaveRun as Python\n'
        '        from behave_fire.components.fuel_models import FuelModels\n'
        '        assert Python is Legacy\n'
        '        assert bf.BehaveRun is not Legacy\n'
        '        assert bf.FuelModels is FuelModels\n'
        '        class Subclass(Python): pass\n'
        '        assert isinstance(Subclass(FuelModels()), Legacy)\n'
        '        model = bf.BehaveRun(FuelModels(), backend="python")\n'
        '        assert not isinstance(model, Legacy)\n'
        '        assert repr(model).startswith("<behave_fire.auto.BehaveRun object")\n'
        '        try: model.backend_in_use = "rust"\n'
        '        except AttributeError: pass\n'
        '        else: raise AssertionError("backend mutable")\n'
        '        names = [name for name in vars(Legacy) if not name.startswith("_")]\n'
        '        assert len(names) == 11\n'
        '        for name in names:\n'
        '            assert inspect.signature(getattr(bf.BehaveRun, name)) == '
        'inspect.signature(getattr(Legacy, name)), name\n'
        '            assert inspect.signature(getattr(model, name)) == '
        'inspect.signature(getattr(Legacy(FuelModels()), name)), name\n'
        '    '
    )


@pytest.mark.parametrize(
    "capabilities", [None, (), ("unrelated.operation",), "not-an-operation-list", (1, 2), "one-missing"],
)
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
