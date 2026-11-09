"""Native automatic-model snapshots survive copies, pickle, and process transfer."""

import copy
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest

pytestmark = pytest.mark.rust
native = pytest.importorskip("behave_fire._rust", exc_type=ImportError)


def _environment():
    return dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))


@pytest.fixture(params=["python", "rust"])
def _native_model(monkeypatch, request):
    from behave_fire import BehaveRun, auto, get_backend

    monkeypatch.setattr(native, "__backend_complete__", True)
    monkeypatch.setattr(auto, "_PROBE_RESULT", None)
    monkeypatch.setattr(auto, "_WARNING_EMITTED", False)
    registry = get_backend(request.param).FuelModels()
    model = BehaveRun(registry, backend="rust", threads=3)
    registry.fuel_models_[1]["fuel_bed_depth"] = 777.
    assert model._delegate._lut["depth"][1] == 1.
    return model


def test_native_automatic_copy_and_deepcopy_preserve_cached_state(_native_model):
    model = _native_model
    shallow, deep = copy.copy(model), copy.deepcopy(model)
    assert shallow is not model
    assert shallow._delegate is not model._delegate
    assert shallow._delegate._lut is model._delegate._lut
    assert shallow._delegate._fuel_models is model._delegate._fuel_models
    assert deep._delegate._lut is not model._delegate._lut
    assert deep._delegate._fuel_models is not model._delegate._fuel_models
    assert not np.shares_memory(deep._delegate._lut["depth"], model._delegate._lut["depth"])
    assert deep.backend_in_use == "rust"
    assert deep._threads == deep._delegate.threads == 3
    deep._delegate._lut["depth"][1] = 999.
    assert model._delegate._lut["depth"][1] == 1.
    assert model._delegate._fuel_models.fuel_models_[1]["fuel_bed_depth"] == 777.


@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
def test_native_automatic_pickle_retains_selected_backend_threads_and_snapshot(_native_model, protocol):
    model = _native_model
    restored = pickle.loads(pickle.dumps(model, protocol=protocol))
    assert restored.backend_in_use == "rust"
    assert restored._threads == restored._delegate.threads == 3
    assert restored.__getstate__()["version"] == 2
    assert restored._delegate.__getstate__()["schema_version"] == 1
    assert restored._delegate._fuel_models.fuel_models_[1]["fuel_bed_depth"] == 777.
    np.testing.assert_array_equal(restored._delegate._lut["depth"], model._delegate._lut["depth"])
    assert restored._delegate._lut["depth"][1] == 1.
    inputs = dict(fuel_model_grid=np.array([1]), m1h=np.array([.06]), m10h=np.array([.07]),
                  m100h=np.array([.08]), mlh=np.array([.6]), mlw=np.array([.9]),
                  wind_speed=np.array([8.]), wind_speed_units=0, wind_direction=np.array([35.]),
                  wind_orientation_mode="RelativeToUpslope", slope=np.array([12.]))
    actual, expected = restored.do_surface_run(**inputs), model.do_surface_run(**inputs)
    for field in expected:
        np.testing.assert_array_equal(actual[field], expected[field])
    with pytest.raises(AttributeError):
        restored.backend_in_use = "python"


@pytest.mark.parametrize("extension", ["compatible", "absent", "incompatible"])
def test_native_automatic_restore_in_fresh_process_never_reselects(_native_model, extension, tmp_path):
    payload = tmp_path / "auto-native.pickle"
    payload.write_bytes(pickle.dumps(_native_model))
    source = textwrap.dedent('''
        import importlib.abc, pickle, sys, types
        import numpy as np
        import behave_fire.auto as auto
        from pathlib import Path
        mode = sys.argv[2]
        def poison(): raise AssertionError("native restore attempted backend reselection")
        auto._probe = poison
        if mode == "absent":
            class Absent(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname == "behave_fire._rust": raise ImportError("native extension deliberately absent")
            sys.meta_path.insert(0, Absent())
        elif mode == "incompatible":
            extension = types.ModuleType("behave_fire._rust")
            extension.__binding_api_version__ = 999
            sys.modules["behave_fire._rust"] = extension
        else:
            from behave_fire import _rust
            _rust.__backend_complete__ = False
        try:
            model = pickle.loads(Path(sys.argv[1]).read_bytes())
        except ImportError as error:
            assert mode in ("absent", "incompatible")
            assert "Rust" in str(error) or "API" in str(error)
            assert "install" in str(error).lower() or "rebuild" in str(error).lower()
        else:
            assert mode == "compatible"
            from behave_fire import _rust
            assert _rust.__backend_complete__ is False
            assert model.backend_in_use == "rust"
            assert model._threads == model._delegate.threads == 3
            assert model._delegate._lut["depth"][1] == 1.
            assert model._delegate._fuel_models.fuel_models_[1]["fuel_bed_depth"] == 777.
            original = _rust.component_batch
            requests = []
            def record(*args, **kwargs):
                requests.append(kwargs.get("threads", 1))
                return original(*args, **kwargs)
            _rust.component_batch = record
            np.testing.assert_array_equal(model.calculate_fire_length(np.array([1.]), .5, 0, 3., 0), np.array([4.5]))
            assert requests and set(requests) == {3}
        assert auto._PROBE_RESULT is None
    ''')
    result = subprocess.run([sys.executable, "-c", source, str(payload), extension], env=_environment(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""


def test_native_automatic_windows_spawn_preserves_snapshot(_native_model, tmp_path):
    payload = tmp_path / "auto-native-spawn.pickle"
    payload.write_bytes(pickle.dumps(_native_model))
    script = tmp_path / "spawn_native_dispatch.py"
    script.write_text(textwrap.dedent('''
        import multiprocessing as mp
        from pathlib import Path
        import pickle
        import sys

        def inspect_model(payload, queue):
            try:
                model = pickle.loads(Path(payload).read_bytes())
                queue.put((model.backend_in_use, model._threads, model._delegate.threads,
                           float(model._delegate._lut["depth"][1]), model._delegate._fuel_models.fuel_models_[1]["fuel_bed_depth"]))
            except BaseException as error:
                queue.put((type(error).__name__, str(error)))

        if __name__ == "__main__":
            context = mp.get_context("spawn")
            queue = context.Queue()
            process = context.Process(target=inspect_model, args=(sys.argv[1], queue))
            process.start()
            try:
                result = queue.get(timeout=30)
                process.join(timeout=30)
                assert process.exitcode == 0, process.exitcode
                assert result == ("rust", 3, 3, 1., 777.), result
            finally:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10)
                queue.close()
    '''), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script), str(payload)], env=_environment(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stderr == ""
