"""Automatic facade snapshots, legacy pickles, copies, and process transfer."""

import copy
import os
from pathlib import Path
import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from behave_fire import BehaveRun, FuelModels
from behave_fire.behave import BehaveRun as PythonBehaveRun


def _environment():
    return dict(os.environ, PYTHONPATH=str(Path(__file__).parents[1] / "src"))


def _model():
    registry = FuelModels()
    model = BehaveRun(registry, backend="python", threads=7)
    before = model._delegate._lut["depth"].copy()
    registry.fuel_models_[1]["fuel_bed_depth"] = 777.0
    assert model._delegate._lut["depth"][1] != 777.0
    return model, before


def test_copy_preserves_reference_sharing_and_deepcopy_independence():
    model, expected = _model()
    shallow = copy.copy(model)
    deep = copy.deepcopy(model)
    assert shallow is not model
    assert shallow._delegate is not model._delegate
    assert shallow._delegate._lut is model._delegate._lut
    assert shallow._delegate._fuel_models is model._delegate._fuel_models
    assert deep._delegate._fuel_models is not model._delegate._fuel_models
    assert deep.backend_in_use == "python"
    assert deep._threads == 7
    np.testing.assert_array_equal(deep._delegate._lut["depth"], expected)
    deep._delegate._lut["depth"][1] = 999.0
    assert model._delegate._lut["depth"][1] != 999.0


def test_fresh_process_unpickle_preserves_snapshot(tmp_path):
    model, expected = _model()
    payload_path = tmp_path / "snapshot.pickle"
    payload_path.write_bytes(pickle.dumps(model))
    source = textwrap.dedent(f'''
        import importlib.util, pickle
        from pathlib import Path
        original = importlib.util.find_spec
        def poison(name, *args):
            if name == "behave_fire._rust": raise AssertionError("Python unpickle probed native")
            return original(name, *args)
        importlib.util.find_spec = poison
        model = pickle.loads(Path({str(payload_path)!r}).read_bytes())
        assert model.backend_in_use == "python"
        assert model._threads == 7
        assert model._delegate._lut["depth"][1] == {float(expected[1])!r}
        assert model._delegate._fuel_models.fuel_models_[1]["fuel_bed_depth"] == 777.0
    ''')
    result = subprocess.run([sys.executable, "-c", source], env=_environment(), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_historical_python_pickle_remains_loadable(protocol):
    model = PythonBehaveRun(FuelModels())
    payload = pickle.dumps(model, protocol=protocol)
    assert b"behave_fire.behave" in payload
    restored = pickle.loads(payload)
    assert type(restored) is PythonBehaveRun
    np.testing.assert_array_equal(restored._lut["depth"], model._lut["depth"])


@pytest.mark.parametrize("protocol", range(2, pickle.HIGHEST_PROTOCOL + 1))
def test_pickle_preserves_backend_threads_and_cached_snapshot(protocol):
    model, expected = _model()
    restored = pickle.loads(pickle.dumps(model, protocol=protocol))
    assert restored.backend_in_use == "python"
    assert restored._threads == 7
    assert restored.__getstate__()["version"] == 2
    np.testing.assert_array_equal(restored._delegate._lut["depth"], expected)
    assert restored._delegate._fuel_models.fuel_models_[1]["fuel_bed_depth"] == 777.0
    with pytest.raises(AttributeError):
        restored.backend_in_use = "rust"


def test_python_route_converts_a_native_registry_snapshot_without_probe():
    class Registry:
        def __init__(self):
            self.python = FuelModels()

        def to_python(self):
            return copy.deepcopy(self.python)

    registry = Registry()
    model = BehaveRun(registry, backend="python")
    registry.python.fuel_models_[1]["fuel_bed_depth"] = 777.0
    assert model._delegate._fuel_models.fuel_models_[1]["fuel_bed_depth"] != 777.0


def test_state_rejects_unknown_version_or_backend():
    model, _ = _model()
    state = model.__getstate__()
    with pytest.raises(ValueError, match="version"):
        model.__setstate__(dict(state, version=999))
    with pytest.raises(ValueError, match="version"):
        model.__setstate__(dict(state, version=1))
    with pytest.raises(ValueError, match="backend"):
        model.__setstate__(dict(state, backend="auto"))


def test_windows_spawn_transfers_python_snapshot(tmp_path):
    script = tmp_path / "spawn_dispatch.py"
    script.write_text(textwrap.dedent('''
        import multiprocessing as mp
        import numpy as np
        from behave_fire import BehaveRun, FuelModels

        def inspect_model(model, queue):
            queue.put((model.backend_in_use, model._threads, float(model._delegate._lut["depth"][1]), model._delegate._fuel_models.fuel_models_[1]["fuel_bed_depth"]))

        if __name__ == "__main__":
            registry = FuelModels()
            model = BehaveRun(registry, backend="python", threads=3)
            before = float(model._delegate._lut["depth"][1])
            registry.fuel_models_[1]["fuel_bed_depth"] = 777.0
            context = mp.get_context("spawn")
            queue = context.Queue()
            process = context.Process(target=inspect_model, args=(model, queue))
            process.start()
            result = queue.get(timeout=30)
            process.join(timeout=30)
            assert process.exitcode == 0
            assert result == ("python", 3, before, 777.0)
    '''), encoding="utf-8")
    result = subprocess.run([sys.executable, str(script)], env=_environment(), capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
