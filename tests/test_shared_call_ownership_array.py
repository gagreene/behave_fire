"""Automatic preflight and native stages reuse one backend-neutral snapshot."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_capture_and_invoke_share_owned_arrays_with_nested_native_ownership():
    from behave_fire._auto_routing import capture_call
    from behave_fire._ownership import _owned_inputs
    from behave_fire.python import surface

    original = np.array([3., 4.], dtype=np.float32)
    call = capture_call('surface.calculate_fire_area', surface.calculate_fire_area,
                        (2., 1., 2., original), {})
    captured = call.arguments['elapsed_min']
    original[:] = 99.

    def selected(forward_ros, backing_ros, lwr, elapsed_min, is_crown=False):
        with _owned_inputs() as owner:
            assert elapsed_min is captured
            assert owner.snapshot(elapsed_min) is captured
        return elapsed_min

    assert call.invoke(selected) is captured
    np.testing.assert_array_equal(captured, [3., 4.])


def test_importing_shared_owner_never_discovers_native_extension():
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    source = '''
import importlib.util
original = importlib.util.find_spec
def find(name, *args):
    if name == 'behave_fire._rust': raise AssertionError('native discovered')
    return original(name, *args)
importlib.util.find_spec = find
from behave_fire._ownership import _owned_inputs
with _owned_inputs() as owner:
    assert owner.snapshot(1.) == 1.
'''
    result = subprocess.run([sys.executable, '-c', source], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
