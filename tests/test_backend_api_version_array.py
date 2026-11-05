"""Reject old complete native modules before any numerical operation."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'pytest.ini').is_file())


def test_binding_api_one_is_rejected_by_every_native_entry_route():
    source = '''
        import importlib, importlib.machinery, sys, types, warnings
        import behave_fire as bf
        from behave_fire._backend_contract import REQUIRED_CAPABILITIES
        native = types.ModuleType('behave_fire._rust')
        native.__spec__ = importlib.machinery.ModuleSpec('behave_fire._rust', loader=None)
        native.__binding_api_version__ = 1
        native.__backend_complete__ = True
        native.__capabilities__ = REQUIRED_CAPABILITIES
        sys.modules['behave_fire._rust'] = native
        assert bf.available_backends() == ['python']
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            assert bf.BehaveRun(bf.FuelModels()).backend_in_use == 'python'
        assert len(caught) == 1
        assert 'expected 2' in str(caught[0].message)
        for call in (lambda: bf.get_backend('rust'), lambda: importlib.import_module('behave_fire.rust')):
            try:
                call()
            except ImportError as error:
                assert 'API' in str(error) or 'native' in str(error)
            else:
                raise AssertionError('old complete API 1 extension was accepted')
    '''
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(source)],
                            env=dict(os.environ, PYTHONPATH=str(ROOT / 'src')),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.rust
def test_native_restore_rejects_binding_api_one_state():
    from behave_fire.rust import FireSize

    original = FireSize()
    state = original.__getstate__()
    state["binding_api_version"] = 1
    with pytest.raises(ValueError, match="API"):
        FireSize.__new__(FireSize).__setstate__(state)
