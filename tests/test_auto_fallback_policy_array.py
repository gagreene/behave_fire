"""Automatic native absence is a visible, once-per-process Python fallback."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


def _run(source):
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'))
    result = subprocess.run([sys.executable, '-c', textwrap.dedent(source)],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('entry', ['facade', 'namespace'])
def test_absence_error_filter_prevents_delegate_and_does_not_consume_warning(entry):
    _run(f'''
        import importlib.util, types, warnings
        import behave_fire as bf
        import behave_fire.auto as auto
        calls = []
        importlib.util.find_spec = lambda name: None
        original = importlib.import_module
        def load(name):
            if name == 'behave_fire.python':
                calls.append(name)
                return types.SimpleNamespace(__name__='behave_fire.python', BehaveRun=lambda fuels: object())
            return original(name)
        importlib.import_module = load
        def select():
            return bf.get_backend('auto') if {entry!r} == 'namespace' else bf.BehaveRun(bf.FuelModels())
        warnings.simplefilter('error', bf.BehaveBackendWarning)
        for _ in range(3):
            try: select()
            except bf.BehaveBackendWarning: pass
            else: raise AssertionError('fallback warning did not prevent delegation')
        assert not calls
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', bf.BehaveBackendWarning)
            select()
            select()
            assert len(caught) == 1
            assert 'Python' in str(caught[0].message)
            assert caught[0].filename == '<string>'
        assert calls
    ''')


def test_absence_warns_once_across_concurrent_namespace_and_model_selection():
    _run('''
        from concurrent.futures import ThreadPoolExecutor
        import importlib.util, warnings
        import behave_fire as bf
        importlib.util.find_spec = lambda name: None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', bf.BehaveBackendWarning)
            assert bf.available_backends() == ['python']
            assert not caught
            def choose(index):
                if index % 2: return bf.get_backend('auto')
                return bf.BehaveRun(bf.FuelModels())
            with ThreadPoolExecutor(max_workers=4) as workers:
                assert len(list(workers.map(choose, range(12)))) == 12
            assert len(caught) == 1
            assert caught[0].category is bf.BehaveBackendWarning
    ''')
