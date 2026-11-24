"""Check selection of the live C++ lane through pytest's public CLI."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# ---- live lane selection tests ----

def test_cpp_live_requires_the_environment_and_exact_marker():
    cases = (
        ({}, (), 5, 'deselected'),
        ({'BEHAVE_CPP_LIVE': '1'}, ('-m', 'not slow'), 5, 'deselected'),
        ({}, ('-m', 'cpp_live'), 4, 'BEHAVE_CPP_LIVE=1'),
        ({'BEHAVE_CPP_LIVE': '1'}, ('-m', 'cpp_live'), 0, 'test_cpp_live_inventory_and_goldens'),
    )
    for env, markers, exit_code, expected in cases:
        environment = os.environ.copy()
        environment.pop('BEHAVE_CPP_LIVE', None)
        environment.pop('PYTEST_ADDOPTS', None)
        environment.update(env)
        result = subprocess.run(
            [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider', '--collect-only',
             'tests/cpp_live', *markers],
            cwd=ROOT, env=environment, text=True, capture_output=True, check=False,
        )
        assert result.returncode == exit_code, result.stdout + result.stderr
        assert expected in (result.stdout + result.stderr), result.stdout + result.stderr
