"""Opt-in replay of the registered C++ public API, goldens, and entry guards."""

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.cpp_live


# ---- live C++ qualification tests ----

def test_cpp_live_inventory_and_goldens(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    from cpp_calibrate import measure_cpp

    compiler = os.environ.get('BEHAVE_CPP_LIVE_COMPILER')
    if compiler not in (None, 'GNU', 'MSVC'):
        raise pytest.UsageError('BEHAVE_CPP_LIVE_COMPILER must be GNU or MSVC.')
    output = Path(os.environ.get('BEHAVE_CPP_LIVE_OUT', tmp_path))
    report = measure_cpp(ROOT, output, compiler)
    assert report['passed'] is True
    assert report['public_api']['public_functions'] > 0
