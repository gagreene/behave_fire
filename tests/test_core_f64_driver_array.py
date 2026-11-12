"""Confirmation must use a passing frozen screen and new samples."""
import importlib.util
from pathlib import Path

import pytest


def _driver():
    path = Path(__file__).resolve().parents[1] / 'tools/benchmark_core_f64.py'
    spec = importlib.util.spec_from_file_location('benchmark_core_f64', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmation_rejects_failed_or_changed_screen():
    driver = _driver()
    identity = dict(source_hash='a' * 64, binary_hash='b' * 64,
                    environment=dict(numpy='2.1.3', host='same'))
    screen = dict(identity, phase='screen', assessment=dict(passed=True))
    driver.validate_confirmation(screen, identity)
    for changed in (dict(screen, assessment=dict(passed=False)),
                    dict(screen, source_hash='c' * 64),
                    dict(screen, binary_hash='d' * 64),
                    dict(screen, environment=dict(numpy='different')),
                    dict(screen, phase='confirmation')):
        with pytest.raises(ValueError):
            driver.validate_confirmation(changed, identity)
