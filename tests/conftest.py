"""Native test selection with an explicit required-extension CI mode."""

from importlib import import_module

import pytest


def pytest_addoption(parser):
    parser.addoption("--require-rust", action="store_true", default=False, help="Require the native extension and binding API version 3.")


def pytest_collection_modifyitems(config, items):
    if config._behave_native_error is None:
        return
    marker = pytest.mark.skip(reason=str(config._behave_native_error))
    for item in items:
        if item.get_closest_marker("rust") is not None:
            item.add_marker(marker)


def pytest_configure(config):
    error = None
    try:
        native = import_module("behave_fire._rust")
        if getattr(native, "__binding_api_version__", None) != 3:
            raise ImportError("behave_fire._rust has an incompatible binding API; expected version 2")
    except Exception as caught:
        error = caught
    config._behave_native_error = error
    if error is not None and config.getoption("--require-rust"):
        raise pytest.UsageError(f"--require-rust requires a usable native extension: {error}") from error
