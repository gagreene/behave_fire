"""Native test selection with an explicit required-extension CI mode."""

from importlib import import_module
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def cpp_snapshot_template():
    """Capture pristine evidence bytes while retaining fresh per-case validation."""
    from cpp_snapshot_support import capture_snapshot

    return capture_snapshot(Path(__file__).resolve().parents[1])


def pytest_addoption(parser):
    parser.addoption("--require-rust", action="store_true", default=False,
        help="Require the native extension and binding API version 4.")


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
        if getattr(native, "__binding_api_version__", None) != 4:
            raise ImportError("behave_fire._rust has an incompatible binding API; expected version 4")
    except Exception as caught:
        error = caught
    config._behave_native_error = error
    if error is not None and config.getoption("--require-rust"):
        raise pytest.UsageError(f"--require-rust requires a usable native extension: {error}") from error
