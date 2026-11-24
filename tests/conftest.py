"""Test tier selection and native-extension requirements."""

import os
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


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Select the live lane explicitly, apply tier markers, and skip unavailable native tests.

    :param config: Active pytest configuration with native availability recorded.
    :param items: Tests discovered by pytest.
    :return: None. Updates test markers in place.
    """
    if config.getoption("markexpr").strip() != "cpp_live":
        live_items = [item for item in items if item.get_closest_marker("cpp_live") is not None]
        if live_items:
            items[:] = [item for item in items if item.get_closest_marker("cpp_live") is None]
            config.hook.pytest_deselected(items=live_items)
    test_root = config.rootpath / "tests"
    tier_marks = {"integration": pytest.mark.integration, "regression": pytest.mark.regression}
    native_error = config._behave_native_error
    for item in items:
        if item.path.is_relative_to(test_root):
            tier = item.path.relative_to(test_root).parts[0]
            if tier in tier_marks:
                item.add_marker(tier_marks[tier])
        if native_error is not None and item.get_closest_marker("rust") is not None:
            item.add_marker(pytest.mark.skip(reason=str(native_error)))


def pytest_configure(config):
    if config.getoption("markexpr").strip() == "cpp_live" and os.environ.get("BEHAVE_CPP_LIVE") != "1":
        raise pytest.UsageError("The cpp_live lane requires BEHAVE_CPP_LIVE=1 and -m cpp_live.")
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
