"""Pure selector and explicit Python worker-keyword contracts."""
import pytest


def test_python_routes_keep_distinct_thread_interfaces():
    from behave_fire import BehaveRun as AutoRun
    from behave_fire.python import BehaveRun as PythonRun, FuelModels as PythonFuels

    model = AutoRun(PythonFuels(), backend="python", threads="intentionally ignored")
    assert model.backend_in_use == "python"
    with pytest.raises(TypeError):
        PythonRun(PythonFuels(), threads=4)
