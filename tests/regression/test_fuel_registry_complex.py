"""Complex values are rejected in fuel-registry data on every backend, and only there."""

import numpy as np
import pytest

from behave_fire import BehaveRun, FuelModels
from behave_fire.components.fuel_models import FuelModels as PythonFuelModels

COMPLEX_VALUES = [12 + 0j, 12 + 5j, np.complex128(12 + 0j), np.complex128(12 + 5j), np.complex64(12 + 1j)]
SETTER_ARGS = (0, .2, 0, 8000, 8000, 0, .1, .05, .05, .02, .02, 0, 2000, 1500, 1500, 0, False)
BACKENDS = ["python", pytest.param("rust", marks=pytest.mark.rust), pytest.param("auto", marks=pytest.mark.rust)]


def _cell():
    return dict(fuel_model_grid=np.array([[256]]), m1h=np.array([[.06]]), m10h=np.array([[.07]]),
                m100h=np.array([[.08]]), mlh=np.array([[.6]]), mlw=np.array([[.9]]), wind_speed=np.array([[10.]]),
                wind_speed_units=6, wind_direction=np.array([[225.]]), wind_orientation_mode="RelativeToNorth",
                slope=np.array([[10.]]), slope_units=1, aspect=np.array([[180.]]), canopy_cover=np.array([[.4]]),
                canopy_height=np.array([[15.]]), crown_ratio=np.array([[.55]]), wind_height_mode="TenMeter")


def _registry_with_model_256():
    registry = FuelModels()
    assert registry.set_custom_fuel_model(256, "ZZ1", "custom", 3., 0, *SETTER_ARGS[1:])
    return registry


# ---- fuel registry complex value tests ----

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("field", ["fuel_bed_depth", "dead_1h", "savr_live_woody", "is_dynamic"])
@pytest.mark.parametrize("value", COMPLEX_VALUES)
def test_complex_value_in_a_stored_registry_is_rejected_on_every_backend(backend, field, value):
    registry = _registry_with_model_256()
    registry.fuel_models_[256][field] = value
    with pytest.raises(TypeError, match=rf"Fuel model 256 field {field}: complex values are not accepted"):
        BehaveRun(registry, backend=backend).do_surface_run(**_cell())


@pytest.mark.rust
@pytest.mark.parametrize("value", COMPLEX_VALUES)
def test_complex_value_is_rejected_by_the_rust_setter(value):
    from behave_fire.rust.fuel_models import FuelModels as RustFuelModels

    with pytest.raises(TypeError, match="Fuel model 256 field fuel_bed_depth: complex values are not accepted"):
        RustFuelModels().set_custom_fuel_model(256, "ZZ1", "custom", value, *SETTER_ARGS)


@pytest.mark.parametrize("value", COMPLEX_VALUES)
def test_complex_value_is_rejected_by_the_setter(value):
    with pytest.raises(TypeError, match="Fuel model 256 field fuel_bed_depth: complex values are not accepted"):
        PythonFuelModels().set_custom_fuel_model(256, "ZZ1", "custom", value, *SETTER_ARGS)


def test_complex_value_in_a_later_setter_field_names_that_field():
    args = list(SETTER_ARGS)
    args[7] = np.complex128(.05)  # the 10-hr load
    with pytest.raises(TypeError, match="Fuel model 256 field dead_10h: complex values are not accepted"):
        PythonFuelModels().set_custom_fuel_model(256, "ZZ1", "custom", 3., *args)


@pytest.mark.parametrize("registry_class", [PythonFuelModels, pytest.param("rust", marks=pytest.mark.rust)])
def test_real_numpy_scalars_remain_accepted_in_the_registry(registry_class):
    if registry_class == "rust":
        from behave_fire.rust.fuel_models import FuelModels as registry_class
    assert registry_class().set_custom_fuel_model(256, "ZZ1", "custom", np.float64(3.), *SETTER_ARGS)


def test_complex_calculation_inputs_are_not_registry_data():
    cell = _cell()
    cell.update(fuel_model_grid=np.array([[124]]), wind_speed=np.array([[10. + 0j]]))
    with pytest.warns(np.exceptions.ComplexWarning):
        result = BehaveRun(FuelModels(), backend="python").do_surface_run(**cell)
    assert np.isfinite(result["spread_rate"]).all()
