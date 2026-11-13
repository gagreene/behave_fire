"""API4 advertises one physical numeric contract for all operation families."""
import numpy as np
import pytest

pytestmark = pytest.mark.rust


def test_manifest_covers_each_operation_with_one_float64_contract():
    from behave_fire import _rust
    from behave_fire._backend_contract import BINDING_API_VERSION
    from behave_fire.rust._contracts import numeric_contract

    assert BINDING_API_VERSION == _rust.__binding_api_version__ == 4
    manifest = _rust.__numeric_contracts__
    assert len(manifest) == 88
    for name, contract in manifest.items():
        assert contract == numeric_contract(name)
    assert set(manifest.values()) == {'f64'}


def test_stateless_physical_values_widen_and_missing_controls_raise():
    from behave_fire.rust import spot, vapor_pressure_deficit_calculator as vpd

    actual = spot.calculate_spotting_from_burning_pile(np.array([2.], np.float32), 5., 20.)
    assert actual.dtype == np.float64
    with pytest.raises(ValueError, match='missing'):
        vpd.calculate_vpd(25., np.ma.array([1], mask=True), .5, 0)
    with pytest.raises(TypeError):
        spot.calculate_spotting_from_burning_pile(np.array([2.], object), 5., 20.)
