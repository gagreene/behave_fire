"""Public layouts include dtype-neutral masks and an additional mixed layout."""
import numpy as np
import pytest

from parity_support import ROOT as ROOT

# ---- public layout tests ----

@pytest.mark.rust
@pytest.mark.parametrize('cells', [1, 16, 1031])
@pytest.mark.parametrize('dataset', ['uniform', 'heterogeneous', 'continuous'])
@pytest.mark.parametrize('operation', ['surface', 'crown', 'typed-crown', 'typed-crown-b',
                                     'masked-crown', 'masked-false-crown', 'typed-crown-c'])
def test_gate_layouts_validate_every_native_output(cells, dataset, operation):
    from raster_cases import validate_layout

    record = validate_layout(operation, dataset, cells)
    assert record['passed'], record.get('error')
    assert record['output_fields'] == len(record['fields']) == (14 if operation == 'surface' else 22)
    assert record['native_boundary_calls'] == ['model_batch']


def test_model_call_rejects_unknown_backends():
    from raster_cases import model_call

    with pytest.raises(ValueError, match='rsut'):
        model_call('surface', 'uniform', 4, 'rsut')


def test_model_call_rejects_normalized_native_reference():
    from raster_cases import model_call

    with pytest.raises(ValueError):
        model_call('surface', 'uniform', 4, 'rust', normalize_reference=True)


@pytest.mark.parametrize('operation', ['masked-crown', 'masked-false-crown', 'typed-crown-c'])
def test_public_layouts_are_distinct(operation):
    from behave_fire.python import BehaveRun, FuelModels
    from raster_cases import build_inputs, supplied_surface

    inputs = build_inputs(16, 'uniform')
    values = supplied_surface(inputs, operation, BehaveRun(FuelModels()))
    if operation == 'typed-crown-c':
        assert values['spread_rate'].dtype == np.int16
        assert values['heat_per_unit_area'].dtype == np.int32
        assert values['fireline_intensity'].dtype == np.float32
        assert np.ma.getmaskarray(values['fireline_intensity']).any()
    else:
        assert np.ma.isMaskedArray(values['heat_per_unit_area'])
        assert values['heat_per_unit_area'].dtype == np.float64
        assert bool(np.ma.getmaskarray(values['heat_per_unit_area']).any()) == (operation == 'masked-crown')
