"""Public gates include dtype-neutral masks and an additional mixed layout."""
import numpy as np
import pytest
from parity_support import ROOT


@pytest.mark.rust
@pytest.mark.parametrize('cells', [1, 16, 1031])
@pytest.mark.parametrize('dataset', ['uniform', 'heterogeneous', 'continuous'])
@pytest.mark.parametrize('operation', ['surface', 'crown', 'typed-crown', 'typed-crown-b',
                                     'masked-crown', 'masked-false-crown', 'typed-crown-c'])
def test_gate_layouts_validate_every_native_output(cells, dataset, operation):
    from types import SimpleNamespace
    from benchmark_backends import _validation_worker

    record = _validation_worker(SimpleNamespace(operation=operation, dataset=dataset,
        cells=cells, threads=1, seed=20261105))
    assert record['passed'], record.get('error')
    assert record['output_fields'] == len(record['fields']) == (14 if operation == 'surface' else 22)
    assert record['native_boundary_calls'] == ['model_batch']


@pytest.mark.parametrize('operation', ['masked-crown', 'masked-false-crown', 'typed-crown-c'])
def test_public_layouts_are_distinct(operation):
    from behave_fire.python import BehaveRun, FuelModels
    from performance_cases import supplied_surface, build_inputs

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
