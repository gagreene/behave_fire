"""Independent Python f64 science plus the reviewed missing-data truth rules."""
import hashlib
from pathlib import Path
import sys

import numpy as np

from performance_cases import build_inputs, supplied_surface

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def _missing_rules(outputs, supplied):
    """Correct old NumPy threshold defaults using the reviewed new contract.

    :param outputs: Ordinary Python result arrays on widened, NaN-filled inputs.
    :param supplied: Actual normalized Surface fields consumed by Crown.
    :return: Corrected complete output mapping without calling Rust.
    """
    heat_missing = np.isnan(supplied['heat_per_unit_area'])
    rate_missing = np.isnan(supplied['spread_rate'])
    code = outputs['fire_type']
    if np.any(heat_missing):
        for field in ('crown_fire_heat_per_unit_area', 'crown_fire_line_intensity', 'crown_flame_length',
                      'surface_fire_critical_spread_rate', 'crown_fraction_burned',
                      'passive_crown_fire_spread_rate', 'passive_crown_fire_heat_per_unit_area',
                      'passive_crown_fire_line_intensity', 'passive_crown_fire_flame_length',
                      'final_heat_per_unit_area'):
            outputs[field][heat_missing] = np.nan
        outputs['final_spread_rate'][heat_missing & (code == 1)] = np.nan
        for field in ('final_fireline_intensity', 'final_flame_length'):
            outputs[field][heat_missing & np.isin(code, [1, 3])] = np.nan
    if np.any(rate_missing):
        unknown_fraction = rate_missing & (
            outputs['crowning_surface_fire_spread_rate'] - outputs['surface_fire_critical_spread_rate'] > 1e-7)
        outputs['crown_fraction_burned'][unknown_fraction] = np.nan
        outputs['passive_crown_fire_heat_per_unit_area'][unknown_fraction] = np.nan
        for field in ('passive_crown_fire_spread_rate', 'passive_crown_fire_line_intensity',
                      'passive_crown_fire_flame_length'):
            outputs[field][rate_missing] = np.nan
        outputs['final_spread_rate'][rate_missing & np.isin(code, [0, 1, 2])] = np.nan
        for field in ('final_fireline_intensity', 'final_flame_length'):
            outputs[field][rate_missing & (code == 1)] = np.nan
        outputs['final_heat_per_unit_area'][unknown_fraction & (code == 1)] = np.nan
    return outputs


def expected_outputs(operation, cells, dataset, seed):
    """Calculate complete common-grid references without importing the Rust backend.

    :param operation: Surface, Crown, or original mixed-input layout name.
    :param cells: Common native raster size.
    :param dataset: Original uniform, heterogeneous, or continuous input recipe.
    :param seed: Original deterministic recipe seed.
    :return: Full field dictionary, widened scientific dtype and typed categories.
    """
    from behave_fire.python import BehaveRun, FuelModels

    inputs = build_inputs(cells, dataset, seed)
    model = BehaveRun(FuelModels())
    with np.errstate(all='ignore'):
        if operation == 'surface':
            result = model.do_surface_run(**inputs)
        else:
            supplied = supplied_surface(inputs, operation, model, uniform=dataset == 'uniform')
            supplied = {field: np.asarray(np.ma.asarray(value, dtype=np.float64).filled(np.nan))
                        for field, value in supplied.items()}
            canopy = dict(canopy_base_height=4., canopy_height=30., canopy_bulk_density=.01, moisture_foliar=100.)
            if dataset == 'heterogeneous':
                phase = ((np.arange(cells) + seed) % 17) / 16
                canopy = dict(canopy_base_height=1 + 5 * phase, canopy_height=15 + 25 * phase,
                              canopy_bulk_density=.005 + .02 * phase, moisture_foliar=80 + 40 * phase)
            result = model.do_crown_run(surface_results=supplied, **inputs, **canopy)
        result = {field: np.broadcast_to(np.asarray(value), (cells,)).astype(
                    np.int32 if field == 'fire_type' else np.float64, copy=True)
                  for field, value in result.items()}
        if operation != 'surface':
            result = _missing_rules(result, supplied)
    return result


def validate_export(directory, operation, cells, dataset, seed):
    """Compare every stored native field to an independent complete Python reference.

    :param directory: Directory of actual typed binary column exports.
    :param operation: Original workload route.
    :param cells: Expected complete native row count.
    :param dataset: Input recipe name.
    :param seed: Deterministic input recipe identity.
    :return: Complete validation record and exact exported field checksums.
    """
    directory = Path(directory)
    expected = expected_outputs(operation, cells, dataset, seed)
    files = sorted(directory.glob('*.f64')) + sorted(directory.glob('*.i32'))
    if {path.stem for path in files} != set(expected) or len(files) != len(expected):
        raise AssertionError('Incomplete/duplicate stored output inventory')
    hashes = {}
    for path in files:
        field = path.stem
        suffix = '.i32' if field == 'fire_type' else '.f64'
        if path.suffix != suffix:
            raise AssertionError(f'Wrong typed output storage: {field}')
        actual = np.fromfile(path, dtype='<i4' if field == 'fire_type' else '<f8')
        if actual.shape != (cells,):
            raise AssertionError(f'Wrong stored output length: {field}')
        wanted = expected[field]
        if field == 'fire_type':
            np.testing.assert_array_equal(actual, wanted, err_msg=field)
        else:
            np.testing.assert_array_equal(np.isnan(actual), np.isnan(wanted), err_msg=field)
            np.testing.assert_array_equal(np.isposinf(actual), np.isposinf(wanted), err_msg=field)
            np.testing.assert_array_equal(np.isneginf(actual), np.isneginf(wanted), err_msg=field)
            zeros = (actual == 0) & (wanted == 0)
            np.testing.assert_array_equal(np.signbit(actual[zeros]), np.signbit(wanted[zeros]), err_msg=field)
            np.testing.assert_allclose(actual, wanted, rtol=1e-9, atol=1e-12, equal_nan=True, err_msg=field)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(complete=True, independent_python=True, truth_contract='f64-nan-v1',
                all_fields=True, typed_codes=True, fields=sorted(expected), export_hashes=hashes)