"""Structural array normalization for explicit native calculations."""
from importlib import import_module
from threading import Lock
import numpy as np
from .._backend_contract import BINDING_API_VERSION
from ._threads import _THREADS
try:
    _native = import_module('behave_fire._rust')
except ImportError as error:
    raise ImportError('Rust backend unavailable; install a native behave-fire wheel or build the extension.') from error
if getattr(_native, '__binding_api_version__', None) != BINDING_API_VERSION:
    raise ImportError('Rust backend has an incompatible binding API; rebuild or reinstall behave-fire.')
_ATAN2_PROFILE_CACHE = None
_ATAN2_PROFILE_LOCK = Lock()
# NumPy AVX512 gather/scatter limit after its unsigned sizeof(float) division.
# Negative byte strides fail this predicate in the supported NumPy sources.
_ATAN2_MAX_STRIDE_BYTES = (0x7fffffff // 16) * 4 + 3
_RETAIN = {'calculate_backing_spread_rate', 'calculate_eccentricity', 'calculate_fire_length', 'calculate_fire_length_to_width_ratio', 'calculate_flame_length', 'calculate_flanking_spread_rate', 'calculate_no_wind_no_slope_spread_rate', 'calculate_residence_time', 'calculate_wind_phi', 'calculate_effective_wind_speed', 'calculate_surface_wind_coefficients', 'calculate_fuelbed_intermediates', 'calculate_reaction_intensity', 'calculate_direction_of_max_spread', 'calculate_slope_phi', 'calculate_surface_rate_vector'}
_PARTICLES = ('load_dead', 'load_live', 'savr_dead', 'savr_live', 'moisture_dead', 'moisture_live')
_PARTICLE_FIELDS = _PARTICLES + ('hoc_dead', 'hoc_live', 'depth', 'moe_dead', 'is_defined')
_REACTION_FIELDS = ('sigma', 'relative_packing_ratio', 'w_moist_dead', 'w_moist_live', 'moe_dead', 'moe_live', 'w_silica_dead', 'w_silica_live', 'wfl_dead', 'wfl_live', 'w_heat_dead', 'w_heat_live')
_SPREAD_FIELDS = ('sigma', 'relative_packing_ratio', 'propagating_flux', 'heat_sink', 'depth', 'packing_ratio', 'is_defined')
_LOOKUP_FIELDS = ('depth', 'moe_dead', 'hoc_dead', 'hoc_live', 'dead_1h', 'dead_10h', 'dead_100h', 'live_herb', 'live_woody', 'savr_1h', 'savr_lh', 'savr_lw', 'is_dynamic', 'is_defined')
_FUEL_KEYS = {'depth': 'fuel_bed_depth', 'moe_dead': 'moisture_of_extinction_dead', 'hoc_dead': 'heat_of_combustion_dead', 'hoc_live': 'heat_of_combustion_live', 'savr_lh': 'savr_live_herb', 'savr_lw': 'savr_live_woody'}
_UNIT_COUNTS = {'Area': 6, 'Length': 8, 'Loading': 4, 'Pressure': 9, 'SurfaceAreaToVolume': 4, 'Speed': 7, 'Density': 2, 'HeatPerUnitArea': 3, 'HeatSourceAndReactionIntensity': 5, 'FirelineIntensity': 5, 'Time': 5}
_EMPTY_KEYS = {'calculate_safety_zone': ('separation_distance', 'radius', 'area'), 'run_contain_sim_array': ('status', 'contained_time', 'final_perimeter', 'final_area'), 'calculate_crown_scorch_mortality': ('crown_length_scorch', 'crown_volume_scorch', 'probability_mortality'), 'calculate_vpd': ('vpd', 'actual_vp', 'saturated_vp'), 'calculate_surface_rate_vector': ('x_component', 'y_component', 'rate_vector'), 'calculate_surface_wind_coefficients': ('safe_sigma', 'wind_c', 'wind_b', 'wind_e')}
_EMPTY_KEYS['run_surface_fire'] = ('spread_rate', 'backing_spread_rate', 'flanking_spread_rate', 'flame_length', 'fireline_intensity', 'heat_per_unit_area', 'effective_wind_speed', 'fire_length_to_width_ratio', 'eccentricity', 'direction_of_max_spread', 'residence_time', 'reaction_intensity', 'midflame_wind_speed', 'no_wind_no_slope_spread_rate')
_EMPTY_KEYS['build_particle_arrays'] = ('hoc_dead', 'hoc_live', 'depth', 'moe_dead', 'is_defined') + tuple((f'{key}.{j}' for key in _PARTICLES for j in range(5)))
_EMPTY_KEYS['calculate_fuelbed_intermediates'] = ('sigma', 'bulk_density', 'packing_ratio', 'relative_packing_ratio', 'heat_sink', 'propagating_flux', 'w_heat_dead', 'w_heat_live', 'w_silica_dead', 'w_silica_live', 'w_moist_dead', 'w_moist_live', 'wfl_dead', 'wfl_live', 'moe_dead', 'moe_live', 'frac_dead', 'frac_live', 'depth', 'is_defined')
_EMPTY_KEYS.update({'assign_final_crown_fire_behavior': ('final_spread_rate', 'final_heat_per_unit_area', 'final_fireline_intensity', 'final_flame_length'), 'calculate_crown_fire_ratios': ('crown_fire_transition_ratio', 'crown_fire_active_ratio'), 'calculate_crown_heat_and_intensity': ('crown_fuel_load', 'canopy_heat_per_unit_area', 'crown_fire_heat_per_unit_area', 'crown_fire_line_intensity', 'crown_flame_length'), 'calculate_passive_crown_fire_behavior': ('passive_crown_fire_spread_rate', 'passive_crown_fire_heat_per_unit_area', 'passive_crown_fire_line_intensity', 'passive_crown_fire_flame_length'), 'classify_crown_fire_type': ('fire_type', 'is_surface_fire', 'is_passive_crown', 'is_active_crown'), 'run_crown_fire': ('crown_fire_spread_rate', 'crown_critical_fire_spread_rate', 'crown_flame_length', 'crown_fire_line_intensity', 'crown_critical_surface_fire_line_intensity', 'crown_fire_heat_per_unit_area', 'canopy_heat_per_unit_area', 'crown_fire_transition_ratio', 'crown_fire_active_ratio', 'crown_length_to_width_ratio', 'fire_type', 'surface_fire_critical_spread_rate', 'crowning_surface_fire_spread_rate', 'crown_fraction_burned', 'passive_crown_fire_spread_rate', 'passive_crown_fire_heat_per_unit_area', 'passive_crown_fire_line_intensity', 'passive_crown_fire_flame_length', 'final_spread_rate', 'final_heat_per_unit_area', 'final_fireline_intensity', 'final_flame_length')})
_EMPTY_KEYS['calculate_crown_surface_fire'] = tuple(('p_crown.' + key for key in _EMPTY_KEYS['build_particle_arrays'])) + tuple(('ib_crown.' + key for key in _EMPTY_KEYS['calculate_fuelbed_intermediates'])) + tuple(('crown_surface.' + key for key in _EMPTY_KEYS['run_surface_fire'])) + ('ri_crown', 'crown_fire_spread_rate', 'crown_length_to_width_ratio')


def _atan2_math_profile():
    """Identify the linked NumPy atan2 kernel once from fixed intrinsic probes.

    :return: One for guarded SVML or zero for scalar libm; unknown kernels raise.
    """
    global _ATAN2_PROFILE_CACHE
    with _ATAN2_PROFILE_LOCK:
        if _ATAN2_PROFILE_CACHE is None:
            try:
                _ATAN2_PROFILE_CACHE = (_calibrate_atan2_profile(), None)
            except Exception as error:
                _ATAN2_PROFILE_CACHE = (None, 'Rust backend cannot identify the NumPy float32 atan2 kernel: '
                                        f'{error}. Rebuild or reinstall compatible NumPy and behave-fire, then restart '
                                        'the process, or select backend="python" explicitly.')
        selected, failure = _ATAN2_PROFILE_CACHE
        if failure is not None:
            # Store only text, never an exception/traceback that could retain caller arrays.
            raise RuntimeError(failure) from None
        return selected


def _broadcast_buffer(value, shape):
    """Validate broadcasting while retaining scalar buffers as single values.

    :param value: Normalized numeric or metadata array.
    :param shape: Checked result broadcast shape.
    :return: Logical broadcast view, or the original one-element buffer.
    """
    broadcast = np.broadcast_to(value, shape)
    return value if value.size == 1 else broadcast


def _calibrate_atan2_profile():
    """Compare fixed finite NumPy primitive results with native candidate kernels.

    :return: The unique matching native profile; rejects malformed or ambiguous candidates.
    """
    record = _native.atan2_profile_candidates()
    if not isinstance(record, (tuple, list)) or len(record) != 4:
        raise RuntimeError('malformed native atan2 configuration record')
    y_bits, x_bits, libm_bits, svml_bits = record
    y = _calibration_bits(y_bits, 'y operands').view(np.float32)
    x = _calibration_bits(x_bits, 'x operands').view(np.float32)
    libm = _calibration_bits(libm_bits, 'libm results')
    svml = None if svml_bits is None else _calibration_bits(svml_bits, 'SVML results')
    if not (np.isfinite(y).all() and np.isfinite(x).all()):
        raise RuntimeError('native atan2 configuration operands must be finite')
    # Configuration identification only: no caller inputs or fire formulas enter this ufunc.
    with np.errstate(all='ignore'):
        observed = np.arctan2(y, x)
    if not isinstance(observed, np.ndarray) or observed.dtype != np.float32 or observed.shape != (16,):
        raise RuntimeError('unexpected NumPy atan2 configuration result')
    bits = observed.view(np.uint32)
    matches = [index for index, candidate in enumerate((libm, svml))
               if candidate is not None and np.array_equal(bits, candidate)]
    if len(matches) != 1:
        raise RuntimeError('native atan2 candidates are ambiguous' if matches else 'no native atan2 candidate matches NumPy')
    return matches[0]


def _calibration_bits(value, name):
    """Validate the bounded native configuration protocol before constructing arrays.

    :param value: Sixteen unsigned binary32 payloads from the no-argument native helper.
    :param name: Diagnostic field name.
    :return: Owned uint32 array; malformed records raise without executing a probe.
    """
    if (not isinstance(value, (tuple, list)) or len(value) != 16
            or any(type(bit) is not int or not 0 <= bit <= 0xffffffff for bit in value)):
        raise RuntimeError(f'malformed native atan2 {name}')
    return np.array(value, dtype=np.uint32)


def _call(operation, arguments):
    """Own a public operation's inputs before its first detached native stage.

    :param operation: Fully qualified explicit component operation.
    :param arguments: Bound argument mapping in public signature order.
    :return: Fresh result computed exclusively from the shared input snapshot.
    """
    from ._ownership import _owned_inputs

    direction_profile = 0
    if operation == 'surface.calculate_direction_of_max_spread':
        normalized, single = _normalize_direction(arguments)
        shape_fields = ('x_component', 'y_component', 'rate_vector')
        if 'north' in str(arguments['wind_orientation_mode']).lower():
            shape_fields += ('aspect',)
        shape = np.broadcast_shapes(*(np.shape(normalized[key]) for key in shape_fields))
        if single and all(shape):
            # Select using original layout: a contiguous copy can select a
            # different NumPy float32 atan2 kernel for reversed/strided inputs.
            direction_profile = _direction_math_profile(normalized)
    with _owned_inputs() as owner:
        result = _call_owned(operation, owner.snapshot(arguments), direction_profile)
        return owner.remember(result)


def _call_owned(operation, arguments, direction_profile):
    """Normalize structural inputs and call the owned native batch boundary.

    :param operation: Fully qualified explicit component operation.
    :param arguments: Bound argument mapping in public signature order.
    :param direction_profile: Atan2 selector captured before copying original operands.
    :return: Fresh arrays in the reference result structure.
    """
    arrays = {}
    integers = {}
    mask_metadata = {}
    strings = {}
    precision = []
    arguments = dict(arguments)
    original_arguments = dict(arguments)
    name = operation.rsplit('.', 1)[1]
    direction_float32 = False
    if operation == 'surface.calculate_direction_of_max_spread':
        arguments, direction_float32 = _normalize_direction(arguments)
    if operation == 'contain.run_contain_sim_array':
        import operator
        arguments['max_steps'] = operator.index(arguments['max_steps'])
    if operation == 'ignite.calculate_lightning_ignition_probability':
        charge = arguments['charge_type']
        arguments['charge_type'] = 0 if charge == 0 else 1 if charge == 1 else 2
    if name in ('calculate_fire_area', 'calculate_fire_perimeter'):
        arguments['is_crown'] = bool(arguments['is_crown'])
    partition = arguments.pop('_native_partition', False)
    if operation == 'crown.calculate_crown_fire_ratios' and (not partition):
        fields = {'crown_fire_transition_ratio': ('surface_fli', 'critical_surface_fli'), 'crown_fire_active_ratio': ('crown_ros', 'critical_crown_ros')}
        return _partition_fields(operation, arguments, fields)
    if operation == 'crown.assign_final_crown_fire_behavior' and (not partition):
        fields = {'final_spread_rate': ('is_surface_fire', 'is_passive_crown', 'surface_ros', 'passive_ros', 'crown_ros'), 'final_heat_per_unit_area': ('is_surface_fire', 'is_passive_crown', 'surface_hpua', 'passive_hpua', 'crown_hpua'), 'final_fireline_intensity': ('is_surface_fire', 'is_passive_crown', 'surface_fli', 'passive_fli', 'crown_fli'), 'final_flame_length': ('is_surface_fire', 'is_passive_crown', 'surface_fli', 'passive_fl', 'crown_fl')}
        return _partition_fields(operation, arguments, fields)
    for mode in ('wind_orientation_mode', 'wind_height_mode', 'waf_method'):
        if mode in arguments:
            arguments[mode] = str(arguments[mode])
    for field, kind in (('wind_speed_units', 'Speed'), ('slope_units', 'Slope'), ('temp_units', 'Temperature'), ('rh_units', 'Fraction')):
        if field in arguments:
            arguments[field] = _unit_code(kind, arguments[field])
    if operation == 'vapor_pressure_deficit_calculator.calculate_vpd':
        arguments['output_units'] = _unit_code('Pressure', arguments['output_units'])
    if name in ('calculate_direction_of_max_spread', 'calculate_surface_rate_vector', 'run_surface_fire', 'run_surface', 'run_crown_fire') and 'north' not in arguments['wind_orientation_mode'].lower():
        np.asarray(arguments['aspect'], dtype=float)
        arguments['aspect'] = 0.0
    if name in ('calculate_surface_wind_adjustment_factor', 'run_surface_fire', 'run_surface'):
        supplied = arguments.get('user_waf') is not None and arguments['waf_method'].replace('_', '').lower() == 'userinput'
        if supplied and name == 'calculate_surface_wind_adjustment_factor':
            return _call('units.Fraction.to_base', {'value': arguments['user_waf'], 'units': 0})
        if supplied:
            for key in ('canopy_cover', 'canopy_height', 'crown_ratio'):
                arguments[key] = 0.0
        else:
            arguments['user_waf'] = None
    if operation.startswith('units.'):
        kind = operation.split('.')[1]
        arguments['units'] = _unit_code(kind, arguments['units'])
    if operation == 'crown.get_surface_fire_inputs':
        return {key: np.asanyarray(_normalize(arguments['surface_results'][field])).copy() for key, field in (('surface_ros', 'spread_rate'), ('surface_fli', 'fireline_intensity'), ('surface_hpua', 'heat_per_unit_area'))}
    if operation == 'crown.coerce_crown_fire_inputs':
        from .behave_units import slope_to_base
        mapping = {'fm_grid': 'fuel_model_grid', 'wind_speed': 'wind_speed', 'wind_dir': 'wind_direction', 'aspect': 'aspect', 'cbh': 'canopy_base_height', 'ch': 'canopy_height', 'cbd': 'canopy_bulk_density', 'mf_pct': 'moisture_foliar'}
        result = {key: np.atleast_1d(np.asarray(_normalize(arguments[field]), dtype=np.int32 if key == 'fm_grid' else float)).copy() for key, field in mapping.items()}
        result['slope_deg'] = slope_to_base(np.atleast_1d(_normalize(arguments['slope'])), arguments['slope_units'])
        result['S'] = result['fm_grid'].shape
        return result
    if operation == 'mortality.build_mortality_lookup':
        return np.array(_native.defaults('mortality'), dtype=np.float64)
    if operation in ('surface.build_particle_arrays', 'model.run_surface'):
        lookup = arguments.pop('lut')
        grid = np.atleast_1d(np.asarray(arguments.pop('fuel_model_grid'), dtype=np.int32))
        index = np.clip(grid, 0, lookup['depth'].shape[0] - 1)
        for key in _LOOKUP_FIELDS:
            arguments['model.' + _FUEL_KEYS.get(key, key)] = np.asanyarray(lookup[key])[index]
        for key in ('m1h', 'm10h', 'm100h', 'mlh', 'mlw'):
            arguments[key] = np.atleast_1d(np.asarray(_normalize(arguments[key]), dtype=float))
            if arguments[key].shape != grid.shape:
                raise ValueError('moisture arrays must match the fuel grid shape')
    if operation.startswith('crown.') and 'lut' in arguments:
        lookup = arguments.pop('lut')
        shape = tuple(arguments.pop('S')) if 'S' in arguments else np.atleast_1d(np.asarray(arguments['fuel_model_grid'])).shape
        if 'slope_deg' in arguments:
            arguments['slope'] = arguments.pop('slope_deg')
            arguments['slope_units'] = 0
        for key in _LOOKUP_FIELDS:
            arguments['model.' + _FUEL_KEYS.get(key, key)] = np.asanyarray(lookup[key])[10]
        for key in ('m1h', 'm10h', 'm100h', 'mlh', 'mlw'):
            arguments[key] = np.atleast_1d(np.asarray(_normalize(arguments[key]), dtype=float))
            if arguments[key].shape != shape:
                raise ValueError('moisture arrays must match the fuel grid shape')
    if 'ib_crown' in arguments:
        arguments['ib'] = arguments.pop('ib_crown')
    supplied_shape = arguments.pop('S', None)
    if supplied_shape is not None:
        if not tuple(supplied_shape):
            mask_metadata['_scalar_grid'] = np.asarray(1.0)
    if 'coeffs' in arguments:
        coefficients = np.asarray(arguments.pop('coeffs'))
        equations = np.clip(np.atleast_1d(np.asarray(arguments['equation_number_grid'], dtype=np.int32)), 0, 120)
        for column in range(3):
            arguments[f'coefficients.{column}'] = coefficients[equations, column]
    for key, value in list(arguments.items()):
        if isinstance(value, dict):
            del arguments[key]
            fields = _PARTICLE_FIELDS if key == 'p' else (_REACTION_FIELDS if name == 'calculate_reaction_intensity' else _SPREAD_FIELDS) if key == 'ib' else ('spread_rate', 'fireline_intensity', 'heat_per_unit_area') if key == 'surface_results' else tuple(value)
            for field in fields:
                data = value[field]
                if data is None:
                    raise TypeError(f'{key}.{field} must be numeric')
                if key == 'p' and field in _PARTICLES:
                    data = np.asanyarray(_normalize(data))
                    if data.ndim < 1 or data.shape[0] != 5:
                        raise ValueError('particle arrays require five leading rows')
                    if data.ndim == 1:
                        mask_metadata['_particle_scalar'] = np.asarray(1.0)
                    for j in range(5):
                        if np.ma.isMaskedArray(data):
                            row_value = np.ma.array(np.ma.getdata(data)[j], mask=np.ma.getmaskarray(data)[j])
                            row_value._fill_value = data._fill_value
                        else:
                            row_value = data[j]
                        arguments[f'{key}.{field}.{j}'] = row_value
                else:
                    arguments[f'{key}.{field}'] = data
    for name, value in arguments.items():
        if value is None and name == 'user_waf':
            continue
        if name in ('wind_orientation_mode', 'wind_height_mode', 'waf_method'):
            strings[name] = str(value)
        else:
            value = _normalize(value)
            if name in ('is_surface_fire', 'is_passive_crown', 'is_active_crown'):
                value = np.asarray(value, dtype=bool)
            dtype = np.asanyarray(value).dtype
            if dtype.kind not in 'biuf' or (dtype.kind == 'f' and dtype.itemsize > 8):
                coercion = _real_coercion(operation, name)
                if coercion is None:
                    if dtype.kind != 'O' or not _numeric_object(value):
                        if operation == 'surface.calculate_direction_of_max_spread':
                            raise AttributeError('object input does not provide the NumPy atan2 operation')
                        raise TypeError('Arbitrary-precision or opaque object arithmetic requires the explicit Python backend.')
                else:
                    value = np.asarray(value, dtype=coercion)
            numeric = np.asanyarray(value)
            if numeric.size == 0:
                mask_metadata['_empty.' + name] = np.asarray(1.0)
            if numeric.dtype.kind == 'O':
                kinds, bits, numpy_payload, type_numbers = _object_buffers(numeric)
                mask_metadata['_object.' + name] = np.asarray(1.0)
                mask_metadata['_object_kind.' + name] = kinds
                mask_metadata['_integer_type.' + name] = type_numbers
                mask_metadata['_object_numpy.' + name] = numpy_payload
                integers[name] = ('uint64', bits)
            if numeric.dtype.kind in 'biu':
                mask_metadata['_integer_type.' + name] = np.asarray(numeric.dtype.num, dtype=float)
                integers[name] = (numeric.dtype.name, np.asarray(np.ma.getdata(numeric), dtype=np.uint64))
            if np.ma.isMaskedArray(value):
                mask_metadata['_masked.' + name] = np.asarray(1.0)
                mask_metadata['_mask.' + name] = np.asarray(np.ma.getmaskarray(value), dtype=float)
                if value._fill_value is not None:
                    mask_metadata['_fill.' + name] = np.asarray(value._fill_value, dtype=float)
            if np.ndim(value) == 0:
                mask_metadata['_scalar.' + name] = np.asarray(1.0)
            if isinstance(value, np.generic):
                mask_metadata['_numpy_scalar.' + name] = np.asarray(1.0)
            if isinstance(value, (int, float, bool)) and (not isinstance(value, np.generic)):
                mask_metadata['_weak.' + name] = np.asarray(1.0)
                if isinstance(value, bool):
                    mask_metadata['_weak_bool.' + name] = np.asarray(1.0)
                elif isinstance(value, int):
                    mask_metadata['_weak_integer.' + name] = np.asarray(1.0)
            if numeric.dtype.kind == 'f' and numeric.dtype.itemsize == 4:
                mask_metadata['_single.' + name] = np.asarray(1.0)
            if name in ('is_surface_fire', 'is_passive_crown', 'is_active_crown'):
                value = np.asarray(value, dtype=bool)
            if name.endswith('_grid') or name.endswith('_i'):
                value = np.asarray(value, dtype=np.int32)
                if operation == 'mortality.calculate_crown_scorch_mortality':
                    value = np.atleast_1d(value)
            try:
                arrays[name] = np.asarray(value, dtype=np.float64)
            except ValueError as error:
                function = operation.rsplit('.', 1)[1]
                if function == 'calculate_direction_of_max_spread':
                    raise AttributeError(str(error)) from error
                if operation.startswith('surface.') and function in _RETAIN | {'run_surface_fire'} and (function != 'calculate_slope_phi') or (operation.startswith('crown.') and function not in ('run_crown_fire', 'calculate_crown_surface_fire')):
                    raise TypeError(str(error)) from error
                raise
            if not name.endswith('units') and name != 'max_steps' and (not name.endswith('is_defined')) and (not name.startswith('coeffs.')):
                precision.append(np.asarray(value) if isinstance(value, (tuple, list)) else value)
    name = operation.rsplit('.', 1)[1]
    object_input = any((np.asanyarray(value).dtype == object for value in precision))
    if name == 'calculate_wind_phi' and np.asarray(arguments['wind_e']).dtype == bool and isinstance(arguments['wind_e'], (np.ndarray, np.generic)):
        raise TypeError('boolean negative is not supported')
    if name in ('calculate_fire_length', 'calculate_flanking_spread_rate') and all((np.asarray(value).dtype == bool for value in precision[:2])) and (name == 'calculate_fire_length' or any((isinstance(value, (np.ndarray, np.generic)) for value in precision[:2]))):
        arrays['_bool'] = np.asarray(1.0)
    if name == 'calculate_direction_of_max_spread':
        precision = [arguments[key] for key in ('x_component', 'y_component', 'rate_vector')]
    elif name == 'calculate_slope_phi':
        precision = [arguments[key] for key in ('packing_ratio', 'reaction_intensity')]
    elif name == 'calculate_surface_rate_vector':
        precision = [arguments[key] for key in ('no_wind_no_slope_spread_rate', 'wind_phi', 'slope_phi')]
    single = name in _RETAIN and bool(precision) and (np.result_type(*precision) == np.dtype('float32'))
    if operation.startswith('crown.'):
        single = bool(precision) and np.result_type(*precision) == np.dtype('float32')
        pairs = {'calculate_crown_fraction_burned': (('crowning_surface_ros', 'surface_critical_ros'), ('surface_ros', 'surface_critical_ros')), 'calculate_crown_heat_and_intensity': (('canopy_height', 'canopy_base_height'),), 'calculate_passive_crown_fire_behavior': (('crown_ros', 'surface_ros'),)}
        for pair in pairs.get(name, ()):
            values = [arguments[key] for key in pair]
            if all((np.asarray(value).dtype == bool for value in values)) and any((isinstance(value, (np.ndarray, np.generic)) for value in values)):
                raise TypeError('boolean subtraction is not supported')
        mixed_fields = {'calculate_critical_surface_fireline_intensity': ('canopy_base_height',), 'calculate_surface_fire_critical_spread_rate': ('critical_surface_fli',), 'assign_final_crown_fire_behavior': ('surface_ros', 'surface_hpua', 'surface_fli'), 'calculate_crowning_surface_fire_spread_rate': ('critical_crown_ros',)}
        if name in mixed_fields and all((np.asarray(_normalize(arguments[key])).dtype == np.float32 for key in mixed_fields[name])):
            arrays['_mixed'] = np.asarray(1.0)
    if name == 'run_surface_fire':
        single = np.asarray(_normalize(arguments['ri'])).dtype == np.float32
    if operation.startswith('contain.'):
        shape = np.atleast_1d(arrays['report_spread_rate']).shape
    else:
        shape = np.broadcast_shapes(*(value.shape for value in arrays.values()))
    input_shapes = {key: value.shape for key, value in arrays.items()}
    arrays.update(mask_metadata)
    arrays['_simd_exp'] = np.asarray(_math_profile(), dtype=float)
    arrays['_svml_atan2'] = np.asarray(direction_profile if direction_float32 and all(shape) else 0, dtype=float)
    arrays = {key: _broadcast_buffer(value, shape) for key, value in arrays.items()}
    integers = {key: (kind, _broadcast_buffer(bits, shape)) for key, (kind, bits) in integers.items()}
    result = _native.batch(operation, arrays, strings, shape, single, integers, threads=_THREADS.get())
    if not result and 0 in shape:
        result = {key: np.empty(shape) for key in _EMPTY_KEYS.get(name, ('',))}
    promoted = name in ('calculate_effective_wind_speed', 'run_surface_fire', 'calculate_fuelbed_intermediates', 'calculate_slope_phi', 'calculate_surface_rate_vector') or (name == 'calculate_direction_of_max_spread' and 'north' in strings['wind_orientation_mode'].lower())
    if single and (not promoted):
        result = {key: value if key.startswith('__') or '__single.' + key in result else value.astype(np.float32) for key, value in result.items()}
    if name == 'assign_final_crown_fire_behavior':
        for field, sources in (('final_spread_rate', ('surface_ros', 'passive_ros', 'crown_ros')), ('final_heat_per_unit_area', ('surface_hpua', 'passive_hpua', 'crown_hpua')), ('final_fireline_intensity', ('surface_fli', 'passive_fli', 'crown_fli'))):
            dtype = np.result_type(*[_normalize(arguments[key]) for key in sources])
            if '__single.' + field not in result or dtype == object:
                result[field] = result[field].astype(dtype)
    if name == 'calculate_passive_crown_fire_behavior':
        for field, sources in (('passive_crown_fire_spread_rate', ('surface_ros', 'crown_ros', 'cfb')), ('passive_crown_fire_heat_per_unit_area', ('surface_hpua', 'canopy_hpua', 'cfb'))):
            dtype = np.result_type(*[_normalize(arguments[key]) for key in sources])
            if '__single.' + field not in result or dtype == object:
                result[field] = result[field].astype(dtype)
    if 'is_defined' in result:
        dtype = np.asanyarray(original_arguments['p']['is_defined']).dtype if 'p' in original_arguments else bool
        result['is_defined'] = result['is_defined'].astype(dtype)
    if name == 'build_particle_arrays':
        for field in _PARTICLES:
            result[field] = np.stack([result.pop(f'{field}.{j}') for j in range(5)])
    if 'status' in result:
        result['status'] = result['status'].astype(np.int32)
    if 'fire_type' in result:
        result['fire_type'] = result['fire_type'].astype(np.int32)
    for field in ('is_surface_fire', 'is_passive_crown', 'is_active_crown'):
        if field in result:
            result[field] = result[field].astype(bool)
    if name == 'calculate_crown_surface_fire':
        for prefix in ('p_crown', 'ib_crown', 'crown_surface'):
            result[prefix] = {key[len(prefix) + 1:]: result.pop(key) for key in list(result) if key.startswith(prefix + '.')}
        for field in _PARTICLES:
            result['p_crown'][field] = np.stack([result['p_crown'].pop(f'{field}.{j}') for j in range(5)])
        for prefix in ('p_crown', 'ib_crown'):
            result[prefix]['is_defined'] = result[prefix]['is_defined'].astype(bool)
    dependencies = {'calculate_crown_scorch_mortality': {'crown_length_scorch': ('scorch_height_ft', 'tree_height_ft', 'crown_ratio'), 'crown_volume_scorch': ('scorch_height_ft', 'tree_height_ft', 'crown_ratio')}, 'calculate_safety_zone': {'separation_distance': ('flame_height_ft',)}, 'calculate_vpd': {'saturated_vp': ('temperature',)}, 'calculate_crown_fire_ratios': {'crown_fire_transition_ratio': ('surface_fli', 'critical_surface_fli'), 'crown_fire_active_ratio': ('crown_ros', 'critical_crown_ros')}, 'calculate_crown_heat_and_intensity': {'crown_fuel_load': ('canopy_base_height', 'canopy_height', 'canopy_bulk_density'), 'canopy_heat_per_unit_area': ('canopy_base_height', 'canopy_height', 'canopy_bulk_density'), 'crown_fire_heat_per_unit_area': ('surface_hpua', 'canopy_base_height', 'canopy_height', 'canopy_bulk_density')}, 'calculate_passive_crown_fire_behavior': {'passive_crown_fire_spread_rate': ('surface_ros', 'crown_ros', 'cfb'), 'passive_crown_fire_heat_per_unit_area': ('surface_hpua', 'canopy_hpua', 'cfb')}}
    dependencies['calculate_vpd']['saturated_vp'] = ('temperature', 'temp_units', 'output_units')
    if name == 'run_crown_fire':
        dependencies[name] = _crown_dependencies(arguments['wind_orientation_mode'])
    for field, sources in dependencies.get(name, {}).items():
        if name == 'run_crown_fire':
            target = np.broadcast_shapes(*(np.shape(original_arguments['surface_results'][key.partition('.')[2]]) if key.startswith('surface_results.') else np.atleast_1d(original_arguments[key]).shape for key in sources))
        else:
            target = np.broadcast_shapes(*(input_shapes.get(key, np.shape(original_arguments[key])) for key in sources))
        if result[field].size == 0 and all((size > 0 for size in target)):
            result[field] = _recompute_field(operation, original_arguments, field, sources)
            for key in list(result):
                if key.startswith('__') and key.partition('.')[2] == field:
                    del result[key]
        else:
            result[field] = _restore_shape(result[field], target)
            for key in list(result):
                if key.startswith('__') and key.partition('.')[2] == field and result[key].ndim:
                    result[key] = _restore_shape(result[key], target)
    metadata = {key: result.pop(key) for key in list(result) if key.startswith('__')}
    for key in result:
        result[key] = _restore_payload(result[key], metadata, key)
        if '__masked.' + key in metadata and np.any(metadata['__masked.' + key]):
            result[key] = np.ma.array(result[key], mask=metadata['__mask.' + key].astype(bool))
            if '__fill_present.' + key in metadata and np.any(metadata['__fill_present.' + key]):
                result[key].set_fill_value(metadata['__fill.' + key].ravel()[0])
    box_surface = object_input and name in _RETAIN and (name not in ('calculate_effective_wind_speed', 'calculate_reaction_intensity', 'calculate_fuelbed_intermediates'))
    box_crown = object_input and operation.startswith('crown.') and (name not in ('run_crown_fire', 'calculate_crown_surface_fire', 'calculate_crowning_surface_fire_spread_rate', 'classify_crown_fire_type'))
    if box_surface or box_crown:
        result = {key: value if '__object.' + key in metadata else value.astype(object) for key, value in result.items()}
    return result[''] if '' in result else result


def _crown_dependencies(wind_orientation_mode):
    """Describe each crown output's independent NumPy broadcast dependencies.

    :param wind_orientation_mode: Direction convention deciding whether aspect participates.
    :return: Output names mapped to original public input names; no numerical work occurs.
    """
    fuel = ('fuel_model_grid',)
    spread = fuel + ('wind_speed', 'wind_direction', 'slope') + (('aspect',) if 'north' in wind_orientation_mode.lower() else ())
    canopy_heat = ('canopy_base_height', 'canopy_height', 'canopy_bulk_density')
    critical_intensity = ('canopy_base_height', 'moisture_foliar')
    critical_spread = ('canopy_bulk_density',)
    surface_ros = ('surface_results.spread_rate',)
    surface_fli = ('surface_results.fireline_intensity',)
    surface_hpua = ('surface_results.heat_per_unit_area',)
    crown_heat = surface_hpua + canopy_heat
    crown_intensity = crown_heat + spread
    transition = surface_fli + critical_intensity
    active = spread + critical_spread
    classification = transition + active
    surface_critical = critical_intensity + surface_hpua
    crowning_surface = fuel + critical_spread
    fraction_burned = surface_ros + surface_critical + crowning_surface
    passive_spread = surface_ros + fraction_burned + spread
    passive_heat = surface_hpua + fraction_burned + canopy_heat
    passive_intensity = passive_spread + passive_heat
    return {
        'crown_fire_spread_rate': spread,
        'crown_flame_length': crown_intensity,
        'crown_fire_line_intensity': crown_intensity,
        'crown_fire_heat_per_unit_area': crown_heat,
        'canopy_heat_per_unit_area': canopy_heat,
        'crown_critical_surface_fire_line_intensity': critical_intensity,
        'crown_critical_fire_spread_rate': critical_spread,
        'crown_fire_transition_ratio': transition,
        'crown_fire_active_ratio': active,
        'crown_length_to_width_ratio': spread,
        'fire_type': classification,
        'surface_fire_critical_spread_rate': surface_critical,
        'crowning_surface_fire_spread_rate': crowning_surface,
        'crown_fraction_burned': fraction_burned,
        'passive_crown_fire_spread_rate': passive_spread,
        'passive_crown_fire_heat_per_unit_area': passive_heat,
        'passive_crown_fire_line_intensity': passive_intensity,
        'passive_crown_fire_flame_length': passive_intensity,
        'final_spread_rate': classification + surface_ros + passive_spread + spread,
        'final_heat_per_unit_area': classification + surface_hpua + passive_heat + crown_heat,
        'final_fireline_intensity': classification + surface_fli + passive_intensity + crown_intensity,
        'final_flame_length': classification + surface_fli + passive_intensity + crown_intensity,
    }


def _direction_math_profile(arguments):
    """Preserve the reference atan2 inner-loop eligibility before owned copying.

    :param arguments: Normalized original direction operands, before rate/aspect broadcasting.
    :return: Scalar kernel selector or per-cell selectors when iterator chunks differ.
    """
    if _atan2_math_profile() == 0:
        return 0
    operands = [np.asarray(arguments[key]) for key in ('y_component', 'x_component')]
    if any(value.size == 0 for value in operands):
        return 0
    buffer_size = np.getbufsize()
    shapes = [value.shape for value in operands if value.ndim]
    trivial = (all(value.ndim == 0 or (value.ndim == 1 and
                   ((value.dtype == np.float32 and value.flags.aligned) or value.size <= buffer_size))
                   for value in operands)
               and all(shape == shapes[0] for shape in shapes))
    if trivial:
        # NumPy pre-copies scalar/small 1-D cast or unaligned operands, while
        # retaining even singleton negative/large strides of aligned float32.
        strides = [0 if not value.ndim else value.strides[0]
                   if value.dtype == np.float32 and value.flags.aligned else 4 for value in operands]
        return int(all(0 <= stride <= _ATAN2_MAX_STRIDE_BYTES for stride in strides))
    if all(0 <= stride <= _ATAN2_MAX_STRIDE_BYTES for value in operands for stride in value.strides):
        return 1
    # The buffered K-order iterator determines effective strides after casting,
    # alignment copies, broadcasting, and axis ordering. No mathematical ufunc
    # executes on these operands; the allocated output stores selectors only.
    with np.errstate(all='ignore'):
        with np.nditer(operands + [None],
                       flags=['external_loop', 'buffered', 'grow_inner', 'zerosize_ok', 'refs_ok', 'copy_if_overlap'],
                       op_flags=[['readonly', 'aligned'], ['readonly', 'aligned'],
                                 ['writeonly', 'allocate', 'aligned', 'no_broadcast', 'no_subtype']],
                       op_dtypes=[np.float32] * 3, order='K', casting='unsafe', buffersize=buffer_size) as iterator:
            result = iterator.operands[2]
            first = None
            mixed = False
            for chunk in iterator:
                selected = int(all(0 <= value.strides[0] <= _ATAN2_MAX_STRIDE_BYTES for value in chunk))
                chunk[2].fill(selected)
                if first is None:
                    first = selected
                else:
                    mixed |= selected != first
    return result if mixed else int(first)


def _math_profile():
    """Read NumPy's active float32 exponential implementation metadata.

    :return: One for NumPy's x86 SIMD kernel, zero for the baseline libm route.
    """
    module = import_module('numpy._core._multiarray_umath')
    current = getattr(module, '__cpu_targets_info__', {}).get('exp', {}).get('ff', {}).get('current')
    if current is not None:
        return int(current.startswith(('AVX2', 'AVX512', 'X86_V3', 'X86_V4')))
    features = getattr(module, '__cpu_features__', {})
    targets = getattr(module, '__cpu_dispatch__', ())
    grouped = any((target in targets and features.get(target, False) for target in ('X86_V3', 'X86_V4')))
    return int(grouped or ('AVX2' in targets and features.get('AVX2', False) and features.get('FMA3', False)))


def _normalize(value):
    """Widen float16 before each entry point's ordinary coercion.

    :param value: Scalar or array input.
    :return: Input preserving its shape and mask, with float16 widened.
    """
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind == 'f' and value.dtype.itemsize == 2:
        return value.astype(np.float32)
    return value


def _normalize_direction(arguments):
    """Widen operands when NumPy would select an internal float16 atan2 loop.

    :param arguments: Bound direction-helper inputs.
    :return: Normalized inputs and whether the resulting atan2 loop uses float32.
    """
    result = dict(arguments)
    operands = [_normalize(result[key]) for key in ('y_component', 'x_component')]
    dtypes = [float if isinstance(value, float) and (not isinstance(value, np.generic)) else int if isinstance(value, int) and (not isinstance(value, (bool, np.generic))) else np.asanyarray(value).dtype for value in operands]
    resolved = np.arctan2.resolve_dtypes((*dtypes, None))
    for key, value in zip(('y_component', 'x_component'), operands):
        if resolved[2] == np.dtype('float16'):
            result[key] = value.astype(np.float32) if isinstance(value, (np.ndarray, np.generic)) else np.float32(value)
        else:
            result[key] = value
    return result, resolved[2] in (np.dtype('float16'), np.dtype('float32'))


def _numeric_kind(value):
    """Classify scalar arithmetic without coercing away its precision.

    :param value: Public scalar state or input.
    :return: Native Python-float, NumPy-float64, or NumPy-float32 code.
    """
    if isinstance(value, (np.ndarray, np.generic)):
        return 2 if value.dtype in (np.dtype('float16'), np.dtype('float32')) else 1
    return 0


def _numeric_object(value):
    """Recognize supported boxed numeric payloads without executing custom operators.

    :param value: Candidate object scalar or array.
    :return: Whether every payload belongs to the supported real numeric domain.
    """
    integer_types = (bool, int, np.bool_, np.int8, np.int16, np.intc, np.int32, np.int64, np.longlong, np.uint8, np.uint16, np.uintc, np.uint32, np.uint64, np.ulonglong)
    floating_types = (float, np.float16, np.float32, np.float64)
    for item in np.ma.getdata(np.asanyarray(value)).flat:
        if type(item) in integer_types:
            if not -2 ** 63 <= int(item) <= 2 ** 64 - 1:
                return False
        elif type(item) not in floating_types:
            return False
    return True


def _object_buffers(value):
    """Encode supported object payloads without numerical arithmetic.

    :param value: Object array containing supported real scalar payloads.
    :return: Per-element dtype codes, exact integer bits, NumPy scalar flags, and exact integer type numbers.
    """
    values = np.asarray(np.ma.getdata(value), dtype=object)
    kinds = np.full(values.shape, 9.0)
    bits = np.zeros(values.shape, dtype=np.uint64)
    numpy_payload = np.zeros(values.shape, dtype=float)
    type_numbers = np.full(values.shape, -1.0)
    names = ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool')
    for index in np.ndindex(values.shape):
        item = values[index]
        numpy_payload[index] = isinstance(item, np.generic)
        if isinstance(item, (np.integer, np.bool_)):
            type_numbers[index] = item.dtype.num
        if isinstance(item, (bool, np.bool_)):
            kinds[index] = 8
            bits[index] = int(item)
        elif isinstance(item, (int, np.integer)):
            kinds[index] = names.index(item.dtype.name) if isinstance(item, np.integer) else 3 if item <= np.iinfo(np.int64).max else 7
            bits[index] = int(item) & (1 << 64) - 1
        elif isinstance(item, np.float32):
            kinds[index] = 10
    return (kinds, bits, numpy_payload, type_numbers)


def _partition_fields(operation, arguments, fields):
    """Marshal independent output dependency groups without forcing shared axes.

    :param operation: Native operation route.
    :param arguments: Complete bound arguments.
    :param fields: Output fields and the source inputs each field depends on.
    :return: Owned native results with independent reference shapes and dtypes.
    """
    for key, value in arguments.items():
        if key in ('is_surface_fire', 'is_passive_crown', 'is_active_crown'):
            value = np.asarray(value, dtype=bool)
        dtype = np.asanyarray(_normalize(value)).dtype
        if dtype.kind not in 'biuf' or (dtype.kind == 'f' and dtype.itemsize > 8):
            if dtype.kind != 'O' or not _numeric_object(value):
                raise TypeError('Arbitrary-precision or opaque object arithmetic requires the explicit Python backend.')
    result = {}
    for field, dependencies in fields.items():
        scoped = {key: value if key in dependencies else 1.0 for key, value in arguments.items()}
        scoped['_native_partition'] = True
        result[field] = _call(operation, scoped)[field]
    return result


def _real_coercion(operation, parameter):
    """Describe explicit reference coercion before numerical arithmetic.

    :param operation: Native public calculation route.
    :param parameter: Flattened public input name.
    :return: Supported real dtype for explicit coercion, or None for raw arithmetic.
    """
    if parameter.endswith(('_grid', '_i')):
        return np.int32
    module, name = operation.split('.', 1)
    if module in ('units', 'safety', 'spot', 'ignite', 'contain', 'vapor_pressure_deficit_calculator', 'fine_dead_fuel_moisture_tool'):
        return np.float64
    if module == 'mortality':
        return None if parameter.startswith('coefficients.') else np.float64
    if module == 'crown':
        if name in ('run_crown_fire', 'calculate_crown_surface_fire') and (not parameter.startswith(('surface_results.', 'model.'))):
            return np.float64
        return None
    normalized = {'build_particle_arrays': ('m1h', 'm10h', 'm100h', 'mlh', 'mlw'), 'calculate_direction_of_max_spread': ('aspect',), 'calculate_fire_area': ('forward_ros', 'backing_ros', 'lwr'), 'calculate_fire_perimeter': ('forward_ros', 'backing_ros', 'lwr'), 'calculate_fire_width': ('forward_ros', 'backing_ros', 'lwr'), 'calculate_fireline_intensity': ('spread_rate', 'heat_per_unit_area'), 'calculate_forward_spread_rate': ('no_wind_no_slope_spread_rate', 'rate_vector'), 'calculate_heat_per_unit_area': ('reaction_intensity', 'residence_time'), 'calculate_midflame_wind_speed': ('waf', 'twenty_foot_wind_speed'), 'calculate_reaction_intensity_output': ('reaction_intensity',), 'calculate_slope_phi': ('slope',), 'calculate_surface_rate_vector': ('wind_direction', 'aspect'), 'calculate_surface_wind_adjustment_factor': ('canopy_cover', 'canopy_height', 'crown_ratio', 'depth', 'user_waf'), 'calculate_twenty_foot_wind_speed': ('wind_speed',), 'calculate_wind_adjustment_factor': ('canopy_cover', 'canopy_height', 'crown_ratio', 'depth'), 'run_surface_fire': ('wind_speed', 'wind_direction', 'slope', 'aspect', 'canopy_cover', 'canopy_height', 'crown_ratio', 'user_waf', 'ib.depth')}
    return np.float64 if parameter in normalized.get(name, ()) else None


def _recompute_field(operation, arguments, field, dependencies):
    """Evaluate an independent nonempty field when another output has zero cells.

    :param operation: Native calculation route.
    :param arguments: Original public inputs.
    :param field: Output field to retain.
    :param dependencies: Public inputs that determine this output field.
    :return: Owned field computed in the native core over its own input shape.
    """
    if operation == 'crown.run_crown_fire':
        scoped = dict(arguments)
        retain_fuel = 'fuel_model_grid' in dependencies
        if not retain_fuel:
            scoped['fuel_model_grid'] = np.array([10], dtype=np.int32)
            for key in ('m1h', 'm10h', 'm100h', 'mlh', 'mlw'):
                scoped[key] = np.ones(1)
        for key in ('wind_speed', 'wind_direction', 'slope', 'aspect', 'canopy_base_height', 'canopy_height', 'canopy_bulk_density', 'moisture_foliar'):
            if key not in dependencies:
                scoped[key] = 1.0
        scoped['surface_results'] = {key: value if 'surface_results.' + key in dependencies else 1.0 for key, value in arguments['surface_results'].items()}
        scoped['_native_partition'] = True
        return _call(operation, scoped)[field]
    scoped = {key: value if key in dependencies or key.endswith('units') or isinstance(value, str) else 1.0 for key, value in arguments.items()}
    if 'coeffs' in arguments:
        scoped['coeffs'] = np.array(_native.defaults('mortality'))
        scoped['equation_number_grid'] = 0
    scoped['_native_partition'] = True
    return _call(operation, scoped)[field]


def _restore_payload(value, metadata, key):
    """Reconstruct owned typed outputs from native scalar metadata.

    :param value: Owned real-value storage returned by the native extension.
    :param metadata: Native per-field dtype, exact bits, and object metadata.
    :param key: Result field name.
    :return: An owned numeric or object array with exact native payload types.
    """
    if '__single.' + key not in metadata:
        return value
    names = ('int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool')
    integer = metadata.get('__integer_dtype.' + key, np.asarray(-1.0))
    type_numbers = metadata.get('__integer_type.' + key, np.asarray(-1.0))
    type_chars = '?bBhHiIlLqQ'
    single = metadata['__single.' + key]
    complex_kind = metadata.get('__complex.' + key, np.asarray(0.0))
    object_kind = metadata.get('__object.' + key, np.asarray(0.0))
    bits = None
    if np.any(integer >= 0) and value.size:
        bits = metadata['__integer_low.' + key].astype(np.uint64) | metadata['__integer_high.' + key].astype(np.uint64) << np.uint64(32)
    if np.any(object_kind):
        result = np.empty(value.shape, dtype=object)
        flags = np.broadcast_to(metadata.get('__object_numpy.' + key, np.asarray(0.0)), value.shape)
        integers = np.broadcast_to(integer, value.shape)
        exact_types = np.broadcast_to(type_numbers, value.shape)
        singles = np.broadcast_to(single, value.shape)
        complexes = np.broadcast_to(complex_kind, value.shape)
        imaginary = np.broadcast_to(metadata.get('__imag.' + key, np.asarray(0.0)), value.shape)
        for index in np.ndindex(value.shape):
            if complexes[index]:
                payload = np.asarray(complex(value[index], imaginary[index]), dtype=np.complex64 if singles[index] else np.complex128)[()]
            elif integers[index] >= 0:
                dtype = type_chars[int(exact_types[index])] if exact_types[index] >= 0 else names[int(integers[index])]
                payload = bits[index].astype(dtype)
            else:
                payload = np.asarray(value[index], dtype=np.float32 if singles[index] else np.float64)[()]
            result[index] = payload if flags[index] else payload.item()
        return result
    if np.any(complex_kind):
        result = np.empty(value.shape, dtype=np.complex64 if np.all(single) else np.complex128)
        result.real = value
        result.imag = metadata.get('__imag.' + key, 0.0)
        return result
    if np.any(integer >= 0):
        code = int(integer.ravel()[0])
        if not 0 <= code <= 8 or not np.all(integer == code):
            raise ValueError('native output has inconsistent integer dtype metadata')
        exact_type = int(type_numbers.ravel()[0]) if type_numbers.size else -1
        dtype = type_chars[exact_type] if exact_type >= 0 else names[code]
        return np.empty(value.shape, dtype=dtype) if bits is None else bits.astype(dtype)
    return value.astype(np.float32 if np.all(single) else np.float64) if key != 'is_defined' else value


def _restore_shape(value, shape):
    """Restore an output's independent broadcast shape.

    :param value: Array computed over the union of input dimensions.
    :param shape: Reference shape determined by this field's input dependencies.
    :return: Owned array with the field's public shape.
    """
    if value.shape == shape:
        return value
    selectors = (0,) * (value.ndim - len(shape)) + tuple((slice(0, 1) if size == 1 else slice(None) for size in shape))
    return np.array(value[selectors], copy=True).reshape(shape)


def _state(instance, domain, method, arguments):
    """Copy helper state into a checked native operation and restore its snapshot.

    :param instance: Public native state wrapper.
    :param domain: Native helper module name.
    :param method: Selected public helper method.
    :param arguments: Bound scalar arguments, including the receiver.
    :return: Native scalar result, or None after mutation.
    """
    arguments = {key: value for key, value in arguments.items() if key != 'self'}
    state = dict(vars(instance))
    if domain == 'fire_size':
        state['spread_rates_are_numpy_'] = isinstance(state.get('forward_spread_rate_'), (np.ndarray, np.generic))
        state['effective_wind_is_numpy_'] = isinstance(state.get('effective_wind_speed_'), (np.ndarray, np.generic))
        state['ratio_is_numpy_'] = isinstance(state.get('fire_length_to_width_ratio_'), (np.ndarray, np.generic))
        for flag, key in (('effective_wind_numeric_kind_', 'effective_wind_speed_'), ('ratio_numeric_kind_', 'fire_length_to_width_ratio_'), ('eccentricity_numeric_kind_', 'eccentricity_')):
            state[flag] = _numeric_kind(state.get(key))
        if method == 'calculate_fire_basic_dimensions':
            arguments['_wind_kind'] = _numeric_kind(arguments['effective_wind_speed'])
    updated, result, error = _native.state_call(domain, method, state, arguments)
    if domain == 'fire_size':
        for flag, fields in (('spread_rates_are_numpy_', ('forward_spread_rate_', 'backing_spread_rate_', 'flanking_spread_rate_')), ('effective_wind_is_numpy_', ('effective_wind_speed_',)), ('ratio_is_numpy_', ('fire_length_to_width_ratio_',))):
            if updated.pop(flag):
                for key in fields:
                    updated[key] = np.float64(updated[key])
        for flag, key in (('effective_wind_numeric_kind_', 'effective_wind_speed_'), ('ratio_numeric_kind_', 'fire_length_to_width_ratio_'), ('eccentricity_numeric_kind_', 'eccentricity_')):
            kind = updated.pop(flag)
            if kind:
                updated[key] = (np.float32 if kind == 2 else np.float64)(updated[key])
    vars(instance).update(updated)
    if error:
        category = {'Domain': ValueError, 'Overflow': OverflowError, 'DivisionByZero': ZeroDivisionError}.get(error, RuntimeError)
        raise category(error)
    if method in ('get_fire_length_to_width_ratio', 'get_eccentricity', 'get_heading_to_backing_ratio'):
        return updated[{'get_fire_length_to_width_ratio': 'fire_length_to_width_ratio_', 'get_eccentricity': 'eccentricity_', 'get_heading_to_backing_ratio': 'heading_to_backing_ratio_'}[method]]
    if method.startswith('get_number_of_') or method == 'get_representative_fraction_at_index':
        return int(result)
    return result


def _unit_code(kind, units):
    """Normalize the reference's scalar branch or table-index unit convention.

    :param kind: Native unit family name.
    :param units: Caller-supplied unit selector.
    :return: Declared native unit code or codes, preserving index exceptions.
    """
    if units == 0:
        return 0
    if kind in ('Fraction', 'Slope', 'BasalArea', 'HeatOfCombustion', 'HeatSink'):
        return 1
    if kind == 'Temperature':
        return 1 if units == 1 else 2 if units == 2 else 0
    indices = list(range(_UNIT_COUNTS[kind])) if kind == 'SurfaceAreaToVolume' else np.arange(_UNIT_COUNTS[kind])
    return indices[units]


def _validate_threads(threads):
    """Validate the explicit native worker-count option.

    :param threads: Integer worker count; zero requests available parallelism.
    :return: Validated nonnegative Python integer.
    """
    import operator
    threads = operator.index(threads)
    if threads < 0:
        raise ValueError('threads must be nonnegative')
    return threads
