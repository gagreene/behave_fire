"""Structural array normalization for explicit native calculations."""
from importlib import import_module
import numpy as np
from .._backend_contract import BINDING_API_VERSION
from ._threads import _THREADS
try:
    _native = import_module('behave_fire._rust')
except ImportError as error:
    raise ImportError('Rust backend unavailable; install a native behave-fire wheel or build the extension.') from error
if getattr(_native, '__binding_api_version__', None) != BINDING_API_VERSION:
    raise ImportError('Rust backend has an incompatible binding API; rebuild or reinstall behave-fire.')
_PARTICLES = ('load_dead', 'load_live', 'savr_dead', 'savr_live', 'moisture_dead', 'moisture_live')
_PARTICLE_FIELDS = _PARTICLES + ('hoc_dead', 'hoc_live', 'depth', 'moe_dead', 'is_defined')
_REACTION_FIELDS = ('sigma', 'relative_packing_ratio', 'w_moist_dead', 'w_moist_live', 'moe_dead', 'moe_live', 'w_silica_dead', 'w_silica_live', 'wfl_dead', 'wfl_live', 'w_heat_dead', 'w_heat_live')
_SPREAD_FIELDS = ('sigma', 'relative_packing_ratio', 'propagating_flux', 'heat_sink', 'depth', 'packing_ratio', 'is_defined')
_LOOKUP_FIELDS = ('depth', 'moe_dead', 'hoc_dead', 'hoc_live', 'dead_1h', 'dead_10h', 'dead_100h', 'live_herb', 'live_woody', 'savr_1h', 'savr_lh', 'savr_lw', 'is_dynamic', 'is_defined')
_FUEL_KEYS = {'depth': 'fuel_bed_depth', 'moe_dead': 'moisture_of_extinction_dead', 'hoc_dead': 'heat_of_combustion_dead', 'hoc_live': 'heat_of_combustion_live', 'savr_lh': 'savr_live_herb', 'savr_lw': 'savr_live_woody'}
_UNIT_COUNTS = {'Area': 6, 'Length': 8, 'Loading': 4, 'Pressure': 9, 'SurfaceAreaToVolume': 4, 'Speed': 7, 'Density': 2, 'HeatPerUnitArea': 3, 'HeatSourceAndReactionIntensity': 5, 'FirelineIntensity': 5, 'Time': 5}
_EMPTY_KEYS = {'calculate_safety_zone': ('separation_distance', 'radius', 'area'), 'calculate_crown_scorch_mortality': ('crown_length_scorch', 'crown_volume_scorch', 'probability_mortality'), 'calculate_vpd': ('vpd', 'actual_vp', 'saturated_vp'), 'calculate_surface_rate_vector': ('x_component', 'y_component', 'rate_vector'), 'calculate_surface_wind_coefficients': ('safe_sigma', 'wind_c', 'wind_b', 'wind_e')}
_EMPTY_KEYS['run_surface_fire'] = ('spread_rate', 'backing_spread_rate', 'flanking_spread_rate', 'flame_length', 'fireline_intensity', 'heat_per_unit_area', 'effective_wind_speed', 'fire_length_to_width_ratio', 'eccentricity', 'direction_of_max_spread', 'residence_time', 'reaction_intensity', 'midflame_wind_speed', 'no_wind_no_slope_spread_rate')
_EMPTY_KEYS['build_particle_arrays'] = ('hoc_dead', 'hoc_live', 'depth', 'moe_dead', 'is_defined') + tuple((f'{key}.{j}' for key in _PARTICLES for j in range(5)))
_EMPTY_KEYS['calculate_fuelbed_intermediates'] = ('sigma', 'bulk_density', 'packing_ratio', 'relative_packing_ratio', 'heat_sink', 'propagating_flux', 'w_heat_dead', 'w_heat_live', 'w_silica_dead', 'w_silica_live', 'w_moist_dead', 'w_moist_live', 'wfl_dead', 'wfl_live', 'moe_dead', 'moe_live', 'frac_dead', 'frac_live', 'depth', 'is_defined')
_EMPTY_KEYS.update({'assign_final_crown_fire_behavior': ('final_spread_rate', 'final_heat_per_unit_area', 'final_fireline_intensity', 'final_flame_length'), 'calculate_crown_fire_ratios': ('crown_fire_transition_ratio', 'crown_fire_active_ratio'), 'calculate_crown_heat_and_intensity': ('crown_fuel_load', 'canopy_heat_per_unit_area', 'crown_fire_heat_per_unit_area', 'crown_fire_line_intensity', 'crown_flame_length'), 'calculate_passive_crown_fire_behavior': ('passive_crown_fire_spread_rate', 'passive_crown_fire_heat_per_unit_area', 'passive_crown_fire_line_intensity', 'passive_crown_fire_flame_length'), 'classify_crown_fire_type': ('fire_type', 'is_surface_fire', 'is_passive_crown', 'is_active_crown'), 'run_crown_fire': ('crown_fire_spread_rate', 'crown_critical_fire_spread_rate', 'crown_flame_length', 'crown_fire_line_intensity', 'crown_critical_surface_fire_line_intensity', 'crown_fire_heat_per_unit_area', 'canopy_heat_per_unit_area', 'crown_fire_transition_ratio', 'crown_fire_active_ratio', 'crown_length_to_width_ratio', 'fire_type', 'surface_fire_critical_spread_rate', 'crowning_surface_fire_spread_rate', 'crown_fraction_burned', 'passive_crown_fire_spread_rate', 'passive_crown_fire_heat_per_unit_area', 'passive_crown_fire_line_intensity', 'passive_crown_fire_flame_length', 'final_spread_rate', 'final_heat_per_unit_area', 'final_fireline_intensity', 'final_flame_length')})
_EMPTY_KEYS['calculate_crown_surface_fire'] = tuple(('p_crown.' + key for key in _EMPTY_KEYS['build_particle_arrays'])) + tuple(('ib_crown.' + key for key in _EMPTY_KEYS['calculate_fuelbed_intermediates'])) + tuple(('crown_surface.' + key for key in _EMPTY_KEYS['run_surface_fire'])) + ('ri_crown', 'crown_fire_spread_rate', 'crown_length_to_width_ratio')


def _call(operation, arguments):
    """Own a public operation's inputs before its first detached native stage.

    :param operation: Fully qualified explicit component operation.
    :param arguments: Bound argument mapping in public signature order.
    :return: Fresh result computed exclusively from the shared input snapshot.
    """
    from .._ownership import _owned_inputs

    _validate_raw_arithmetic(operation, arguments)
    with _owned_inputs() as owner:
        result = _call_owned(operation, owner.snapshot(arguments))
        return owner.remember(result)


def _call_owned(operation, arguments):
    """Dispatch owned arguments through the configured model or primitive component boundary.

    :param operation: Fully qualified public operation.
    :param arguments: Independently owned original operands.
    :return: Float64 physical results and separately typed codes or flags.
    """
    if operation in ('model.run_surface', 'crown.run_crown_fire'):
        from ._model_api4 import execute_model
        return execute_model('surface' if operation == 'model.run_surface' else 'crown',
                             arguments, threads=_THREADS.get())
    from ._component_api4 import execute_component
    from ._contracts import normalize_arguments
    arguments = normalize_arguments(operation, arguments)
    if operation == 'mortality.build_mortality_lookup':
        return np.array(_native.defaults('mortality'), dtype=np.float64)
    if operation == 'crown.get_surface_fire_inputs':
        return {name: np.array(arguments['surface_results'][key], dtype=np.float64, copy=True)
                for name, key in [('surface_ros', 'spread_rate'), ('surface_fli', 'fireline_intensity'),
                                  ('surface_hpua', 'heat_per_unit_area')]}
    if operation == 'crown.coerce_crown_fire_inputs':
        from .behave_units import slope_to_base
        mapping = {'fm_grid': 'fuel_model_grid', 'wind_speed': 'wind_speed', 'wind_dir': 'wind_direction',
                   'aspect': 'aspect', 'cbh': 'canopy_base_height', 'ch': 'canopy_height',
                   'cbd': 'canopy_bulk_density', 'mf_pct': 'moisture_foliar'}
        result = {key: np.atleast_1d(arguments[field]).copy() for key, field in mapping.items()}
        result['slope_deg'] = slope_to_base(np.atleast_1d(arguments['slope']), arguments['slope_units'])
        result['S'] = result['fm_grid'].shape
        return result
    return execute_component(operation, arguments, threads=_THREADS.get())


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


def _real_coercion(operation, parameter):
    """Describe explicit reference coercion before numerical arithmetic.

    :param operation: Native public calculation route.
    :param parameter: Flattened public input name.
    :return: Supported real dtype for explicit coercion, or None for raw arithmetic.
    """
    if parameter.endswith(('_grid', '_i')):
        return np.int32
    module, name = operation.split('.', 1)
    if module in ('units', 'safety', 'spot', 'ignite', 'vapor_pressure_deficit_calculator', 'fine_dead_fuel_moisture_tool'):
        return np.float64
    if module == 'mortality':
        return None if parameter.startswith('coefficients.') else np.float64
    if module == 'crown':
        if name in ('coerce_crown_fire_inputs', 'run_crown_fire', 'calculate_crown_surface_fire') and (not parameter.startswith(('surface_results.', 'model.'))):
            return np.float64
        return None
    normalized = {'build_particle_arrays': ('m1h', 'm10h', 'm100h', 'mlh', 'mlw'), 'calculate_direction_of_max_spread': ('aspect',), 'calculate_fire_area': ('forward_ros', 'backing_ros', 'lwr'), 'calculate_fire_perimeter': ('forward_ros', 'backing_ros', 'lwr'), 'calculate_fire_width': ('forward_ros', 'backing_ros', 'lwr'), 'calculate_fireline_intensity': ('spread_rate', 'heat_per_unit_area'), 'calculate_forward_spread_rate': ('no_wind_no_slope_spread_rate', 'rate_vector'), 'calculate_heat_per_unit_area': ('reaction_intensity', 'residence_time'), 'calculate_midflame_wind_speed': ('waf', 'twenty_foot_wind_speed'), 'calculate_reaction_intensity_output': ('reaction_intensity',), 'calculate_slope_phi': ('slope',), 'calculate_surface_rate_vector': ('wind_direction', 'aspect'), 'calculate_surface_wind_adjustment_factor': ('canopy_cover', 'canopy_height', 'crown_ratio', 'depth', 'user_waf'), 'calculate_twenty_foot_wind_speed': ('wind_speed',), 'calculate_wind_adjustment_factor': ('canopy_cover', 'canopy_height', 'crown_ratio', 'depth'), 'run_surface_fire': ('wind_speed', 'wind_direction', 'slope', 'aspect', 'canopy_cover', 'canopy_height', 'crown_ratio', 'user_waf', 'ib.depth')}
    return np.float64 if parameter in normalized.get(name, ()) else None


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
    """Own physical state as float64 and keep unit/index controls integral.

    :param instance: Public native scalar-state wrapper.
    :param domain: Native scalar helper domain.
    :param method: Selected public helper method.
    :param arguments: Bound scalar arguments, including the receiver.
    :return: Scalar result or None; replaces physical state without dtype provenance.
    """
    from ._numeric import normalize_control, normalize_flag, normalize_physical, normalize_selector

    unit_fields = {'area_units': 'Area', 'contour_units': 'Length', 'distance_units': 'Length',
                   'map_distance_units': 'Length', 'elevation_units': 'Length',
                   'length_units': 'Length', 'slope_units': 'Slope', 'spread_rate_units': 'Speed',
                   'time_units': 'Time', 'wind_speed_units': 'Speed'}
    raw = {name: value for name, value in arguments.items() if name != 'self'}
    coerced = {'elapsed_time'} if domain == 'fire_size' else set()
    if domain == 'fire_size' and method == 'calculate_fire_basic_dimensions':
        coerced.add('forward_spread_rate')
        if raw['wind_speed_units'] != 5:
            coerced.add('effective_wind_speed')
    elif domain == 'slope_tool':
        coerced.update({'calculate_horizontal_distance': ('calculated_map_distance', 'max_slope_steepness'),
                        'calculate_slope_from_map_measurements': ('map_distance',)}.get(method, ()))
    _validate_raw_arithmetic(domain + '.' + method, {name: value for name, value in raw.items() if name not in coerced})
    _validate_raw_arithmetic(domain + '.' + method, dict(vars(instance)))
    # Each normalizer owns its data before state_call detaches. A generic deep
    # snapshot here would duplicate every immutable scale-table list first.
    original = raw
    arguments = {}
    for name, value in original.items():
        if name in unit_fields:
            column = normalize_selector(unit_fields[name], value)
        elif name == 'is_crown':
            column = normalize_flag(value)
        elif name == 'index':
            column = normalize_control(value, np.int64)
        else:
            if name in coerced:
                if np.asarray(value).dtype.kind not in 'biuf':
                    raise TypeError('Rust accepts real numeric inputs for explicit unit coercion.')
                value = (np.ma.asarray(value, dtype=np.float64) if np.ma.isMaskedArray(value)
                         else np.asarray(value, dtype=np.float64))
            column = normalize_physical(value)
        if column.size != 1:
            raise TypeError(f'{domain}.{method} requires scalar {name}')
        arguments[name] = column.item()
    state = {}
    for name, value in dict(vars(instance)).items():
        column = normalize_control(value, np.int32) if name == 'representative_fractions_' else normalize_physical(value)
        state[name] = column.tolist() if column.ndim else column.item()
    updated, result, error = _native.state_call(domain, method, state, arguments)
    vars(instance).update(updated)
    if error:
        raise RuntimeError(error)
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


def _validate_raw_arithmetic(operation, arguments, prefix=''):
    """Reject numeric subclasses before array coercion can erase their operators.

    :param operation: Fully qualified native calculation route.
    :param arguments: Original named inputs, including nested scientific fields.
    :param prefix: Dotted path of an enclosing input dictionary.
    :return: None; unsupported custom arithmetic raises TypeError before native calls.
    """
    controls = {'S', 'charge_type', 'index', 'is_crown', 'is_surface_fire',
                'is_passive_crown', 'is_active_crown',
                'wind_orientation_mode', 'wind_height_mode', 'waf_method'}
    for name, value in arguments.items():
        parameter = prefix + name
        if isinstance(value, dict):
            _validate_raw_arithmetic(operation, value, parameter + '.')
        elif not (name.endswith('units') or name in controls
                  or _real_coercion(operation, parameter) is not None):
            _validate_raw_scalar(value)


def _validate_raw_scalar(value):
    """Check raw scalar payloads before NumPy can discard their Python subclass.

    :param value: Scalar or sequence used in non-coercing native arithmetic.
    :return: None; custom numeric subclasses raise TypeError.
    """
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_raw_scalar(item)
    elif isinstance(value, np.generic):
        if type(value) is not np.dtype(value.dtype).type:
            raise TypeError('Unsupported opaque custom numeric arithmetic requires the explicit Python backend.')
    elif isinstance(value, (int, float)) and type(value) not in (bool, int, float):
        raise TypeError('Unsupported opaque custom numeric arithmetic requires the explicit Python backend.')


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
