"""Owned primitive component transport with structural record and shape handling."""
import numpy as np

from ._bridge import _native
from ._model_api4 import _buffer
from ._numeric import broadcast_shape

_CODE_FIELDS = {'units', 'wind_speed_units', 'slope_units', 'temp_units', 'rh_units', 'output_units',
                'max_steps', 'charge_type', 'fuel_model_grid', 'fuel_bed_type_grid'}
_FLAG_FIELDS = {'is_crown', 'is_surface_fire', 'is_passive_crown', 'is_active_crown'}
_RATIO_FIELDS = {'crown_fire_transition_ratio': ('surface_fli', 'critical_surface_fli'),
                 'crown_fire_active_ratio': ('crown_ros', 'critical_crown_ros')}
_FINAL_FIELDS = {'final_spread_rate': ('is_surface_fire', 'is_passive_crown', 'surface_ros', 'passive_ros', 'crown_ros'),
                 'final_heat_per_unit_area': ('is_surface_fire', 'is_passive_crown', 'surface_hpua', 'passive_hpua', 'crown_hpua'),
                 'final_fireline_intensity': ('is_surface_fire', 'is_passive_crown', 'surface_fli', 'passive_fli', 'crown_fli'),
                 'final_flame_length': ('is_surface_fire', 'is_passive_crown', 'surface_fli', 'passive_fl', 'crown_fl')}


def _flatten(operation, operands):
    """Expand actual record fields and prebind numerical lookup rows once.

    :param operation: Exact component operation.
    :param operands: Owned role-normalized input mapping.
    :return: Flat actual columns and scalar mode settings.
    """
    from ._bridge import _FUEL_KEYS, _LOOKUP_FIELDS, _PARTICLES, _REACTION_FIELDS, _SPREAD_FIELDS

    arguments = dict(operands)
    settings = {name: str(arguments.pop(name)) for name in ('wind_orientation_mode', 'wind_height_mode', 'waf_method') if name in arguments}
    if 'lut' in arguments:
        lookup = arguments.pop('lut')
        if operation == 'surface.build_particle_arrays':
            grid = np.atleast_1d(arguments.pop('fuel_model_grid'))
            index = np.clip(grid, 0, len(lookup['depth']) - 1)
            shape = grid.shape
        else:
            shape = tuple(arguments.pop('S')) if 'S' in arguments else np.atleast_1d(arguments['fuel_model_grid']).shape
            arguments.pop('fuel_model_grid', None)
            index = 10
            if 'slope_deg' in arguments:
                arguments['slope'] = arguments.pop('slope_deg')
                arguments['slope_units'] = np.array(0, dtype=np.int32)
        for name in _LOOKUP_FIELDS:
            arguments['model.' + _FUEL_KEYS.get(name, name)] = np.asarray(lookup[name])[index]
        for name in ('m1h', 'm10h', 'm100h', 'mlh', 'mlw'):
            arguments[name] = np.atleast_1d(arguments[name])
            if arguments[name].shape != shape:
                raise ValueError('moisture arrays must match the fuel grid shape')
    arguments.pop('S', None)
    arguments.pop('_native_partition', None)
    arguments.pop('_pipeline', None)
    if operation == 'contain.run_contain_sim_array':
        arguments['report_spread_rate'] = np.atleast_1d(arguments['report_spread_rate'])
    if 'coeffs' in arguments:
        coefficients = arguments.pop('coeffs')
        equations = np.clip(np.atleast_1d(arguments.pop('equation_number_grid')), 0, 120)
        for index in range(3):
            arguments[f'coefficients.{index}'] = coefficients[equations, index]
    if 'ib_crown' in arguments:
        arguments['ib'] = arguments.pop('ib_crown')
    flat = {}
    for name, value in arguments.items():
        if value is None:
            continue
        if not isinstance(value, dict):
            flat[name] = np.asarray(value)
            continue
        for field, column in value.items():
            if name == 'ib' and field not in (_REACTION_FIELDS if operation == 'surface.calculate_reaction_intensity' else _SPREAD_FIELDS):
                continue
            if name == 'p' and field in _PARTICLES:
                if column.ndim < 1 or column.shape[0] != 5:
                    raise ValueError('particle arrays require five leading rows')
                for index in range(5):
                    flat[f'{name}.{field}.{index}'] = np.asarray(column[index])
            else:
                flat[f'{name}.{field}'] = np.asarray(column)
    if 'north' not in settings.get('wind_orientation_mode', '').lower() and operation.rsplit('.', 1)[1] in (
            'calculate_direction_of_max_spread', 'calculate_surface_rate_vector', 'run_surface_fire', 'calculate_crown_surface_fire'):
        flat.pop('aspect', None)
    if settings.get('waf_method', '').replace('_', '').lower() == 'userinput' and 'user_waf' in flat:
        for field in ('canopy_cover', 'canopy_height', 'crown_ratio'):
            flat.pop(field, None)
        if operation == 'surface.calculate_surface_wind_adjustment_factor':
            flat.pop('depth', None)
    if operation == 'crown.calculate_crowning_surface_fire_spread_rate':
        flat.pop('canopy_height', None)
    if operation == 'crown.calculate_crown_surface_fire':
        flat.pop('canopy_height', None)
    return flat, settings


def _invoke(operation, arguments, settings, threads, outputs=None):
    """Capture a single dependency group with exact control storage.

    :param operation: Component operation.
    :param arguments: Actual flat operands for this group.
    :param settings: Scalar wind conventions.
    :param threads: Worker request.
    :param outputs: Optional selected output names.
    :return: Owned named arrays on this group's broadcast shape.
    """
    physical, codes, flags = {}, {}, {}
    for name, value in arguments.items():
        if name in _CODE_FIELDS or name.endswith('_i'):
            codes[name] = np.array(value, dtype=np.int64, copy=True, order='C')
        elif name in _FLAG_FIELDS or name.endswith(('.is_defined', '.is_dynamic')):
            flags[name] = np.array(value, dtype=np.bool_, copy=True, order='C')
        else:
            physical[name] = np.asarray(value, dtype=np.float64)
    shape = broadcast_shape(*(value.shape for value in (*physical.values(), *codes.values(), *flags.values())))
    return _native.component_batch(operation,
        {name: _buffer(value, shape) for name, value in physical.items()},
        {name: _buffer(value, shape) for name, value in codes.items()},
        {name: _buffer(value, shape) for name, value in flags.items()}, shape,
        threads=threads, settings=settings, outputs=outputs)


def _project(value, shape):
    """Restore axes contributed by this output's actual inputs.

    :param value: Array calculated on a shared valid broadcast grid.
    :param shape: Actual result shape.
    :return: Independently owned correctly shaped array.
    """
    padded = (1,) * (value.ndim - len(shape)) + tuple(shape)
    selected = value[tuple(slice(0, 1) if axis == 1 else slice(None) for axis in padded)]
    return np.array(selected.reshape(shape), copy=True)


def execute_component(operation, arguments, *, threads=1):
    """Execute normalized component data without dtype, mask, or object payloads.

    :param operation: Exact component identity.
    :param arguments: Independently owned role-normalized inputs.
    :param threads: Validated worker request.
    :return: Plain float64 physical arrays with separately typed codes and flags.
    """
    from ._bridge import _PARTICLES

    flat, settings = _flatten(operation, arguments)
    if operation == 'surface.calculate_fuelbed_intermediates':
        dependencies = _native.component_dependencies()
        shapes = {name: broadcast_shape(*(flat[key].shape for key in fields)) for name, fields in dependencies.items()}
        if len(set(shapes.values())) > 1:
            groups = {}
            for name, fields in dependencies.items():
                groups.setdefault(tuple(fields), []).append(name)
            result = {}
            for fields, outputs in groups.items():
                result.update(_invoke(operation, {name: flat[name] for name in fields}, settings, threads, outputs))
            return result
    if operation == 'crown.calculate_crown_surface_fire':
        from ._bridge import _EMPTY_KEYS

        fuel = {key: value for key, value in flat.items() if key.startswith('model.') or key in ('m1h', 'm10h', 'm100h', 'mlh', 'mlw')}
        fuel_shape = broadcast_shape(*(value.shape for value in fuel.values()))
        weather_shape = broadcast_shape(*(value.shape for value in flat.values()))
        if fuel_shape != weather_shape:
            fuel_outputs = [name for name in _EMPTY_KEYS['calculate_crown_surface_fire'] if name.startswith(('p_crown.', 'ib_crown.')) or name == 'ri_crown']
            weather_outputs = [name for name in _EMPTY_KEYS['calculate_crown_surface_fire'] if name not in fuel_outputs]
            result = _invoke(operation, fuel, settings, threads, fuel_outputs)
            result.update(_invoke(operation, flat, settings, threads, weather_outputs))
            for prefix in ('p_crown', 'ib_crown', 'crown_surface'):
                result[prefix] = {key[len(prefix) + 1:]: result.pop(key) for key in list(result) if key.startswith(prefix + '.')}
            for field in _PARTICLES:
                result['p_crown'][field] = np.stack([result['p_crown'].pop(f'{field}.{i}') for i in range(5)])
            return result
    dependencies = _RATIO_FIELDS if operation == 'crown.calculate_crown_fire_ratios' else (
        _FINAL_FIELDS if operation == 'crown.assign_final_crown_fire_behavior' else None)
    if dependencies is not None:
        result = {}
        for name, fields in dependencies.items():
            result.update(_invoke(operation, {key: flat[key] for key in fields}, settings, threads, [name]))
        return result
    name = operation.rsplit('.', 1)[1]
    independent = {
        'calculate_crown_scorch_mortality': {'crown_length_scorch': ('scorch_height_ft', 'tree_height_ft', 'crown_ratio'),
                                           'crown_volume_scorch': ('scorch_height_ft', 'tree_height_ft', 'crown_ratio')},
        'calculate_safety_zone': {'separation_distance': ('flame_height_ft',)},
        'calculate_vpd': {'saturated_vp': ('temperature', 'temp_units', 'output_units')},
        'calculate_crown_heat_and_intensity': {'crown_fuel_load': ('canopy_base_height', 'canopy_height', 'canopy_bulk_density'),
            'canopy_heat_per_unit_area': ('canopy_base_height', 'canopy_height', 'canopy_bulk_density'),
            'crown_fire_heat_per_unit_area': ('surface_hpua', 'canopy_base_height', 'canopy_height', 'canopy_bulk_density')},
        'calculate_passive_crown_fire_behavior': {'passive_crown_fire_spread_rate': ('surface_ros', 'crown_ros', 'cfb'),
            'passive_crown_fire_heat_per_unit_area': ('surface_hpua', 'canopy_hpua', 'cfb')},
    }
    overrides = independent.get(name, {})
    if overrides:
        from ._bridge import _EMPTY_KEYS

        fields = {field: tuple(overrides.get(field, tuple(flat))) for field in _EMPTY_KEYS[name]}
        shapes = {field: broadcast_shape(*(np.shape(flat[key]) for key in inputs)) for field, inputs in fields.items()}
        if len(set(shapes.values())) == 1:
            result = _invoke(operation, flat, settings, threads)
        else:
            groups = {}
            for field, inputs in fields.items():
                groups.setdefault(inputs, []).append(field)
            result = {}
            for inputs, outputs in groups.items():
                result.update(_invoke(operation, {key: flat[key] for key in inputs}, settings, threads, outputs))
    else:
        result = _invoke(operation, flat, settings, threads)
    if name == 'build_particle_arrays':
        for field in _PARTICLES:
            result[field] = np.stack([result.pop(f'{field}.{i}') for i in range(5)])
    if name == 'calculate_crown_surface_fire':
        for prefix in ('p_crown', 'ib_crown', 'crown_surface'):
            result[prefix] = {key[len(prefix) + 1:]: result.pop(key) for key in list(result) if key.startswith(prefix + '.')}
        for field in _PARTICLES:
            result['p_crown'][field] = np.stack([result['p_crown'].pop(f'{field}.{i}') for i in range(5)])
    return result[''] if '' in result else result
