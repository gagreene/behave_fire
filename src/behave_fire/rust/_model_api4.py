"""Primitive float64 transport for an explicitly planned common model grid.

The public dispatcher switches atomically only after dependency groups, unit
descriptors, and the complete per-operation contract manifest are ready.
"""
import numpy as np

from ._bridge import _native
from ._numeric import broadcast_shape, normalize_control, normalize_identifier, normalize_physical, normalize_selector, plan_shapes

_SHARED = ('m1h', 'm10h', 'm100h', 'mlh', 'mlw', 'wind_speed', 'wind_direction', 'slope', 'aspect')
_FIELDS = {
    'surface': _SHARED + ('canopy_cover', 'canopy_height', 'crown_ratio'),
    'crown': _SHARED + ('canopy_base_height', 'canopy_height', 'canopy_bulk_density',
                        'moisture_foliar', 'surface_ros', 'surface_fli', 'surface_hpua'),
}
_MOISTURE = ('m1h', 'm10h', 'm100h', 'mlh', 'mlw')
_FUEL_OUTPUTS = ('reaction_intensity', 'residence_time', 'heat_per_unit_area', 'no_wind_no_slope_spread_rate')
_SURFACE_OUTPUTS = _FUEL_OUTPUTS + ('midflame_wind_speed', 'spread_rate', 'backing_spread_rate',
    'flanking_spread_rate', 'flame_length', 'fireline_intensity', 'effective_wind_speed',
    'fire_length_to_width_ratio', 'eccentricity', 'direction_of_max_spread')


def _buffer(value, shape):
    """Keep scalar storage compact and materialize only genuine broadcast columns.

    :param value: Owned normalized primitive array.
    :param shape: Validated common logical grid shape.
    :return: Contiguous flat storage, with scalar values retained as one element.
    """
    if value.size == 1:
        return value.reshape(1)
    if value.shape == shape:
        return np.ascontiguousarray(value).reshape(-1)
    return np.ascontiguousarray(np.broadcast_to(value, shape)).reshape(-1)


def _dependencies(kind, settings, derive_ratio, has_user_waf):
    """Describe actual mathematical and structural output dependencies.

    :param kind: Model family.
    :param settings: Validated scalar wind configuration.
    :param derive_ratio: Whether the ratio depends on actual canopy heights.
    :param has_user_waf: Whether a fixed adjustment factor is bound.
    :return: Canonical field names for every output dependency closure.
    """
    if kind == 'crown':
        from ._bridge import _crown_dependencies

        aliases = {'surface_results.spread_rate': 'surface_ros',
                   'surface_results.fireline_intensity': 'surface_fli',
                   'surface_results.heat_per_unit_area': 'surface_hpua'}
        original = _crown_dependencies('north' if settings['wind_orientation'] else 'upslope')
        return {output: tuple(dict.fromkeys(aliases.get(name, name) for name in names))
                for output, names in original.items()}
    fuel = ('fuel_model_grid',)
    canopy = ('user_waf',) if settings['waf_method'] and has_user_waf else (
        ('canopy_cover', 'canopy_height', 'canopy_base_height') if derive_ratio
        else ('canopy_cover', 'canopy_height', 'crown_ratio'))
    wind = fuel + ('wind_speed',) + canopy
    spread = wind + ('wind_direction', 'slope') + (('aspect',) if settings['wind_orientation'] else ())
    # The public Surface pipeline uses a spread-shaped fill operand in every
    # final np.where, including its otherwise independent fuel-stage values.
    return {name: spread for name in _SURFACE_OUTPUTS}


def _prepare(kind, arguments):
    """Normalize original operands once, preserving their independent logical shapes.

    :param kind: Surface or Crown model kind.
    :param arguments: Actual public model operands and configuration.
    :return: Physical columns, typed controls, conversion descriptors, settings, and derivation flag.
    """
    if kind not in _FIELDS:
        raise ValueError('unknown common model kind')
    arguments = dict(arguments)
    if kind == 'crown' and 'surface_results' in arguments:
        for name, key in [('surface_ros', 'spread_rate'), ('surface_fli', 'fireline_intensity'),
                          ('surface_hpua', 'heat_per_unit_area')]:
            arguments[name] = arguments['surface_results'][key]
    pipeline = arguments.get('_pipeline') or {}
    derive_ratio = kind == 'surface' and bool(pipeline.get('derive'))
    inputs = {name: np.atleast_1d(normalize_physical(arguments[name])) for name in _FIELDS[kind]
              if not (derive_ratio and name == 'crown_ratio')}
    if derive_ratio:
        inputs['canopy_base_height'] = np.atleast_1d(normalize_physical(arguments['canopy_base_height']))
    if kind == 'surface' and arguments.get('user_waf') is not None:
        inputs['user_waf'] = np.atleast_1d(normalize_physical(arguments['user_waf']))
    controls = {'fuel_model_grid': np.atleast_1d(normalize_identifier(arguments['fuel_model_grid']))}
    if arguments.get('validity') is not None:
        controls['validity'] = np.atleast_1d(normalize_control(arguments['validity'], np.bool_))
    descriptors = {}
    for field, (family, codes) in pipeline.get('conversions', {}).items():
        if field in inputs:
            descriptors[field] = (family, normalize_control(codes, np.int32))
    for field, family, selector in [('wind_speed', 'Speed', 'wind_speed_units'), ('slope', 'Slope', 'slope_units')]:
        if selector in arguments and field not in descriptors:
            descriptors[field] = (family, normalize_selector(family, arguments[selector]))
    for field, (_, codes) in descriptors.items():
        inputs[field] = np.broadcast_to(inputs[field], broadcast_shape(inputs[field].shape, codes.shape))
    # Fuel particle assignment follows the ID grid, not the weather union grid.
    for name in _MOISTURE:
        if inputs[name].shape != controls['fuel_model_grid'].shape:
            raise ValueError('moisture arrays must match the fuel grid shape')
    return inputs, controls, descriptors, _settings(arguments), derive_ratio


def _project(value, shape):
    """Select only axes added by the common computation grid.

    :param value: Owned native result on the union grid.
    :param shape: Actual independent result shape.
    :return: Correctly shaped view without copying physical results.
    """
    padded = (1,) * (value.ndim - len(shape)) + tuple(shape)
    selected = value[tuple(slice(None) if size != 1 else slice(0, 1) for size in padded)]
    return selected.reshape(shape)


def _registry(lookup):
    """Own physical fuel values and exact boolean flags without tagged payloads.

    :param lookup: Existing numerical fuel lookup arrays.
    :return: Native-endian float64 properties and typed boolean flag columns.
    """
    return {name: normalize_control(value, np.bool_) if name in ('is_dynamic', 'is_defined')
            else normalize_physical(value) for name, value in lookup.items()}


def _settings(arguments):
    """Resolve existing scalar wind conventions without changing their fallbacks.

    :param arguments: Original model configuration values.
    :return: Exact small integer configuration codes for the common model core.
    """
    orientation = str(arguments.get('wind_orientation_mode', 'RelativeToUpslope')).lower()
    height = str(arguments.get('wind_height_mode', 'TwentyFoot')).replace('_', '').lower()
    waf = str(arguments.get('waf_method', 'CrownRatio')).replace('_', '').lower()
    return dict(wind_orientation=int('north' in orientation),
                wind_height=int('10' in height or 'ten' in height),
                waf_method=int(waf == 'userinput'))


def execute_common(kind, arguments, *, threads=1, owned_callback=None):
    """Execute an already chosen common-grid model on primitive owned columns.

    :param kind: Surface or Crown model kind.
    :param arguments: Actual base-unit physical values, typed IDs, registry, and scalar configuration.
    :param threads: Validated native worker request.
    :param owned_callback: Optional ownership-test callback after native snapshots are captured.
    :return: Complete float64 physical arrays and int32 classifications in one native compute call.
    """
    inputs, controls, descriptors, settings, derive_ratio = _prepare(kind, arguments)
    shapes = [value.shape for value in (*inputs.values(), *controls.values())]
    shapes.extend(codes.shape for _, codes in descriptors.values())
    shape = broadcast_shape(*shapes)
    inputs = {name: _buffer(value, shape) for name, value in inputs.items()}
    controls = {name: _buffer(value, shape) for name, value in controls.items()}
    descriptors = {name: (family, _buffer(codes, shape)) for name, (family, codes) in descriptors.items()}
    return _native.model_batch(kind, inputs, controls, _registry(arguments['lut']), shape,
                               settings, threads=threads, conversions=descriptors,
                               derive_ratio=derive_ratio, _owned_callback=owned_callback)


def execute_model(kind, arguments, *, threads=1, owned_callback=None):
    """Execute common or independent model grids through one owned native boundary.

    :param kind: Surface or Crown model kind.
    :param arguments: Actual model operands, registry, and scalar configuration.
    :param threads: Native worker count.
    :param owned_callback: Optional callback after every native operand is owned.
    :return: Independent float64 output arrays and int32 classifications.
    """
    inputs, controls, descriptors, settings, derive = _prepare(kind, arguments)
    dependencies = _dependencies(kind, settings, derive, 'user_waf' in inputs)
    if 'validity' in controls:
        dependencies = {name: fields + ('validity',) for name, fields in dependencies.items()}
    shaped = dict(inputs, **controls)
    groups = plan_shapes(shaped, dependencies)
    shapes = {name: group.shape for group in groups for name in group.outputs}
    union = broadcast_shape(*(group.shape for group in groups))
    registry = _registry(arguments['lut'])
    if all(union):
        active = set().union(*dependencies.values())
        if kind == 'crown' or 'fuel_model_grid' in active:
            active.update(_MOISTURE)
        batch = _native.model_batch(kind,
            {name: _buffer(value, union) for name, value in inputs.items() if name in active},
            {name: _buffer(value, union) for name, value in controls.items()}, registry, union, settings,
            threads=threads, conversions={name: (family, _buffer(codes, union))
                for name, (family, codes) in descriptors.items() if name in active},
            derive_ratio=derive and 'canopy_base_height' in active, _owned_callback=owned_callback)
        return {name: _project(value, shapes[name]) for name, value in batch.items() if not name.startswith('__')}
    planned = []
    empty = {}
    for group in groups:
        if not all(group.shape):
            empty.update({name: np.empty(group.shape, dtype=np.int32 if name == 'fire_type' else np.float64)
                          for name in group.outputs})
            continue
        active = set(group.inputs)
        if 'fuel_model_grid' in active and (kind == 'crown' or any(name in _FUEL_OUTPUTS for name in group.outputs)):
            active.update(_MOISTURE)
        planned.append(dict(outputs=group.outputs, shape=group.shape,
            inputs={name: _buffer(value, group.shape) for name, value in inputs.items() if name in active},
            controls={name: _buffer(value, group.shape) for name, value in controls.items()
                      if name in active or name == 'validity'},
            conversions={name: (family, _buffer(codes, group.shape))
                         for name, (family, codes) in descriptors.items() if name in active},
            derive_ratio=derive and 'canopy_base_height' in active))
    batch = _native.model_groups(kind, planned, registry, settings, threads=threads,
                                 _owned_callback=owned_callback)
    return dict(batch, **empty)
