"""Uniform float64 physical contract and first-boundary typed input roles."""
import numpy as np

from ._numeric import _validate_value, normalize_control, normalize_flag, normalize_identifier, normalize_physical, normalize_selector

def _physical_record(value):
    """Own nested scientific records with separate exact definition flags.

    :param value: Physical scalar/array or nested fuel/particle/fuelbed record.
    :return: Structurally equivalent owned float64 record with boolean flags.
    """
    if isinstance(value, dict):
        return {name: normalize_control(item, np.bool_) if name in ('is_defined', 'is_dynamic')
                else _physical_record(item) for name, item in value.items()}
    return normalize_physical(value)


def normalize_arguments(operation, arguments):
    """Normalize component operands before primitive structural marshalling.

    :param operation: Exact native operation name.
    :param arguments: Owned original argument mapping.
    :return: Float64 physical values and exact controls; no mask or dtype sidecars.
    """
    if operation == 'mortality.build_mortality_lookup':
        return arguments
    from ._bridge import _real_coercion

    result = {}
    selectors = {'wind_speed_units': 'Speed', 'slope_units': 'Slope',
                 'temp_units': 'Temperature', 'rh_units': 'Fraction', 'output_units': 'Pressure'}
    for name, value in arguments.items():
        if name in selectors:
            result[name] = normalize_selector(selectors[name], value)
        elif name == 'units':
            result[name] = normalize_selector(operation.split('.')[1], value)
        elif name.endswith('_grid'):
            result[name] = normalize_identifier(value)
        elif name.endswith('_i'):
            result[name] = normalize_identifier(value)
        elif name == 'charge_type':
            _validate_value(value)
            if np.ndim(value):
                raise TypeError('charge_type must be a scalar control')
            if np.ma.getmaskarray(value).any() or np.isnan(value):
                raise ValueError('charge_type cannot be missing')
            # The public enum explicitly treats every other value as unknown.
            result[name] = 0 if value == 0 else 1 if value == 1 else 2
        elif name == 'max_steps':
            data = normalize_control(value, np.int64)
            if data.ndim:
                raise TypeError(f'{name} must be a scalar control')
            result[name] = data.item()
        elif name == 'S':
            result[name] = value
        elif name in ('is_surface_fire', 'is_passive_crown', 'is_active_crown'):
            result[name] = normalize_flag(value)
        elif name in ('wind_orientation_mode', 'wind_height_mode', 'waf_method'):
            result[name] = str(value)
        elif name == 'is_crown':
            result[name] = normalize_flag(value).item()
        elif value is None or name.startswith('_') or isinstance(value, str):
            result[name] = value
        else:
            if operation == 'crown.coerce_crown_fire_inputs':
                _validate_value(value)
            coercion = _real_coercion(operation, name)
            if coercion is not None and not isinstance(value, dict):
                if np.asarray(value).dtype.kind not in 'biuf':
                    raise TypeError('Rust accepts real numeric inputs for explicit real coercion.')
                value = (np.ma.asarray(value, dtype=coercion) if np.ma.isMaskedArray(value)
                         else np.asarray(value, dtype=coercion))
            result[name] = _physical_record(value)
    return result


def numeric_contract(operation):
    """Return the uniform physical numeric contract of an operation.

    :param operation: Exact native operation identity.
    :return: f64 for every supported operation.
    """
    return 'f64'
