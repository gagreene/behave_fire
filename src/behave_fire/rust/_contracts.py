"""Operation-level numeric migration status and first-boundary input roles."""
import numpy as np

from ._numeric import _validate_value, normalize_control, normalize_identifier, normalize_physical, normalize_selector

_STATELESS = frozenset(('contain', 'ignite', 'mortality', 'safety', 'spot', 'units',
                       'vapor_pressure_deficit_calculator'))


def normalize_arguments(operation, arguments):
    """Normalize migrated stateless operands before legacy structural marshalling.

    :param operation: Exact native operation name.
    :param arguments: Owned original argument mapping.
    :return: Float64 physical values and exact controls; no mask or dtype sidecars.
    """
    if numeric_contract(operation) != 'f64-stateless' or operation == 'mortality.build_mortality_lookup':
        return arguments
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
        elif name.startswith('_') or isinstance(value, str):
            result[name] = value
        else:
            result[name] = normalize_physical(value)
    return result


def numeric_contract(operation):
    """Return the precise current migration status of an operation.

    :param operation: Exact native operation identity.
    :return: Model, stateless, or explicitly unmigrated numeric contract name.
    """
    if operation in ('model.run_surface', 'crown.run_crown_fire'):
        return 'f64-model'
    return 'f64-stateless' if operation.partition('.')[0] in _STATELESS else 'unmigrated'
