"""
fuel_models_array.py — Vectorized Fuel Property Lookup (V2)

Pre-builds NumPy arrays of fuel properties indexed by fuel model number.
Built once at startup; cells are looked up via NumPy fancy indexing.

Usage
-----
    from .fuel_models import FuelModels
    from .fuel_models_array import build_fuel_lookup_arrays

    fm  = FuelModels()
    lut = build_fuel_lookup_arrays(fm)

    # For a grid of fuel model numbers:
    depth_grid = lut['depth'][fuel_model_grid]   # shape matches fuel_model_grid
"""

from decimal import InvalidOperation

import numpy as np

try:
    from .fuel_models import _REGISTRY_VALUE_FIELDS, FuelModels, _reject_complex_registry_value
except ImportError:
    from fuel_models import _REGISTRY_VALUE_FIELDS, FuelModels, _reject_complex_registry_value

# Maximum fuel model number supported in the lookup table.
_MAX_FM = 256


def _fuel_model_indices(value, rows=_MAX_FM + 1):
    """Map fuel identifiers outside the supported table to undefined row zero.

    :param value: Fuel identifier scalar or array, including nonfinite real values.
    :param rows: Actual lookup row count, including the undefined row.
    :return: Int32 indices without narrowing invalid real identifiers first.
    """
    data = np.asarray(value)
    if data.dtype == object:
        return _object_fuel_indices(data, rows)
    if data.dtype.kind not in 'biuf':
        data = np.asarray(value, dtype=np.int32)
    valid = (data >= 1) & (data <= min(_MAX_FM, rows - 1))
    return np.asarray(np.where(valid, data, 0), dtype=np.int32)


def _object_fuel_indices(data, rows):
    """Map object-array identifiers by comparing each original value, never a rounded copy.

    A float64 conversion would round ``Decimal("256.000000000000000001")`` onto model 256 and overflow for
    ``10**400``. Exact comparisons keep every value outside the table on the undefined row.

    :param data: Object array of fuel identifiers.
    :param rows: Actual lookup row count, including the undefined row.
    :return: Int32 indices; complex elements raise ``TypeError`` and text that is not a number raises
        ``ValueError`` through the historical integer conversion.
    """
    limit = min(_MAX_FM, rows - 1)
    flat = data.reshape(-1)
    indices = np.zeros(flat.shape, dtype=np.int32)
    for position, item in enumerate(flat):
        if isinstance(item, (complex, np.complexfloating)):
            raise TypeError('Fuel identifiers must be real numbers')
        try:
            number = item
            inside = 1 <= number <= limit
        except InvalidOperation:
            continue
        except TypeError:
            number = int(np.asarray(item, dtype=np.int32))
            inside = 1 <= number <= limit
        if inside:
            indices[position] = int(number)
    return indices.reshape(data.shape)


def build_fuel_lookup_arrays(fuel_models: FuelModels) -> dict:
    """
    Pre-build NumPy arrays of fuel properties indexed by fuel model number.

    Called once at startup.  Each array has shape (``_MAX_FM`` + 1,); index 0
    is unused (no fuel model 0).  Unrecognised or reserved models get
    ``0.0`` / ``False``.

    :param fuel_models: Populated ``FuelModels`` instance.
    :return: dict with keys:

        - ``'depth'``      — fuel bed depth (ft)
        - ``'moe_dead'``   — moisture of extinction for dead fuel (fraction)
        - ``'hoc_dead'``   — heat of combustion for dead fuel (BTU/lb)
        - ``'hoc_live'``   — heat of combustion for live fuel (BTU/lb)
        - ``'dead_1h'``    — 1-hr dead fuel load (lb/ft²)
        - ``'dead_10h'``   — 10-hr dead fuel load (lb/ft²)
        - ``'dead_100h'``  — 100-hr dead fuel load (lb/ft²)
        - ``'live_herb'``  — live herbaceous fuel load (lb/ft²)
        - ``'live_woody'`` — live woody fuel load (lb/ft²)
        - ``'savr_1h'``    — 1-hr dead SAVR (ft²/ft³)
        - ``'savr_lh'``    — live herbaceous SAVR (ft²/ft³)
        - ``'savr_lw'``    — live woody SAVR (ft²/ft³)
        - ``'is_dynamic'`` — bool, True for dynamic (herb-transfer) models
        - ``'is_defined'`` — bool, True for valid populated models

        Each array has shape (``_MAX_FM`` + 1,).
    """
    depth      = np.zeros(_MAX_FM + 1, dtype=float)
    moe_dead   = np.zeros(_MAX_FM + 1, dtype=float)
    hoc_dead   = np.zeros(_MAX_FM + 1, dtype=float)
    hoc_live   = np.zeros(_MAX_FM + 1, dtype=float)
    dead_1h    = np.zeros(_MAX_FM + 1, dtype=float)
    dead_10h   = np.zeros(_MAX_FM + 1, dtype=float)
    dead_100h  = np.zeros(_MAX_FM + 1, dtype=float)
    live_herb  = np.zeros(_MAX_FM + 1, dtype=float)
    live_woody = np.zeros(_MAX_FM + 1, dtype=float)
    savr_1h    = np.zeros(_MAX_FM + 1, dtype=float)
    savr_lh    = np.zeros(_MAX_FM + 1, dtype=float)
    savr_lw    = np.zeros(_MAX_FM + 1, dtype=float)
    is_dynamic = np.zeros(_MAX_FM + 1, dtype=bool)
    is_defined = np.zeros(_MAX_FM + 1, dtype=bool)

    fm_dict = fuel_models.fuel_models_
    for number, record in fm_dict.items():
        for field in _REGISTRY_VALUE_FIELDS:
            _reject_complex_registry_value(number, field, record.get(field))

    for n in range(1, _MAX_FM + 1):
        m = fm_dict.get(n)
        if m is None:
            continue
        if not m.get('is_defined', False):
            continue
        # Copy fuel model properties into the lookup arrays
        depth[n]      = m['fuel_bed_depth']
        moe_dead[n]   = m['moisture_of_extinction_dead']
        hoc_dead[n]   = m['heat_of_combustion_dead']
        hoc_live[n]   = m['heat_of_combustion_live']
        dead_1h[n]    = m['dead_1h']
        dead_10h[n]   = m['dead_10h']
        dead_100h[n]  = m['dead_100h']
        live_herb[n]  = m['live_herb']
        live_woody[n] = m['live_woody']
        savr_1h[n]    = m['savr_1h']
        savr_lh[n]    = m['savr_live_herb']
        savr_lw[n]    = m['savr_live_woody']
        is_dynamic[n] = bool(m['is_dynamic'])
        is_defined[n] = True

    return {
        'depth':      depth,
        'moe_dead':   moe_dead,
        'hoc_dead':   hoc_dead,
        'hoc_live':   hoc_live,
        'dead_1h':    dead_1h,
        'dead_10h':   dead_10h,
        'dead_100h':  dead_100h,
        'live_herb':  live_herb,
        'live_woody': live_woody,
        'savr_1h':    savr_1h,
        'savr_lh':    savr_lh,
        'savr_lw':    savr_lw,
        'is_dynamic': is_dynamic,
        'is_defined': is_defined,
    }

