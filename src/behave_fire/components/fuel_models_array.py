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

import numpy as np

try:
    from .fuel_models import FuelModels
except ImportError:
    from fuel_models import FuelModels

# Maximum fuel model number supported in the lookup table.
_MAX_FM = 260


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

