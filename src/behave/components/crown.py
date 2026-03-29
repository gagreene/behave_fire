"""
crown_array.py — Vectorized Crown Fire Pipeline (V6)

Crown fire uses FM10 as the crown fuel model, WAF=0.4, flat terrain.
The crown L:W ratio uses the Rothermel 1991 (eq. 10) formula — different
from the surface formula (§13.2 Gap A fix).

Public API
----------
calculate_crown_fire(surface_results, lut, fuel_model_grid,
                     m1h, m10h, m100h, mlh, mlw,
                     wind_speed, wind_speed_units,
                     wind_direction, wind_orientation_mode,
                     slope_deg, aspect,
                     canopy_base_height, canopy_height,
                     canopy_bulk_density, moisture_foliar)
    → dict of (*S) arrays
"""

import numpy as np


def calculate_crown_fire(surface_results, lut,
                          fuel_model_grid,
                          m1h, m10h, m100h, mlh, mlw,
                          wind_speed, wind_speed_units,
                          wind_direction, wind_orientation_mode,
                          slope_deg, aspect,
                          canopy_base_height, canopy_height,
                          canopy_bulk_density, moisture_foliar):
    """
    Vectorized crown fire calculations.

    Parameters
    ----------
    surface_results    : dict from calculate_spread_rate() — the surface run
    lut                : dict from build_fuel_lookup_arrays()
    fuel_model_grid    : int (*S)
    m1h .. mlw         : moisture fraction arrays (*S)
    wind_speed         : (*S) or scalar
    wind_speed_units   : scalar int
    wind_direction     : (*S) or scalar, degrees
    wind_orientation_mode : str
    slope_deg          : (*S) or scalar, degrees
    aspect             : (*S) or scalar, degrees
    canopy_base_height : (*S) — feet
    canopy_height      : (*S) — feet
    canopy_bulk_density: (*S) — lb/ft³
    moisture_foliar    : (*S) or scalar — percent

    Returns
    -------
    dict with keys:
        crown_fire_spread_rate, crown_flame_length,
        crown_fire_line_intensity, crown_fire_heat_per_unit_area,
        canopy_heat_per_unit_area,
        crown_fire_transition_ratio, crown_fire_active_ratio,
        fire_type  (0=Surface, 1=Torching, 2=Crowning),
        crown_critical_surface_fire_line_intensity,
        crown_critical_fire_spread_rate,
        crown_length_to_width_ratio
    """
    try:
        from .surface import (
            build_particle_arrays,
            calculate_fuelbed_intermediates,
            calculate_reaction_intensity,
            calculate_spread_rate,
        )
        from .behave_units import speed_from_base
    except ImportError:
        from surface import (
            build_particle_arrays,
            calculate_fuelbed_intermediates,
            calculate_reaction_intensity,
            calculate_spread_rate,
        )
        from behave_units import speed_from_base

    # --- Coerce spatial inputs ---
    fm_grid   = np.atleast_1d(np.asarray(fuel_model_grid,    dtype=np.int32))
    slope_deg = np.atleast_1d(np.asarray(slope_deg,          dtype=float))
    aspect    = np.atleast_1d(np.asarray(aspect,             dtype=float))
    cbh       = np.atleast_1d(np.asarray(canopy_base_height, dtype=float))
    ch        = np.atleast_1d(np.asarray(canopy_height,      dtype=float))
    cbd       = np.atleast_1d(np.asarray(canopy_bulk_density, dtype=float))
    mf_pct    = np.atleast_1d(np.asarray(moisture_foliar,    dtype=float))
    wind_speed = np.atleast_1d(np.asarray(wind_speed,        dtype=float))
    wind_dir   = np.atleast_1d(np.asarray(wind_direction,    dtype=float))
    S = fm_grid.shape

    # --- Crown fuel: FM10 everywhere, flat terrain, WAF=0.4 ---
    fm10 = np.full(S, 10, dtype=np.int32)
    p_crown  = build_particle_arrays(lut, fm10, m1h, m10h, m100h, mlh, mlw)
    ib_crown = calculate_fuelbed_intermediates(p_crown)
    ri_crown = calculate_reaction_intensity(ib_crown)

    crown_surface = calculate_spread_rate(
        ri_crown, ib_crown,
        wind_speed, wind_speed_units,
        np.zeros(S), 'RelativeToUpslope',   # no direction offset
        np.zeros(S), np.zeros(S),            # flat terrain
        canopy_cover=np.zeros(S),
        canopy_height=ch,
        crown_ratio=np.zeros(S),
        waf_method='UserInput',
        user_waf=np.full(S, 0.4),
    )

    # Crown ROS = 3.34 × surface-equivalent spread rate under FM10
    crown_ros = 3.34 * crown_surface['spread_rate']

    # --- Crown L:W ratio (Rothermel 1991 eq. 10 — §13.2 Gap A fix) ---
    # Uses EWS from the crown run (mph), not the surface formula
    crown_ews_mph = crown_surface['effective_wind_speed']
    crown_lwr = np.where(
        crown_ews_mph > 1e-7,
        1.0 + 0.125 * crown_ews_mph,
        1.0
    )

    # --- Crown fuel load and heat ---
    crown_fuel_load = np.maximum(cbd * (ch - cbh), 0.0)
    LOW_HOC = 8000.0
    canopy_hpua = crown_fuel_load * LOW_HOC
    crown_hpua  = surface_results['heat_per_unit_area'] + canopy_hpua

    # --- Crown fireline intensity and flame length ---
    crown_fli = (crown_ros / 60.0) * crown_hpua
    safe_cfl  = np.where(crown_fli > 0, crown_fli, 0.0)
    crown_fl  = np.where(crown_fli > 0, 0.2 * (safe_cfl ** (2.0 / 3.0)), 0.0)

    # --- Critical surface FLI (Van Wagner) ---
    mf_pct_safe = np.maximum(mf_pct, 30.0)
    cbh_m = cbh * 0.3048
    cbh_m = np.maximum(cbh_m, 0.1)
    crit_fli_kw = (0.010 * cbh_m * (460.0 + 25.9 * mf_pct_safe)) ** 1.5
    crit_fli = crit_fli_kw * 0.2886719   # kW/m → BTU/ft/s

    # --- Critical crown spread rate ---
    cbd_kg_m3 = cbd * 16.0185            # lb/ft³ → kg/m³
    safe_cbd  = np.where(cbd_kg_m3 > 1e-7, cbd_kg_m3, 1.0)
    crit_crown_ros = np.where(
        cbd_kg_m3 > 1e-7,
        (3.0 / safe_cbd) * 3.28084,      # m/min → ft/min
        0.0
    )

    # --- Transition and active ratios ---
    safe_crit_fli = np.where(crit_fli > 1e-7, crit_fli, 1.0)
    transition_ratio = np.where(
        crit_fli > 1e-7,
        surface_results['fireline_intensity'] / safe_crit_fli,
        0.0
    )
    safe_crit_crown = np.where(crit_crown_ros > 1e-7, crit_crown_ros, 1.0)
    active_ratio = np.where(
        crit_crown_ros > 1e-7,
        crown_ros / safe_crit_crown,
        0.0
    )

    # --- Fire type (0=Surface, 1=Torching, 2=Crowning) ---
    fire_type = np.where(
        transition_ratio < 1.0, 0,
        np.where(active_ratio < 1.0, 1, 2)
    ).astype(np.int32)

    return {
        'crown_fire_spread_rate':                       crown_ros,
        'crown_flame_length':                           crown_fl,
        'crown_fire_line_intensity':                    crown_fli,
        'crown_fire_heat_per_unit_area':                crown_hpua,
        'canopy_heat_per_unit_area':                    canopy_hpua,
        'crown_fire_transition_ratio':                  transition_ratio,
        'crown_fire_active_ratio':                      active_ratio,
        'fire_type':                                    fire_type,
        'crown_critical_surface_fire_line_intensity':   crit_fli,
        'crown_critical_fire_spread_rate':              crit_crown_ros,
        'crown_length_to_width_ratio':                  crown_lwr,
    }


