"""
crown.py — Vectorized Crown Fire Pipeline (V6)

Crown fire uses FM10 as the crown fuel model and a fixed WAF=0.4.
Wind direction, orientation mode, slope, and aspect are passed through to
``calculate_spread_rate()`` so that ``direction_of_max_spread`` is computed
correctly for the crown run.  The crown L:W ratio uses the Rothermel 1991
(eq. 10) formula — different from the surface formula (§13.2 Gap A fix).

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
from typing import Union


def calculate_crown_fire(
        surface_results: dict,
        lut: dict,
        fuel_model_grid: Union[int, np.ndarray],
        m1h: Union[float, np.ndarray],
        m10h: Union[float, np.ndarray],
        m100h: Union[float, np.ndarray],
        mlh: Union[float, np.ndarray],
        mlw: Union[float, np.ndarray],
        wind_speed: Union[float, np.ndarray],
        wind_speed_units: int,
        wind_direction: Union[float, np.ndarray],
        wind_orientation_mode: str,
        slope: Union[float, np.ndarray],
        slope_units: int = 0,
        aspect: Union[float, np.ndarray] = 0.0,
        canopy_base_height: Union[float, np.ndarray] = 0.0,
        canopy_height: Union[float, np.ndarray] = 0.0,
        canopy_bulk_density: Union[float, np.ndarray] = 0.0,
        moisture_foliar: Union[float, np.ndarray] = 100.0
) -> dict:
    """
    Vectorized crown fire calculations (Van Wagner 1977 / Rothermel 1991).

    Crown spread uses FM10 everywhere with WAF=0.4 and flat terrain.
    Critical surface FLI uses Van Wagner's canopy ignition threshold.

    :param surface_results: Output dict from ``calculate_spread_rate()`` for the
        surface fire run (must include ``'fireline_intensity'``
        and ``'heat_per_unit_area'``).
    :param lut: Fuel property lookup dict from ``build_fuel_lookup_arrays()``.
    :param fuel_model_grid: Integer fuel model number array (*S) or scalar.
        Used only to derive the spatial shape ``S``; crown fuel properties
        always come from FM10 regardless of this value.
    :param m1h: 1-hr dead fuel moisture as fraction (*S) or scalar.
    :param m10h: 10-hr dead fuel moisture as fraction (*S) or scalar.
    :param m100h: 100-hr dead fuel moisture as fraction (*S) or scalar.
    :param mlh: Live herbaceous fuel moisture as fraction (*S) or scalar.
    :param mlw: Live woody fuel moisture as fraction (*S) or scalar.
    :param wind_speed: Wind speed (*S) or scalar, in ``wind_speed_units``.
    :param wind_speed_units: Scalar integer ``SpeedUnitsEnum`` value.
    :param wind_direction: Wind direction in degrees (*S) or scalar.
        Passed through to ``calculate_spread_rate()`` to compute
        ``direction_of_max_spread`` for the crown run.
    :param wind_orientation_mode: ``'RelativeToUpslope'`` or ``'RelativeToNorth'``.
        Passed through to ``calculate_spread_rate()`` alongside
        ``wind_direction``.
    :param slope: Slope (*S) or scalar, in the units given by ``slope_units``.
        Passed through to ``calculate_spread_rate()`` to compute
        ``direction_of_max_spread`` for the crown run.
    :param slope_units: Scalar integer ``SlopeUnitsEnum`` value
        (0 = Degrees [default], 1 = Percent).
    :param aspect: Terrain aspect in degrees (*S) or scalar (0 = north, clockwise).
        Passed through to ``calculate_spread_rate()`` alongside ``slope``.
    :param canopy_base_height: Height to base of canopy (ft) (*S) or scalar.
    :param canopy_height: Total canopy height (ft) (*S) or scalar.
    :param canopy_bulk_density: Canopy bulk density (lb/ft³) (*S) or scalar.
    :param moisture_foliar: Foliar moisture content (%) (*S) or scalar.
    :return: dict of (*S) ndarrays with keys:
        ``crown_fire_spread_rate`` (ft/min),
        ``crown_flame_length`` (ft),
        ``crown_fire_line_intensity`` (BTU/ft/s),
        ``crown_fire_heat_per_unit_area`` (BTU/ft²),
        ``canopy_heat_per_unit_area`` (BTU/ft²),
        ``crown_fire_transition_ratio`` (dimensionless),
        ``crown_fire_active_ratio`` (dimensionless),
        ``fire_type`` (int32: 0=Surface, 1=Torching, 2=Crowning),
        ``crown_critical_surface_fire_line_intensity`` (BTU/ft/s),
        ``crown_critical_fire_spread_rate`` (ft/min),
        ``crown_length_to_width_ratio`` (dimensionless).
    """
    try:
        from .surface import (
            build_particle_arrays,
            calculate_fuelbed_intermediates,
            calculate_reaction_intensity,
            calculate_spread_rate,
        )
        from .behave_units import speed_from_base, slope_to_base
    except ImportError:
        from surface import (
            build_particle_arrays,
            calculate_fuelbed_intermediates,
            calculate_reaction_intensity,
            calculate_spread_rate,
        )
        from behave_units import speed_from_base, slope_to_base

    # --- Coerce spatial inputs to at-least-1D ndarrays ---
    fm_grid   = np.atleast_1d(np.asarray(fuel_model_grid,    dtype=np.int32))
    slope_deg = slope_to_base(
        np.atleast_1d(np.asarray(slope, dtype=float)), slope_units
    )
    aspect    = np.atleast_1d(np.asarray(aspect,             dtype=float))
    wind_dir  = np.atleast_1d(np.asarray(wind_direction,     dtype=float))
    cbh       = np.atleast_1d(np.asarray(canopy_base_height, dtype=float))   # ft
    ch        = np.atleast_1d(np.asarray(canopy_height,      dtype=float))   # ft
    cbd       = np.atleast_1d(np.asarray(canopy_bulk_density, dtype=float))  # lb/ft³
    mf_pct    = np.atleast_1d(np.asarray(moisture_foliar,    dtype=float))   # percent
    wind_speed = np.atleast_1d(np.asarray(wind_speed,        dtype=float))
    S = fm_grid.shape

    # --- Crown fuel: FM10 everywhere, WAF=0.4 ---
    # Terrain (slope_deg, aspect) and wind direction are passed through so that
    # direction_of_max_spread is computed correctly for the crown run.
    fm10 = np.full(S, 10, dtype=np.int32)
    p_crown = build_particle_arrays(lut, fm10, m1h, m10h, m100h, mlh, mlw)
    ib_crown = calculate_fuelbed_intermediates(p_crown)
    ri_crown = calculate_reaction_intensity(ib_crown)

    crown_surface = calculate_spread_rate(
        ri_crown, ib_crown,
        wind_speed, wind_speed_units,
        wind_dir, wind_orientation_mode,   # actual wind direction and orientation
        slope_deg, 0,                       # already converted to degrees above
        aspect,                             # actual terrain — affects direction_of_max_spread
        canopy_cover=np.zeros(S),
        canopy_height=ch,
        crown_ratio=np.zeros(S),
        waf_method='UserInput',
        user_waf=np.full(S, 0.4),           # fixed WAF for crown fire
    )

    # Crown ROS = 3.34 × surface-equivalent spread rate under FM10 (Rothermel 1991)
    crown_ros = 3.34 * crown_surface['spread_rate']

    # --- Crown L:W ratio (Rothermel 1991 eq. 10 — §13.2 Gap A fix) ---
    # Uses EWS from the crown run (mph) — different from surface L:W formula
    crown_ews_mph = crown_surface['effective_wind_speed']
    crown_lwr = np.where(
        crown_ews_mph > 1e-7,
        1.0 + 0.125 * crown_ews_mph,
        1.0
    )

    # --- Crown fuel load and heat per unit area ---
    crown_fuel_load = np.maximum(cbd * (ch - cbh), 0.0)   # lb/ft²
    LOW_HOC = 8000.0   # low heat of combustion for canopy foliage (BTU/lb)
    canopy_hpua = crown_fuel_load * LOW_HOC                 # BTU/ft²
    crown_hpua = surface_results['heat_per_unit_area'] + canopy_hpua

    # --- Crown fireline intensity and flame length ---
    crown_fli = (crown_ros / 60.0) * crown_hpua   # BTU/ft/s
    safe_cfl = np.where(crown_fli > 0, crown_fli, 0.0)
    crown_fl = np.where(crown_fli > 0, 0.2 * (safe_cfl ** (2.0 / 3.0)), 0.0)   # ft

    # --- Critical surface fireline intensity for crown ignition (Van Wagner 1977) ---
    mf_pct_safe = np.maximum(mf_pct, 30.0)    # clamp to 30% minimum
    cbh_m = cbh * 0.3048                        # ft → m
    cbh_m = np.maximum(cbh_m, 0.1)             # avoid zero canopy base height
    crit_fli_kw = (0.010 * cbh_m * (460.0 + 25.9 * mf_pct_safe)) ** 1.5   # kW/m
    crit_fli = crit_fli_kw * 0.2886719         # kW/m → BTU/ft/s

    # --- Critical crown spread rate (Van Wagner 1977) ---
    cbd_kg_m3 = cbd * 16.0185   # lb/ft³ → kg/m³
    safe_cbd = np.where(cbd_kg_m3 > 1e-7, cbd_kg_m3, 1.0)
    crit_crown_ros = np.where(
        cbd_kg_m3 > 1e-7,
        (3.0 / safe_cbd) * 3.28084,   # m/min → ft/min
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

    # --- Fire type classification (0=Surface, 1=Torching, 2=Crowning) ---
    fire_type = np.where(
        transition_ratio < 1.0, 0,
        np.where(active_ratio < 1.0, 1, 2)
    ).astype(np.int32)

    return {
        'crown_fire_spread_rate':                    crown_ros,
        'crown_flame_length':                        crown_fl,
        'crown_fire_line_intensity':                 crown_fli,
        'crown_fire_heat_per_unit_area':             crown_hpua,
        'canopy_heat_per_unit_area':                 canopy_hpua,
        'crown_fire_transition_ratio':               transition_ratio,
        'crown_fire_active_ratio':                   active_ratio,
        'fire_type':                                 fire_type,
        'crown_critical_surface_fire_line_intensity': crit_fli,
        'crown_critical_fire_spread_rate':           crit_crown_ros,
        'crown_length_to_width_ratio':               crown_lwr,
    }
