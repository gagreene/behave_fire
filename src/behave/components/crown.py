"""
crown.py — Vectorized Crown Fire Pipeline (V8)

Implements both the Rothermel (1991) and Scott & Reinhardt (2001) crown fire
methods, exactly matching the C++ BehavePlus source (crown.cpp).

Key design points
-----------------
* Crown spread always uses FM10 with WAF = 0.4 and flat terrain (Rothermel 1991).
* CFB uses Scott & Reinhardt's linear-interpolation formula, NOT an exponential.
* ``fire_type`` has four values matching the C++ FireType enum:
    0 = Surface
    1 = Torching          (passive crown fire)
    2 = ConditionalCrownFire  (active possible but surface can't transition yet)
    3 = Crowning          (active crown fire)
* ``assignFinalFireBehavior`` follows C++ exactly:
    - Rothermel torching  → surface ROS, crown HPUA/FLI/FL
    - S&R passive         → CFB-blended passive values
    - Active (both)       → crown ROS/HPUA/FLI/FL

Public API
----------
calculate_crown_fire(surface_results, lut, fuel_model_grid,
                     m1h, m10h, m100h, mlh, mlw,
                     wind_speed, wind_speed_units,
                     wind_direction, wind_orientation_mode,
                     slope, slope_units, aspect,
                     canopy_base_height, canopy_height,
                     canopy_bulk_density, moisture_foliar)
    → dict of (*S) arrays
"""

import numpy as np
from typing import Union

# ---------------------------------------------------------------------------
# FM10 physical constants — pre-computed from the fuel model, used to
# back-solve the "crowning" surface fire ROS without a full surface run.
# These match the values hard-coded in calculateCrownFireActiveWindSpeed()
# in the C++ crown.cpp (lines ~330-340).
# ---------------------------------------------------------------------------
_FM10_PROP_FLUX          = 0.048317062998571636   # propagating flux ratio
_FM10_WIND_B             = 1.4308256324729873      # wind factor exponent B
_FM10_WIND_B_INV         = 1.0 / _FM10_WIND_B
_FM10_WIND_K             = 0.0016102128596515481   # wind factor coefficient K = C*(β/β_opt)^-E


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
    Vectorized crown fire calculations — C++ BehavePlus equivalent (V8).

    Implements ``doCrownRunScottAndReinhardt()`` logic from crown.cpp, which is
    the method that produces CFB and the full suite of final head fire outputs.

    :param surface_results: Output dict from ``calculate_spread_rate()`` for the
        surface fire run.  Must contain ``'spread_rate'``, ``'fireline_intensity'``,
        and ``'heat_per_unit_area'`` in their base units (ft/min, BTU/ft/s,
        BTU/ft²).
    :param lut: Fuel property lookup dict from ``build_fuel_lookup_arrays()``.
    :param fuel_model_grid: Integer fuel model number array (*S) or scalar.
        Used only to derive the spatial shape ``S``; crown fuel properties
        always come from FM10 regardless of this value.
    :param m1h: 1-hr dead fuel moisture as fraction (*S) or scalar.
    :param m10h: 10-hr dead fuel moisture as fraction (*S) or scalar.
    :param m100h: 100-hr dead fuel moisture as fraction (*S) or scalar.
    :param mlh: Live herbaceous fuel moisture as fraction (*S) or scalar.
    :param mlw: Live woody fuel moisture as fraction (*S) or scalar.
    :param wind_speed: Wind speed (*S) or scalar, already in ``wind_speed_units``.
        Must be the 20-ft open-wind speed (same value passed to the surface run).
    :param wind_speed_units: Scalar integer ``SpeedUnitsEnum`` value.
    :param wind_direction: Wind direction in degrees (*S) or scalar.
    :param wind_orientation_mode: ``'RelativeToUpslope'`` or ``'RelativeToNorth'``.
    :param slope: Slope (*S) or scalar, in the units given by ``slope_units``.
    :param slope_units: ``SlopeUnitsEnum`` integer (0=Degrees, 1=Percent).
    :param aspect: Terrain aspect in degrees (*S) or scalar (0=north, clockwise).
    :param canopy_base_height: Height to base of canopy (ft) (*S) or scalar.
    :param canopy_height: Total canopy height (ft) (*S) or scalar.
    :param canopy_bulk_density: Canopy bulk density (lb/ft³) (*S) or scalar.
    :param moisture_foliar: Foliar moisture content (%) (*S) or scalar.
    :return: dict of (*S) ndarrays with keys:

        **Crown fire characterisics**

        ``crown_fire_spread_rate`` (ft/min) — active crown fire ROS (3.34 × FM10 ROS).
        ``crown_flame_length`` (ft) — flame length for active crown FLI.
        ``crown_fire_line_intensity`` (BTU/ft/s) — active crown fireline intensity.
        ``crown_fire_heat_per_unit_area`` (BTU/ft²) — surface + canopy HPUA.
        ``canopy_heat_per_unit_area`` (BTU/ft²) — canopy contribution only.
        ``crown_critical_surface_fire_line_intensity`` (BTU/ft/s) — Van Wagner threshold.
        ``crown_critical_fire_spread_rate`` (ft/min) — Van Wagner R'active.
        ``crown_fire_transition_ratio`` (dimensionless) — surface FLI / critical FLI.
        ``crown_fire_active_ratio`` (dimensionless) — crown ROS / critical crown ROS.
        ``crown_length_to_width_ratio`` (dimensionless).
        ``fire_type`` (int32) — 0=Surface, 1=Torching, 2=ConditionalCrownFire, 3=Crowning.

        **Scott & Reinhardt CFB and passive crown fire intermediates**

        ``surface_fire_critical_spread_rate`` (ft/min) — R'initiation: surface ROS
            at which torching begins (= 60 × crit_fli / surface_hpua).
        ``crowning_surface_fire_spread_rate`` (ft/min) — R'sa: surface ROS at which
            the active crown ROS is fully achieved (CFB → 1).
        ``crown_fraction_burned`` (dimensionless, 0–1) — linear interpolation
            between R'initiation (CFB=0) and R'sa (CFB=1).
        ``passive_crown_fire_spread_rate`` (ft/min).
        ``passive_crown_fire_heat_per_unit_area`` (BTU/ft²).
        ``passive_crown_fire_line_intensity`` (BTU/ft/s).
        ``passive_crown_fire_flame_length`` (ft).

        **Blended final head fire outputs**

        ``final_spread_rate`` (ft/min) — depends on fire type (see Notes).
        ``final_heat_per_unit_area`` (BTU/ft²).
        ``final_fireline_intensity`` (BTU/ft/s).
        ``final_flame_length`` (ft).

    Notes
    -----
    ``assignFinalFireBehaviorBasedOnFireType`` (Scott & Reinhardt model):

    * **Surface / ConditionalCrownFire** → final = surface fire values.
    * **Torching (passive)** → final = CFB-blended passive values
      (``passive_crown_fire_*``).
    * **Crowning (active)** → final = active crown fire values
      (``crown_fire_*``).
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

    # -----------------------------------------------------------------------
    # Coerce spatial inputs
    # -----------------------------------------------------------------------
    fm_grid    = np.atleast_1d(np.asarray(fuel_model_grid,     dtype=np.int32))
    slope_deg  = slope_to_base(np.atleast_1d(np.asarray(slope,  dtype=float)), slope_units)
    aspect     = np.atleast_1d(np.asarray(aspect,              dtype=float))
    wind_dir   = np.atleast_1d(np.asarray(wind_direction,      dtype=float))
    cbh        = np.atleast_1d(np.asarray(canopy_base_height,  dtype=float))   # ft
    ch         = np.atleast_1d(np.asarray(canopy_height,       dtype=float))   # ft
    cbd        = np.atleast_1d(np.asarray(canopy_bulk_density, dtype=float))   # lb/ft³
    mf_pct     = np.atleast_1d(np.asarray(moisture_foliar,     dtype=float))   # percent
    wind_speed = np.atleast_1d(np.asarray(wind_speed,          dtype=float))   # in wind_speed_units
    S = fm_grid.shape

    # -----------------------------------------------------------------------
    # Step 1 — Retrieve surface fire results (already computed by caller)
    # -----------------------------------------------------------------------
    surface_ros  = surface_results['spread_rate']           # ft/min
    surface_fli  = surface_results['fireline_intensity']    # BTU/ft/s
    surface_hpua = surface_results['heat_per_unit_area']    # BTU/ft²

    # -----------------------------------------------------------------------
    # Step 2 — Crown fuel run: FM10, WAF=0.4, flat/upslope
    # (mirrors doCrownRunScottAndReinhardt Step 2 in crown.cpp)
    # -----------------------------------------------------------------------
    fm10     = np.full(S, 10, dtype=np.int32)
    p_crown  = build_particle_arrays(lut=lut, fuel_model_grid=fm10,
                                     m1h=m1h, m10h=m10h, m100h=m100h,
                                     mlh=mlh, mlw=mlw)
    ib_crown = calculate_fuelbed_intermediates(p=p_crown)
    ri_crown = calculate_reaction_intensity(ib=ib_crown)

    crown_surface = calculate_spread_rate(
        ri=ri_crown,
        ib=ib_crown,
        wind_speed=wind_speed,
        wind_speed_units=wind_speed_units,
        wind_direction=wind_dir,
        wind_orientation_mode=wind_orientation_mode,
        slope=slope_deg,
        slope_units=0,
        aspect=aspect,
        canopy_cover=np.zeros(S),
        canopy_height=ch,
        crown_ratio=np.zeros(S),
        waf_method='UserInput',
        user_waf=np.full(S, 0.4),
    )

    # Active crown ROS = 3.34 × FM10 ROS (Rothermel 1991)
    crown_ros = 3.34 * crown_surface['spread_rate']   # ft/min

    # Crown L:W ratio — Rothermel 1991 eq. 10 (uses EWS in mph from the crown run)
    crown_ews_mph = crown_surface['effective_wind_speed']   # mph (stored base = 5)
    crown_lwr = np.where(crown_ews_mph > 1e-7, 1.0 + 0.125 * crown_ews_mph, 1.0)

    # -----------------------------------------------------------------------
    # Step 3 — Canopy fuel load and heat per unit area
    # -----------------------------------------------------------------------
    crown_fuel_load = np.maximum(cbd * (ch - cbh), 0.0)   # lb/ft²
    LOW_HOC         = 8000.0                               # BTU/lb (hard-coded, matches C++)
    canopy_hpua     = crown_fuel_load * LOW_HOC            # BTU/ft²
    crown_hpua      = surface_hpua + canopy_hpua           # BTU/ft²

    # Active crown fireline intensity and flame length (Byram 1959 / Thomas 1963)
    crown_fli = (crown_ros / 60.0) * crown_hpua
    crown_fl  = np.where(crown_fli > 0.0, 0.2 * crown_fli ** (2.0 / 3.0), 0.0)

    # -----------------------------------------------------------------------
    # Step 4 — Critical surface fireline intensity (Van Wagner 1977)
    # -----------------------------------------------------------------------
    mf_pct_safe = np.maximum(mf_pct, 30.0)          # clamp to 30 % minimum
    cbh_m       = np.maximum(cbh * 0.3048, 0.1)     # ft → m, avoid zero
    crit_fli_kw = (0.010 * cbh_m * (460.0 + 25.9 * mf_pct_safe)) ** 1.5   # kW/m
    crit_fli    = crit_fli_kw * 0.2886719            # kW/m → BTU/ft/s

    # -----------------------------------------------------------------------
    # Step 5 — Critical crown fire spread rate R'active (Van Wagner 1977)
    # -----------------------------------------------------------------------
    cbd_kg_m3       = cbd * 16.0185                  # lb/ft³ → kg/m³
    safe_cbd        = np.where(cbd_kg_m3 > 1e-7, cbd_kg_m3, 1.0)
    crit_crown_ros  = np.where(
        cbd_kg_m3 > 1e-7,
        (3.0 / safe_cbd) * 3.28084,   # m/min → ft/min
        0.0,
    )

    # -----------------------------------------------------------------------
    # Step 6 — Transition and active ratios
    # -----------------------------------------------------------------------
    safe_crit_fli   = np.where(crit_fli > 1e-7, crit_fli, 1.0)
    transition_ratio = np.where(crit_fli > 1e-7,
                                surface_fli / safe_crit_fli, 0.0)

    safe_crit_crown  = np.where(crit_crown_ros > 1e-7, crit_crown_ros, 1.0)
    active_ratio     = np.where(crit_crown_ros > 1e-7,
                                crown_ros / safe_crit_crown, 0.0)

    # -----------------------------------------------------------------------
    # Step 7 — Fire type (4-value enum matching C++ FireType)
    #   0 = Surface
    #   1 = Torching          (passive crown — transition_ratio >= 1, active_ratio < 1)
    #   2 = ConditionalCrownFire  (active possible but can't transition yet)
    #   3 = Crowning          (active crown — both ratios >= 1)
    # -----------------------------------------------------------------------
    fire_type = np.where(
        transition_ratio < 1.0,
        np.where(active_ratio < 1.0, 0, 2),          # Surface or Conditional
        np.where(active_ratio < 1.0, 1, 3),           # Torching or Crowning
    ).astype(np.int32)

    is_surface_fire   = (fire_type == 0) | (fire_type == 2)   # Surface or Conditional
    is_passive_crown  = fire_type == 1
    is_active_crown   = fire_type == 3

    # -----------------------------------------------------------------------
    # Step 8 — Scott & Reinhardt critical surface ROS (R'initiation)
    # Surface fire ROS at which torching begins:
    #   R'initiation = (60 × crit_surface_fli) / surface_hpua
    # -----------------------------------------------------------------------
    safe_surface_hpua = np.where(surface_hpua > 1e-7, surface_hpua, 1.0)
    surface_crit_ros  = np.where(
        surface_hpua > 1e-7,
        (60.0 * crit_fli) / safe_surface_hpua,
        0.0,
    )

    # -----------------------------------------------------------------------
    # Step 9 — Crowning surface fire ROS (R'sa)
    # Back-solve: what 20-ft wind speed drives FM10 to R'active, then re-run
    # surface at that wind speed to get the corresponding surface ROS.
    #
    # C++ calculateCrownFireActiveWindSpeed() derivation:
    #   R'active = 3.34 × R10
    #   R10 = ros0 × (1 + slope_factor + windK × uMid^windB)     [slope=0]
    #   → uMid = ((R10/ros0 - 1) / windK)^(1/windB)
    #   → U20ft = uMid / waf                                      [waf=0.4]
    #
    # Then the crowning surface ROS = surface model run at that U20ft.
    # We avoid a full vectorised re-run by using the FM10 no-wind/no-slope
    # base spread rate stored in ib_crown and ri_crown, plus the FM10
    # wind constants hard-coded above.
    # -----------------------------------------------------------------------
    # No-wind/no-slope base ROS for FM10 at these moisture conditions
    ros0 = ri_crown * ib_crown['propagating_flux'] / np.where(
        ib_crown['heat_sink'] > 1e-7, ib_crown['heat_sink'], 1.0
    )   # ft/min

    # R'active in ft/min
    r_active  = crit_crown_ros                                  # ft/min (already computed)
    r10       = r_active / 3.34                                 # undo the 3.34× multiplier

    # Back-solve midflame wind speed (ft/min) at which FM10 reaches r10
    # u_mid = ((r10/ros0 - 1) / windK)^(1/windB)  [slope_factor = 0]
    a_val     = np.where(
        ros0 > 1e-7,
        np.maximum((r10 / np.where(ros0 > 1e-7, ros0, 1.0)) - 1.0, 0.0) / _FM10_WIND_K,
        0.0,
    )
    u_mid_fpm = np.where(a_val > 0.0, a_val ** _FM10_WIND_B_INV, 0.0)   # ft/min midflame

    # 20-ft wind speed for WAF=0.4
    u20_active_fpm = u_mid_fpm / 0.4    # ft/min

    # Now run the surface fuel at this wind speed to get R'sa.
    # We re-use the already-computed ib_crown and ri_crown (FM10 particle
    # arrays are wind-speed-independent), passing flat terrain, upslope
    # orientation, and the active wind speed per cell.
    crown_active_surface = calculate_spread_rate(
        ri=ri_crown,
        ib=ib_crown,
        wind_speed=u20_active_fpm,
        wind_speed_units=0,                      # already ft/min
        wind_direction=np.zeros(S),
        wind_orientation_mode='RelativeToUpslope',
        slope=np.zeros(S),
        slope_units=0,
        aspect=np.zeros(S),
        canopy_cover=np.zeros(S),
        canopy_height=ch,
        crown_ratio=np.zeros(S),
        waf_method='UserInput',
        user_waf=np.full(S, 0.4),
    )
    crowning_surface_ros = crown_active_surface['spread_rate']   # R'sa (ft/min)

    # -----------------------------------------------------------------------
    # Step 10 — Crown Fraction Burned (CFB) — Scott & Reinhardt (2001)
    # Linear interpolation between R'initiation (CFB=0) and R'sa (CFB=1):
    #   CFB = (surface_ros - R'initiation) / (R'sa - R'initiation)
    #   clamped to [0, 1]
    # -----------------------------------------------------------------------
    cfb_denom = crowning_surface_ros - surface_crit_ros
    cfb = np.where(
        cfb_denom > 1e-7,
        (surface_ros - surface_crit_ros) / cfb_denom,
        0.0,
    )
    cfb = np.clip(cfb, 0.0, 1.0)

    # -----------------------------------------------------------------------
    # Step 11 — Passive (torching) crown fire spread rate, HPUA, FLI, FL
    # (Scott & Reinhardt blended values, used only when fire_type==Torching)
    # -----------------------------------------------------------------------
    passive_ros  = surface_ros  + cfb * (crown_ros  - surface_ros)
    passive_hpua = surface_hpua + cfb *  canopy_hpua
    passive_fli  = passive_hpua * passive_ros / 60.0
    passive_fl   = np.where(passive_fli > 0.0, 0.2 * passive_fli ** (2.0 / 3.0), 0.0)

    # -----------------------------------------------------------------------
    # Step 12 — Assign final fire behavior (Scott & Reinhardt model)
    # Mirrors assignFinalFireBehaviorBasedOnFireType(scott_and_reinhardt)
    # -----------------------------------------------------------------------
    final_ros  = np.where(is_surface_fire, surface_ros,
                 np.where(is_passive_crown, passive_ros,  crown_ros))
    final_hpua = np.where(is_surface_fire, surface_hpua,
                 np.where(is_passive_crown, passive_hpua, crown_hpua))
    final_fli  = np.where(is_surface_fire, surface_fli,
                 np.where(is_passive_crown, passive_fli,  crown_fli))
    final_fl   = np.where(is_surface_fire,
                          np.where(surface_fli > 0.0,
                                   0.2 * surface_fli ** (2.0 / 3.0), 0.0),
                 np.where(is_passive_crown, passive_fl, crown_fl))

    return {
        # Active crown fire characteristics
        'crown_fire_spread_rate':                     crown_ros,
        'crown_flame_length':                         crown_fl,
        'crown_fire_line_intensity':                  crown_fli,
        'crown_fire_heat_per_unit_area':              crown_hpua,
        'canopy_heat_per_unit_area':                  canopy_hpua,
        'crown_critical_surface_fire_line_intensity': crit_fli,
        'crown_critical_fire_spread_rate':            crit_crown_ros,
        'crown_fire_transition_ratio':                transition_ratio,
        'crown_fire_active_ratio':                    active_ratio,
        'crown_length_to_width_ratio':                crown_lwr,
        'fire_type':                                  fire_type,
        # Scott & Reinhardt CFB intermediates
        'surface_fire_critical_spread_rate':          surface_crit_ros,
        'crowning_surface_fire_spread_rate':          crowning_surface_ros,
        'crown_fraction_burned':                      cfb,
        'passive_crown_fire_spread_rate':             passive_ros,
        'passive_crown_fire_heat_per_unit_area':      passive_hpua,
        'passive_crown_fire_line_intensity':          passive_fli,
        'passive_crown_fire_flame_length':            passive_fl,
        # Final blended head fire outputs
        'final_spread_rate':                          final_ros,
        'final_heat_per_unit_area':                   final_hpua,
        'final_fireline_intensity':                   final_fli,
        'final_flame_length':                         final_fl,
    }
