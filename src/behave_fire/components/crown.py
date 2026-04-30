"""
crown.py - Vectorized Crown Fire Pipeline (V8)

Implements both the Rothermel (1991) and Scott & Reinhardt (2001) crown fire
methods, exactly matching the C++ BehavePlus source (crown.cpp).

Key design points
-----------------
* Crown spread always uses FM10 with WAF = 0.4 and flat terrain (Rothermel 1991).
* CFB uses Scott & Reinhardt's linear-interpolation formula, NOT an exponential.
* ``fire_type`` has four values matching the C++ FireType enum:
    0 = Surface
    1 = Torching
    2 = ConditionalCrownFire
    3 = Crowning
* ``assignFinalFireBehavior`` follows C++ exactly:
    - Rothermel torching -> surface ROS, crown HPUA/FLI/FL
    - S&R passive        -> CFB-blended passive values
    - Active             -> crown ROS/HPUA/FLI/FL
"""

import numpy as np
from typing import Union

# ---------------------------------------------------------------------------
# FM10 physical constants - pre-computed from the fuel model, used to
# back-solve the "crowning" surface fire ROS without a full surface run.
# These match the values hard-coded in calculateCrownFireActiveWindSpeed()
# in the C++ crown.cpp (lines ~330-340).
# ---------------------------------------------------------------------------
_FM10_PROP_FLUX = 0.048317062998571636
_FM10_WIND_B = 1.4308256324729873
_FM10_WIND_B_INV = 1.0 / _FM10_WIND_B
_FM10_WIND_K = 0.0016102128596515481


def _load_slope_to_base():
    try:
        from .behave_units import slope_to_base
    except ImportError:
        from behave_units import slope_to_base
    return slope_to_base


def _load_surface_modules():
    try:
        from .surface import (
            build_particle_arrays,
            calculate_fuelbed_intermediates,
            calculate_reaction_intensity,
            run_surface_fire,
        )
    except ImportError:
        from surface import (
            build_particle_arrays,
            calculate_fuelbed_intermediates,
            calculate_reaction_intensity,
            run_surface_fire,
        )
    return (
        build_particle_arrays,
        calculate_fuelbed_intermediates,
        calculate_reaction_intensity,
        run_surface_fire,
    )


def assign_final_crown_fire_behavior(
        is_surface_fire: Union[bool, np.ndarray],
        is_passive_crown: Union[bool, np.ndarray],
        surface_ros: Union[float, np.ndarray],
        surface_hpua: Union[float, np.ndarray],
        surface_fli: Union[float, np.ndarray],
        passive_ros: Union[float, np.ndarray],
        passive_hpua: Union[float, np.ndarray],
        passive_fli: Union[float, np.ndarray],
        passive_fl: Union[float, np.ndarray],
        crown_ros: Union[float, np.ndarray],
        crown_hpua: Union[float, np.ndarray],
        crown_fli: Union[float, np.ndarray],
        crown_fl: Union[float, np.ndarray],
) -> dict:
    """Step 12 - assign final head fire behavior based on fire type."""
    final_ros = np.where(is_surface_fire, surface_ros, np.where(is_passive_crown, passive_ros, crown_ros))
    final_hpua = np.where(is_surface_fire, surface_hpua, np.where(is_passive_crown, passive_hpua, crown_hpua))
    final_fli = np.where(is_surface_fire, surface_fli, np.where(is_passive_crown, passive_fli, crown_fli))
    surface_fl = np.where(surface_fli > 0.0, 0.2 * surface_fli ** (2.0 / 3.0), 0.0)
    final_fl = np.where(is_surface_fire, surface_fl, np.where(is_passive_crown, passive_fl, crown_fl))
    return {
        'final_spread_rate': final_ros,
        'final_heat_per_unit_area': final_hpua,
        'final_fireline_intensity': final_fli,
        'final_flame_length': final_fl,
    }


def calculate_critical_crown_fire_spread_rate(
        canopy_bulk_density: Union[float, np.ndarray],
) -> np.ndarray:
    """Step 5 - calculate Van Wagner critical active crown spread rate."""
    cbd_kg_m3 = canopy_bulk_density * 16.0185
    safe_cbd = np.where(cbd_kg_m3 > 1e-7, cbd_kg_m3, 1.0)
    return np.where(cbd_kg_m3 > 1e-7, (3.0 / safe_cbd) * 3.28084, 0.0)


def calculate_critical_surface_fireline_intensity(
        canopy_base_height: Union[float, np.ndarray],
        moisture_foliar: Union[float, np.ndarray],
) -> np.ndarray:
    """Step 4 - calculate Van Wagner critical surface fireline intensity."""
    mf_pct_safe = np.maximum(moisture_foliar, 30.0)
    cbh_m = np.maximum(canopy_base_height * 0.3048, 0.1)
    crit_fli_kw = (0.010 * cbh_m * (460.0 + 25.9 * mf_pct_safe)) ** 1.5
    return crit_fli_kw * 0.2886719


def calculate_crown_fire_ratios(
        surface_fli: Union[float, np.ndarray],
        crown_ros: Union[float, np.ndarray],
        critical_surface_fli: Union[float, np.ndarray],
        critical_crown_ros: Union[float, np.ndarray],
) -> dict:
    """Step 6 - calculate transition and active crown fire ratios."""
    safe_crit_fli = np.where(critical_surface_fli > 1e-7, critical_surface_fli, 1.0)
    transition_ratio = np.where(
        critical_surface_fli > 1e-7,
        surface_fli / safe_crit_fli,
        0.0,
    )
    safe_crit_crown = np.where(critical_crown_ros > 1e-7, critical_crown_ros, 1.0)
    active_ratio = np.where(
        critical_crown_ros > 1e-7,
        crown_ros / safe_crit_crown,
        0.0,
    )
    return {
        'crown_fire_transition_ratio': transition_ratio,
        'crown_fire_active_ratio': active_ratio,
    }


def calculate_crown_fraction_burned(
        surface_ros: Union[float, np.ndarray],
        surface_critical_ros: Union[float, np.ndarray],
        crowning_surface_ros: Union[float, np.ndarray],
) -> np.ndarray:
    """Step 10 - calculate Scott & Reinhardt crown fraction burned."""
    cfb_denom = crowning_surface_ros - surface_critical_ros
    cfb = np.where(
        cfb_denom > 1e-7,
        (surface_ros - surface_critical_ros) / cfb_denom,
        0.0,
    )
    return np.clip(cfb, 0.0, 1.0)


def calculate_crown_heat_and_intensity(
        surface_hpua: Union[float, np.ndarray],
        crown_ros: Union[float, np.ndarray],
        canopy_base_height: Union[float, np.ndarray],
        canopy_height: Union[float, np.ndarray],
        canopy_bulk_density: Union[float, np.ndarray],
) -> dict:
    """
    Step 3 - calculate canopy load, crown heat per unit area, active crown
    fireline intensity, and active crown flame length.
    """
    crown_fuel_load = np.maximum(canopy_bulk_density * (canopy_height - canopy_base_height), 0.0)
    low_hoc = 8000.0
    canopy_hpua = crown_fuel_load * low_hoc
    crown_hpua = surface_hpua + canopy_hpua
    crown_fli = (crown_ros / 60.0) * crown_hpua
    crown_fl = np.where(crown_fli > 0.0, 0.2 * crown_fli ** (2.0 / 3.0), 0.0)
    return {
        'crown_fuel_load': crown_fuel_load,
        'canopy_heat_per_unit_area': canopy_hpua,
        'crown_fire_heat_per_unit_area': crown_hpua,
        'crown_fire_line_intensity': crown_fli,
        'crown_flame_length': crown_fl,
    }


def calculate_crown_surface_fire(
        lut: dict,
        S: tuple,
        m1h: Union[float, np.ndarray],
        m10h: Union[float, np.ndarray],
        m100h: Union[float, np.ndarray],
        mlh: Union[float, np.ndarray],
        mlw: Union[float, np.ndarray],
        wind_speed: Union[float, np.ndarray],
        wind_speed_units: int,
        wind_direction: Union[float, np.ndarray],
        wind_orientation_mode: str,
        slope_deg: Union[float, np.ndarray],
        aspect: Union[float, np.ndarray],
        canopy_height: Union[float, np.ndarray],
) -> dict:
    """
    Step 2 - run the crown fuel spread calculation using FM10, WAF=0.4.

    Returns FM10 particle/intermediate arrays, crown surface results, active
    crown ROS, and crown length-to-width ratio.
    """
    (
        build_particle_arrays,
        calc_fuelbed_intermediates,
        calc_reaction_intensity,
        run_surface_fire,
    ) = _load_surface_modules()

    fm10 = np.full(S, 10, dtype=np.int32)
    p_crown = build_particle_arrays(
        lut=lut,
        fuel_model_grid=fm10,
        m1h=m1h,
        m10h=m10h,
        m100h=m100h,
        mlh=mlh,
        mlw=mlw,
    )
    ib_crown = calc_fuelbed_intermediates(p=p_crown)
    ri_crown = calc_reaction_intensity(ib=ib_crown)

    crown_surface = run_surface_fire(
        ri=ri_crown,
        ib=ib_crown,
        wind_speed=wind_speed,
        wind_speed_units=wind_speed_units,
        wind_direction=wind_direction,
        wind_orientation_mode=wind_orientation_mode,
        slope=slope_deg,
        slope_units=0,
        aspect=aspect,
        canopy_cover=np.zeros(S),
        canopy_height=canopy_height,
        crown_ratio=np.zeros(S),
        waf_method='UserInput',
        user_waf=np.full(S, 0.4),
    )

    crown_ros = 3.34 * crown_surface['spread_rate']
    crown_ews_mph = crown_surface['effective_wind_speed']
    crown_lwr = np.where(crown_ews_mph > 1e-7, 1.0 + 0.125 * crown_ews_mph, 1.0)

    return {
        'p_crown': p_crown,
        'ib_crown': ib_crown,
        'ri_crown': ri_crown,
        'crown_surface': crown_surface,
        'crown_fire_spread_rate': crown_ros,
        'crown_length_to_width_ratio': crown_lwr,
    }


def calculate_crowning_surface_fire_spread_rate(
        ri_crown: Union[float, np.ndarray],
        ib_crown: dict,
        critical_crown_ros: Union[float, np.ndarray],
        S: tuple,
        canopy_height: Union[float, np.ndarray],
) -> np.ndarray:
    """
    Step 9 - calculate R'sa, the surface ROS at which active crown ROS is
    fully achieved.
    """
    _, _, _, run_surface_fire = _load_surface_modules()

    ros0 = ri_crown * ib_crown['propagating_flux'] / np.where(
        ib_crown['heat_sink'] > 1e-7,
        ib_crown['heat_sink'],
        1.0,
    )
    r10 = critical_crown_ros / 3.34
    a_val = np.where(
        ros0 > 1e-7,
        np.maximum((r10 / np.where(ros0 > 1e-7, ros0, 1.0)) - 1.0, 0.0) / _FM10_WIND_K,
        0.0,
    )
    u_mid_fpm = np.where(a_val > 0.0, a_val ** _FM10_WIND_B_INV, 0.0)
    u20_active_fpm = u_mid_fpm / 0.4

    crown_active_surface = run_surface_fire(
        ri=ri_crown,
        ib=ib_crown,
        wind_speed=u20_active_fpm,
        wind_speed_units=0,
        wind_direction=np.zeros(S),
        wind_orientation_mode='RelativeToUpslope',
        slope=np.zeros(S),
        slope_units=0,
        aspect=np.zeros(S),
        canopy_cover=np.zeros(S),
        canopy_height=canopy_height,
        crown_ratio=np.zeros(S),
        waf_method='UserInput',
        user_waf=np.full(S, 0.4),
    )
    return crown_active_surface['spread_rate']


def calculate_passive_crown_fire_behavior(
        surface_ros: Union[float, np.ndarray],
        surface_hpua: Union[float, np.ndarray],
        crown_ros: Union[float, np.ndarray],
        canopy_hpua: Union[float, np.ndarray],
        cfb: Union[float, np.ndarray],
) -> dict:
    """Step 11 - calculate passive crown fire spread, HPUA, FLI, and FL."""
    passive_ros = surface_ros + cfb * (crown_ros - surface_ros)
    passive_hpua = surface_hpua + cfb * canopy_hpua
    passive_fli = passive_hpua * passive_ros / 60.0
    passive_fl = np.where(passive_fli > 0.0, 0.2 * passive_fli ** (2.0 / 3.0), 0.0)
    return {
        'passive_crown_fire_spread_rate': passive_ros,
        'passive_crown_fire_heat_per_unit_area': passive_hpua,
        'passive_crown_fire_line_intensity': passive_fli,
        'passive_crown_fire_flame_length': passive_fl,
    }


def calculate_surface_fire_critical_spread_rate(
        critical_surface_fli: Union[float, np.ndarray],
        surface_hpua: Union[float, np.ndarray],
) -> np.ndarray:
    """Step 8 - calculate surface ROS at which torching begins."""
    safe_surface_hpua = np.where(surface_hpua > 1e-7, surface_hpua, 1.0)
    return np.where(surface_hpua > 1e-7, (60.0 * critical_surface_fli) / safe_surface_hpua, 0.0)


def classify_crown_fire_type(
        transition_ratio: Union[float, np.ndarray],
        active_ratio: Union[float, np.ndarray],
) -> dict:
    """
    Step 7 - classify fire type.

    Fire type values: 0=Surface, 1=Torching, 2=ConditionalCrownFire, 3=Crowning.
    """
    fire_type = np.where(
        transition_ratio < 1.0,
        np.where(active_ratio < 1.0, 0, 2),
        np.where(active_ratio < 1.0, 1, 3),
    ).astype(np.int32)
    return {
        'fire_type': fire_type,
        'is_surface_fire': (fire_type == 0) | (fire_type == 2),
        'is_passive_crown': fire_type == 1,
        'is_active_crown': fire_type == 3,
    }


def coerce_crown_fire_inputs(
        fuel_model_grid: Union[int, np.ndarray],
        wind_speed: Union[float, np.ndarray],
        wind_direction: Union[float, np.ndarray],
        slope: Union[float, np.ndarray],
        slope_units: int = 0,
        aspect: Union[float, np.ndarray] = 0.0,
        canopy_base_height: Union[float, np.ndarray] = 0.0,
        canopy_height: Union[float, np.ndarray] = 0.0,
        canopy_bulk_density: Union[float, np.ndarray] = 0.0,
        moisture_foliar: Union[float, np.ndarray] = 100.0,
) -> dict:
    """
    Step 0 - coerce crown fire spatial inputs to arrays and base units.

    Returns a dict containing ``fm_grid``, ``slope_deg``, ``aspect``,
    ``wind_dir``, ``cbh``, ``ch``, ``cbd``, ``mf_pct``, ``wind_speed``, and
    spatial shape ``S``.
    """
    slope_to_base = _load_slope_to_base()
    fm_grid = np.atleast_1d(np.asarray(fuel_model_grid, dtype=np.int32))
    slope_deg = slope_to_base(np.atleast_1d(np.asarray(slope, dtype=float)), slope_units)
    return {
        'fm_grid': fm_grid,
        'slope_deg': slope_deg,
        'aspect': np.atleast_1d(np.asarray(aspect, dtype=float)),
        'wind_dir': np.atleast_1d(np.asarray(wind_direction, dtype=float)),
        'cbh': np.atleast_1d(np.asarray(canopy_base_height, dtype=float)),
        'ch': np.atleast_1d(np.asarray(canopy_height, dtype=float)),
        'cbd': np.atleast_1d(np.asarray(canopy_bulk_density, dtype=float)),
        'mf_pct': np.atleast_1d(np.asarray(moisture_foliar, dtype=float)),
        'wind_speed': np.atleast_1d(np.asarray(wind_speed, dtype=float)),
        'S': fm_grid.shape,
    }


def get_surface_fire_inputs(surface_results: dict) -> dict:
    """
    Step 1 - retrieve surface fire results needed by the crown pipeline.

    ``surface_results`` must contain values in component base units:
    ``spread_rate`` in ft/min, ``fireline_intensity`` in BTU/ft/s, and
    ``heat_per_unit_area`` in BTU/ft^2.
    """
    return {
        'surface_ros': surface_results['spread_rate'],
        'surface_fli': surface_results['fireline_intensity'],
        'surface_hpua': surface_results['heat_per_unit_area'],
    }


def run_crown_fire(
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
    Vectorized crown fire calculations.

    This driver runs the step functions in order for a continuous crown fire
    run. Each step function is also available for callers that need a specific
    intermediate component.
    """
    inputs = coerce_crown_fire_inputs(
        fuel_model_grid=fuel_model_grid,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        slope=slope,
        slope_units=slope_units,
        aspect=aspect,
        canopy_base_height=canopy_base_height,
        canopy_height=canopy_height,
        canopy_bulk_density=canopy_bulk_density,
        moisture_foliar=moisture_foliar,
    )
    surface = get_surface_fire_inputs(surface_results)

    crown_surface = calculate_crown_surface_fire(
        lut=lut,
        S=inputs['S'],
        m1h=m1h,
        m10h=m10h,
        m100h=m100h,
        mlh=mlh,
        mlw=mlw,
        wind_speed=inputs['wind_speed'],
        wind_speed_units=wind_speed_units,
        wind_direction=inputs['wind_dir'],
        wind_orientation_mode=wind_orientation_mode,
        slope_deg=inputs['slope_deg'],
        aspect=inputs['aspect'],
        canopy_height=inputs['ch'],
    )
    crown_ros = crown_surface['crown_fire_spread_rate']

    crown_heat = calculate_crown_heat_and_intensity(
        surface_hpua=surface['surface_hpua'],
        crown_ros=crown_ros,
        canopy_base_height=inputs['cbh'],
        canopy_height=inputs['ch'],
        canopy_bulk_density=inputs['cbd'],
    )
    canopy_hpua = crown_heat['canopy_heat_per_unit_area']
    crown_hpua = crown_heat['crown_fire_heat_per_unit_area']
    crown_fli = crown_heat['crown_fire_line_intensity']
    crown_fl = crown_heat['crown_flame_length']

    crit_fli = calculate_critical_surface_fireline_intensity(
        canopy_base_height=inputs['cbh'],
        moisture_foliar=inputs['mf_pct'],
    )
    crit_crown_ros = calculate_critical_crown_fire_spread_rate(
        canopy_bulk_density=inputs['cbd'],
    )
    ratios = calculate_crown_fire_ratios(
        surface_fli=surface['surface_fli'],
        crown_ros=crown_ros,
        critical_surface_fli=crit_fli,
        critical_crown_ros=crit_crown_ros,
    )
    fire_type = classify_crown_fire_type(
        transition_ratio=ratios['crown_fire_transition_ratio'],
        active_ratio=ratios['crown_fire_active_ratio'],
    )
    surface_crit_ros = calculate_surface_fire_critical_spread_rate(
        critical_surface_fli=crit_fli,
        surface_hpua=surface['surface_hpua'],
    )
    crowning_surface_ros = calculate_crowning_surface_fire_spread_rate(
        ri_crown=crown_surface['ri_crown'],
        ib_crown=crown_surface['ib_crown'],
        critical_crown_ros=crit_crown_ros,
        S=inputs['S'],
        canopy_height=inputs['ch'],
    )
    cfb = calculate_crown_fraction_burned(
        surface_ros=surface['surface_ros'],
        surface_critical_ros=surface_crit_ros,
        crowning_surface_ros=crowning_surface_ros,
    )
    passive = calculate_passive_crown_fire_behavior(
        surface_ros=surface['surface_ros'],
        surface_hpua=surface['surface_hpua'],
        crown_ros=crown_ros,
        canopy_hpua=canopy_hpua,
        cfb=cfb,
    )
    final = assign_final_crown_fire_behavior(
        is_surface_fire=fire_type['is_surface_fire'],
        is_passive_crown=fire_type['is_passive_crown'],
        surface_ros=surface['surface_ros'],
        surface_hpua=surface['surface_hpua'],
        surface_fli=surface['surface_fli'],
        passive_ros=passive['passive_crown_fire_spread_rate'],
        passive_hpua=passive['passive_crown_fire_heat_per_unit_area'],
        passive_fli=passive['passive_crown_fire_line_intensity'],
        passive_fl=passive['passive_crown_fire_flame_length'],
        crown_ros=crown_ros,
        crown_hpua=crown_hpua,
        crown_fli=crown_fli,
        crown_fl=crown_fl,
    )

    return {
        'crown_fire_spread_rate': crown_ros,
        'crown_flame_length': crown_fl,
        'crown_fire_line_intensity': crown_fli,
        'crown_fire_heat_per_unit_area': crown_hpua,
        'canopy_heat_per_unit_area': canopy_hpua,
        'crown_critical_surface_fire_line_intensity': crit_fli,
        'crown_critical_fire_spread_rate': crit_crown_ros,
        'crown_fire_transition_ratio': ratios['crown_fire_transition_ratio'],
        'crown_fire_active_ratio': ratios['crown_fire_active_ratio'],
        'crown_length_to_width_ratio': crown_surface['crown_length_to_width_ratio'],
        'fire_type': fire_type['fire_type'],
        'surface_fire_critical_spread_rate': surface_crit_ros,
        'crowning_surface_fire_spread_rate': crowning_surface_ros,
        'crown_fraction_burned': cfb,
        'passive_crown_fire_spread_rate': passive['passive_crown_fire_spread_rate'],
        'passive_crown_fire_heat_per_unit_area': passive['passive_crown_fire_heat_per_unit_area'],
        'passive_crown_fire_line_intensity': passive['passive_crown_fire_line_intensity'],
        'passive_crown_fire_flame_length': passive['passive_crown_fire_flame_length'],
        'final_spread_rate': final['final_spread_rate'],
        'final_heat_per_unit_area': final['final_heat_per_unit_area'],
        'final_fireline_intensity': final['final_fireline_intensity'],
        'final_flame_length': final['final_flame_length'],
    }
