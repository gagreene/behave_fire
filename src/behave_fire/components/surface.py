"""
surface.py — Vectorized Surface Fire Pipeline (V3 / V4 / V5 / fire-size)

Pure functions: no class state, no self.  All inputs are NumPy arrays or
Python scalars that are coerced to ndarrays at the entry points.

Public API
----------
build_particle_arrays(lut, fuel_model_grid, m1h, m10h, m100h, mlh, mlw)
    → dict of (5, *S) arrays

calculate_fuelbed_intermediates(p)
    → dict of (*S) arrays

calculate_reaction_intensity(ib)
    → (*S) array

calculate_wind_adjustment_factor(canopy_cover, canopy_height, crown_ratio, depth)
    → (*S) array

run_surface_fire(ri, ib, wind_speed, wind_speed_units, wind_direction,
                 wind_orientation_mode, slope, slope_units, aspect,
                 canopy_cover, canopy_height, crown_ratio,
                 wind_height_mode, waf_method, user_waf)
    → dict of (*S) arrays

calculate_fire_area(forward_ros, backing_ros, lwr, elapsed_min, is_crown)
    → (*S) array

calculate_fire_perimeter(forward_ros, backing_ros, lwr, elapsed_min, is_crown)
    → (*S) array

Notes on units
--------------
* All spatial inputs must already be in the base units used by the scalar path:
    - moisture:    fraction (e.g. 0.06, not 6)
    - slope:       degrees by default (SlopeUnitsEnum.Degrees=0); pass
                   SlopeUnitsEnum.Percent=1 and provide percent values to
                   convert automatically via ``slope_units``
    - wind_speed:  in the units specified by wind_speed_units
    - canopy_*:    canopy_cover & crown_ratio as fraction (0–1); heights in feet
"""

import numpy as np
from typing import Union

# ---------------------------------------------------------------------------
# V3 — Particle array construction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# V5 — Full spread rate and fire shape pipeline
# ---------------------------------------------------------------------------

def _load_surface_unit_converters():
    try:
        from .behave_units import speed_to_base, speed_from_base, slope_to_base
    except ImportError:
        from behave_units import speed_to_base, speed_from_base, slope_to_base
    return speed_to_base, speed_from_base, slope_to_base

def _mask_surface_result(
        value: Union[float, np.ndarray],
        is_defined: Union[bool, np.ndarray],
        fill_value: Union[float, np.ndarray] = 0.0,
) -> np.ndarray:
    """Mask undefined fuel model cells in a surface output array."""
    value_arr = np.asarray(value, dtype=float)
    fill_arr = np.asarray(fill_value, dtype=float)
    return np.where(is_defined, value_arr, fill_arr)

def _size_sorted_wfl(
        savr_arr: np.ndarray,
        frac_arr: np.ndarray,
        wn_arr: np.ndarray
) -> np.ndarray:
    """
    Compute size-class weighted fuel load (Albini five-bin scheme).

    :param savr_arr: Surface-area-to-volume ratio array, shape (5, *S) (ft²/ft³).
    :param frac_arr: Per-particle SA fraction array, shape (5, *S) (dimensionless).
    :param wn_arr: Net fuel load array, shape (5, *S) (lb/ft²).
    :return: (*S) ndarray — size-class weighted net fuel load (lb/ft²).

    The five SAVR size bins (Albini) with output bin indices:

    .. code-block:: text

        SAVR >= 1200          → bin 0
        192  <= SAVR < 1200   → bin 1
        96   <= SAVR < 192    → bin 2
        48   <= SAVR < 96     → bin 3
        16   <= SAVR < 48     → bin 4
    """
    S_shape = savr_arr.shape[1:]
    BIN_BOUNDS = [(1200.0, None), (192.0, 1200.0), (96.0, 192.0), (48.0, 96.0), (16.0, 48.0)]

    # Accumulate fraction sum per bin: shape (5-bins, *S)
    summed = np.zeros((5,) + S_shape)
    for p_idx in range(5):
        sv = savr_arr[p_idx]   # (*S)
        fa = frac_arr[p_idx]   # (*S)
        for b_idx, (lo, hi) in enumerate(BIN_BOUNDS):
            if hi is None:
                mask = sv >= lo
            else:
                mask = (sv >= lo) & (sv < hi)
            summed[b_idx] += np.where(mask, fa, 0.0)

    # Map each particle's SAVR back to the corresponding summed bin fraction
    wfl = np.zeros(S_shape)
    for p_idx in range(5):
        sv = savr_arr[p_idx]
        assigned = np.zeros(S_shape)
        for b_idx, (lo, hi) in enumerate(BIN_BOUNDS):
            if hi is None:
                mask = sv >= lo
            else:
                mask = (sv >= lo) & (sv < hi)
            assigned += np.where(mask, summed[b_idx], 0.0)
        wfl += assigned * wn_arr[p_idx]
    return wfl

def build_particle_arrays(
        lut: dict,
        fuel_model_grid: Union[int, np.ndarray],
        m1h: Union[float, np.ndarray],
        m10h: Union[float, np.ndarray],
        m100h: Union[float, np.ndarray],
        mlh: Union[float, np.ndarray],
        mlw: Union[float, np.ndarray]
) -> dict:
    """
    Build (5, *S)-shaped particle arrays from fuel lookup tables and moisture grids.

    :param lut: Fuel property lookup dict from ``build_fuel_lookup_arrays()``.
    :param fuel_model_grid: Integer fuel model number array of shape (*S) or scalar.
    :param m1h: 1-hr dead fuel moisture as fraction (*S) or scalar (e.g. 0.06 = 6%).
    :param m10h: 10-hr dead fuel moisture as fraction (*S) or scalar.
    :param m100h: 100-hr dead fuel moisture as fraction (*S) or scalar.
    :param mlh: Live herbaceous fuel moisture as fraction (*S) or scalar.
    :param mlw: Live woody fuel moisture as fraction (*S) or scalar.
    :return: dict of arrays keyed by particle category:
        ``load_dead``, ``load_live``, ``savr_dead``, ``savr_live``,,
        ``moisture_dead``, ``moisture_live`` each of shape (5, *S);
        ``hoc_dead``, ``hoc_live``, ``depth``, ``moe_dead``, ``is_defined``
        each of shape (*S).
    """
    # --- Coerce to at-least-1D ndarray (G2 fix) ---
    fm    = np.atleast_1d(np.asarray(fuel_model_grid, dtype=np.int32))
    m1h   = np.atleast_1d(np.asarray(m1h,   dtype=float))
    m10h  = np.atleast_1d(np.asarray(m10h,  dtype=float))
    m100h = np.atleast_1d(np.asarray(m100h, dtype=float))
    mlh   = np.atleast_1d(np.asarray(mlh,   dtype=float))
    mlw   = np.atleast_1d(np.asarray(mlw,   dtype=float))
    S = fm.shape        # spatial shape — always at least (1,) after atleast_1d

    # Clamp fuel model numbers to valid range
    fm_safe = np.clip(fm, 0, lut['depth'].shape[0] - 1)

    # --- Fuel loads — shape (5, *S) ---
    load_dead = np.stack([
        lut['dead_1h'][fm_safe],      # particle 0: 1-hr dead
        lut['dead_10h'][fm_safe],     # particle 1: 10-hr dead
        lut['dead_100h'][fm_safe],    # particle 2: 100-hr dead
        np.zeros(S),                  # particle 3: dynamic transfer (filled below)
        np.zeros(S),                  # particle 4: unused
    ])
    load_live = np.stack([
        lut['live_herb'][fm_safe],    # particle 0: live herb
        lut['live_woody'][fm_safe],   # particle 1: live woody
        np.zeros(S), np.zeros(S), np.zeros(S),
    ])

    # --- SAVR — shape (5, *S) ---
    # 10-hr SAVR = 109, 100-hr SAVR = 30 are physical constants (G2)
    savr_dead = np.stack([
        lut['savr_1h'][fm_safe],
        np.full(S, 109.0),
        np.full(S, 30.0),
        lut['savr_lh'][fm_safe],      # transferred herb uses live-herb SAVR
        np.zeros(S),
    ])
    savr_live = np.stack([
        lut['savr_lh'][fm_safe],
        lut['savr_lw'][fm_safe],
        np.zeros(S), np.zeros(S), np.zeros(S),
    ])

    # --- Moistures — shape (5, *S); clamp to 0.01 minimum ---
    m1h_   = np.maximum(m1h,   0.01)
    m10h_  = np.maximum(m10h,  0.01)
    m100h_ = np.maximum(m100h, 0.01)
    mlh_   = np.maximum(mlh,   0.01)
    mlw_   = np.maximum(mlw,   0.01)

    moisture_dead = np.stack([m1h_, m10h_, m100h_, m1h_, np.zeros(S)])
    moisture_live = np.stack([mlh_, mlw_, np.zeros(S), np.zeros(S), np.zeros(S)])

    # --- Dynamic load transfer (G3) ---
    is_dyn    = lut['is_dynamic'][fm_safe]   # bool (*S)
    m_herb    = mlh_                          # (*S)
    full_xfer = is_dyn & (m_herb < 0.30)
    part_xfer = is_dyn & (m_herb >= 0.30) & (m_herb <= 1.20)

    transferred = np.where(
        part_xfer,
        load_live[0] * (1.333 - 1.11 * m_herb),
        0.0
    )
    load_dead[3] = np.where(full_xfer, load_live[0], transferred)
    load_live[0] = np.where(
        full_xfer, 0.0,
        np.where(part_xfer, load_live[0] - transferred, load_live[0])
    )

    return {
        'load_dead':    load_dead,      # (5, *S) — lb/ft²
        'load_live':    load_live,      # (5, *S) — lb/ft²
        'savr_dead':    savr_dead,      # (5, *S) — ft²/ft³
        'savr_live':    savr_live,      # (5, *S) — ft²/ft³
        'moisture_dead': moisture_dead, # (5, *S) — fraction
        'moisture_live': moisture_live, # (5, *S) — fraction
        # Per-cell scalars (*S):
        'hoc_dead':   lut['hoc_dead'][fm_safe],   # BTU/lb
        'hoc_live':   lut['hoc_live'][fm_safe],   # BTU/lb
        'depth':      lut['depth'][fm_safe],       # ft
        'moe_dead':   lut['moe_dead'][fm_safe],    # fraction
        'is_defined': lut['is_defined'][fm_safe],  # bool
    }

def calculate_backing_spread_rate(
        spread_rate: Union[float, np.ndarray],
        eccentricity: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate the backing spread rate (ft/min)."""
    safe_ecc_denom = 1.0 + eccentricity
    return np.where(
        safe_ecc_denom > 1e-7,
        spread_rate * (1.0 - eccentricity) / safe_ecc_denom,
        0.0,
    )

def calculate_direction_of_max_spread(
        x_component: Union[float, np.ndarray],
        y_component: Union[float, np.ndarray],
        rate_vector: Union[float, np.ndarray],
        wind_orientation_mode: str,
        aspect: Union[float, np.ndarray] = 0.0,
) -> np.ndarray:
    """Calculate direction of maximum spread (degrees)."""
    aspect = np.asarray(aspect, dtype=float)
    dir_deg = np.where(rate_vector > 1e-7, np.degrees(np.arctan2(y_component, x_component)), 0.0)
    dir_deg = np.where(np.abs(dir_deg) < 0.5, 0.0, dir_deg)
    dir_deg = np.where(dir_deg < -1e-20, dir_deg + 360.0, dir_deg)
    if 'north' in str(wind_orientation_mode).lower():
        dir_deg = (dir_deg + aspect + 180.0) % 360.0
    return dir_deg

def calculate_eccentricity(
        fire_length_to_width_ratio: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate ellipse eccentricity from the fire length-to-width ratio."""
    x_ecc = fire_length_to_width_ratio ** 2 - 1.0
    return np.where(
        x_ecc > 0,
        np.sqrt(np.maximum(x_ecc, 0.0)) / fire_length_to_width_ratio,
        0.0,
    )

def calculate_effective_wind_speed(
        spread_rate: Union[float, np.ndarray],
        no_wind_no_slope_spread_rate: Union[float, np.ndarray],
        wind_b: Union[float, np.ndarray],
        wind_c: Union[float, np.ndarray],
        relative_packing_ratio: Union[float, np.ndarray],
        wind_e: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate effective wind speed (mph) used for fire shape."""
    _, speed_from_base, _ = _load_surface_unit_converters()
    safe_r0 = np.where(no_wind_no_slope_spread_rate > 1e-7, no_wind_no_slope_spread_rate, 1.0)
    phi_eff = np.where(
        no_wind_no_slope_spread_rate > 1e-7,
        spread_rate / safe_r0 - 1.0,
        0.0,
    )
    safe_rpr = np.where(relative_packing_ratio > 1e-7, relative_packing_ratio, 1.0)
    safe_wc = np.where(wind_c > 1e-7, wind_c, 1.0)
    safe_wb = np.where(wind_b > 1e-7, wind_b, 1.0)
    ews_fpm = np.where(
        (phi_eff > 0) & (wind_b > 1e-7) & (wind_c > 1e-7) & (relative_packing_ratio > 1e-7),
        ((phi_eff * (safe_rpr ** wind_e)) / safe_wc) ** (1.0 / safe_wb),
        0.0,
    )
    return speed_from_base(ews_fpm, 5)

# ---------------------------------------------------------------------------
# V5-ext — Fire area and perimeter (§13.2 Gap B)
# ---------------------------------------------------------------------------

def calculate_fire_area(
        forward_ros: Union[float, np.ndarray],
        backing_ros: Union[float, np.ndarray],
        lwr: Union[float, np.ndarray],
        elapsed_min: float,
        is_crown: bool = False
) -> np.ndarray:
    """
    Calculate elliptical fire area.

    :param forward_ros: Forward rate of spread (ft/min) (*S).
    :param backing_ros: Backing rate of spread (ft/min) (*S).
    :param lwr: Fire length-to-width ratio (dimensionless) (*S).
    :param elapsed_min: Elapsed time (minutes) — scalar float.
    :param is_crown: If ``True``, use circular crown fire approximation.
        Defaults to ``False``.
    :return: (*S) ndarray — fire area (ft²).
    """
    fros = np.asarray(forward_ros, dtype=float)
    bros = np.asarray(backing_ros, dtype=float)
    lwr  = np.asarray(lwr,         dtype=float)
    safe_lwr = np.where(lwr > 1e-7, lwr, 1.0)

    if is_crown:
        # Crown fire: circular approximation using forward spread only
        d = fros * elapsed_min
        return np.where(lwr > 1e-7, np.pi * d ** 2 / (4.0 * safe_lwr), 0.0)
    else:
        # Surface fire: ellipse area = π × a × b
        ell_b = (fros + bros) / 2.0 * elapsed_min
        ell_a = np.where(lwr > 1e-7, ell_b / safe_lwr, 0.0)
        return np.pi * ell_a * ell_b

def calculate_fire_length(
        forward_ros: Union[float, np.ndarray],
        backing_ros: Union[float, np.ndarray],
        elapsed_min: float
) -> np.ndarray:
    """
    Calculate fire ellipse length (major axis × 2).

    :param forward_ros: Forward rate of spread (ft/min).
    :param backing_ros: Backing rate of spread (ft/min).
    :param elapsed_min: Elapsed time (minutes).
    :return: Fire length (ft).
    """
    return (np.asarray(forward_ros) + np.asarray(backing_ros)) * elapsed_min

def calculate_fire_length_to_width_ratio(
        effective_wind_speed: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate the surface fire length-to-width ratio."""
    return np.where(
        effective_wind_speed > 1e-7,
        np.minimum(
            0.936 * np.exp(0.1147 * effective_wind_speed) +
            0.461 * np.exp(-0.0692 * effective_wind_speed) - 0.397,
            8.0,
        ),
        1.0,
    )

def calculate_fire_perimeter(
        forward_ros: Union[float, np.ndarray],
        backing_ros: Union[float, np.ndarray],
        lwr: Union[float, np.ndarray],
        elapsed_min: float,
        is_crown: bool = False
) -> np.ndarray:
    """
    Calculate elliptical fire perimeter using Ramanujan's approximation.

    :param forward_ros: Forward rate of spread (ft/min) (*S).
    :param backing_ros: Backing rate of spread (ft/min) (*S).
    :param lwr: Fire length-to-width ratio (dimensionless) (*S).
    :param elapsed_min: Elapsed time (minutes) — scalar float.
    :param is_crown: If ``True``, use circular crown fire approximation.
        Defaults to ``False``.
    :return: (*S) ndarray — fire perimeter (ft).
    """
    fros = np.asarray(forward_ros, dtype=float)
    bros = np.asarray(backing_ros, dtype=float)
    lwr  = np.asarray(lwr,         dtype=float)
    safe_lwr = np.where(lwr > 1e-7, lwr, 1.0)

    if is_crown:
        # Crown fire: semi-circle perimeter approximation
        d = fros * elapsed_min
        return np.where(lwr > 1e-7, 0.5 * np.pi * d * (1.0 + 1.0 / safe_lwr), 0.0)
    else:
        # Surface fire: Ramanujan ellipse perimeter approximation
        ell_b = (fros + bros) / 2.0 * elapsed_min
        ell_a = np.where(lwr > 1e-7, ell_b / safe_lwr, 0.0)
        apb   = ell_a + ell_b
        safe_apb2 = np.where(apb > 1e-7, apb ** 2, 1.0)
        h = np.where(apb > 1e-7, (ell_a - ell_b) ** 2 / safe_apb2, 0.0)
        return np.where(
            apb > 1e-7,
            np.pi * apb * (1.0 + h / 4.0 + h ** 2 / 64.0),
            0.0
        )

def calculate_fire_width(
        forward_ros: Union[float, np.ndarray],
        backing_ros: Union[float, np.ndarray],
        lwr: Union[float, np.ndarray],
        elapsed_min: float
) -> np.ndarray:
    """
    Calculate fire ellipse width (minor axis × 2).

    :param forward_ros: Forward rate of spread (ft/min).
    :param backing_ros: Backing rate of spread (ft/min).
    :param lwr: Fire length-to-width ratio (dimensionless).
    :param elapsed_min: Elapsed time (minutes).
    :return: Fire width (ft).
    """
    fros = np.asarray(forward_ros, dtype=float)
    bros = np.asarray(backing_ros, dtype=float)
    lwr  = np.asarray(lwr,         dtype=float)
    length = (fros + bros) * elapsed_min
    safe_lwr = np.where(lwr > 1e-7, lwr, 1.0)
    return np.where(lwr > 1e-7, length / safe_lwr, 0.0)

def calculate_fireline_intensity(
        spread_rate: Union[float, np.ndarray],
        heat_per_unit_area: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate fireline intensity (BTU/ft/s)."""
    return np.asarray(spread_rate, dtype=float) * np.asarray(heat_per_unit_area, dtype=float) / 60.0

def calculate_flame_length(
        fireline_intensity: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate flame length (ft)."""
    safe_fli = np.where(fireline_intensity > 0, fireline_intensity, 0.0)
    return np.where(fireline_intensity > 0, 0.45 * (safe_fli ** 0.46), 0.0)

def calculate_flanking_spread_rate(
        spread_rate: Union[float, np.ndarray],
        backing_spread_rate: Union[float, np.ndarray],
        fire_length_to_width_ratio: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate the flanking spread rate (ft/min)."""
    safe_lwr = np.where(fire_length_to_width_ratio > 1e-7, fire_length_to_width_ratio, 1.0)
    return np.where(
        fire_length_to_width_ratio > 1e-7,
        (spread_rate + backing_spread_rate) / (2.0 * safe_lwr),
        0.0,
    )

def calculate_forward_spread_rate(
        no_wind_no_slope_spread_rate: Union[float, np.ndarray],
        rate_vector: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate the forward surface spread rate (ft/min)."""
    return np.asarray(no_wind_no_slope_spread_rate, dtype=float) + np.asarray(rate_vector, dtype=float)

# ---------------------------------------------------------------------------
# V3 (continued) — Fuelbed intermediates
# ---------------------------------------------------------------------------

def calculate_fuelbed_intermediates(p: dict) -> dict:
    """
    Compute per-cell fuelbed intermediate quantities from particle arrays.

    :param p: Particle dict from ``build_particle_arrays()``.
    :return: dict of (*S) ndarrays with keys:
        ``sigma`` (characteristic SAVR, ft²/ft³),
        ``bulk_density`` (lb/ft³),
        ``packing_ratio`` (dimensionless),
        ``relative_packing_ratio`` (dimensionless),
        ``heat_sink`` (BTU/ft³),
        ``propagating_flux`` (dimensionless),
        ``w_heat_dead``, ``w_heat_live`` (weighted heat of combustion, BTU/lb),
        ``w_silica_dead``, ``w_silica_live`` (weighted silica content, fraction),
        ``w_moist_dead``, ``w_moist_live`` (weighted moisture, fraction),
        ``wfl_dead``, ``wfl_live`` (size-class weighted fuel load, lb/ft²),
        ``moe_dead``, ``moe_live`` (moisture of extinction, fraction),
        ``frac_dead``, ``frac_live`` (SA fractions, dimensionless),
        ``depth`` (ft),
        ``is_defined`` (bool).
    """
    FUEL_DENSITY = 32.0     # Albini constant (lb/ft³)
    SILICA_TOTAL = 0.0555   # total silica content (fraction)

    load_d  = p['load_dead']      # (5, *S)
    load_l  = p['load_live']      # (5, *S)
    savr_d  = p['savr_dead']      # (5, *S)
    savr_l  = p['savr_live']      # (5, *S)
    moist_d = p['moisture_dead']  # (5, *S)
    moist_l = p['moisture_live']  # (5, *S)

    # --- Surface areas: SA = load * savr / FUEL_DENSITY ---
    sa_dead = load_d * savr_d / FUEL_DENSITY   # (5, *S)
    sa_live = load_l * savr_l / FUEL_DENSITY   # (5, *S)

    tsa_dead  = sa_dead.sum(axis=0)            # (*S)
    tsa_live  = sa_live.sum(axis=0)            # (*S)
    tsa_total = tsa_dead + tsa_live            # (*S)

    # Fraction of total SA for each life state
    safe_tsa = np.where(tsa_total > 1e-7, tsa_total, 1.0)
    frac_dead = np.where(tsa_total > 1e-7, tsa_dead / safe_tsa, 0.0)
    frac_live = 1.0 - frac_dead

    # Per-particle fraction within each life state
    safe_d = np.where(tsa_dead > 1e-7, tsa_dead, 1.0)
    safe_l = np.where(tsa_live > 1e-7, tsa_live, 1.0)
    frac_p_dead = sa_dead / safe_d   # (5, *S)
    frac_p_live = sa_live / safe_l   # (5, *S)

    # --- Live MOE ---
    safe_savr_d = np.where(savr_d > 1e-7, savr_d, 1.0)
    safe_savr_l = np.where(savr_l > 1e-7, savr_l, 1.0)
    exp_d138 = np.where(savr_d > 1e-7, np.exp(-138.0 / safe_savr_d), 0.0)
    exp_l500 = np.where(savr_l > 1e-7, np.exp(-500.0 / safe_savr_l), 0.0)

    fine_dead   = (load_d * exp_d138).sum(axis=0)             # (*S)
    fine_dead_m = (load_d * exp_d138 * moist_d).sum(axis=0)   # (*S)
    fine_live   = (load_l * exp_l500).sum(axis=0)             # (*S)

    safe_fine_dead = np.where(fine_dead > 1e-7, fine_dead, 1.0)
    fine_dead_moist = np.where(fine_dead > 1e-7, fine_dead_m / safe_fine_dead, 0.0)

    safe_fine_live = np.where(fine_live > 1e-7, fine_live, 1.0)
    fdol = np.where(fine_live > 1e-7, fine_dead / safe_fine_live, 0.0)

    moe_dead = p['moe_dead']                              # (*S)
    safe_moe_dead = np.where(moe_dead > 1e-7, moe_dead, 1.0)
    moe_live_raw = 2.9 * fdol * (1.0 - fine_dead_moist / safe_moe_dead) - 0.226
    moe_live = np.maximum(moe_live_raw, moe_dead)

    # --- Characteristic SAVR (sigma) ---
    hoc_d_broad = p['hoc_dead'][np.newaxis]   # (1, *S) → broadcasts over (5, *S)
    hoc_l_broad = p['hoc_live'][np.newaxis]

    # Effective silica constants per particle (Albini)
    EFF_SIL_D = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
    EFF_SIL_L = np.array([0.01, 0.01, 0.00, 0.00, 0.00])
    # Reshape to (5, 1) so they broadcast over spatial dims
    eff_sil_d = EFF_SIL_D.reshape(5, *([1] * (load_d.ndim - 1)))
    eff_sil_l = EFF_SIL_L.reshape(5, *([1] * (load_l.ndim - 1)))

    w_heat_dead   = (frac_p_dead * hoc_d_broad).sum(axis=0)   # (*S)
    w_heat_live   = (frac_p_live * hoc_l_broad).sum(axis=0)
    w_silica_dead = (frac_p_dead * eff_sil_d).sum(axis=0)
    w_silica_live = (frac_p_live * eff_sil_l).sum(axis=0)
    w_moist_dead  = (frac_p_dead * moist_d).sum(axis=0)
    w_moist_live  = (frac_p_live * moist_l).sum(axis=0)
    w_savr_dead   = (frac_p_dead * savr_d).sum(axis=0)
    w_savr_live   = (frac_p_live * savr_l).sum(axis=0)

    sigma = frac_dead * w_savr_dead + frac_live * w_savr_live   # (*S)

    # --- Net fuel load (wn = load * (1 - total_silica)) ---
    wn_dead = load_d * (1.0 - SILICA_TOTAL)
    wn_live = load_l * (1.0 - SILICA_TOTAL)

    # --- Size-class weighted fuel loads ---
    wfl_dead = _size_sorted_wfl(savr_d, frac_p_dead, wn_dead)
    wfl_live = _size_sorted_wfl(savr_l, frac_p_live, wn_live)

    # --- Bulk density ---
    total_load = load_d.sum(axis=0) + load_l.sum(axis=0)
    depth       = p['depth']
    safe_depth  = np.where(depth > 1e-7, depth, 1.0)
    bulk_density = np.where(depth > 1e-7, total_load / safe_depth, 0.0)

    # --- Packing ratio ---
    safe_depth_broad = np.where(depth > 1e-7, depth, 1.0)[np.newaxis]
    packing_ratio = np.where(
        depth > 1e-7,
        (load_d / (safe_depth_broad * FUEL_DENSITY)).sum(axis=0) +
        (load_l / (safe_depth_broad * FUEL_DENSITY)).sum(axis=0),
        0.0
    )

    # --- Relative packing ratio ---
    safe_sigma = np.where(sigma > 1e-7, sigma, 1.0)
    opt     = np.where(sigma > 1e-7, 3.348 / (safe_sigma ** 0.8189), 1.0)
    safe_opt = np.where(opt > 1e-7, opt, 1.0)
    rpr = np.where(
        (sigma > 1e-7) & (opt > 1e-7),
        packing_ratio / safe_opt,
        0.0
    )

    # --- Heat sink ---
    qig_dead = 250.0 + 1116.0 * moist_d   # (5, *S) — BTU/lb
    qig_live = 250.0 + 1116.0 * moist_l   # (5, *S) — BTU/lb
    frac_d_b = frac_dead[np.newaxis]       # (1, *S)
    frac_l_b = frac_live[np.newaxis]       # (1, *S)
    exp_l138 = np.where(savr_l > 1e-7, np.exp(-138.0 / safe_savr_l), 0.0)
    heat_sink = (
        (frac_d_b * frac_p_dead * qig_dead * exp_d138).sum(axis=0) +
        (frac_l_b * frac_p_live * qig_live * exp_l138).sum(axis=0)
    ) * bulk_density

    # --- Propagating flux ---
    prop_flux = np.where(
        sigma > 1e-7,
        np.exp(
            (0.792 + 0.681 * np.sqrt(safe_sigma)) * (packing_ratio + 0.1)
        ) / (192.0 + 0.2595 * safe_sigma),
        0.0
    )

    return {
        'sigma':                  sigma,
        'bulk_density':           bulk_density,
        'packing_ratio':          packing_ratio,
        'relative_packing_ratio': rpr,
        'heat_sink':              heat_sink,
        'propagating_flux':       prop_flux,
        'w_heat_dead':            w_heat_dead,
        'w_heat_live':            w_heat_live,
        'w_silica_dead':          w_silica_dead,
        'w_silica_live':          w_silica_live,
        'w_moist_dead':           w_moist_dead,
        'w_moist_live':           w_moist_live,
        'wfl_dead':               wfl_dead,
        'wfl_live':               wfl_live,
        'moe_dead':               moe_dead,
        'moe_live':               moe_live,
        'frac_dead':              frac_dead,
        'frac_live':              frac_live,
        'depth':                  depth,
        'is_defined':             p['is_defined'],
    }

def calculate_heat_per_unit_area(
        reaction_intensity: Union[float, np.ndarray],
        residence_time: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate heat per unit area (BTU/ft^2)."""
    return np.asarray(reaction_intensity, dtype=float) * np.asarray(residence_time, dtype=float)

def calculate_midflame_wind_speed(
        waf: Union[float, np.ndarray],
        twenty_foot_wind_speed: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate the midflame wind speed (ft/min)."""
    return np.asarray(waf, dtype=float) * np.asarray(twenty_foot_wind_speed, dtype=float)

def calculate_no_wind_no_slope_spread_rate(
        ri: Union[float, np.ndarray],
        propagating_flux: Union[float, np.ndarray],
        heat_sink: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate the no-wind, no-slope spread rate (ft/min)."""
    safe_hs = np.where(heat_sink > 1e-7, heat_sink, 1.0)
    return np.where(heat_sink > 1e-7, ri * propagating_flux / safe_hs, 0.0)

# ---------------------------------------------------------------------------
# V4 — Reaction Intensity
# ---------------------------------------------------------------------------

def calculate_reaction_intensity(ib: dict) -> np.ndarray:
    """
    Compute reaction intensity from fuelbed intermediates.

    :param ib: Fuelbed intermediate dict from ``calculate_fuelbed_intermediates()``.
    :return: (*S) ndarray — reaction intensity (BTU/ft²/min).
    """
    sigma = ib['sigma']
    rpr   = ib['relative_packing_ratio']
    valid = (sigma > 1e-7) & (rpr > 0)

    safe_sigma = np.where(valid, sigma, 1.0)
    safe_rpr   = np.where(valid, rpr,   1.0)

    # Rothermel optimum reaction velocity coefficients
    aa        = 133.0 / (safe_sigma ** 0.7913)
    gamma_max = (safe_sigma ** 1.5) / (495.0 + 0.0594 * (safe_sigma ** 1.5))
    gamma     = gamma_max * (safe_rpr ** aa) * np.exp(aa * (1.0 - safe_rpr))

    def _eta_m(moisture, moe):
        """Moisture damping coefficient (0–1)."""
        rm = np.where(moe > 1e-7, moisture / np.where(moe > 1e-7, moe, 1.0), 1.0)
        extinguished = (moe < 1e-7) | (rm >= 1.0)
        return np.where(
            extinguished, 0.0,
            1.0 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3
        )

    def _eta_s(silica):
        """Mineral damping coefficient (0–1)."""
        safe_sil = np.where(silica > 1e-7, silica, 1.0)
        return np.where(silica > 1e-7, np.minimum(0.174 / (safe_sil ** 0.19), 1.0), 0.0)

    eta_m_dead = _eta_m(ib['w_moist_dead'], ib['moe_dead'])
    eta_m_live = _eta_m(ib['w_moist_live'], ib['moe_live'])
    eta_s_dead = _eta_s(ib['w_silica_dead'])
    eta_s_live = _eta_s(ib['w_silica_live'])

    # Reaction intensity = reaction velocity × net load × heat × moisture damping × mineral damping
    ri_dead = gamma * ib['wfl_dead'] * ib['w_heat_dead'] * eta_m_dead * eta_s_dead
    ri_live = gamma * ib['wfl_live'] * ib['w_heat_live'] * eta_m_live * eta_s_live

    return np.where(valid, ri_dead + ri_live, 0.0)

def calculate_reaction_intensity_output(
        reaction_intensity: Union[float, np.ndarray],
) -> np.ndarray:
    """Return reaction intensity as a named surface output."""
    return np.asarray(reaction_intensity, dtype=float)

def calculate_residence_time(
        sigma: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate residence time (min)."""
    safe_sigma = np.where(sigma > 1e-7, sigma, 1.0)
    return np.where(sigma > 1e-7, 384.0 / safe_sigma, 0.0)

def calculate_slope_phi(
        slope: Union[float, np.ndarray],
        slope_units: int,
        packing_ratio: Union[float, np.ndarray],
        reaction_intensity: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate the slope phi factor."""
    _, _, slope_to_base = _load_surface_unit_converters()
    slope_deg = slope_to_base(slope, slope_units)
    slope_tan = np.tan(np.radians(np.asarray(slope_deg, dtype=float)))
    safe_pr = np.where(packing_ratio > 1e-7, packing_ratio, 1.0)
    phi_s = np.where(
        (slope_tan > 1e-7) & (packing_ratio > 1e-7),
        5.275 * (safe_pr ** -0.3) * (slope_tan ** 2),
        0.0,
    )
    return np.minimum(phi_s, 0.9 * reaction_intensity)

def calculate_surface_rate_vector(
        no_wind_no_slope_spread_rate: Union[float, np.ndarray],
        wind_phi: Union[float, np.ndarray],
        slope_phi: Union[float, np.ndarray],
        wind_direction: Union[float, np.ndarray],
        wind_orientation_mode: str,
        aspect: Union[float, np.ndarray] = 0.0,
) -> dict:
    """Calculate the x/y spread vector components and resultant rate vector."""
    wind_direction = np.asarray(wind_direction, dtype=float)
    aspect = np.asarray(aspect, dtype=float)
    if 'north' in str(wind_orientation_mode).lower():
        corrected_wind_direction = wind_direction - aspect
    else:
        corrected_wind_direction = wind_direction
    wd_rad = np.radians(corrected_wind_direction)

    slope_rate = no_wind_no_slope_spread_rate * slope_phi
    wind_rate = no_wind_no_slope_spread_rate * wind_phi
    x_component = slope_rate + wind_rate * np.cos(wd_rad)
    y_component = wind_rate * np.sin(wd_rad)
    return {
        'x_component': x_component,
        'y_component': y_component,
        'rate_vector': np.sqrt(x_component ** 2 + y_component ** 2),
    }

def calculate_surface_wind_adjustment_factor(
        canopy_cover: Union[float, np.ndarray],
        canopy_height: Union[float, np.ndarray],
        crown_ratio: Union[float, np.ndarray],
        depth: Union[float, np.ndarray],
        waf_method: str = 'UseCrownRatio',
        user_waf: Union[float, np.ndarray, None] = None,
) -> np.ndarray:
    """Calculate the wind adjustment factor used to derive midflame wind speed."""
    if str(waf_method).lower().replace('_', '') == 'userinput' and user_waf is not None:
        return np.asarray(user_waf, dtype=float)
    return calculate_wind_adjustment_factor(canopy_cover, canopy_height, crown_ratio, depth)

def calculate_surface_wind_coefficients(
        sigma: Union[float, np.ndarray],
) -> dict:
    """Calculate Albini/Rothermel wind coefficients from sigma."""
    safe_sigma = np.where(sigma > 1e-7, sigma, 1.0)
    return {
        'safe_sigma': safe_sigma,
        'wind_c': 7.47 * np.exp(-0.133 * (safe_sigma ** 0.55)),
        'wind_b': 0.02526 * (safe_sigma ** 0.54),
        'wind_e': 0.715 * np.exp(-0.000359 * safe_sigma),
    }

def calculate_twenty_foot_wind_speed(
        wind_speed: Union[float, np.ndarray],
        wind_speed_units: int,
        wind_height_mode: str = 'TwentyFoot',
) -> np.ndarray:
    """Convert wind speed to 20-ft wind speed in ft/min."""
    speed_to_base, _, _ = _load_surface_unit_converters()
    ws_fpm = speed_to_base(wind_speed, wind_speed_units)
    if '10' in str(wind_height_mode).lower() or 'ten' in str(wind_height_mode).lower():
        ws_fpm = ws_fpm / 1.15
    return ws_fpm

# ---------------------------------------------------------------------------
# V5 — Wind adjustment factor
# ---------------------------------------------------------------------------

def calculate_wind_adjustment_factor(
        canopy_cover: Union[float, np.ndarray],
        canopy_height: Union[float, np.ndarray],
        crown_ratio: Union[float, np.ndarray],
        depth: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Compute the wind adjustment factor (WAF) for midflame wind speed.

    Converts 20-ft (or 10-m) open-wind speed to midflame wind speed.
    Uses an open-fuel-bed formula when canopy is absent or sparse, and
    a sheltered-canopy formula otherwise (Andrews 2012).

    :param canopy_cover: Canopy cover fraction (0–1) (*S) or scalar.
    :param canopy_height: Canopy height (ft) (*S) or scalar.
    :param crown_ratio: Crown ratio fraction (0–1) (*S) or scalar.
    :param depth: Fuel bed depth (ft) (*S) or scalar.
    :return: (*S) ndarray — wind adjustment factor (dimensionless, typically 0.1–1.0).
    """
    canopy_cover  = np.asarray(canopy_cover,  dtype=float)
    canopy_height = np.asarray(canopy_height, dtype=float)
    crown_ratio   = np.asarray(crown_ratio,   dtype=float)
    depth         = np.asarray(depth,         dtype=float)

    # Crown cover fraction = crown_ratio * canopy_cover / 3
    ccf = crown_ratio * canopy_cover / 3.0
    no_canopy = (canopy_cover < 1e-7) | (ccf < 0.05) | (canopy_height < 6.0)

    # Open WAF: 1.83 / ln((20 + 0.36*d) / (0.13*d))
    safe_depth = np.where(depth > 1e-7, depth, 1.0)
    waf_open = np.where(
        depth > 1e-7,
        1.83 / np.log((20.0 + 0.36 * safe_depth) / (0.13 * safe_depth)),
        1.0
    )

    # Canopy WAF: 0.555 / (sqrt(ccf * ch) * ln((20 + 0.36*ch)/(0.13*ch)))
    safe_ch  = np.where(canopy_height > 1e-7, canopy_height, 1.0)
    safe_ccf = np.where(ccf > 1e-7, ccf, 1.0)
    denom = np.sqrt(safe_ccf * safe_ch) * np.log(
        (20.0 + 0.36 * safe_ch) / (0.13 * safe_ch)
    )
    waf_canopy = np.where(denom > 1e-7, 0.555 / denom, 1.0)

    return np.where(no_canopy, waf_open, waf_canopy)

def calculate_wind_phi(
        sigma: Union[float, np.ndarray],
        midflame_wind_speed: Union[float, np.ndarray],
        wind_b: Union[float, np.ndarray],
        wind_c: Union[float, np.ndarray],
        relative_packing_ratio: Union[float, np.ndarray],
        wind_e: Union[float, np.ndarray],
) -> np.ndarray:
    """Calculate the wind phi factor."""
    safe_rpr = np.where(relative_packing_ratio > 1e-7, relative_packing_ratio, 1.0)
    return np.where(
        (sigma > 1e-7) & (midflame_wind_speed > 1e-7),
        (midflame_wind_speed ** wind_b) * wind_c * (safe_rpr ** (-wind_e)),
        0.0,
    )

def run_surface_fire(
        ri: Union[float, np.ndarray],
        ib: dict,
        wind_speed: Union[float, np.ndarray],
        wind_speed_units: int,
        wind_direction: Union[float, np.ndarray],
        wind_orientation_mode: str,
        slope: Union[float, np.ndarray],
        slope_units: int = 0,
        aspect: Union[float, np.ndarray] = 0.0,
        canopy_cover: Union[float, np.ndarray] = 0.0,
        canopy_height: Union[float, np.ndarray] = 0.0,
        crown_ratio: Union[float, np.ndarray] = 0.0,
        wind_height_mode: str = 'TwentyFoot',
        waf_method: str = 'UseCrownRatio',
        user_waf: Union[float, np.ndarray, None] = None
) -> dict:
    """
    Compute fire spread rate, fire shape, and intensity from reaction intensity
    and fuelbed intermediates (Rothermel 1972).

    :param ri: Reaction intensity array (*S) (BTU/ft²/min) from
        ``calculate_reaction_intensity()``.
    :param ib: Fuelbed intermediate dict from ``calculate_fuelbed_intermediates()``.
    :param wind_speed: Wind speed (*S) or scalar, in ``wind_speed_units``.
    :param wind_speed_units: Scalar integer ``SpeedUnitsEnum`` value.
    :param wind_direction: Wind direction in degrees (*S) or scalar.
        Interpretation depends on ``wind_orientation_mode``.
    :param wind_orientation_mode: ``'RelativeToUpslope'`` or ``'RelativeToNorth'``.
    :param slope: Slope (*S) or scalar, in the units given by ``slope_units``.
    :param slope_units: Scalar integer ``SlopeUnitsEnum`` value
        (0 = Degrees [default], 1 = Percent).
    :param aspect: Terrain aspect in degrees (*S) or scalar (0 = north, clockwise).
    :param canopy_cover: Canopy cover fraction (0–1) (*S) or scalar.
    :param canopy_height: Canopy height (ft) (*S) or scalar.
    :param crown_ratio: Crown ratio fraction (0–1) (*S) or scalar.
    :param wind_height_mode: ``'TwentyFoot'`` (default) or ``'TenMeter'``.
    :param waf_method: ``'UseCrownRatio'`` (default) or ``'UserInput'``.
    :param user_waf: User-supplied WAF (*S) or scalar, used when
        ``waf_method='UserInput'``. Ignored otherwise.
    :return: dict of (*S) ndarrays with keys:
        ``spread_rate`` (ft/min), ``backing_spread_rate`` (ft/min),
        ``flanking_spread_rate`` (ft/min), ``flame_length`` (ft),
        ``fireline_intensity`` (BTU/ft/s), ``heat_per_unit_area`` (BTU/ft²),
        ``effective_wind_speed`` (mph), ``fire_length_to_width_ratio``
        (dimensionless), ``eccentricity`` (dimensionless),
        ``direction_of_max_spread`` (degrees), ``residence_time`` (min),
        ``reaction_intensity`` (BTU/ft²/min), ``midflame_wind_speed`` (ft/min),
        ``no_wind_no_slope_spread_rate`` (ft/min).
    """
    sigma = ib['sigma']
    wind_coeffs = calculate_surface_wind_coefficients(sigma)
    no_wind_no_slope_spread_rate = calculate_no_wind_no_slope_spread_rate(
        ri=ri,
        propagating_flux=ib['propagating_flux'],
        heat_sink=ib['heat_sink'],
    )
    twenty_foot_wind_speed = calculate_twenty_foot_wind_speed(
        wind_speed=wind_speed,
        wind_speed_units=wind_speed_units,
        wind_height_mode=wind_height_mode,
    )
    waf = calculate_surface_wind_adjustment_factor(
        canopy_cover=canopy_cover,
        canopy_height=canopy_height,
        crown_ratio=crown_ratio,
        depth=ib['depth'],
        waf_method=waf_method,
        user_waf=user_waf,
    )
    midflame_wind_speed = calculate_midflame_wind_speed(
        waf=waf,
        twenty_foot_wind_speed=twenty_foot_wind_speed,
    )
    wind_phi = calculate_wind_phi(
        sigma=sigma,
        midflame_wind_speed=midflame_wind_speed,
        wind_b=wind_coeffs['wind_b'],
        wind_c=wind_coeffs['wind_c'],
        relative_packing_ratio=ib['relative_packing_ratio'],
        wind_e=wind_coeffs['wind_e'],
    )
    slope_phi = calculate_slope_phi(
        slope=slope,
        slope_units=slope_units,
        packing_ratio=ib['packing_ratio'],
        reaction_intensity=ri,
    )
    rate_components = calculate_surface_rate_vector(
        no_wind_no_slope_spread_rate=no_wind_no_slope_spread_rate,
        wind_phi=wind_phi,
        slope_phi=slope_phi,
        wind_direction=wind_direction,
        wind_orientation_mode=wind_orientation_mode,
        aspect=aspect,
    )
    spread_rate = calculate_forward_spread_rate(
        no_wind_no_slope_spread_rate=no_wind_no_slope_spread_rate,
        rate_vector=rate_components['rate_vector'],
    )
    effective_wind_speed = calculate_effective_wind_speed(
        spread_rate=spread_rate,
        no_wind_no_slope_spread_rate=no_wind_no_slope_spread_rate,
        wind_b=wind_coeffs['wind_b'],
        wind_c=wind_coeffs['wind_c'],
        relative_packing_ratio=ib['relative_packing_ratio'],
        wind_e=wind_coeffs['wind_e'],
    )
    fire_length_to_width_ratio = calculate_fire_length_to_width_ratio(effective_wind_speed)
    eccentricity = calculate_eccentricity(fire_length_to_width_ratio)
    backing_spread_rate = calculate_backing_spread_rate(spread_rate, eccentricity)
    flanking_spread_rate = calculate_flanking_spread_rate(
        spread_rate=spread_rate,
        backing_spread_rate=backing_spread_rate,
        fire_length_to_width_ratio=fire_length_to_width_ratio,
    )
    residence_time = calculate_residence_time(sigma)
    heat_per_unit_area = calculate_heat_per_unit_area(ri, residence_time)
    fireline_intensity = calculate_fireline_intensity(spread_rate, heat_per_unit_area)
    flame_length = calculate_flame_length(fireline_intensity)
    direction_of_max_spread = calculate_direction_of_max_spread(
        x_component=rate_components['x_component'],
        y_component=rate_components['y_component'],
        rate_vector=rate_components['rate_vector'],
        wind_orientation_mode=wind_orientation_mode,
        aspect=aspect,
    )
    reaction_intensity = calculate_reaction_intensity_output(ri)

    defined = ib['is_defined']
    zero = np.zeros_like(spread_rate)
    return {
        'spread_rate': _mask_surface_result(spread_rate, defined, zero),
        'backing_spread_rate': _mask_surface_result(backing_spread_rate, defined, zero),
        'flanking_spread_rate': _mask_surface_result(flanking_spread_rate, defined, zero),
        'flame_length': _mask_surface_result(flame_length, defined, zero),
        'fireline_intensity': _mask_surface_result(fireline_intensity, defined, zero),
        'heat_per_unit_area': _mask_surface_result(heat_per_unit_area, defined, zero),
        'effective_wind_speed': _mask_surface_result(effective_wind_speed, defined, zero),
        'fire_length_to_width_ratio': _mask_surface_result(
            fire_length_to_width_ratio,
            defined,
            np.ones_like(fire_length_to_width_ratio),
        ),
        'eccentricity': _mask_surface_result(eccentricity, defined, zero),
        'direction_of_max_spread': _mask_surface_result(direction_of_max_spread, defined, zero),
        'residence_time': _mask_surface_result(residence_time, defined, zero),
        'reaction_intensity': _mask_surface_result(reaction_intensity, defined, zero),
        'midflame_wind_speed': _mask_surface_result(midflame_wind_speed, defined, zero),
        'no_wind_no_slope_spread_rate': _mask_surface_result(
            no_wind_no_slope_spread_rate,
            defined,
            zero,
        ),
    }
