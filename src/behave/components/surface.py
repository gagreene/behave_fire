"""
surface_array.py — Vectorized Surface Fire Pipeline (V3 / V4 / V5 / fire-size)

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

calculate_spread_rate(ri, ib, wind_speed, wind_speed_units, wind_direction,
                      wind_orientation_mode, slope_deg, aspect,
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
    - slope_deg:   DEGREES  (not percent) — convert before calling
    - wind_speed:  in the units specified by wind_speed_units
    - canopy_*:    canopy_cover & crown_ratio as fraction (0–1); heights in feet
"""

import numpy as np

# ---------------------------------------------------------------------------
# V3 — Particle array construction
# ---------------------------------------------------------------------------

def build_particle_arrays(lut, fuel_model_grid, m1h, m10h, m100h, mlh, mlw):
    """
    Build (5, *S)-shaped particle arrays from fuel lookup tables and moisture grids.

    Parameters
    ----------
    lut            : dict from build_fuel_lookup_arrays()
    fuel_model_grid: int array (*S) or scalar
    m1h .. mlw     : float arrays (*S) or scalar — moisture fractions

    Returns
    -------
    dict of arrays, each shape (5, *S) unless noted:
        'load_dead', 'load_live', 'savr_dead', 'savr_live',
        'moisture_dead', 'moisture_live',
        'hoc_dead', 'hoc_live', 'depth', 'moe_dead', 'is_defined'
        (the last five are (*S), not (5, *S))
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
        'load_dead':    load_dead,     # (5, *S)
        'load_live':    load_live,     # (5, *S)
        'savr_dead':    savr_dead,     # (5, *S)
        'savr_live':    savr_live,     # (5, *S)
        'moisture_dead': moisture_dead, # (5, *S)
        'moisture_live': moisture_live, # (5, *S)
        # Per-cell scalars ((*S)):
        'hoc_dead':   lut['hoc_dead'][fm_safe],
        'hoc_live':   lut['hoc_live'][fm_safe],
        'depth':      lut['depth'][fm_safe],
        'moe_dead':   lut['moe_dead'][fm_safe],
        'is_defined': lut['is_defined'][fm_safe],
    }


# ---------------------------------------------------------------------------
# V3 (continued) — Fuelbed intermediates
# ---------------------------------------------------------------------------

def calculate_fuelbed_intermediates(p):
    """
    Compute per-cell fuelbed intermediate quantities.

    Parameters
    ----------
    p : dict from build_particle_arrays()

    Returns
    -------
    dict of (*S) arrays (and one (5, *S) for savr_dead/savr_live indirectly
    used in V4).  Keys include:
        sigma, bulk_density, packing_ratio, relative_packing_ratio,
        heat_sink, propagating_flux,
        w_heat_dead, w_heat_live, w_silica_dead, w_silica_live,
        w_moist_dead, w_moist_live,
        wfl_dead, wfl_live,
        moe_dead, moe_live, frac_dead, frac_live,
        depth, is_defined
    """
    FUEL_DENSITY = 32.0     # Albini constant
    SILICA_TOTAL = 0.0555   # total silica content

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
    qig_dead = 250.0 + 1116.0 * moist_d   # (5, *S)
    qig_live = 250.0 + 1116.0 * moist_l   # (5, *S)
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


def _size_sorted_wfl(savr_arr, frac_arr, wn_arr):
    """
    Compute size-class weighted fuel load.

    savr_arr, frac_arr, wn_arr : shape (5, *S)
    Returns (*S) array.

    The five SAVR size bins (Albini) with output bin indices:
        SAVR >= 1200  → bin 0
        192 <= SAVR < 1200 → bin 1
        96  <= SAVR < 192  → bin 2
        48  <= SAVR < 96   → bin 3
        16  <= SAVR < 48   → bin 4
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


# ---------------------------------------------------------------------------
# V4 — Reaction Intensity
# ---------------------------------------------------------------------------

def calculate_reaction_intensity(ib):
    """
    Compute reaction intensity from fuelbed intermediates.

    Parameters
    ----------
    ib : dict from calculate_fuelbed_intermediates()

    Returns
    -------
    (*S) ndarray — reaction intensity in BTU/ft²/min
    """
    sigma = ib['sigma']
    rpr   = ib['relative_packing_ratio']
    valid = (sigma > 1e-7) & (rpr > 0)

    safe_sigma = np.where(valid, sigma, 1.0)
    safe_rpr   = np.where(valid, rpr,   1.0)

    aa        = 133.0 / (safe_sigma ** 0.7913)
    gamma_max = (safe_sigma ** 1.5) / (495.0 + 0.0594 * (safe_sigma ** 1.5))
    gamma     = gamma_max * (safe_rpr ** aa) * np.exp(aa * (1.0 - safe_rpr))

    def _eta_m(moisture, moe):
        rm = np.where(moe > 1e-7, moisture / np.where(moe > 1e-7, moe, 1.0), 1.0)
        extinguished = (moe < 1e-7) | (rm >= 1.0)
        return np.where(
            extinguished, 0.0,
            1.0 - 2.59 * rm + 5.11 * rm ** 2 - 3.52 * rm ** 3
        )

    def _eta_s(silica):
        safe_sil = np.where(silica > 1e-7, silica, 1.0)
        return np.where(silica > 1e-7, np.minimum(0.174 / (safe_sil ** 0.19), 1.0), 0.0)

    eta_m_dead = _eta_m(ib['w_moist_dead'], ib['moe_dead'])
    eta_m_live = _eta_m(ib['w_moist_live'], ib['moe_live'])
    eta_s_dead = _eta_s(ib['w_silica_dead'])
    eta_s_live = _eta_s(ib['w_silica_live'])

    ri_dead = gamma * ib['wfl_dead'] * ib['w_heat_dead'] * eta_m_dead * eta_s_dead
    ri_live = gamma * ib['wfl_live'] * ib['w_heat_live'] * eta_m_live * eta_s_live

    return np.where(valid, ri_dead + ri_live, 0.0)


# ---------------------------------------------------------------------------
# V5 — Wind adjustment factor
# ---------------------------------------------------------------------------

def calculate_wind_adjustment_factor(canopy_cover, canopy_height, crown_ratio, depth):
    """
    Compute wind adjustment factor (WAF) for midflame wind speed.

    All inputs are (*S) arrays or scalars.  Returns (*S) ndarray.
    """
    canopy_cover  = np.asarray(canopy_cover,  dtype=float)
    canopy_height = np.asarray(canopy_height, dtype=float)
    crown_ratio   = np.asarray(crown_ratio,   dtype=float)
    depth         = np.asarray(depth,         dtype=float)

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


# ---------------------------------------------------------------------------
# V5 — Full spread rate and fire shape pipeline
# ---------------------------------------------------------------------------

def calculate_spread_rate(ri, ib,
                           wind_speed, wind_speed_units,
                           wind_direction, wind_orientation_mode,
                           slope_deg, aspect,
                           canopy_cover, canopy_height, crown_ratio,
                           wind_height_mode='TwentyFoot',
                           waf_method='UseCrownRatio',
                           user_waf=None):
    """
    Compute fire spread rate, shape, and intensity from reaction intensity and
    fuelbed intermediates.

    Parameters
    ----------
    ri                  : (*S) array — reaction intensity (BTU/ft²/min)
    ib                  : dict from calculate_fuelbed_intermediates()
    wind_speed          : (*S) or scalar — wind speed in wind_speed_units
    wind_speed_units    : scalar int — SpeedUnits enum value
    wind_direction      : (*S) or scalar — wind direction in degrees
    wind_orientation_mode : str — 'RelativeToUpslope' or 'RelativeToNorth'
    slope_deg           : (*S) or scalar — slope in DEGREES (not percent)
    aspect              : (*S) or scalar — aspect in degrees (0 = north, clockwise)
    canopy_cover        : (*S) or scalar — fraction (0–1)
    canopy_height       : (*S) or scalar — feet
    crown_ratio         : (*S) or scalar — fraction (0–1)
    wind_height_mode    : str — 'TwentyFoot' or 'TenMeter'
    waf_method          : str — 'UseCrownRatio' or 'UserInput'
    user_waf            : (*S) or scalar or None — WAF when waf_method='UserInput'

    Returns
    -------
    dict of (*S) arrays with keys:
        spread_rate, backing_spread_rate, flanking_spread_rate,
        flame_length, fireline_intensity, heat_per_unit_area,
        effective_wind_speed, fire_length_to_width_ratio, eccentricity,
        direction_of_max_spread, residence_time, reaction_intensity

    IMPORTANT — slope_deg must be in DEGREES.
    Convert percent slope before calling:
        slope_deg = np.degrees(np.arctan(slope_pct / 100.0))
    """
    try:
        from .behave_units import speed_to_base, speed_from_base
    except ImportError:
        from behave_units import speed_to_base, speed_from_base

    sigma     = ib['sigma']
    rpr       = ib['relative_packing_ratio']
    prop_flux = ib['propagating_flux']
    heat_sink = ib['heat_sink']

    # No-wind no-slope rate of spread
    safe_hs = np.where(heat_sink > 1e-7, heat_sink, 1.0)
    r0 = np.where(heat_sink > 1e-7, ri * prop_flux / safe_hs, 0.0)

    # Wind coefficients from Albini / Rothermel (per sigma)
    safe_sigma = np.where(sigma > 1e-7, sigma, 1.0)
    wind_c = 7.47   * np.exp(-0.133   * (safe_sigma ** 0.55))
    wind_b = 0.02526 * (safe_sigma ** 0.54)
    wind_e = 0.715  * np.exp(-0.000359 * safe_sigma)

    # Convert wind speed to ft/min
    ws_fpm = speed_to_base(wind_speed, wind_speed_units)
    if '10' in str(wind_height_mode).lower() or 'ten' in str(wind_height_mode).lower():
        ws_fpm = ws_fpm / 1.15

    # WAF
    if str(waf_method).lower().replace('_', '') == 'userinput' and user_waf is not None:
        waf = np.asarray(user_waf, dtype=float)
    else:
        waf = calculate_wind_adjustment_factor(
            canopy_cover, canopy_height, crown_ratio, ib['depth']
        )

    midflame_ws = waf * ws_fpm

    # Wind phi factor
    safe_rpr = np.where(rpr > 1e-7, rpr, 1.0)
    phi_w = np.where(
        (sigma > 1e-7) & (midflame_ws > 1e-7),
        (midflame_ws ** wind_b) * wind_c * (safe_rpr ** (-wind_e)),
        0.0
    )

    # Slope phi factor (slope_deg already in degrees)
    slope_tan = np.tan(np.radians(np.asarray(slope_deg, dtype=float)))
    pr = ib['packing_ratio']
    safe_pr = np.where(pr > 1e-7, pr, 1.0)
    phi_s = np.where(
        (slope_tan > 1e-7) & (pr > 1e-7),
        5.275 * (safe_pr ** -0.3) * (slope_tan ** 2),
        0.0
    )
    # Cap phi_s at wind-speed limit
    wind_speed_limit = 0.9 * ri
    phi_s = np.minimum(phi_s, wind_speed_limit)

    # Direction of max spread via vector composition
    wind_direction = np.asarray(wind_direction, dtype=float)
    aspect         = np.asarray(aspect,         dtype=float)
    if 'north' in str(wind_orientation_mode).lower():
        corrected_wd = wind_direction - aspect
    else:
        corrected_wd = wind_direction
    wd_rad = np.radians(corrected_wd)

    slope_rate = r0 * phi_s
    wind_rate  = r0 * phi_w
    x = slope_rate + wind_rate * np.cos(wd_rad)
    y = wind_rate * np.sin(wd_rad)
    rate_vector          = np.sqrt(x ** 2 + y ** 2)
    forward_spread_rate  = r0 + rate_vector

    # Effective wind speed (mph) for fire shape
    safe_r0 = np.where(r0 > 1e-7, r0, 1.0)
    phi_eff = np.where(r0 > 1e-7, forward_spread_rate / safe_r0 - 1.0, 0.0)
    safe_wc = np.where(wind_c > 1e-7, wind_c, 1.0)
    safe_wb = np.where(wind_b > 1e-7, wind_b, 1.0)
    ews_fpm = np.where(
        (phi_eff > 0) & (wind_b > 1e-7) & (wind_c > 1e-7) & (rpr > 1e-7),
        ((phi_eff * (safe_rpr ** wind_e)) / safe_wc) ** (1.0 / safe_wb),
        0.0
    )
    ews_mph = speed_from_base(ews_fpm, 5)   # 5 = MilesPerHour

    # Fire L:W ratio (surface formula — crown uses different formula in crown_array.py)
    lwr = np.where(
        ews_mph > 1e-7,
        np.minimum(
            0.936 * np.exp(0.1147 * ews_mph) + 0.461 * np.exp(-0.0692 * ews_mph) - 0.397,
            8.0
        ),
        1.0
    )

    # Eccentricity
    x_ecc = lwr ** 2 - 1.0
    ecc = np.where(x_ecc > 0, np.sqrt(np.maximum(x_ecc, 0.0)) / lwr, 0.0)

    # Backing and flanking rates
    safe_ecc_denom = 1.0 + ecc
    backing  = np.where(
        safe_ecc_denom > 1e-7,
        forward_spread_rate * (1.0 - ecc) / safe_ecc_denom,
        0.0
    )
    safe_lwr = np.where(lwr > 1e-7, lwr, 1.0)
    flanking = np.where(
        lwr > 1e-7,
        (forward_spread_rate + backing) / (2.0 * safe_lwr),
        0.0
    )

    # Heat per unit area and flame length
    residence_time     = np.where(sigma > 1e-7, 384.0 / safe_sigma, 0.0)
    hpua               = ri * residence_time
    fireline_intensity = forward_spread_rate * hpua / 60.0
    safe_fli = np.where(fireline_intensity > 0, fireline_intensity, 0.0)
    flame_length = np.where(fireline_intensity > 0, 0.45 * (safe_fli ** 0.46), 0.0)

    # Direction of max spread
    dir_deg = np.where(rate_vector > 1e-7, np.degrees(np.arctan2(y, x)), 0.0)
    dir_deg = np.where(np.abs(dir_deg) < 0.5, 0.0, dir_deg)
    dir_deg = np.where(dir_deg < -1e-20, dir_deg + 360.0, dir_deg)
    if 'north' in str(wind_orientation_mode).lower():
        dir_deg = (dir_deg + aspect + 180.0) % 360.0

    # Mask undefined fuel model cells
    defined = ib['is_defined']
    zero    = np.zeros_like(forward_spread_rate)
    return {
        'spread_rate':                    np.where(defined, forward_spread_rate, zero),
        'backing_spread_rate':            np.where(defined, backing,             zero),
        'flanking_spread_rate':           np.where(defined, flanking,            zero),
        'flame_length':                   np.where(defined, flame_length,        zero),
        'fireline_intensity':             np.where(defined, fireline_intensity,  zero),
        'heat_per_unit_area':             np.where(defined, hpua,               zero),
        'effective_wind_speed':           np.where(defined, ews_mph,            zero),
        'fire_length_to_width_ratio':     np.where(defined, lwr,  np.ones_like(lwr)),
        'eccentricity':                   np.where(defined, ecc,               zero),
        'direction_of_max_spread':        np.where(defined, dir_deg,           zero),
        'residence_time':                 np.where(defined, residence_time,    zero),
        'reaction_intensity':             np.where(defined, ri,               zero),
        'midflame_wind_speed':            np.where(defined, midflame_ws,       zero),
        'no_wind_no_slope_spread_rate':   np.where(defined, r0,               zero),
    }


# ---------------------------------------------------------------------------
# V5-ext — Fire area and perimeter (§13.2 Gap B)
# ---------------------------------------------------------------------------

def calculate_fire_area(forward_ros, backing_ros, lwr, elapsed_min, is_crown=False):
    """
    Calculate elliptical fire area in square feet.

    Parameters
    ----------
    forward_ros  : (*S) — forward rate of spread (ft/min)
    backing_ros  : (*S) — backing rate of spread (ft/min)
    lwr          : (*S) — fire length-to-width ratio
    elapsed_min  : scalar float — elapsed time in minutes
    is_crown     : bool — True for circular crown fire approximation

    Returns
    -------
    (*S) ndarray — fire area in square feet
    """
    fros = np.asarray(forward_ros, dtype=float)
    bros = np.asarray(backing_ros, dtype=float)
    lwr  = np.asarray(lwr,         dtype=float)
    safe_lwr = np.where(lwr > 1e-7, lwr, 1.0)

    if is_crown:
        d = fros * elapsed_min
        return np.where(lwr > 1e-7, np.pi * d ** 2 / (4.0 * safe_lwr), 0.0)
    else:
        ell_b = (fros + bros) / 2.0 * elapsed_min
        ell_a = np.where(lwr > 1e-7, ell_b / safe_lwr, 0.0)
        return np.pi * ell_a * ell_b


def calculate_fire_perimeter(forward_ros, backing_ros, lwr, elapsed_min, is_crown=False):
    """
    Calculate elliptical fire perimeter in feet (Ramanujan approximation).

    Parameters
    ----------
    forward_ros  : (*S) — forward rate of spread (ft/min)
    backing_ros  : (*S) — backing rate of spread (ft/min)
    lwr          : (*S) — fire length-to-width ratio
    elapsed_min  : scalar float — elapsed time in minutes
    is_crown     : bool — True for circular crown fire approximation

    Returns
    -------
    (*S) ndarray — fire perimeter in feet
    """
    fros = np.asarray(forward_ros, dtype=float)
    bros = np.asarray(backing_ros, dtype=float)
    lwr  = np.asarray(lwr,         dtype=float)
    safe_lwr = np.where(lwr > 1e-7, lwr, 1.0)

    if is_crown:
        d = fros * elapsed_min
        return np.where(lwr > 1e-7, 0.5 * np.pi * d * (1.0 + 1.0 / safe_lwr), 0.0)
    else:
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


def calculate_fire_length(forward_ros, backing_ros, elapsed_min):
    """Fire ellipse length (major axis × 2) in feet."""
    return (np.asarray(forward_ros) + np.asarray(backing_ros)) * elapsed_min


def calculate_fire_width(forward_ros, backing_ros, lwr, elapsed_min):
    """Fire ellipse width (minor axis × 2) in feet."""
    fros = np.asarray(forward_ros, dtype=float)
    bros = np.asarray(backing_ros, dtype=float)
    lwr  = np.asarray(lwr,         dtype=float)
    length = (fros + bros) * elapsed_min
    safe_lwr = np.where(lwr > 1e-7, lwr, 1.0)
    return np.where(lwr > 1e-7, length / safe_lwr, 0.0)



