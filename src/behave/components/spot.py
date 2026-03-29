"""
spot_array.py — Vectorized Spotting Distance (V8)

Public API
----------
calculate_spotting_from_surface_fire(flame_length_ft, wind_mph, cover_height_ft)
calculate_spotting_from_burning_pile(flame_height_ft, wind_mph, cover_height_ft)
calculate_spotting_from_torching_trees(dbh_in, height_ft, count,
                                        wind_mph, cover_height_ft)
"""

import numpy as np


def calculate_spotting_from_surface_fire(flame_length_ft, wind_mph, cover_height_ft):
    """Flat-terrain spotting distance from surface fire (Albini 1979). Returns feet."""
    fl = np.asarray(flame_length_ft, dtype=float)
    ws = np.asarray(wind_mph,        dtype=float)
    ch = np.asarray(cover_height_ft, dtype=float)

    active  = (ws > 1e-7) & (fl > 1e-7)
    safe_ws = np.where(ws > 1e-7, ws, 1.0)
    safe_fl = np.where(fl > 1e-7, fl, 1.0)

    f      = 322.0 * ((0.474 * safe_ws) ** -1.01)
    byrams = (safe_fl / 0.45) ** (1.0 / 0.46)
    val    = f * byrams
    fb_ht  = np.where(active & (val > 1e-7), 1.055 * np.sqrt(val), 0.0)

    safe_fb    = np.where(fb_ht > 1e-7, fb_ht, 1.0)
    crit_cover = np.where(
        fb_ht > 1e-7,
        np.maximum(ch, 2.2 * (safe_fb ** 0.337) - 4.0),
        0.0
    )
    safe_cc = np.where(crit_cover > 1e-7, crit_cover, 1.0)
    drift   = 0.000278 * ws * (safe_fb ** 0.643)
    dist_mi = np.where(
        crit_cover > 1e-7,
        0.000718 * ws * np.sqrt(safe_cc) *
        (0.362 + np.sqrt(safe_fb / safe_cc) / 2.0 * np.log(safe_fb / safe_cc))
        + drift,
        0.0
    )
    return dist_mi * 5280.0


def calculate_spotting_from_burning_pile(flame_height_ft, wind_mph, cover_height_ft):
    """Flat-terrain spotting distance from burning pile (Albini 1979). Returns feet."""
    fh = np.asarray(flame_height_ft, dtype=float)
    ws = np.asarray(wind_mph,        dtype=float)
    ch = np.asarray(cover_height_ft, dtype=float)

    active  = (ws > 1e-7) & (fh > 1e-7)
    safe_fh = np.where(fh > 1e-7, fh, 1.0)
    safe_ws = np.where(ws > 1e-7, ws, 1.0)

    fb_ht = np.where(active, 1.055 * np.sqrt(safe_fh * safe_ws), 0.0)

    safe_fb    = np.where(fb_ht > 1e-7, fb_ht, 1.0)
    crit_cover = np.where(
        fb_ht > 1e-7,
        np.maximum(ch, 2.2 * (safe_fb ** 0.337) - 4.0),
        0.0
    )
    safe_cc = np.where(crit_cover > 1e-7, crit_cover, 1.0)
    drift   = 0.000278 * ws * (safe_fb ** 0.643)
    dist_mi = np.where(
        crit_cover > 1e-7,
        0.000718 * ws * np.sqrt(safe_cc) *
        (0.362 + np.sqrt(safe_fb / safe_cc) / 2.0 * np.log(safe_fb / safe_cc))
        + drift,
        0.0
    )
    return dist_mi * 5280.0


def calculate_spotting_from_torching_trees(dbh_in, height_ft, count,
                                            wind_mph, cover_height_ft):
    """Spotting distance from torching trees. Returns feet (*S)."""
    dbh = np.asarray(dbh_in,          dtype=float)
    ht  = np.asarray(height_ft,       dtype=float)
    cnt = np.asarray(count,           dtype=float)
    ws  = np.asarray(wind_mph,        dtype=float)
    ch  = np.asarray(cover_height_ft, dtype=float)

    safe_cnt = np.where(cnt > 0, cnt, 1.0)
    flame_ht = np.where(
        (dbh > 1e-7) & (ht > 1e-7) & (cnt > 0),
        0.5 * ht + 2.0 * dbh * np.sqrt(safe_cnt),
        0.0
    )
    return calculate_spotting_from_burning_pile(flame_ht, ws, ch)

