"""
fine_dead_fuel_moisture_tool_array.py — Vectorized FDMF Lookup (§13.5)

Pure 2D fancy-index table lookup — the simplest vectorization in the library.

Public API
----------
calculate_fine_dead_fuel_moisture(dry_bulb_i, rh_i, slope_i, aspect_i,
                                   shading_i, month_i, elev_i, time_i)
    All integer index inputs (*S) or scalar.
    Returns moisture as fraction (*S).

Index ranges (mirror FineDeadFuelMoistureTool):
    dry_bulb_i : 0–5   (6 temperature classes)
    rh_i       : 0–20  (21 relative humidity classes)
    slope_i    : 0–1   (2 slope classes)
    aspect_i   : 0–3   (4 aspect classes)
    shading_i  : 0–1   (0=unshaded, 1=shaded)
    month_i    : 0–2   (3 month-group classes; 0=May-Jul, 1=Aug-Oct, 2=Nov-Apr)
    elev_i     : 0–2   (3 elevation classes)
    time_i     : 0–5   (6 time-of-day classes)
"""

import numpy as np
from typing import Union

# ---------------------------------------------------------------------------
# Lookup tables (inlined from the original FineDeadFuelMoistureTool class).
# _REF  shape (6, 21)   — indexed by [dry_bulb_i, rh_i]
# _CORR shape (36, 18)  — indexed by [row, col]
# ---------------------------------------------------------------------------
_REF = np.array([
    [0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06, 0.07, 0.08, 0.08, 0.08, 0.09, 0.09, 0.1, 0.11, 0.12, 0.12, 0.13, 0.13, 0.14],
    [0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06, 0.07, 0.07, 0.07, 0.08, 0.09, 0.09, 0.1, 0.1, 0.11, 0.12, 0.13, 0.13, 0.13],
    [0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06, 0.06, 0.07, 0.07, 0.08, 0.08, 0.09, 0.09, 0.1, 0.11, 0.12, 0.12, 0.12, 0.13],
    [0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06, 0.07, 0.07, 0.08, 0.08, 0.08, 0.09, 0.1, 0.1, 0.11, 0.12, 0.12, 0.13],
    [0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.04, 0.05, 0.06, 0.07, 0.07, 0.08, 0.08, 0.08, 0.09, 0.1, 0.1, 0.11, 0.12, 0.12, 0.13],
    [0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.04, 0.05, 0.06, 0.07, 0.07, 0.08, 0.08, 0.08, 0.09, 0.1, 0.1, 0.11, 0.12, 0.12, 0.12],
], dtype=float)

_CORR = np.array([
    [0.02, 0.03, 0.04, 0.01, 0.01, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04],
    [0.03, 0.04, 0.04, 0.01, 0.02, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.02, 0.02, 0.03, 0.04, 0.04],
    [0.02, 0.02, 0.03, 0.01, 0.01, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.01, 0.02, 0.03, 0.04, 0.04],
    [0.01, 0.02, 0.02, 0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.04, 0.05, 0.06],
    [0.02, 0.03, 0.03, 0.01, 0.01, 0.01, 0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.03],
    [0.02, 0.03, 0.03, 0.01, 0.01, 0.02, 0.0, 0.01, 0.01, 0.0, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03],
    [0.02, 0.03, 0.04, 0.01, 0.01, 0.02, 0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.0, 0.01, 0.01, 0.02, 0.03, 0.03],
    [0.04, 0.05, 0.06, 0.02, 0.03, 0.04, 0.01, 0.01, 0.02, 0.0, 0.0, 0.01, 0.0, 0.0, 0.01, 0.01, 0.02, 0.02],
    [0.04, 0.05, 0.05, 0.03, 0.04, 0.05, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.04, 0.05, 0.04, 0.05, 0.05],
    [0.04, 0.04, 0.05, 0.03, 0.04, 0.05, 0.03, 0.03, 0.04, 0.03, 0.04, 0.04, 0.03, 0.04, 0.05, 0.04, 0.05, 0.06],
    [0.04, 0.04, 0.05, 0.03, 0.04, 0.05, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.04, 0.05, 0.04, 0.05, 0.05],
    [0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.03, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.04, 0.05, 0.04, 0.04, 0.05],
    [0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.02, 0.03, 0.03, 0.04, 0.05],
    [0.03, 0.04, 0.05, 0.03, 0.03, 0.04, 0.02, 0.03, 0.04, 0.02, 0.03, 0.04, 0.03, 0.03, 0.04, 0.03, 0.04, 0.05],
    [0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.02, 0.03, 0.03, 0.04, 0.05],
    [0.03, 0.03, 0.04, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.03, 0.04, 0.05, 0.03, 0.04, 0.06],
    [0.03, 0.04, 0.05, 0.01, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.03, 0.04, 0.05],
    [0.03, 0.04, 0.05, 0.01, 0.02, 0.02, 0.0, 0.01, 0.01, 0.0, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05],
    [0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03, 0.03, 0.04, 0.05],
    [0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.01, 0.02, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.03, 0.03, 0.04],
    [0.04, 0.05, 0.06, 0.04, 0.05, 0.05, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.04, 0.05, 0.05, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.03, 0.04, 0.05, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.02, 0.03, 0.04, 0.02, 0.03, 0.04, 0.03, 0.04, 0.05, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.03, 0.04, 0.04, 0.02, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.04, 0.05, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.02, 0.03, 0.04, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.02, 0.03, 0.03, 0.02, 0.02, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.02, 0.03, 0.03, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.03, 0.04, 0.05, 0.02, 0.03, 0.03, 0.02, 0.03, 0.03, 0.03, 0.04, 0.04, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.03, 0.04, 0.04, 0.02, 0.02, 0.03, 0.02, 0.03, 0.04, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06],
    [0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06, 0.04, 0.05, 0.06],
], dtype=float)


def calculate_fine_dead_fuel_moisture(
        dry_bulb_i: Union[int, np.ndarray],
        rh_i: Union[int, np.ndarray],
        slope_i: Union[int, np.ndarray],
        aspect_i: Union[int, np.ndarray],
        shading_i: Union[int, np.ndarray],
        month_i: Union[int, np.ndarray],
        elev_i: Union[int, np.ndarray],
        time_i: Union[int, np.ndarray]
) -> np.ndarray:
    """
    Vectorized 1-hr fine dead fuel moisture table lookup.

    All inputs are integer class indices, not raw physical values.
    Use the ``FineDeadFuelMoistureTool`` class (scalar) to convert raw
    observations to index values before calling this function.

    :param dry_bulb_i: Dry bulb temperature class index 0–5 (*S) or scalar.
    :param rh_i: Relative humidity class index 0–20 (*S) or scalar.
    :param slope_i: Slope class index 0–1 (*S) or scalar.
        0 = ≤ 30%, 1 = > 30%.
    :param aspect_i: Aspect class index 0–3 (*S) or scalar.
        0 = N/NE, 1 = E/SE, 2 = S/SW, 3 = W/NW.
    :param shading_i: Shading class index 0–1 (*S) or scalar.
        0 = unshaded, 1 = shaded.
    :param month_i: Month-group class index 0–2 (*S) or scalar.
        0 = May–Jul, 1 = Aug–Oct, 2 = Nov–Apr.
    :param elev_i: Elevation class index 0–2 (*S) or scalar.
        0 = below, 1 = same, 2 = above reference elevation.
    :param time_i: Time-of-day class index 0–5 (*S) or scalar.
    :return: (*S) ndarray — 1-hr fine dead fuel moisture as fraction
        (e.g. 0.05 = 5%). Returns ``-0.02`` (sentinel) for out-of-range
        inputs, mirroring the scalar class behaviour.
    """
    db  = np.asarray(dry_bulb_i, dtype=np.int32)
    rh  = np.asarray(rh_i,       dtype=np.int32)
    sl  = np.asarray(slope_i,    dtype=np.int32)
    asp = np.asarray(aspect_i,   dtype=np.int32)
    sh  = np.asarray(shading_i,  dtype=np.int32)
    mo  = np.asarray(month_i,    dtype=np.int32)
    el  = np.asarray(elev_i,     dtype=np.int32)
    ti  = np.asarray(time_i,     dtype=np.int32)

    # Valid-range masks
    valid = (
        (db  >= 0) & (db  < 6)  &
        (rh  >= 0) & (rh  < 21) &
        (sl  >= 0) & (sl  < 2)  &
        (asp >= 0) & (asp < 4)  &
        (sh  >= 0) & (sh  < 2)  &
        (mo  >= 0) & (mo  < 3)  &
        (el  >= 0) & (el  < 3)  &
        (ti  >= 0) & (ti  < 6)
    )

    # Clamp to safe ranges to avoid index errors even for invalid inputs
    db_s  = np.clip(db,  0, 5)
    rh_s  = np.clip(rh,  0, 20)
    sl_s  = np.clip(sl,  0, 1)
    asp_s = np.clip(asp, 0, 3)
    sh_s  = np.clip(sh,  0, 1)
    mo_s  = np.clip(mo,  0, 2)
    el_s  = np.clip(el,  0, 2)
    ti_s  = np.clip(ti,  0, 5)

    # Reference moisture: indexed by [dry_bulb, rh]
    ref = _REF[db_s, rh_s]

    # Correction row:
    #   unshaded (sh==0): row = slope_i + 2*aspect_i
    #   shaded   (sh==1): row = 8 + aspect_i
    # Then offset by month_group * 12 (each month group occupies 12 rows)
    row_base = np.where(sh_s == 0, sl_s + 2 * asp_s, 8 + asp_s)
    row      = row_base + 12 * mo_s

    # Correction column: elev_i + 3 * time_i
    col = el_s + 3 * ti_s

    row = np.clip(row, 0, _CORR.shape[0] - 1)
    col = np.clip(col, 0, _CORR.shape[1] - 1)

    corr = _CORR[row, col]

    moisture = ref + corr
    return np.where(valid, moisture, -0.02)

