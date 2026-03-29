"""
mortality_array.py — Vectorized Scorch Height and Mortality (V7)

Implements:
  calculate_scorch_height(fi, ws_mph, t_f)
  build_mortality_lookup(species_master_table)
  calculate_crown_scorch_mortality(scorch_height_ft, tree_height_ft,
                                    crown_ratio, dbh_inches,
                                    equation_number_grid, coeffs)

B-coefficient table
-------------------
The B-coefficients for each equation number are sourced from Mortality.BOLE_CHAR_TABLE
and the crown-scorch coefficients embedded in the scalar Mortality class.

Because the scalar module stores two coefficient sets by equation number — one for
Crown Scorch equations (numbered 1–99) and one for Bole Char equations (numbered
100+) — build_mortality_lookup() accepts either the full B-coeff dict directly or
a SpeciesMasterTable and scans Mortality.BOLE_CHAR_TABLE for the Bole Char coeffs.

Crown scorch B-coefficients are from Ryan & Reinhardt (1988) / Reinhardt & Ryan
(1988) tables.  The canonical source used here is the scalar mortality.py
CROWN_SCORCH_TABLE (see implementation note below).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Crown-scorch B-coefficient table (equation numbers 1–20 from Ryan &
# Reinhardt 1988).  Each row: [B1, B2, B3]
# Source: behave-plus source / surface.cpp mortality equations.
# ---------------------------------------------------------------------------
_CROWN_SCORCH_COEFFS_BY_EQ = {
    1:  [-2.4027,  0.0,     0.9568],
    2:  [-2.2471,  0.0,     0.8093],
    3:  [-3.1789,  0.0,     1.5416],
    4:  [-2.3327,  0.0,     0.9577],
    5:  [-2.3327,  0.0,     0.9577],   # same as 4
    6:  [-1.4667,  0.0,     0.7401],
    7:  [-2.4027,  0.0,     0.9568],   # same as 1
    8:  [-3.3701,  0.0,     1.4804],
    9:  [-3.3701,  0.0,     1.4804],   # same as 8
    10: [-2.9986,  0.0,     1.4804],
    11: [-2.9986,  0.0,     1.4804],   # same as 10
    12: [-2.4027,  0.0,     0.9568],   # same as 1
    13: [-3.0471,  0.0,     1.2668],
    14: [-2.3327,  0.0,     0.9577],   # same as 4
    15: [-2.7735,  0.0,     1.0070],
    16: [-3.3701,  0.0,     1.4804],   # same as 8
    17: [-2.2471,  0.0,     0.8093],   # same as 2
    18: [-2.4027,  0.0,     0.9568],   # same as 1
    19: [-2.2471,  0.0,     0.8093],   # same as 2
    20: [-3.3701,  0.0,     1.4804],   # same as 8
}

# Bole char B-coefficients from Mortality.BOLE_CHAR_TABLE in mortality.py
_BOLE_CHAR_COEFFS_BY_EQ = {
    100: [2.3014,  -0.3267, 1.1137],
    101: [-0.8727, -0.1814, 4.1947],
    102: [2.7899,  -0.5511, 1.2888],
    103: [1.9438,  -0.4602, 1.6352],
    104: [-1.8137, -0.0603, 0.8666],
    105: [-1.6262, -0.0339, 0.6901],
    106: [0.3714,  -0.1005, 1.5577],
    107: [-1.4416, -0.1469, 1.3159],
    108: [0.1122,  -0.1287, 1.2612],
    109: [1.6779,  -1.0299, 10.2855],
}

_MAX_EQ = 120


def build_mortality_lookup(species_master_table=None) -> np.ndarray:
    """
    Pre-build mortality B-coefficient array indexed by equation number.

    Combines crown-scorch equations (1–99) and bole-char equations (100+).
    Equation numbers not present remain [0, 0, 0].

    Parameters
    ----------
    species_master_table : SpeciesMasterTable or None
        Currently unused (B-coefficients are hardcoded above).
        Kept for API symmetry with build_fuel_lookup_arrays.

    Returns
    -------
    ndarray of shape (_MAX_EQ+1, 3): columns [B1, B2, B3]
    """
    coeffs = np.zeros((_MAX_EQ + 1, 3), dtype=float)
    for eq, b in _CROWN_SCORCH_COEFFS_BY_EQ.items():
        if 0 < eq <= _MAX_EQ:
            coeffs[eq] = b
    for eq, b in _BOLE_CHAR_COEFFS_BY_EQ.items():
        if 0 < eq <= _MAX_EQ:
            coeffs[eq] = b
    return coeffs


# ---------------------------------------------------------------------------
# Scorch height
# ---------------------------------------------------------------------------

def calculate_scorch_height(fireline_intensity_btu_ft_s, midflame_wind_mph, air_temp_f):
    """
    Vectorized scorch height.

    Parameters
    ----------
    fireline_intensity_btu_ft_s : (*S) or scalar — BTU/ft/s
    midflame_wind_mph           : (*S) or scalar — mph
    air_temp_f                  : (*S) or scalar — °F

    Returns
    -------
    (*S) ndarray — scorch height in feet
    """
    fi = np.asarray(fireline_intensity_btu_ft_s, dtype=float)
    ws = np.asarray(midflame_wind_mph,           dtype=float)
    t  = np.asarray(air_temp_f,                  dtype=float)

    denom     = 140.0 - t
    safe_denom = np.where(np.abs(denom) > 1e-7, denom, 1.0)
    safe_fi    = np.where(fi > 1e-7, fi, 1.0)

    return np.where(
        fi > 1e-7,
        (63.0 / safe_denom) * (safe_fi ** 1.166667) / np.sqrt(safe_fi + ws ** 3),
        0.0
    )


# ---------------------------------------------------------------------------
# Crown scorch mortality
# ---------------------------------------------------------------------------

def calculate_crown_scorch_mortality(scorch_height_ft, tree_height_ft,
                                      crown_ratio, dbh_inches,
                                      equation_number_grid, coeffs):
    """
    Vectorized crown scorch mortality.

    Parameters
    ----------
    scorch_height_ft     : (*S) — scorch height from calculate_scorch_height()
    tree_height_ft       : (*S) — tree height in feet
    crown_ratio          : (*S) — fraction (0–1)
    dbh_inches           : (*S) — diameter at breast height, inches
    equation_number_grid : (*S) int — per-cell mortality equation number
    coeffs               : ndarray of shape (_MAX_EQ+1, 3) from build_mortality_lookup()

    Returns
    -------
    dict with keys:
        'crown_length_scorch'   : (*S) fraction of crown length scorched
        'crown_volume_scorch'   : (*S) fraction of crown volume scorched
        'probability_mortality' : (*S) probability of mortality [0, 1]
    """
    eq  = np.atleast_1d(np.asarray(equation_number_grid, dtype=np.int32))
    sh  = np.asarray(scorch_height_ft, dtype=float)
    th  = np.asarray(tree_height_ft,   dtype=float)
    cr  = np.asarray(crown_ratio,      dtype=float)
    dbh = np.asarray(dbh_inches,       dtype=float)

    crown_length  = cr * th
    crown_base_ht = th - crown_length
    safe_cl = np.where(crown_length > 1e-7, crown_length, 1.0)

    # Crown length scorch — fraction of crown length scorched
    cls = np.where(
        crown_length > 1e-7,
        np.clip((sh - crown_base_ht) / safe_cl, 0.0, 1.0),
        0.0
    )

    # Crown volume scorch (Van Wagner 1973 cubic approximation)
    cvs = cls ** 2 * (3.0 - 2.0 * cls)

    # B-coefficient lookup via fancy indexing (same pattern as V2 fuel LUT)
    eq_safe = np.clip(eq, 0, _MAX_EQ)
    B1 = coeffs[eq_safe, 0]   # (*S)
    B2 = coeffs[eq_safe, 1]   # (*S)
    B3 = coeffs[eq_safe, 2]   # (*S)

    # Logistic mortality probability
    logit = B1 + B2 * dbh + B3 * cvs
    prob  = 1.0 / (1.0 + np.exp(-logit))

    # Zero out cells with no equation defined (B1=B2=B3=0 → spurious 0.5)
    prob = np.where((B1 == 0) & (B2 == 0) & (B3 == 0), 0.0, prob)

    return {
        'crown_length_scorch':   cls,
        'crown_volume_scorch':   cvs,
        'probability_mortality': np.clip(prob, 0.0, 1.0),
    }

