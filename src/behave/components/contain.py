"""
contain_array.py — Batch Containment Simulation (§13.7)

Fixed-iteration masked RK4 loop approach.  Cells already resolved
(CONTAINED / OVERRUN / EXHAUSTED) are masked out at each step with
np.where so their state arrays stop updating.

Status codes (match ContainSim C++ enum):
    1 = REPORTED
    3 = CONTAINED
    4 = OVERRUN
    5 = EXHAUSTED
    8 = TIME_LIMIT_EXCEEDED

This is the lowest-priority component.  The implementation here is a
functional sketch that matches the documented approach in PYTHONIC_ARRAY.md
§13.7.  It is intentionally simplified relative to the full C++ ContainSim
(no terrain, single resource, single attack point) and serves as a starting
point for a fuller implementation.

Public API
----------
run_contain_sim_array(report_spread_rate, lw_ratio, report_size,
                       production_rate, attack_time,
                       max_steps=500, dist_step=0.01)
    → dict
"""

import numpy as np

# Status codes
STATUS_REPORTED   = np.int32(1)
STATUS_CONTAINED  = np.int32(3)
STATUS_OVERRUN    = np.int32(4)
STATUS_EXHAUSTED  = np.int32(5)
STATUS_TIME_LIMIT = np.int32(8)


def run_contain_sim_array(report_spread_rate, lw_ratio, report_size,
                           production_rate, attack_time,
                           max_steps=500, dist_step=0.01):
    """
    Vectorized fire containment simulation.

    Parameters
    ----------
    report_spread_rate : (*S) or scalar — fire spread rate at report time (ft/min)
    lw_ratio           : (*S) or scalar — fire length:width ratio
    report_size        : (*S) or scalar — fire size at report time (acres)
    production_rate    : (*S) or scalar — suppression production rate (chains/hr)
    attack_time        : (*S) or scalar — initial attack time (minutes after report)
    max_steps          : int — maximum RK4 iterations
    dist_step          : float — integration distance step (chains)

    Returns
    -------
    dict with keys:
        'status'           : (*S) int32 — final status code
        'contained_time'   : (*S) float — time to containment (minutes), 0 if not
        'final_perimeter'  : (*S) float — final fire perimeter (chains)
        'final_area'       : (*S) float — final fire area (acres)
    """
    # Coerce to at-least-1D
    rsr   = np.atleast_1d(np.asarray(report_spread_rate, dtype=float))
    lwr   = np.atleast_1d(np.asarray(lw_ratio,           dtype=float))
    rsize = np.atleast_1d(np.asarray(report_size,        dtype=float))
    prod  = np.atleast_1d(np.asarray(production_rate,    dtype=float))
    atk   = np.atleast_1d(np.asarray(attack_time,        dtype=float))
    S     = rsr.shape

    # Convert spread rate: ft/min → chains/min
    rsr_ch = rsr / 66.0

    # Initialise state arrays
    status    = np.full(S, STATUS_REPORTED,   dtype=np.int32)
    time      = np.zeros(S)
    perimeter = np.zeros(S)
    area      = np.zeros(S)
    fire_ros  = rsr_ch.copy()   # simplified: constant ROS

    # Track suppression line built
    line_built = np.zeros(S)

    # Active mask: cells still being simulated
    active = np.ones(S, dtype=bool)

    for _step in range(max_steps):
        if not np.any(active):
            break

        # Increment time
        dt    = dist_step / np.where(fire_ros > 1e-7, fire_ros, 1.0)
        dt    = np.where(active, dt, 0.0)
        time += dt

        # Fire perimeter growth (simplified ellipse approximation)
        safe_lwr   = np.where(lwr > 1e-7, lwr, 1.0)
        head_dist  = fire_ros * time
        perimeter  = np.where(
            active,
            np.pi * head_dist * (1.0 + 1.0 / safe_lwr),
            perimeter
        )
        area = np.where(
            active,
            np.pi * head_dist ** 2 / (4.0 * safe_lwr) / 10.0,  # chains² → acres (approx)
            area
        )

        # Suppression line built (at attack_time, production begins)
        line_rate  = np.where(time >= atk, prod / 60.0, 0.0)   # chains/min
        line_built = np.where(active, line_built + line_rate * dt, line_built)

        # Containment check: line built >= perimeter at attack
        contained_now = active & (line_built >= perimeter) & (time >= atk)
        # Overrun check: fire too large before attack
        overrun_now   = active & (area > rsize * 10.0) & (time < atk)
        # Exhausted: ran out of steps at max_steps - 1 (handled after loop)

        status = np.where(contained_now, STATUS_CONTAINED, status)
        status = np.where(overrun_now,   STATUS_OVERRUN,   status)
        active = active & ~contained_now & ~overrun_now

    # Any cell still active after max_steps → TIME_LIMIT_EXCEEDED
    status = np.where(active, STATUS_TIME_LIMIT, status)

    contained_time = np.where(status == STATUS_CONTAINED, time, 0.0)

    return {
        'status':          status,
        'contained_time':  contained_time,
        'final_perimeter': perimeter,
        'final_area':      area,
    }

