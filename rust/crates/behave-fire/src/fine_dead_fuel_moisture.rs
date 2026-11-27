//! Standalone fine dead fuel moisture lookup, including invalid-index sentinel.
use crate::generated::fine_dead_fuel_moisture::{CORR, REF};

/// Look up moisture fraction using eight integer class indices.
#[allow(clippy::too_many_arguments)]
pub fn calculate_fine_dead_fuel_moisture(
    dry_bulb_i: i32,
    rh_i: i32,
    slope_i: i32,
    aspect_i: i32,
    shading_i: i32,
    month_i: i32,
    elev_i: i32,
    time_i: i32,
) -> f64 {
    let inputs = [
        dry_bulb_i, rh_i, slope_i, aspect_i, shading_i, month_i, elev_i, time_i,
    ];
    let limits = [6, 21, 2, 4, 2, 3, 3, 6];
    if inputs
        .iter()
        .zip(limits)
        .any(|(&value, limit)| value < 0 || value >= limit)
    {
        return -0.02;
    }
    let row_base = if shading_i == 0 {
        slope_i + 2 * aspect_i
    } else {
        8 + aspect_i
    };
    REF[dry_bulb_i as usize][rh_i as usize]
        + CORR[(row_base + 12 * month_i) as usize][(elev_i + 3 * time_i) as usize]
}
