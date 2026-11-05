//! Fuel temperature and ignition probabilities from Python reference 7933789.
use crate::math::{np_clip, np_max, np_min};

/// Fuel temperature in Fahrenheit, including the reference shade adjustment.
pub fn calculate_fuel_temperature(air_temp_f: f64, sun_shade_fraction: f64) -> f64 {
    air_temp_f + (25.0 - 20.0 * sun_shade_fraction)
}

/// Calculate glowing-firebrand ignition probability.
pub fn calculate_firebrand_ignition_probability(
    air_temp_f: f64,
    sun_shade_fraction: f64,
    moisture_1h_fraction: f64,
) -> f64 {
    let t = (calculate_fuel_temperature(air_temp_f, sun_shade_fraction) - 32.0) * 5.0 / 9.0;
    let m = moisture_1h_fraction;
    let hoi = 144.51 - 0.26600 * t - 0.00058 * t.powi(2) - t * m
        + 18.54 * (1.0 - (-15.1 * m).exp())
        + 640.0 * m;
    let x = 0.1 * (400.0 - np_min(hoi, 400.0));
    np_clip(0.000048 * np_max(x, 0.0).powf(4.3) / 50.0, 0.0, 1.0)
}

/// Legacy lightning model, retained unchanged for compatibility only.
/// Upstream removed this model because its scientific basis is unreliable.
#[deprecated(
    note = "legacy lightning model retained for compatibility; upstream removed its scientific model"
)]
pub fn calculate_lightning_ignition_probability(
    fuel_bed_type_grid: i32,
    moisture_100h_fraction: f64,
    duff_depth_inches: f64,
    charge_type: i32,
) -> f64 {
    const COEFFS: [[f64; 4]; 8] = [
        [0.92, -0.087, 1.04, -0.054],
        [0.44, -0.110, 0.59, -0.094],
        [0.86, -0.060, 0.90, -0.056],
        [0.60, -0.011, 0.73, -0.011],
        [5.13, 0.68, 3.84, 0.60],
        [6.69, 1.39, 5.48, 1.28],
        [0.62, -0.050, 0.80, -0.014],
        [0.71, -0.070, 0.84, -0.060],
    ];
    let ft = fuel_bed_type_grid.clamp(0, 7) as usize;
    let m = np_min(moisture_100h_fraction * 100.0, 40.0);
    let d = np_min(duff_depth_inches * 2.54, 10.0);
    let [ap, bp, an, bn] = COEFFS[ft];
    let pos = if ft == 4 || ft == 5 {
        1.0 / (1.0 + (ap - bp.abs() * d).exp())
    } else {
        ap * (bp * m).exp()
    };
    let neg = if ft == 4 || ft == 5 {
        1.0 / (1.0 + (an - bn.abs() * d).exp())
    } else if ft == 3 || ft == 6 {
        an + bn * m
    } else {
        an * (bn * m).exp()
    };
    let probability = match charge_type {
        0 => 0.2 * neg,
        1 => 0.9 * pos,
        _ => 0.723 * 0.2 * neg + 0.277 * 0.9 * pos,
    };
    np_clip(probability, 0.0, 1.0)
}
