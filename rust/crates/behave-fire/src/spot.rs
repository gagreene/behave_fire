//! Flat-terrain spotting equations from the explicit Python reference.
use crate::math::np_max;

/// Finish the shared firebrand-height calculation in feet.
fn distance(fb_ht: f64, wind_mph: f64, cover_height_ft: f64) -> f64 {
    let safe_fb = if fb_ht > 1e-7 { fb_ht } else { 1.0 };
    let crit_cover = if fb_ht > 1e-7 {
        np_max(cover_height_ft, 2.2 * safe_fb.powf(0.337) - 4.0)
    } else {
        0.0
    };
    let safe_cc = if crit_cover > 1e-7 { crit_cover } else { 1.0 };
    let drift = 0.000278 * wind_mph * safe_fb.powf(0.643);
    let miles = if crit_cover > 1e-7 {
        0.000718
            * wind_mph
            * safe_cc.sqrt()
            * (0.362 + (safe_fb / safe_cc).sqrt() / 2.0 * (safe_fb / safe_cc).ln())
            + drift
    } else {
        0.0
    };
    miles * 5280.0
}

/// Spotting distance from a burning pile, in feet.
pub fn calculate_spotting_from_burning_pile(
    flame_height_ft: f64,
    wind_mph: f64,
    cover_height_ft: f64,
) -> f64 {
    let height = if wind_mph > 1e-7 && flame_height_ft > 1e-7 {
        1.055 * (flame_height_ft * wind_mph).sqrt()
    } else {
        0.0
    };
    distance(height, wind_mph, cover_height_ft)
}

/// Spotting distance from a surface fire, in feet.
pub fn calculate_spotting_from_surface_fire(
    flame_length_ft: f64,
    wind_mph: f64,
    cover_height_ft: f64,
) -> f64 {
    let active = wind_mph > 1e-7 && flame_length_ft > 1e-7;
    let safe_ws = if wind_mph > 1e-7 { wind_mph } else { 1.0 };
    let safe_fl = if flame_length_ft > 1e-7 {
        flame_length_ft
    } else {
        1.0
    };
    let f = 322.0 * (0.474 * safe_ws).powf(-1.01);
    let byrams = (safe_fl / 0.45).powf(1.0 / 0.46);
    let val = f * byrams;
    let height = if active && val > 1e-7 {
        1.055 * val.sqrt()
    } else {
        0.0
    };
    distance(height, wind_mph, cover_height_ft)
}

/// Spotting distance from a torching group, in feet.
pub fn calculate_spotting_from_torching_trees(
    dbh_in: f64,
    height_ft: f64,
    count: f64,
    wind_mph: f64,
    cover_height_ft: f64,
) -> f64 {
    let height = if dbh_in > 1e-7 && height_ft > 1e-7 && count > 0.0 {
        0.5 * height_ft + 2.0 * dbh_in * count.sqrt()
    } else {
        0.0
    };
    calculate_spotting_from_burning_pile(height, wind_mph, cover_height_ft)
}
