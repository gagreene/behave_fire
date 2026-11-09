//! Explicit whole-cell unavailable results; these contain no NumPy metadata.
pub(crate) fn surface() -> crate::surface::SurfaceResult {
    crate::surface::SurfaceResult {
        spread_rate: f64::NAN,
        backing_spread_rate: f64::NAN,
        flanking_spread_rate: f64::NAN,
        flame_length: f64::NAN,
        fireline_intensity: f64::NAN,
        heat_per_unit_area: f64::NAN,
        effective_wind_speed: f64::NAN,
        fire_length_to_width_ratio: f64::NAN,
        eccentricity: f64::NAN,
        direction_of_max_spread: f64::NAN,
        residence_time: f64::NAN,
        reaction_intensity: f64::NAN,
        midflame_wind_speed: f64::NAN,
        no_wind_no_slope_spread_rate: f64::NAN,
    }
}
pub(crate) fn crown() -> crate::crown::CrownResult {
    crate::crown::CrownResult {
        crown_fire_spread_rate: f64::NAN,
        crown_flame_length: f64::NAN,
        crown_fire_line_intensity: f64::NAN,
        crown_fire_heat_per_unit_area: f64::NAN,
        canopy_heat_per_unit_area: f64::NAN,
        crown_critical_surface_fire_line_intensity: f64::NAN,
        crown_critical_fire_spread_rate: f64::NAN,
        crown_fire_transition_ratio: f64::NAN,
        crown_fire_active_ratio: f64::NAN,
        crown_length_to_width_ratio: f64::NAN,
        surface_fire_critical_spread_rate: f64::NAN,
        crowning_surface_fire_spread_rate: f64::NAN,
        crown_fraction_burned: f64::NAN,
        passive_crown_fire_spread_rate: f64::NAN,
        passive_crown_fire_heat_per_unit_area: f64::NAN,
        passive_crown_fire_line_intensity: f64::NAN,
        passive_crown_fire_flame_length: f64::NAN,
        final_spread_rate: f64::NAN,
        final_heat_per_unit_area: f64::NAN,
        final_fireline_intensity: f64::NAN,
        final_flame_length: f64::NAN,
        fire_type: -1,
    }
}

/// Resolve a conjunction without turning an unavailable comparison into false.
pub(crate) fn all_above(values: &[(f64, f64)]) -> Option<bool> {
    let mut unknown = false;
    for &(value, limit) in values {
        if value.is_nan() {
            unknown = true;
        } else if value <= limit {
            return Some(false);
        }
    }
    if unknown {
        None
    } else {
        Some(true)
    }
}
/// Resolve a disjunction; a known true branch makes other inputs unnecessary.
pub(crate) fn any_below(values: &[(f64, f64)]) -> Option<bool> {
    let mut unknown = false;
    for &(value, limit) in values {
        if value.is_nan() {
            unknown = true;
        } else if value < limit {
            return Some(true);
        }
    }
    if unknown {
        None
    } else {
        Some(false)
    }
}
/// Share the ordinary WAF formula while checking its selected dependencies.
pub(crate) fn waf(cell: &super::SurfaceCell, depth: f64, options: &super::SurfaceOptions) -> f64 {
    wind_adjustment_factor(
        cell.canopy_cover,
        cell.canopy_height,
        cell.crown_ratio,
        depth,
        options.waf_method,
        cell.user_waf,
    )
}

/// Evaluate only actual WAF inputs under three-valued canopy selection.
pub fn wind_adjustment_factor(
    cover: f64,
    height: f64,
    ratio: f64,
    depth: f64,
    method: super::WafMethod,
    user: Option<f64>,
) -> f64 {
    if method == super::WafMethod::UserInput {
        if let Some(value) = user {
            return value;
        }
    }
    let fraction = ratio * cover / 3.0;
    match any_below(&[(cover, 1e-7), (fraction, 0.05), (height, 6.0)]) {
        None => f64::NAN,
        Some(true) if depth.is_nan() => f64::NAN,
        Some(false) if height.is_infinite() => f64::NAN,
        _ => crate::surface::calculate_wind_adjustment_factor(cover, height, ratio, depth),
    }
}
