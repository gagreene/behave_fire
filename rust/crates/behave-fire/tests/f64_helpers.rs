use behave_fire::{fire_size::FireSize, surface};

#[test]
fn missing_threshold_operands_remain_missing_in_shared_geometry() {
    assert!(surface::calculate_eccentricity(f64::NAN).is_nan());
    assert!(surface::calculate_flame_length(f64::NAN).is_nan());
    assert!(surface::calculate_fire_length_to_width_ratio(f64::NAN).is_nan());
    assert!(surface::calculate_backing_spread_rate(1., f64::NAN).is_nan());
    assert!(surface::calculate_fire_width(1., 1., f64::NAN, 1.).is_nan());
}

#[test]
fn fire_size_uses_ieee_domain_arithmetic_and_float64_state() {
    let mut state = FireSize::new();
    state.effective_wind_speed_ = 1e6;
    state.calculate_surface_fire_length_to_width_ratio();
    assert_eq!(state.fire_length_to_width_ratio_, 8.);
    state.fire_length_to_width_ratio_ = 0.5;
    state.calculate_elliptical_dimensions();
    assert!(state.heading_to_backing_ratio_.is_nan());
    state.effective_wind_speed_ = f64::NAN;
    state.calculate_surface_fire_length_to_width_ratio();
    assert!(state.fire_length_to_width_ratio_.is_nan());
}
