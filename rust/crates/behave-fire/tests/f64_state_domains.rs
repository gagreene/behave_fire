#[test]
fn infinite_physical_slope_produces_ieee_missing_distances() {
    let mut state = behave_fire::slope::SlopeTool::new();
    state.calculate_horizontal_distance(
        100.,
        behave_fire::units::LengthUnit::Feet,
        f64::INFINITY,
        behave_fire::units::SlopeUnit::Degrees,
    );
    assert!(state.horizontal_distances_.iter().all(|v| v.is_nan()));
}
