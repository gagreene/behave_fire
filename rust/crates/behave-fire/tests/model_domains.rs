//! Standalone facade coverage with analytic geometry and independent domain anchors.
use behave_fire::{BehaveRun, FuelModels};

#[test]
fn facade_geometry_has_base_unit_contracts() {
    // Equal forward/backing rates and unit L/W form a circle of radius 20 ft.
    let area = BehaveRun::calculate_fire_area(2.0, 2.0, 1.0, 10.0, false);
    assert!((area - 400.0 * std::f64::consts::PI).abs() < 1e-10);
    let perimeter = BehaveRun::calculate_fire_perimeter(2.0, 2.0, 1.0, 10.0, false);
    assert!((perimeter - 40.0 * std::f64::consts::PI).abs() < 1e-10);
    assert_eq!(BehaveRun::calculate_fire_length(2.0, 2.0, 10.0), 40.0);
    assert_eq!(BehaveRun::calculate_fire_width(2.0, 2.0, 1.0, 10.0), 40.0);
}

#[test]
fn facade_effects_share_owned_coefficients_and_base_units() {
    let model = BehaveRun::new(FuelModels::default());
    // Equation 1 at full crown scorch: logit = -2.4027 + 0.9568.
    let mortality = model.calculate_crown_scorch_mortality(80.0, 80.0, 0.5, 12.0, 1);
    assert_eq!(mortality.crown_volume_scorch, 1.0);
    let expected = 1.0 / (1.0 + 1.4459_f64.exp());
    assert!((mortality.probability_mortality - expected).abs() < 1e-12);
    assert_eq!(BehaveRun::calculate_scorch_height(1.0, 0.0, 77.0), 1.0);
    assert_eq!(
        BehaveRun::calculate_spotting_from_surface_fire(0.0, 5.0, 20.0),
        0.0
    );
    assert_eq!(
        BehaveRun::calculate_spotting_from_burning_pile(10.0, 0.0, 20.0),
        0.0
    );
    assert_eq!(
        BehaveRun::calculate_spotting_from_torching_trees(12.0, 80.0, 0.0, 5.0, 20.0),
        0.0
    );
    assert_eq!(
        model.clone().mortality_coefficients(),
        model.mortality_coefficients()
    );
}
