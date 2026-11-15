//! Approved low-spread geometry and flame-cutoff behavior from Behave f3b2943.

use behave_fire::engine::{PreparedFuel, SurfaceCell, SurfacePlan, SurfaceTweaks};
use behave_fire::{surface, FuelModels};

#[test]
fn effective_wind_preserves_elongation_below_old_spread_cutoff() {
    for r0 in [1e-3, 1e-7, 1e-8, 5.23839427461715e-15] {
        let spread = r0 * 78.44;
        let wind = surface::calculate_effective_wind_speed(spread, r0, 2., 0.01, 1., 1.);
        assert!((wind - 0.99999999968).abs() < 1e-14, "R0={r0}, wind={wind}");
        let ratio = surface::calculate_fire_length_to_width_ratio(wind);
        assert!(
            (ratio - 1.0829361181265).abs() < 1e-13,
            "R0={r0}, ratio={ratio}"
        );
    }
}

#[test]
fn flame_length_uses_strict_cpp_intensity_cutoff() {
    let below = f64::from_bits(1e-7_f64.to_bits() - 1);
    let above = f64::from_bits(1e-7_f64.to_bits() + 1);
    for intensity in [-1., 0., 1e-10, 1e-8, below] {
        assert_eq!(
            surface::calculate_flame_length(intensity),
            0.,
            "intensity={intensity}"
        );
    }
    for (intensity, expected) in [
        (1e-7, 0.0002711518137334609),
        (above, 0.0002711518137334609),
        (1e-6, 0.0007820103729372187),
        (1., 0.45),
    ] {
        let actual = surface::calculate_flame_length(intensity);
        assert!(
            (actual - expected).abs() <= 2e-14 * expected,
            "intensity={intensity}, actual={actual}"
        );
    }
    assert!(surface::calculate_flame_length(f64::NAN).is_nan());
}

#[test]
fn prepared_engine_preserves_wind_at_tiny_positive_spread() {
    let fuels = FuelModels::default();
    let fuel = PreparedFuel::from_model(fuels.for_grid(1));
    let plan = SurfacePlan::default();
    for r0 in [1e-3, 1e-7, 1e-8, 5.23839427461715e-15] {
        for (wind, expected_ratio) in [(0., 1.), (88., 1.0829361181265)] {
            let result = plan.run_cell(
                &SurfaceCell::default(),
                &fuel,
                SurfaceTweaks {
                    no_wind_no_slope_spread_rate: Some(r0),
                    midflame_wind_speed: Some(wind),
                    slope_phi: Some(0.),
                    ..SurfaceTweaks::default()
                },
            );
            assert!(
                (result.fire_length_to_width_ratio - expected_ratio).abs() < 1e-13,
                "R0={r0}, wind={wind}, ratio={}",
                result.fire_length_to_width_ratio
            );
        }
    }
}

#[test]
fn zero_and_negative_denominators_remain_guarded() {
    for (spread, r0) in [(0., 0.), (1., 0.), (-1., -1.)] {
        assert_eq!(
            surface::calculate_effective_wind_speed(spread, r0, 2., 0.01, 1., 1.),
            0.
        );
    }
    assert!(surface::calculate_effective_wind_speed(1., f64::NAN, 2., 0.01, 1., 1.).is_nan());
}
