//! Crown spatial and selected final outputs depend on their physical inputs.
use behave_fire::crown::assign_final_crown_fire_behavior;
use behave_fire::engine::{
    CrownCell, CrownPlan, CrownSpatial, PreparedFuels, SurfaceOptions, WindHeight,
};
use behave_fire::FuelModels;

// ---- crown final flame ----
#[test]
fn crown_final_surface_flame_uses_byram_cutoff_and_selected_intensity() {
    for (intensity, expected) in [
        (1.0, 0.45),
        (0.5e-7, 0.0),
        (1e-7, 0.45 * 1e-7_f64.powf(0.46)),
        (f64::NAN, f64::NAN),
    ] {
        let result = assign_final_crown_fire_behavior(
            true, false, 2., 3., intensity, 4., 5., 6., 7., 8., 9., 10., 11.,
        );
        assert!(
            result.final_flame_length == expected
                || (expected.is_nan() && result.final_flame_length.is_nan()),
            "{intensity}: {} != {expected}",
            result.final_flame_length
        );
    }
}

// ---- spatial spread inputs ----
#[test]
fn crown_ratio_uses_twenty_foot_wind_independent_of_spread() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    for (height, speed, expected) in [
        (WindHeight::TwentyFoot, 1320., 2.8749999994),
        (WindHeight::TenMetre, 1518., 2.8749999994),
        (WindHeight::TwentyFoot, 0., 1.),
        (WindHeight::TwentyFoot, f64::NAN, f64::NAN),
    ] {
        let input = CrownCell {
            wind_speed: speed,
            slope: f64::NAN,
            ..Default::default()
        };
        let plan = CrownPlan::new(SurfaceOptions {
            wind_height: height,
            ..Default::default()
        });
        let result = plan.prepare_surface(&input, fuels.for_grid(10), Default::default());
        let actual = result.crown_length_to_width_ratio;
        assert!(
            (actual - expected).abs() < 1e-14 || (expected.is_nan() && actual.is_nan()),
            "{height:?} {speed}: {actual} != {expected}"
        );
    }
}

#[test]
fn spatial_spread_uses_only_real_weather_and_moisture_inputs() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let input = CrownCell {
        wind_speed: 551.,
        wind_direction: 271.,
        slope: 27.,
        aspect: 113.,
        canopy_height: f64::NAN,
        canopy_base_height: f64::NAN,
        canopy_bulk_density: f64::NAN,
        moisture_foliar: f64::NAN,
        ..CrownCell::default()
    };
    let spatial = CrownSpatial {
        m1h: input.m1h,
        m10h: input.m10h,
        m100h: input.m100h,
        mlh: input.mlh,
        mlw: input.mlw,
        wind_speed: input.wind_speed,
        wind_direction: input.wind_direction,
        slope: input.slope,
        aspect: input.aspect,
    };
    let plan = CrownPlan::new(SurfaceOptions::default());
    let actual = plan.prepare_spread_spatial(&spatial, fuels.for_grid(10));
    let expected = plan.prepare_spread(&input, fuels.for_grid(10), Default::default());
    assert_eq!(
        actual.crown_fire_spread_rate.to_bits(),
        expected.crown_fire_spread_rate.to_bits()
    );
    assert_eq!(
        actual.crown_length_to_width_ratio.to_bits(),
        expected.crown_length_to_width_ratio.to_bits()
    );
    assert!(actual.crown_fire_spread_rate.is_finite());
}
