//! Crown spread has no canopy-geometry or supplied-Surface dependency.
use behave_fire::engine::{CrownCell, CrownPlan, CrownSpatial, PreparedFuels, SurfaceOptions};
use behave_fire::FuelModels;

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
