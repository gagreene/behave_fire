//! General ordinary layouts retain checked Crown promotion and raw masked values.
use behave_fire::crown::{
    run_with_surface_inputs, try_run_compact_surface_inputs, SurfaceFireInputs,
};
use behave_fire::engine::{CrownCell, PreparedFuels, SurfaceOptions};
use behave_fire::surface::masked::{IntegerKind, IntegerValue, MaskedValue};
use behave_fire::FuelModels;

#[test]
fn compact_crown_matches_checked_arithmetic_across_numeric_layouts_and_domains() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let spatial = CrownCell {
        wind_speed: 440.0,
        slope: 12.0,
        canopy_base_height: 4.0,
        canopy_height: 30.0,
        canopy_bulk_density: 0.01,
        ..CrownCell::default()
    };
    for mask in [false, true] {
        for scalar in [false, true] {
            for value in [
                0.0,
                -0.0,
                1e-7,
                300.0,
                -10.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ] {
                let variants = [
                    MaskedValue::plain(value),
                    MaskedValue::plain_f32(value as f32),
                    MaskedValue::masked(value, mask).with_fill_value(Some(-123.0)),
                    MaskedValue::masked_f32(value as f32, mask).with_fill_value(Some(987.0)),
                    MaskedValue::plain(0.0)
                        .with_integer(IntegerValue::from_bits(IntegerKind::U64, (1_u64 << 63) + 7)),
                    MaskedValue::plain(0.0)
                        .with_integer(IntegerValue::from_bits(IntegerKind::I16, 65535)),
                    MaskedValue::plain(0.0)
                        .with_integer(IntegerValue::from_bits(IntegerKind::Bool, 1)),
                ];
                for (index, rate) in variants.into_iter().enumerate() {
                    let mut surface = SurfaceFireInputs::new();
                    surface.surface_ros = rate.with_scalar_array(scalar);
                    surface.surface_fli =
                        variants[(index + 2) % variants.len()].with_scalar_array(scalar);
                    surface.surface_hpua =
                        variants[(index + 3) % variants.len()].with_scalar_array(scalar);
                    let expected = run_with_surface_inputs(
                        &surface,
                        &spatial,
                        fuels.for_grid(10),
                        &SurfaceOptions::default(),
                    )
                    .unwrap();
                    let actual = try_run_compact_surface_inputs(
                        &surface,
                        &spatial,
                        fuels.for_grid(10),
                        &SurfaceOptions::default(),
                    )
                    .expect("ordinary numeric layouts must use compact arithmetic")
                    .unwrap();
                    assert_eq!(actual.fire_type, expected.fire_type);
                    for ((name, mut actual), (_, mut expected)) in actual
                        .named_values()
                        .into_iter()
                        .zip(expected.named_values())
                    {
                        if expected.value.is_nan() {
                            assert!(actual.value.is_nan(), "{name}, layout {index}");
                            actual.value = 0.0;
                            expected.value = 0.0;
                        } else {
                            assert_eq!(
                                actual.value.to_bits(),
                                expected.value.to_bits(),
                                "{name}, layout {index}"
                            );
                        }
                        assert_eq!(actual, expected, "{name}, layout {index}");
                    }
                }
            }
        }
    }
}

#[test]
fn crown_fraction_clip_preserves_negative_zero_at_the_inclusive_lower_bound() {
    let fraction = behave_fire::crown::calculate_crown_fraction_burned(-0.0, 0.0, 1.0);
    assert_eq!(fraction.to_bits(), (-0.0_f64).to_bits());
    let fraction = behave_fire::crown::masked::calculate_crown_fraction_burned(
        MaskedValue::plain(-0.0),
        MaskedValue::plain(0.0),
        MaskedValue::plain(1.0),
    );
    assert_eq!(fraction.value.to_bits(), (-0.0_f64).to_bits());
}
