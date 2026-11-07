//! Prepared physics must preserve arbitrary edited particle and fuel inputs.
use behave_fire::{
    engine::{PreparedFuel, SurfaceCell, SurfacePlan, SurfaceTweaks},
    surface, FuelModels,
};

#[test]
fn prepared_savr_terms_match_particle_path_for_static_dynamic_custom_and_undefined_fuels() {
    let registry = FuelModels::default();
    let plan = SurfacePlan::default();
    for id in 0..=260 {
        let mut fuel = PreparedFuel::from_model(registry.for_grid(id));
        for edit in 0..3 {
            if edit == 1 {
                fuel.savr_live_herb = 191.999;
                fuel.savr_1h = 1200.0;
            }
            if edit == 2 {
                fuel.savr_live_woody = 48.0;
                fuel.is_dynamic = true;
                fuel.live_herb = 0.31;
            }
            for moisture in [0.2, 0.3, 0.6, 1.2, 1.21] {
                let cell = SurfaceCell {
                    mlh: moisture,
                    wind_speed: 7.31,
                    slope: 13.12,
                    ..SurfaceCell::new(id)
                };
                let particles = surface::build_particle_arrays_from_prepared(
                    &fuel, cell.m1h, cell.m10h, cell.m100h, cell.mlh, cell.mlw,
                );
                let expected = surface::calculate_fuelbed_intermediates(&particles);
                let actual = plan.prepare_fuelbed(&cell, &fuel);
                for ((name, actual), (_, expected)) in actual
                    .named_values()
                    .into_iter()
                    .zip(expected.named_values())
                {
                    assert!(
                        actual.to_bits() == expected.to_bits()
                            || (actual.is_nan() && expected.is_nan()),
                        "{id}/{edit}/{moisture}/{name}"
                    );
                }
                let expected = plan.run_from_fuelbed(&cell, &expected, SurfaceTweaks::default());
                let actual = plan.run_cell(&cell, &fuel, SurfaceTweaks::default());
                for ((name, actual), (_, expected)) in actual
                    .named_values()
                    .into_iter()
                    .zip(expected.named_values())
                {
                    assert!(
                        actual.to_bits() == expected.to_bits()
                            || (actual.is_nan() && expected.is_nan()),
                        "final {id}/{edit}/{name}"
                    );
                }
            }
        }
    }
}
