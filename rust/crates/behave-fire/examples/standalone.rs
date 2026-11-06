//! Run surface fire and mortality using only the standalone Rust crate.
use behave_fire::{
    engine::{SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceTweaks},
    fuel_models::CustomFuelModel,
    BehaveRun, FuelModels,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut fuels = FuelModels::default();
    // Register a local variant of standard model 1 in an unreserved slot.
    let mut custom = CustomFuelModel {
        model: fuels.get(1).expect("embedded standard fuel 1").clone(),
        ..CustomFuelModel::default()
    };
    custom.model.code = "LOCAL".into();
    custom.model.name = "Example deeper grass fuel".into();
    custom.model.fuel_bed_depth *= 1.5;
    assert!(fuels.set_custom_fuel_model(14, custom));

    let model = BehaveRun::new(fuels.snapshot());
    // Later caller edits do not alter the model's calculation snapshot.
    assert!(fuels.clear_custom_fuel_model(14));
    for number in [1, 14] {
        let input = SurfaceCell::new(number);
        let surface = model.do_surface_run(
            &SurfacePlan::new(SurfaceOptions::default()),
            &input,
            SurfaceTweaks::default(),
        );
        assert!(surface.spread_rate.is_finite() && surface.spread_rate > 0.0);
        let scorch = BehaveRun::calculate_scorch_height(surface.fireline_intensity, 5.0, 77.0);
        let mortality = model.calculate_crown_scorch_mortality(scorch, 80.0, 0.5, 12.0, 1);
        assert!((0.0..=1.0).contains(&mortality.probability_mortality));
        println!(
            "Fuel {number}: spread={:.6} ft/min, scorch={scorch:.6} ft, mortality={:.6}",
            surface.spread_rate, mortality.probability_mortality,
        );
    }
    Ok(())
}
