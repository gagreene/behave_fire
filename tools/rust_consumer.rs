//! Downstream smoke program compiled against the extracted standalone archive.
use behave_fire::{
    fuel_models::CustomFuelModel, map_chunks, engine::{SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceTweaks}, BehaveError, BehaveRun,
    FuelModels, ParallelError, SpeciesMasterTable,
};

fn run_cells(model: &BehaveRun, threads: usize) -> Result<Vec<f64>, ParallelError<BehaveError>> {
    let fuels = [1, 14, 124, 4];
    let chunks = map_chunks(fuels.len(), threads, |range| {
        range
            .map(|index| {
                let surface = model.do_surface_run(&SurfacePlan::new(SurfaceOptions::default()), &SurfaceCell::new(fuels[index]), SurfaceTweaks::default());
                let scorch = BehaveRun::calculate_scorch_height(surface.fireline_intensity, 5.0, 77.0);
                Ok(model
                    .calculate_crown_scorch_mortality(scorch, 80.0, 0.5, 12.0, 1)
                    .probability_mortality)
            })
            .collect::<Result<Vec<_>, BehaveError>>()
    })?;
    Ok(chunks.into_iter().flatten().collect())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut fuels = FuelModels::default();
    assert!(!SpeciesMasterTable::default().records.is_empty());
    let mut custom = CustomFuelModel {
        model: fuels.get(1).expect("default fuel exists").clone(),
        ..CustomFuelModel::default()
    };
    custom.model.code = "CUSTOM".into();
    custom.model.fuel_bed_depth *= 1.5;
    assert!(fuels.set_custom_fuel_model(14, custom));
    let model = BehaveRun::new(fuels.snapshot());
    assert!(fuels.clear_custom_fuel_model(14));
    let serial = run_cells(&model, 1)?;
    assert!(serial.iter().all(|value| (0.0..=1.0).contains(value)));
    for threads in [0, 2, 4] {
        assert_eq!(serial, run_cells(&model, threads)?);
    }
    println!("Standalone consumer passed: embedded species, default/custom fuels, surface, mortality, threads 0/1/2/4.");
    Ok(())
}
