//! A caller can supply raster windows and tweak stages without any file-I/O dependency.
use behave_fire::engine::{PreparedFuels, SurfaceOptions, SurfacePlan};
use behave_fire::raster::{SurfaceColumns, SurfaceRaster, SurfaceTweakColumns};
use behave_fire::{FuelModels, GridShape, PerCell};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let plan = SurfacePlan::new(SurfaceOptions::default());
    // These borrowed arrays can come from a caller's raster-window reader.
    let models = [1, 10, 101, 124, 4, 0];
    let wind = [0.0, 8.0, 16.0, 32.0, 64.0, 128.0];
    let reaction = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0];
    let validity = [true, true, true, true, true, false];
    let raster = SurfaceRaster::new(
        GridShape::new(vec![2, 3])?,
        SurfaceColumns {
            fuel_model_grid: PerCell::Cells(&models),
            wind_speed: PerCell::Cells(&wind),
            validity: Some(PerCell::Cells(&validity)),
            ..SurfaceColumns::default()
        },
    )?;
    let tweaks = SurfaceTweakColumns {
        reaction_intensity: Some(PerCell::Cells(&reaction)),
        ..SurfaceTweakColumns::default()
    };
    let result = raster.run(&plan, &fuels, &tweaks, 1)?;
    assert_eq!(result.shape().dimensions(), [2, 3]);
    assert_eq!(result.validity.as_deref(), Some(validity.as_slice()));
    assert!(result.spread_rate[5].is_nan());
    // Output-only tiles keep the original logical offsets in the caller.
    let tile = raster.run_tile(2..5, &plan, &fuels, &tweaks, 1)?;
    assert_eq!(tile.spread_rate, result.spread_rate[2..5]);
    println!("Raster window and stage overrides passed");
    Ok(())
}
