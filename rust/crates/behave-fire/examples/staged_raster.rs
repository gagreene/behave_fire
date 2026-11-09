//! Run owned stages, inject base-unit fields, and distinguish completion from restart.
use behave_fire::engine::{
    CrownPlan, Field, GridSession, ModelRun, OutputRequest, PreparedFuels, Stage, SurfacePlan,
    SurfaceSource,
};
use behave_fire::raster::{
    CrownColumns, CrownRaster, CrownTweakColumns, SurfaceColumns, SurfaceRaster,
    SurfaceTweakColumns,
};
use behave_fire::{FuelModels, GridShape, PerCell};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let shape = GridShape::new(vec![2, 3])?;
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let surface = ModelRun::Surface {
        raster: SurfaceRaster::new(
            shape.clone(),
            SurfaceColumns {
                fuel_model_grid: PerCell::Scalar(124),
                wind_speed: PerCell::Scalar(88.),
                ..SurfaceColumns::default()
            },
        )?,
        plan: SurfacePlan::default(),
        source: SurfaceSource::Fuels(&fuels),
        tweaks: SurfaceTweakColumns::default(),
    };
    let mut grid = GridSession::new(&surface, OutputRequest::surface(), &[], 4)?;
    for stage in [Stage::Particles, Stage::Fuelbed, Stage::Reaction] {
        grid.run_stage(stage)?;
    }
    let reaction = grid
        .get(Field::ReactionIntensity)?
        .physical()?
        .iter()
        .map(|v| v * 0.9)
        .collect();
    // Reaction intensity is BTU/ft²/min; run_remaining preserves this injection.
    grid.set_physical(Field::ReactionIntensity, reaction)?;
    grid.run_remaining()?;
    println!(
        "Injected Surface spread: {:?}",
        grid.get(Field::SpreadRate)?.physical()?
    );
    grid.run_fused()?; // Clears the transient injection and restarts the original calculation.
    println!(
        "Original Surface spread: {:?}",
        grid.get(Field::SpreadRate)?.physical()?
    );

    let crown = ModelRun::Crown {
        raster: CrownRaster::new(
            shape,
            CrownColumns {
                canopy_base_height: PerCell::Scalar(3.),
                canopy_height: PerCell::Scalar(30.),
                canopy_bulk_density: PerCell::Scalar(0.02),
                surface_ros: PerCell::Scalar(5.),
                surface_fli: PerCell::Scalar(1000.),
                surface_hpua: PerCell::Scalar(1000.),
                wind_speed: PerCell::Scalar(88.),
                ..CrownColumns::default()
            },
        )?,
        plan: CrownPlan::default(),
        fuels: &fuels,
        tweaks: CrownTweakColumns::default(),
    };
    let mut grid = GridSession::new(&crown, OutputRequest::crown(), &[Field::IsPassiveCrown], 4)?;
    grid.run_stage(Stage::CrownFraction)?;
    grid.set_physical(Field::CrownFractionBurned, vec![0.5])?;
    grid.run_remaining()?;
    println!(
        "Injected Crown final spread: {:?}",
        grid.get(Field::FinalSpreadRate)?.physical()?
    );
    Ok(())
}
