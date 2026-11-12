//! One availability policy for public scalar plans and retained raster interfaces.
use behave_fire::engine::{
    CrownPlan, CrownTweaks, PreparedFuels, SurfaceCell, SurfacePlan, SurfaceTweaks,
};
use behave_fire::output::ColumnValues;
use behave_fire::raster::{CrownColumns, CrownRaster, CrownTweakColumns};
use behave_fire::{FuelModels, GridShape, PerCell};

#[test]
fn public_surface_scalar_plan_preserves_missing_slope() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let mut cell = SurfaceCell::new(124);
    cell.slope = f64::NAN;
    let plan = SurfacePlan::default();
    let result = plan.run_cell(&cell, fuels.for_grid(124), SurfaceTweaks::default());
    assert!(result.spread_rate.is_nan());
    let fuelbed = plan.prepare_fuelbed(&cell, fuels.for_grid(124));
    assert!(plan
        .run_from_fuelbed(&cell, &fuelbed, SurfaceTweaks::default())
        .spread_rate
        .is_nan());
}

#[test]
fn public_crown_raster_invalid_cells_have_missing_results_in_every_entry_point() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let raster = CrownRaster::new(
        GridShape::new(vec![1]).unwrap(),
        CrownColumns {
            validity: Some(PerCell::Scalar(false)),
            ..CrownColumns::default()
        },
    )
    .unwrap();
    let plan = CrownPlan::default();
    let value = raster
        .evaluate_cell(0, &plan, &fuels, CrownTweaks::default())
        .unwrap();
    assert_eq!(value.fire_type, -1);
    assert!(value.named_values().iter().all(|(_, value)| value.is_nan()));
    for result in [
        raster
            .run(&plan, &fuels, &CrownTweakColumns::default(), 1)
            .unwrap(),
        raster
            .run_tile(0..1, &plan, &fuels, &CrownTweakColumns::default(), 1)
            .unwrap(),
    ] {
        for (_, values) in result.into_named_columns() {
            match values {
                ColumnValues::Float64(v) => assert!(v.iter().all(|x| x.is_nan())),
                ColumnValues::Int32(v) => assert_eq!(v, vec![-1]),
                ColumnValues::Bool(v) => assert!(!v.iter().any(|x| *x)),
            }
        }
    }
}
