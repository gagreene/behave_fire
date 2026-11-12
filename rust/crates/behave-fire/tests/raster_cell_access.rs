//! Shared validated cell access supports callers with their own output storage.
use behave_fire::engine::{CrownPlan, CrownTweaks, PreparedFuels, SurfacePlan, SurfaceTweaks};
use behave_fire::raster::{
    CrownColumns, CrownRaster, CrownTweakColumns, SurfaceColumns, SurfaceRaster,
    SurfaceTweakColumns,
};
use behave_fire::{FuelModels, GridShape, PerCell};

#[test]
fn validated_raster_cells_match_full_outputs_and_reject_out_of_range_indexes() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let models = [10, 124];
    let surface = SurfaceRaster::new(
        GridShape::new(vec![2]).unwrap(),
        SurfaceColumns {
            fuel_model_grid: PerCell::Cells(&models),
            ..SurfaceColumns::default()
        },
    )
    .unwrap();
    let plan = SurfacePlan::default();
    let expected = surface
        .run(&plan, &fuels, &SurfaceTweakColumns::default(), 1)
        .unwrap();
    for index in 0..2 {
        let cell = surface
            .evaluate_cell(index, &plan, &fuels, SurfaceTweaks::default())
            .unwrap();
        assert_eq!(cell.spread_rate, expected.spread_rate[index]);
        assert_eq!(cell.heat_per_unit_area, expected.heat_per_unit_area[index]);
    }
    assert!(surface
        .evaluate_cell(2, &plan, &fuels, SurfaceTweaks::default())
        .is_none());
    let crown =
        CrownRaster::new(GridShape::new(vec![2]).unwrap(), CrownColumns::default()).unwrap();
    let plan = CrownPlan::default();
    let expected = crown
        .run(&plan, &fuels, &CrownTweakColumns::default(), 1)
        .unwrap();
    for index in 0..2 {
        let cell = crown
            .evaluate_cell(index, &plan, &fuels, CrownTweaks::default())
            .unwrap();
        assert_eq!(cell.final_spread_rate, expected.final_spread_rate[index]);
        assert_eq!(cell.fire_type, expected.fire_type[index]);
        assert!(crown.input_cell(index).is_some());
    }
    assert!(crown.input_cell(2).is_none());
}
#[test]
fn invalid_and_empty_raster_cells_do_not_read_scientific_inputs() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let invalid = [false, true];
    let surface = SurfaceRaster::new(
        GridShape::new(vec![2]).unwrap(),
        SurfaceColumns {
            validity: Some(PerCell::Cells(&invalid)),
            ..SurfaceColumns::default()
        },
    )
    .unwrap();
    let plan = SurfacePlan::default();
    let full = surface
        .run(&plan, &fuels, &SurfaceTweakColumns::default(), 1)
        .unwrap();
    let skipped = surface
        .evaluate_cell(0, &plan, &fuels, SurfaceTweaks::default())
        .unwrap();
    assert_eq!(skipped.spread_rate, full.spread_rate[0]);
    assert!(surface
        .evaluate_cell(usize::MAX, &plan, &fuels, SurfaceTweaks::default())
        .is_none());
    let crown = CrownRaster::new(
        GridShape::new(vec![2]).unwrap(),
        CrownColumns {
            validity: Some(PerCell::Cells(&invalid)),
            ..CrownColumns::default()
        },
    )
    .unwrap();
    let plan = CrownPlan::default();
    let full = crown
        .run(&plan, &fuels, &CrownTweakColumns::default(), 1)
        .unwrap();
    assert!(crown.input_cell(0).is_none());
    assert!(crown.input_cell(1).is_some());
    assert!(crown.input_cell(usize::MAX).is_none());
    assert_eq!(
        crown
            .evaluate_cell(0, &plan, &fuels, CrownTweaks::default())
            .unwrap()
            .fire_type,
        full.fire_type[0]
    );
    let empty =
        CrownRaster::new(GridShape::new(vec![0]).unwrap(), CrownColumns::default()).unwrap();
    assert!(empty.input_cell(0).is_none());
    assert!(empty
        .evaluate_cell(0, &plan, &fuels, CrownTweaks::default())
        .is_none());
}
