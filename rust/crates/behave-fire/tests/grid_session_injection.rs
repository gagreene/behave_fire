use behave_fire::engine::{
    CrownPlan, Field, GridSession, ModelRun, OutputRequest, PreparedFuels, Stage,
};
use behave_fire::raster::{CrownColumns, CrownRaster, CrownTweakColumns};
use behave_fire::{FuelModels, GridShape, PerCell};

#[test]
fn partial_heat_injection_before_producer_keeps_siblings() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let raster = CrownRaster::new(
        GridShape::new(vec![3]).unwrap(),
        CrownColumns {
            canopy_base_height: PerCell::Scalar(4.),
            canopy_height: PerCell::Scalar(30.),
            canopy_bulk_density: PerCell::Scalar(0.01),
            surface_hpua: PerCell::Scalar(1000.),
            ..CrownColumns::default()
        },
    )
    .unwrap();
    let run = ModelRun::Crown {
        raster,
        plan: CrownPlan::default(),
        fuels: &fuels,
        tweaks: CrownTweakColumns::default(),
    };
    let request = OutputRequest::new(&[Field::CrownFuelLoad, Field::CrownFireHeatPerUnitArea]);
    let mut session = GridSession::new(&run, request, &[], 1).unwrap();
    session
        .set_physical(Field::CanopyHeatPerUnitArea, vec![500.])
        .unwrap();
    session.run_remaining().unwrap();
    assert_eq!(
        session
            .get(Field::CrownFuelLoad)
            .unwrap()
            .physical()
            .unwrap(),
        &[0.26; 3]
    );
    assert_eq!(
        session
            .get(Field::CanopyHeatPerUnitArea)
            .unwrap()
            .physical()
            .unwrap(),
        &[500.]
    );
    assert_eq!(
        session
            .get(Field::CrownFireHeatPerUnitArea)
            .unwrap()
            .physical()
            .unwrap(),
        &[1500.; 3]
    );
    session.run_stage(Stage::CrownHeat).unwrap();
    assert_eq!(
        session
            .get(Field::CanopyHeatPerUnitArea)
            .unwrap()
            .physical()
            .unwrap(),
        &[2080.; 3]
    );
}

#[test]
fn injected_nan_before_producer_is_pinned() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let raster = CrownRaster::new(
        GridShape::new(vec![2]).unwrap(),
        CrownColumns {
            canopy_base_height: PerCell::Scalar(4.),
            canopy_height: PerCell::Scalar(30.),
            canopy_bulk_density: PerCell::Scalar(0.01),
            surface_hpua: PerCell::Scalar(1000.),
            ..CrownColumns::default()
        },
    )
    .unwrap();
    let run = ModelRun::Crown {
        raster,
        plan: CrownPlan::default(),
        fuels: &fuels,
        tweaks: CrownTweakColumns::default(),
    };
    let mut session = GridSession::new(
        &run,
        OutputRequest::new(&[Field::CrownFuelLoad, Field::CrownFireHeatPerUnitArea]),
        &[],
        4,
    )
    .unwrap();
    session
        .set_physical(Field::CanopyHeatPerUnitArea, vec![f64::NAN])
        .unwrap();
    session.run_remaining().unwrap();
    assert_eq!(
        session
            .get(Field::CrownFuelLoad)
            .unwrap()
            .physical()
            .unwrap(),
        &[0.26; 2]
    );
    assert!(session
        .get(Field::CrownFireHeatPerUnitArea)
        .unwrap()
        .physical()
        .unwrap()
        .iter()
        .all(|v| v.is_nan()));
    assert!(session
        .get(Field::CanopyHeatPerUnitArea)
        .unwrap()
        .physical()
        .unwrap()[0]
        .is_nan());
}

#[test]
fn reset_preserves_input_edits_and_persistent_overrides() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let raster = CrownRaster::new(
        GridShape::new(vec![2]).unwrap(),
        CrownColumns {
            canopy_base_height: PerCell::Scalar(4.),
            canopy_height: PerCell::Scalar(30.),
            canopy_bulk_density: PerCell::Scalar(0.01),
            surface_hpua: PerCell::Scalar(1000.),
            ..CrownColumns::default()
        },
    )
    .unwrap();
    let run = ModelRun::Crown {
        raster,
        plan: CrownPlan::default(),
        fuels: &fuels,
        tweaks: CrownTweakColumns::default(),
    };
    let mut session = GridSession::new(
        &run,
        OutputRequest::new(&[Field::CrownFuelLoad, Field::CrownFireHeatPerUnitArea]),
        &[],
        1,
    )
    .unwrap();
    session
        .set_physical(Field::CanopyHeight, vec![40.])
        .unwrap();
    session
        .persistent(
            Field::CanopyHeatPerUnitArea,
            behave_fire::output::ColumnValues::Float64(vec![500.]),
        )
        .unwrap();
    session
        .set_physical(Field::CanopyHeatPerUnitArea, vec![100.])
        .unwrap();
    session.run_remaining().unwrap();
    assert_eq!(
        session
            .get(Field::CrownFireHeatPerUnitArea)
            .unwrap()
            .physical()
            .unwrap(),
        &[1100.; 2]
    );
    session.reset();
    assert_eq!(
        session
            .get(Field::CanopyHeight)
            .unwrap()
            .physical()
            .unwrap(),
        &[40.]
    );
    assert_eq!(
        session
            .get(Field::CanopyHeatPerUnitArea)
            .unwrap()
            .physical()
            .unwrap(),
        &[500.]
    );
    assert!(session.get(Field::CrownFireHeatPerUnitArea).is_err());
    session.run_remaining().unwrap();
    assert_eq!(
        session
            .get(Field::CrownFireHeatPerUnitArea)
            .unwrap()
            .physical()
            .unwrap(),
        &[1500.; 2]
    );
}

#[test]
fn injected_reaction_updates_descendants_only() {
    use behave_fire::engine::{SurfacePlan, SurfaceSource};
    use behave_fire::raster::{SurfaceColumns, SurfaceRaster, SurfaceTweakColumns};
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let raster = SurfaceRaster::new(
        GridShape::new(vec![1031]).unwrap(),
        SurfaceColumns {
            fuel_model_grid: PerCell::Scalar(10),
            wind_speed: PerCell::Scalar(88.),
            ..SurfaceColumns::default()
        },
    )
    .unwrap();
    let run = ModelRun::Surface {
        raster,
        plan: SurfacePlan::default(),
        source: SurfaceSource::Fuels(&fuels),
        tweaks: SurfaceTweakColumns::default(),
    };
    let mut session =
        GridSession::new(&run, OutputRequest::surface(), &[Field::FuelbedSigma], 4).unwrap();
    session.run_remaining().unwrap();
    let ri = session
        .get(Field::ReactionIntensity)
        .unwrap()
        .physical()
        .unwrap()[0];
    let heat = session
        .get(Field::HeatPerUnitArea)
        .unwrap()
        .physical()
        .unwrap()[0];
    let sigma = session
        .get(Field::FuelbedSigma)
        .unwrap()
        .physical()
        .unwrap()[0];
    session
        .set_physical(Field::ReactionIntensity, vec![ri * 2.])
        .unwrap();
    assert_eq!(
        session
            .get(Field::FuelbedSigma)
            .unwrap()
            .physical()
            .unwrap()[0],
        sigma
    );
    assert!(session.get(Field::HeatPerUnitArea).is_err());
    session.run_remaining().unwrap();
    assert!(
        (session
            .get(Field::HeatPerUnitArea)
            .unwrap()
            .physical()
            .unwrap()[0]
            - heat * 2.)
            .abs()
            < 1e-9
    );
    assert_eq!(
        session
            .get(Field::FuelbedSigma)
            .unwrap()
            .physical()
            .unwrap()[0],
        sigma
    );
}
