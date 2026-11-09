use behave_fire::engine::{
    Field, GridSession, ModelRun, OutputRequest, PreparedFuels, Stage, SurfacePlan, SurfaceSource,
};
use behave_fire::output::ColumnValues;
use behave_fire::raster::{SurfaceColumns, SurfaceRaster, SurfaceTweakColumns};
use behave_fire::{FuelModels, GridShape};

fn session() -> GridSession {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let run = ModelRun::Surface {
        raster: SurfaceRaster::new(GridShape::new(vec![3]).unwrap(), SurfaceColumns::default())
            .unwrap(),
        plan: SurfacePlan::default(),
        source: SurfaceSource::Fuels(&fuels),
        tweaks: SurfaceTweakColumns::default(),
    };
    GridSession::new(&run, OutputRequest::surface(), &[], 1).unwrap()
}

#[test]
fn requests_threads_registry_and_override_lifetimes_are_explicit() {
    let mut s = session();
    s.set_outputs(
        OutputRequest::new(&[Field::ReactionIntensity]),
        &[Field::FuelbedSigma],
    )
    .unwrap();
    s.set_threads(4);
    assert_eq!(s.threads(), 4);
    s.persistent(Field::ReactionIntensity, ColumnValues::Float64(vec![100.]))
        .unwrap();
    s.set_physical(Field::ReactionIntensity, vec![200.])
        .unwrap();
    s.run_remaining().unwrap();
    assert_eq!(
        s.get(Field::ReactionIntensity).unwrap().physical().unwrap(),
        &[200.]
    );
    s.clear_injection(Field::ReactionIntensity).unwrap();
    assert_eq!(
        s.get(Field::ReactionIntensity).unwrap().physical().unwrap(),
        &[100.]
    );
    s.clear_persistent(Field::ReactionIntensity).unwrap();
    assert!(s.get(Field::ReactionIntensity).is_err());
    s.run_remaining().unwrap();
    let before = s.get(Field::ReactionIntensity).unwrap().physical().unwrap()[0];
    let original = PreparedFuels::from_registry(&FuelModels::default());
    let mut rows = (0..=260).map(|i| *original.for_grid(i)).collect::<Vec<_>>();
    for row in &mut rows {
        row.heat_of_combustion_dead *= 0.5;
        row.heat_of_combustion_live *= 0.5;
    }
    s.set_fuels(PreparedFuels::from_rows(rows).unwrap())
        .unwrap();
    assert!(s.get(Field::ReactionIntensity).is_err());
    s.run_remaining().unwrap();
    assert!(s.get(Field::ReactionIntensity).unwrap().physical().unwrap()[0] < before);
}

#[test]
fn wrong_pipeline_fields_stages_and_categories_are_rejected_before_mutation() {
    let mut s = session();
    assert!(s.run_stage(Stage::CrownHeat).is_err());
    assert!(s
        .set_physical(Field::CrownFractionBurned, vec![0.5])
        .is_err());
    assert!(s.set_outputs(OutputRequest::crown(), &[]).is_err());
    s.run_remaining().unwrap();
    assert!(s.get(Field::SpreadRate).is_ok());
}
