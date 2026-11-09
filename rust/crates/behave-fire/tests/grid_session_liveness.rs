use behave_fire::engine::{CrownPlan, Field, GridSession, ModelRun, OutputRequest, PreparedFuels};
use behave_fire::raster::{CrownColumns, CrownRaster, CrownTweakColumns};
use behave_fire::{FuelModels, GridShape, PerCell};

#[test]
fn repeated_input_edits_rematerialize_freed_heat_ancestors() {
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
    let mut session = GridSession::new(
        &run,
        OutputRequest::new(&[Field::CrownFireHeatPerUnitArea]),
        &[],
        4,
    )
    .unwrap();
    for height in [30., 40., 50.] {
        session
            .set_physical(Field::CanopyHeight, vec![height])
            .unwrap();
        session.run_remaining().unwrap();
        let expected = 1000. + (height - 4.) * 0.01 * 8000.;
        assert_eq!(
            session
                .get(Field::CrownFireHeatPerUnitArea)
                .unwrap()
                .physical()
                .unwrap(),
            &[expected; 3]
        );
        assert!(session.get(Field::CrownFuelLoad).is_err());
        assert!(session.get(Field::CanopyHeatPerUnitArea).is_err());
    }
}

#[test]
fn injection_survives_unrelated_ancestor_recovery_and_fused_restart_clears_it() {
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
        OutputRequest::new(&[Field::CrownFireHeatPerUnitArea]),
        &[],
        1,
    )
    .unwrap();
    session
        .set_physical(Field::CanopyHeatPerUnitArea, vec![500.])
        .unwrap();
    session.run_remaining().unwrap();
    session
        .set_physical(Field::CanopyHeight, vec![40.])
        .unwrap();
    session.run_remaining().unwrap();
    assert_eq!(
        session
            .get(Field::CanopyHeatPerUnitArea)
            .unwrap()
            .physical()
            .unwrap(),
        &[500.]
    );
    session.run_fused().unwrap();
    assert_eq!(
        session
            .get(Field::CrownFireHeatPerUnitArea)
            .unwrap()
            .physical()
            .unwrap(),
        &[3880.; 2]
    );
}

#[test]
fn final_only_complete_inject_fraction_complete_again_and_keep_two_injections() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let raster = CrownRaster::new(
        GridShape::new(vec![1031]).unwrap(),
        CrownColumns {
            canopy_base_height: PerCell::Scalar(4.),
            canopy_height: PerCell::Scalar(30.),
            canopy_bulk_density: PerCell::Scalar(0.01),
            surface_ros: PerCell::Scalar(2.),
            surface_hpua: PerCell::Scalar(1000.),
            surface_fli: PerCell::Scalar(1e6),
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
        OutputRequest::new(&[
            Field::FinalSpreadRate,
            Field::FinalHeatPerUnitArea,
            Field::FinalFirelineIntensity,
        ]),
        &[Field::CrownFireSpreadRate, Field::FireType],
        4,
    )
    .unwrap();
    session.run_remaining().unwrap();
    assert_eq!(session.get(Field::FireType).unwrap().code().unwrap()[0], 1);
    assert!(session.get(Field::ReactionIntensity).is_err());
    session
        .set_physical(Field::CrownFractionBurned, vec![0.5])
        .unwrap();
    session
        .set_physical(Field::CanopyHeatPerUnitArea, vec![500.])
        .unwrap();
    for moisture in [0.9, 1.1] {
        session.set_physical(Field::Mlw, vec![moisture]).unwrap();
        session.run_remaining().unwrap();
        assert_eq!(
            session
                .get(Field::CrownFractionBurned)
                .unwrap()
                .physical()
                .unwrap(),
            &[0.5]
        );
        assert_eq!(
            session
                .get(Field::CanopyHeatPerUnitArea)
                .unwrap()
                .physical()
                .unwrap(),
            &[500.]
        );
        let rate = session
            .get(Field::CrownFireSpreadRate)
            .unwrap()
            .physical()
            .unwrap()[0];
        let expected = 2. + 0.5 * (rate - 2.);
        assert!(
            (session
                .get(Field::FinalSpreadRate)
                .unwrap()
                .physical()
                .unwrap()[0]
                - expected)
                .abs()
                < 1e-9
        );
        assert_eq!(
            session
                .get(Field::FinalHeatPerUnitArea)
                .unwrap()
                .physical()
                .unwrap(),
            &[1250.; 1031]
        );
        assert!(
            (session
                .get(Field::FinalFirelineIntensity)
                .unwrap()
                .physical()
                .unwrap()[0]
                - 1250. * expected / 60.)
                .abs()
                < 1e-9
        );
    }
}
