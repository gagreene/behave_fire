use behave_fire::engine::{
    Field, GridSession, ModelRun, OutputRequest, PreparedFuels, SurfaceOptions, SurfacePlan,
    SurfaceSource,
};
use behave_fire::raster::{SurfaceColumns, SurfaceRaster, SurfaceTweakColumns};
use behave_fire::{FuelModels, GridShape, PerCell};

#[test]
fn snapshot_owns_original_columns_and_validates_typed_edits() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let mut moisture = vec![0.06; 3];
    let mut session = {
        let raster = SurfaceRaster::new(
            GridShape::new(vec![3]).unwrap(),
            SurfaceColumns {
                m1h: PerCell::Cells(&moisture),
                ..SurfaceColumns::default()
            },
        )
        .unwrap();
        let run = ModelRun::Surface {
            raster,
            plan: SurfacePlan::new(SurfaceOptions::default()),
            source: SurfaceSource::Fuels(&fuels),
            tweaks: SurfaceTweakColumns::default(),
        };
        GridSession::new(&run, OutputRequest::surface(), &[], 4).unwrap()
    };
    moisture.fill(0.9);
    assert_eq!(
        session.get(Field::M1h).unwrap().physical().unwrap(),
        &[0.06; 3]
    );
    assert_eq!(session.shape().len(), 3);
    assert_eq!(session.threads(), 4);
    session
        .set_code(Field::FuelModelGrid, vec![10, 124, 0])
        .unwrap();
    assert_eq!(
        session.get(Field::FuelModelGrid).unwrap().code().unwrap(),
        &[10, 124, 0]
    );
    session
        .set_flag(Field::Validity, vec![false, true, false])
        .unwrap();
    assert_eq!(
        session.get(Field::Validity).unwrap().flag().unwrap(),
        &[false, true, false]
    );
    assert!(session.set_flag(Field::M1h, vec![true]).is_err());
    assert!(session.set_code(Field::FuelModelGrid, vec![1, 2]).is_err());
    assert_eq!(
        session.get(Field::FuelModelGrid).unwrap().code().unwrap(),
        &[10, 124, 0]
    );
}

#[test]
fn fuelbed_and_reaction_stages_match_scalar_composition() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    for workers in [1, 4] {
        let raster = SurfaceRaster::new(
            GridShape::new(vec![3]).unwrap(),
            SurfaceColumns {
                fuel_model_grid: PerCell::Scalar(124),
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
        let mut session = GridSession::new(&run, OutputRequest::surface(), &[], workers).unwrap();
        session
            .run_stage(behave_fire::engine::Stage::Particles)
            .unwrap();
        session
            .run_stage(behave_fire::engine::Stage::Fuelbed)
            .unwrap();
        session
            .run_stage(behave_fire::engine::Stage::Reaction)
            .unwrap();
        let particles = behave_fire::surface::build_particle_arrays_from_prepared(
            fuels.for_grid(124),
            0.06,
            0.07,
            0.08,
            0.6,
            0.9,
        );
        let bed = behave_fire::surface::calculate_fuelbed_intermediates(&particles);
        for (name, value) in bed.named_values() {
            let field = Field::from_name(&format!("ib.{name}")).unwrap();
            let actual = session.get(field).unwrap().physical().unwrap()[0];
            assert!(
                (actual - value).abs() <= 1e-12 * value.abs().max(1.),
                "{name}: {actual} != {value}"
            );
        }
        let reaction = behave_fire::surface::calculate_reaction_intensity(&bed);
        assert!(
            (session
                .get(Field::ReactionIntensity)
                .unwrap()
                .physical()
                .unwrap()[0]
                - reaction)
                .abs()
                < 1e-9
        );
    }
}

#[test]
fn surface_session_matches_configured_full_run_including_undefined_fuel() {
    use behave_fire::engine::{run_model, ModelBatch, WafMethod, WindHeight, WindOrientation};
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let ids = [10, 124, 0];
    for waf in [WafMethod::CrownRatio, WafMethod::UserInput] {
        let raster = SurfaceRaster::new(
            GridShape::new(vec![3]).unwrap(),
            SurfaceColumns {
                fuel_model_grid: PerCell::Cells(&ids),
                wind_speed: PerCell::Scalar(88.),
                wind_direction: PerCell::Scalar(35.),
                slope: PerCell::Scalar(12.),
                canopy_cover: PerCell::Scalar(0.4),
                canopy_height: PerCell::Scalar(30.),
                crown_ratio: PerCell::Scalar(0.5),
                user_waf: Some(PerCell::Scalar(0.4)),
                ..SurfaceColumns::default()
            },
        )
        .unwrap();
        let run = ModelRun::Surface {
            raster,
            plan: SurfacePlan::new(SurfaceOptions {
                wind_orientation: WindOrientation::North,
                wind_height: WindHeight::TenMetre,
                waf_method: waf,
            }),
            source: SurfaceSource::Fuels(&fuels),
            tweaks: SurfaceTweakColumns::default(),
        };
        let request = OutputRequest::surface();
        let ModelBatch::Surface(expected) = run_model(&run, &request, 1).unwrap() else {
            panic!("wrong model")
        };
        let mut session = GridSession::new(&run, request, &[], 4).unwrap();
        session.run_remaining().unwrap();
        for (name, values) in expected.into_named_columns() {
            let behave_fire::output::ColumnValues::Float64(values) = values else {
                panic!("wrong dtype")
            };
            let actual = session
                .get(Field::from_name(name).unwrap())
                .unwrap()
                .physical()
                .unwrap();
            for (a, b) in actual.iter().zip(values) {
                assert!(
                    (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * b.abs().max(1.),
                    "{name}: {a} != {b}"
                );
            }
        }
    }
}

#[test]
fn crown_session_matches_full_run_with_independent_missing_heat() {
    use behave_fire::engine::{run_model, CrownPlan, ModelBatch};
    use behave_fire::raster::{CrownColumns, CrownRaster, CrownTweakColumns};
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let heat = [f64::NAN, 1000.];
    let raster = CrownRaster::new(
        GridShape::new(vec![2]).unwrap(),
        CrownColumns {
            wind_speed: PerCell::Scalar(88.),
            wind_direction: PerCell::Scalar(35.),
            slope: PerCell::Scalar(12.),
            canopy_base_height: PerCell::Scalar(4.),
            canopy_height: PerCell::Scalar(30.),
            canopy_bulk_density: PerCell::Scalar(0.01),
            surface_ros: PerCell::Scalar(2.),
            surface_fli: PerCell::Scalar(200.),
            surface_hpua: PerCell::Cells(&heat),
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
    let request = OutputRequest::crown();
    let ModelBatch::Crown(expected) = run_model(&run, &request, 1).unwrap() else {
        panic!("wrong model")
    };
    let mut session = GridSession::new(&run, request, &[], 4).unwrap();
    session.run_remaining().unwrap();
    for (name, values) in expected.into_named_columns() {
        let actual = session.get(Field::from_name(name).unwrap()).unwrap();
        match values {
            behave_fire::output::ColumnValues::Float64(values) => {
                for (a, b) in actual.physical().unwrap().iter().zip(values) {
                    assert!(
                        (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * b.abs().max(1.),
                        "{name}: {a} != {b}"
                    );
                }
            }
            behave_fire::output::ColumnValues::Int32(values) => {
                assert_eq!(actual.code().unwrap(), values)
            }
            _ => panic!("wrong type"),
        }
    }
}
