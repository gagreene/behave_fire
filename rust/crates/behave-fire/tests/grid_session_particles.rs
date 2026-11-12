use behave_fire::engine::{
    Field, GridSession, ModelRun, OutputRequest, PreparedFuels, Stage, SurfacePlan, SurfaceSource,
};
use behave_fire::raster::{SurfaceColumns, SurfaceRaster, SurfaceTweakColumns};
use behave_fire::{FuelModels, GridShape, PerCell};

#[test]
fn particle_stage_uses_actual_static_and_dynamic_fuel_inputs() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let ids = [10, 124];
    let herb = [0.4, 0.8];
    let raster = SurfaceRaster::new(
        GridShape::new(vec![2]).unwrap(),
        SurfaceColumns {
            fuel_model_grid: PerCell::Cells(&ids),
            mlh: PerCell::Cells(&herb),
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
        GridSession::new(&run, OutputRequest::new(Stage::Particles.writes()), &[], 4).unwrap();
    session.run_stage(Stage::Particles).unwrap();
    for index in 0..2 {
        let expected = behave_fire::surface::build_particle_arrays_from_prepared(
            fuels.for_grid(ids[index]),
            0.06,
            0.07,
            0.08,
            herb[index],
            0.90,
        );
        assert_eq!(
            session
                .get(Field::ParticleLoadDead3)
                .unwrap()
                .physical()
                .unwrap()[index],
            expected.load_dead[3]
        );
        assert_eq!(
            session
                .get(Field::ParticleLoadLive0)
                .unwrap()
                .physical()
                .unwrap()[index],
            expected.load_live[0]
        );
        assert_eq!(
            session
                .get(Field::ParticleSavrDead0)
                .unwrap()
                .physical()
                .unwrap()[index],
            expected.savr_dead[0]
        );
        assert_eq!(
            session
                .get(Field::ParticleMoistureLive0)
                .unwrap()
                .physical()
                .unwrap()[index],
            expected.moisture_live[0]
        );
        assert_eq!(
            session.get(Field::ParticleDefined).unwrap().flag().unwrap()[index],
            expected.is_defined
        );
    }
}
