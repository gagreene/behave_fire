use behave_fire::engine::{
    Field, GridSession, ModelRun, OutputRequest, PreparedFuels, SurfacePlan, SurfaceSource,
};
use behave_fire::output::{RunObserver, RunPhase};
use behave_fire::raster::{SurfaceColumns, SurfaceRaster, SurfaceTweakColumns};
use behave_fire::{FuelModels, GridShape};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Default)]
struct Calls(AtomicUsize);
impl RunObserver for Calls {
    fn observe<T>(&self, phase: RunPhase, work: impl FnOnce() -> T) -> T {
        if phase == RunPhase::Kernel {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
        work()
    }
}
#[test]
fn fused_restart_uses_one_configured_model_kernel_per_cell() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let raster = SurfaceRaster::new(
        GridShape::new(vec![1031]).unwrap(),
        SurfaceColumns::default(),
    )
    .unwrap();
    let run = ModelRun::Surface {
        raster,
        plan: SurfacePlan::default(),
        source: SurfaceSource::Fuels(&fuels),
        tweaks: SurfaceTweakColumns::default(),
    };
    let mut session = GridSession::new(&run, OutputRequest::surface(), &[], 4).unwrap();
    session
        .set_physical(Field::ReactionIntensity, vec![1.])
        .unwrap();
    let calls = Calls::default();
    session.run_fused_observed(&calls).unwrap();
    assert_eq!(calls.0.load(Ordering::Relaxed), 1031);
    assert_ne!(
        session
            .get(Field::ReactionIntensity)
            .unwrap()
            .physical()
            .unwrap()[0],
        1.
    );
    assert!(session.get(Field::FuelbedSigma).is_err());
}
