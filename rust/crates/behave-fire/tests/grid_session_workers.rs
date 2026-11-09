use behave_fire::engine::{
    GridSession, ModelRun, OutputRequest, PreparedFuels, Stage, SurfacePlan, SurfaceSource,
};
use behave_fire::output::{RunObserver, RunPhase};
use behave_fire::raster::{SurfaceColumns, SurfaceRaster, SurfaceTweakColumns};
use behave_fire::{FuelModels, GridShape};
use std::collections::HashSet;
use std::sync::Mutex;

#[derive(Default)]
struct Workers(Mutex<HashSet<std::thread::ThreadId>>);
impl RunObserver for Workers {
    fn observe<T>(&self, phase: RunPhase, work: impl FnOnce() -> T) -> T {
        if phase == RunPhase::Kernel {
            self.0.lock().unwrap().insert(std::thread::current().id());
        }
        work()
    }
}

#[test]
fn staged_columns_execute_on_requested_distinct_workers() {
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
    let workers = Workers::default();
    session
        .run_stage_observed(Stage::Particles, &workers)
        .unwrap();
    assert!(workers.0.lock().unwrap().len() >= 4);
}
