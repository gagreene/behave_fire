//! Connect configured core scopes without adding dependencies to the counter.
#[cfg(not(feature = "allocation-probe"))]
pub(crate) use behave_fire::output::NoopObserver as Observer;

#[cfg(feature = "allocation-probe")]
#[derive(Default)]
pub(crate) struct Observer;

#[cfg(feature = "allocation-probe")]
impl behave_fire::output::RunObserver for Observer {
    fn observe<T>(&self, phase: behave_fire::output::RunPhase, work: impl FnOnce() -> T) -> T {
        let scope = match phase {
            behave_fire::output::RunPhase::Kernel => crate::allocation_probe::KERNEL,
            behave_fire::output::RunPhase::Encoder => crate::allocation_probe::ENCODER,
        };
        let _guard = crate::allocation_probe::Scope::enter(scope);
        work()
    }
}
