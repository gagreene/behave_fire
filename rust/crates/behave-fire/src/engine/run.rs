//! Fully configured borrowed model runs over the existing typed batch owners.
use super::{CrownPlan, OutputRequest, PreparedFuels, SurfacePlan};
use crate::batch::{CrownBatch, SurfaceBatch};
use crate::output::{NoopObserver, RunObserver, RunPhase};
use crate::raster::{
    CrownRaster, CrownTweakColumns, FuelbedColumns, SurfaceRaster, SurfaceTweakColumns,
};
use crate::{BehaveError, ParallelError};

/// Surface starts from immutable model fuels or explicitly supplied fuelbed fields.
#[derive(Clone, Copy, Debug)]
#[allow(clippy::large_enum_variant)] // Borrowed columns stay inline once per call, outside cell loops.
pub enum SurfaceSource<'a> {
    /// Prepared immutable model lookup.
    Fuels(&'a PreparedFuels),
    /// Caller-supplied numeric fuelbed with an independent typed defined flag.
    Fuelbeds(FuelbedColumns<'a>),
}
/// Complete inputs, options, and persistent typed overrides for one model run.
#[derive(Clone, Debug)]
#[allow(clippy::large_enum_variant)] // Configuration is borrowed, never copied per cell or boxed per run.
pub enum ModelRun<'a> {
    /// A Surface raster and its exact configuration/source.
    Surface {
        /// Validated borrowed inputs.
        raster: SurfaceRaster<'a>,
        /// Wind/WAF controls.
        plan: SurfacePlan,
        /// Model fuel lookup or resumed fuelbed.
        source: SurfaceSource<'a>,
        /// Persistent scalar/per-cell overrides.
        tweaks: SurfaceTweakColumns<'a>,
    },
    /// A Crown raster with independent supplied Surface fields.
    Crown {
        /// Validated borrowed inputs.
        raster: CrownRaster<'a>,
        /// Crown wind controls.
        plan: CrownPlan,
        /// Immutable model-10 fuel source.
        fuels: &'a PreparedFuels,
        /// Persistent scalar/per-cell overrides.
        tweaks: CrownTweakColumns<'a>,
    },
}
/// One existing model batch; this enum introduces no second result allocation.
#[derive(Clone, Debug)]
pub enum ModelBatch {
    /// Owned Surface columns.
    Surface(SurfaceBatch),
    /// Owned Crown columns with int32 classification.
    Crown(CrownBatch),
}
/// Execute a configured model directly into final requested columns.
pub fn run_model(
    run: &ModelRun<'_>,
    request: &OutputRequest,
    threads: usize,
) -> Result<ModelBatch, ParallelError<BehaveError>> {
    run_model_observed(run, request, threads, &NoopObserver)
}
/// Execute the same configured model with optional scientific/write scope observation.
pub fn run_model_observed<O: RunObserver>(
    run: &ModelRun<'_>,
    request: &OutputRequest,
    threads: usize,
    observer: &O,
) -> Result<ModelBatch, ParallelError<BehaveError>> {
    match run {
        ModelRun::Surface {
            raster,
            plan,
            source,
            tweaks,
        } => {
            tweaks
                .validate(raster.shape())
                .map_err(ParallelError::Work)?;
            if let SurfaceSource::Fuelbeds(columns) = source {
                columns
                    .validate(raster.shape())
                    .map_err(ParallelError::Work)?;
            }
            SurfaceBatch::run_with_request_observed(
                raster.shape().clone(),
                request,
                threads,
                |index| {
                    let Some(cell) = raster.input_cell(index) else {
                        return Ok(super::missing::surface());
                    };
                    let value = observer.observe(RunPhase::Kernel, || match source {
                        SurfaceSource::Fuels(fuels) => plan.run_available_cell(
                            &cell,
                            fuels.for_grid(cell.fuel_model_grid),
                            tweaks.cell(index),
                        ),
                        SurfaceSource::Fuelbeds(columns) => plan.run_available_from_fuelbed::<0>(
                            &cell,
                            &columns.cell(index),
                            tweaks.cell(index),
                        ),
                    });
                    Ok(value)
                },
                observer,
            )
            .map(ModelBatch::Surface)
        }
        ModelRun::Crown {
            raster,
            plan,
            fuels,
            tweaks,
        } => {
            tweaks
                .validate(raster.shape())
                .map_err(ParallelError::Work)?;
            CrownBatch::run_with_request_observed(
                raster.shape().clone(),
                request,
                threads,
                |index| {
                    let Some(input) = raster.input_cell(index) else {
                        return Ok(super::missing::crown());
                    };
                    let supplied = raster
                        .supplied_surface(index)
                        .expect("validated Crown index");
                    Ok(observer.observe(RunPhase::Kernel, || {
                        plan.run_available_cell(
                            &input,
                            &supplied,
                            fuels.for_grid(10),
                            tweaks.cell(index),
                        )
                    }))
                },
                observer,
            )
            .map(ModelBatch::Crown)
        }
    }
}
