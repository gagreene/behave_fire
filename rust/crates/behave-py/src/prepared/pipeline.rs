//! Whole-call native raster execution and direct typed output.
use super::{crown::PreparedCrown, surface::PreparedSurface, Field};
use crate::column_output::ColumnBundle;
use behave_fire::engine::{CrownTweaks, PreparedFuels, SurfaceTweaks};

pub(crate) fn surface_pipeline(
    prepared: &PreparedSurface<'_>,
    fuels: &PreparedFuels,
    threads: usize,
) -> Result<crate::parallel::BatchEvaluation, String> {
    let raster = prepared.raster()?;
    let count = raster.shape().len();
    let chunks = behave_fire::map_chunks(count, threads, |range| {
        crate::parallel::chunk_with(range, true, false, false, |index, output| {
            let result = {
                #[cfg(feature = "allocation-probe")]
                let _scope = crate::allocation_probe::Scope::enter(crate::allocation_probe::KERNEL);
                raster
                    .evaluate_cell(index, prepared.plan(), fuels, SurfaceTweaks::default())
                    .ok_or_else(|| "invalid prepared Surface index".to_owned())?
            };
            crate::structures::surface_output(output, result);
            Ok(())
        })
    })
    .map_err(crate::parallel::error)?;
    finish(chunks, count)
}

pub(crate) fn crown_pipeline(
    prepared: &PreparedCrown<'_>,
    fuels: &PreparedFuels,
    threads: usize,
) -> Result<crate::parallel::BatchEvaluation, String> {
    let raster = prepared.raster()?;
    let count = raster.shape().len();
    let a = prepared.arguments();
    let compact = !a.has_crown_python_payloads();
    let checked =
        a.has_masks() || a.has_single() || a.has_integers() || a.has_crown_python_payloads();
    let chunks = behave_fire::map_chunks(count, threads, |range| {
        crate::parallel::chunk_with(
            range,
            true,
            false,
            a.metadata_varies(true),
            |index, output| {
                if checked {
                    let mut supplied = behave_fire::crown::SurfaceFireInputs::new();
                    supplied.surface_ros = a.masked(Field::SurfaceResultsSpreadRate, index)?;
                    supplied.surface_fli =
                        a.masked(Field::SurfaceResultsFirelineIntensity, index)?;
                    supplied.surface_hpua =
                        a.masked(Field::SurfaceResultsHeatPerUnitArea, index)?;
                    let result = {
                        #[cfg(feature = "allocation-probe")]
                        let _scope =
                            crate::allocation_probe::Scope::enter(crate::allocation_probe::KERNEL);
                        let input = raster
                            .input_cell(index)
                            .ok_or_else(|| "invalid prepared Crown index".to_owned())?;
                        let compact_result = if compact {
                            behave_fire::crown::try_run_compact_surface_inputs(
                                &supplied,
                                &input,
                                fuels.for_grid(10),
                                prepared.plan().options(),
                            )
                        } else {
                            None
                        };
                        compact_result.unwrap_or_else(|| {
                            behave_fire::crown::run_with_surface_inputs(
                                &supplied,
                                &input,
                                fuels.for_grid(10),
                                prepared.plan().options(),
                            )
                        })
                    }
                    .map_err(|error| error.to_string())?;
                    crate::masked::outputs(output, result.named_values())?;
                    output.i32("fire_type", result.fire_type);
                } else {
                    let result = {
                        #[cfg(feature = "allocation-probe")]
                        let _scope =
                            crate::allocation_probe::Scope::enter(crate::allocation_probe::KERNEL);
                        raster
                            .evaluate_cell(index, prepared.plan(), fuels, CrownTweaks::default())
                            .ok_or_else(|| "invalid prepared Crown index".to_owned())?
                    };
                    crate::dispatch::named(output, result.named_values());
                    output.i32("fire_type", result.fire_type);
                }
                Ok(())
            },
        )
    })
    .map_err(crate::parallel::error)?;
    finish(chunks, count)
}

pub(crate) fn prepare_fuels(
    arguments: &super::Arguments,
    count: usize,
) -> Result<PreparedFuels, String> {
    let fields = super::PreparedArguments::new(
        arguments,
        &behave_fire::GridShape::new(vec![count]).map_err(|error| error.to_string())?,
    )?;
    let mut rows = Vec::new();
    rows.try_reserve_exact(count)
        .map_err(|error| format!("MemoryError: {error}"))?;
    for index in 0..count {
        rows.push(crate::structures::fuel(&fields, index)?);
    }
    PreparedFuels::from_rows(rows).map_err(|error| error.to_string())
}

fn finish(
    chunks: Vec<ColumnBundle>,
    count: usize,
) -> Result<crate::parallel::BatchEvaluation, String> {
    let bytes = chunks.iter().skip(1).map(ColumnBundle::bytes).sum();
    let started = std::time::Instant::now();
    let output = crate::parallel::merge(chunks, count)?;
    Ok((output, started.elapsed().as_secs_f64(), bytes))
}
