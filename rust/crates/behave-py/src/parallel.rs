//! Ordered native chunks writing directly into typed output columns.
use crate::column_output::{ColumnBundle, RowWriter};
use crate::operation::Operation;
use crate::prepared::PreparedArguments;
use std::ops::Range;

type BatchEvaluation = (ColumnBundle, f64, usize);

pub fn surface(
    a: &PreparedArguments<'_>,
    i: usize,
) -> Result<behave_fire::surface::SurfaceResult, String> {
    let mut input = crate::structures::input(a, i)?;
    use crate::prepared::Field;
    input.m1h = a.n(Field::M1h, i)?;
    input.m10h = a.n(Field::M10h, i)?;
    input.m100h = a.n(Field::M100h, i)?;
    input.mlh = a.n(Field::Mlh, i)?;
    input.mlw = a.n(Field::Mlw, i)?;
    let plan = behave_fire::engine::SurfacePlan::new(crate::prepared::controls::surface(a)?);
    let fuel = crate::structures::fuel(a, i)?;
    #[cfg(feature = "allocation-probe")]
    let _scope = crate::allocation_probe::Scope::enter(crate::allocation_probe::KERNEL);
    Ok(plan.run_cell(&input, &fuel, behave_fire::engine::SurfaceTweaks::default()))
}

pub fn crown(
    a: &PreparedArguments<'_>,
    i: usize,
) -> Result<behave_fire::crown::CrownResult, String> {
    let plan = behave_fire::engine::CrownPlan::new(crate::prepared::controls::crown(a));
    let input = crate::crown::input(a, i)?;
    let surface = crate::crown::surface_values(a, i);
    let fuel = crate::crown::fuel(a, i)?;
    #[cfg(feature = "allocation-probe")]
    let _scope = crate::allocation_probe::Scope::enter(crate::allocation_probe::KERNEL);
    Ok(plan.run_cell(
        &input,
        &surface,
        &fuel,
        behave_fire::engine::CrownTweaks::default(),
    ))
}

fn default_single(operation: Operation, a: &PreparedArguments<'_>, single: bool) -> bool {
    if !single || operation.main_model() || operation.family() == crate::operation::Family::Units {
        return false;
    }
    !matches!(
        operation,
        Operation::SurfaceCalculateEffectiveWindSpeed
            | Operation::SurfaceRunSurfaceFire
            | Operation::SurfaceCalculateFuelbedIntermediates
            | Operation::SurfaceCalculateSlopePhi
            | Operation::SurfaceCalculateSurfaceRateVector
    ) && !(operation == Operation::SurfaceCalculateDirectionOfMaxSpread
        && a.crown_options().wind_orientation == behave_fire::engine::WindOrientation::North)
}

pub fn empty(
    operation: Operation,
    arguments: &PreparedArguments<'_>,
    single: bool,
) -> Result<ColumnBundle, String> {
    let mut seeds = Vec::new();
    let mut output = RowWriter::record(&mut seeds);
    crate::dispatch::evaluate(operation, arguments, 0, single, &mut output)?;
    output.finish()?;
    ColumnBundle::new(
        seeds,
        0,
        default_single(operation, arguments, single),
        arguments.metadata_varies(operation == Operation::CrownRunCrownFire),
    )
}

pub fn evaluate(
    operation: Operation,
    arguments: &PreparedArguments<'_>,
    count: usize,
    threads: usize,
) -> Option<Result<BatchEvaluation, String>> {
    crate::prepared::units::PreparedUnits::new(operation.name(), arguments).map(|plan| {
        plan.and_then(|plan| {
            let chunks =
                behave_fire::map_chunks(count, threads, |range| plan.run(range)).map_err(error)?;
            let appended = chunks
                .iter()
                .skip(1)
                .map(|values| values.len() * std::mem::size_of::<f64>())
                .sum();
            let started = std::time::Instant::now();
            let mut chunks = chunks.into_iter();
            let mut values = chunks.next().unwrap_or_default();
            values
                .try_reserve_exact(count - values.len())
                .map_err(|error| format!("MemoryError: {error}"))?;
            for mut chunk in chunks {
                values.append(&mut chunk);
            }
            Ok((
                ColumnBundle::from_f64("", values),
                started.elapsed().as_secs_f64(),
                appended,
            ))
        })
    })
}

pub fn chunk(
    operation: Operation,
    arguments: &PreparedArguments<'_>,
    range: Range<usize>,
    single: bool,
) -> Result<ColumnBundle, String> {
    let mut seeds = Vec::new();
    let mut capture = RowWriter::record(&mut seeds);
    crate::dispatch::evaluate(operation, arguments, range.start, single, &mut capture)?;
    capture.finish()?;
    let mut output = ColumnBundle::new(
        seeds,
        range.len(),
        default_single(operation, arguments, single),
        arguments.metadata_varies(operation == Operation::CrownRunCrownFire),
    )?;
    output.write_first(operation.main_model())?;
    for index in range.start + 1..range.end {
        let mut row = RowWriter::row(&mut output, index - range.start, operation.main_model());
        crate::dispatch::evaluate(operation, arguments, index, single, &mut row)?;
        row.finish()?;
    }
    output.check_lengths()?;
    Ok(output)
}

pub fn merge(chunks: Vec<ColumnBundle>, count: usize) -> Result<ColumnBundle, String> {
    let mut chunks = chunks.into_iter();
    let mut output = chunks
        .next()
        .ok_or_else(|| "missing output chunks".to_owned())?;
    output.reserve_total(count)?;
    for chunk in chunks {
        output.append(chunk)?;
    }
    output.finish()?;
    Ok(output)
}

pub fn error<E: std::fmt::Display>(error: behave_fire::ParallelError<E>) -> String {
    match error {
        behave_fire::ParallelError::Work(error) => error.to_string(),
        behave_fire::ParallelError::Allocation(message) => format!("MemoryError: {message}"),
        other => format!("RuntimeError: {other}"),
    }
}
