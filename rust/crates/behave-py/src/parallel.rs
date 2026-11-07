//! Fused native model pipelines and ordered field-wise chunk concatenation.
use crate::arrays::Arguments;
use behave_fire::{
    batch::{CrownBatch, SurfaceBatch},
    GridShape,
};
use std::collections::BTreeMap;
use std::ops::Range;

/// Named output columns, concatenation duration, and appended byte count.
type BatchEvaluation = (BTreeMap<String, Vec<f64>>, f64, usize);

pub fn surface(a: &Arguments, i: usize) -> Result<behave_fire::surface::SurfaceResult, String> {
    let mut input = crate::structures::input(a, i)?;
    input.m1h = a.n("m1h", i)?;
    input.m10h = a.n("m10h", i)?;
    input.m100h = a.n("m100h", i)?;
    input.mlh = a.n("mlh", i)?;
    input.mlw = a.n("mlw", i)?;
    let plan = behave_fire::engine::SurfacePlan::new(crate::controls::surface(a)?);
    let fuel = crate::structures::fuel(a, i)?;
    #[cfg(feature = "allocation-probe")]
    let _scope = crate::allocation_probe::Scope::enter(crate::allocation_probe::KERNEL);
    Ok(plan.run_cell(&input, &fuel, behave_fire::engine::SurfaceTweaks::default()))
}

pub fn evaluate(
    operation: &str,
    arguments: &Arguments,
    shape: &[usize],
    threads: usize,
) -> Option<Result<BatchEvaluation, String>> {
    let shape = match GridShape::new(shape.to_vec()) {
        Ok(shape) => shape,
        Err(error) => return Some(Err(error.to_string())),
    };
    let result = if operation == "model.run_surface" {
        SurfaceBatch::run_with(shape, threads, |index| surface(arguments, index)).map(|batch| {
            let stats = *batch.statistics();
            (
                batch.into_named_vectors(),
                stats.concatenation_seconds,
                stats.appended_bytes,
            )
        })
    } else if operation == "crown.run_crown_fire"
        && !arguments.has_masks()
        && !arguments.has_single()
        && !arguments.has_integers()
        // Crown normalizes aspect and foliar moisture to f64 arrays; unit codes
        // are validated controls. Other weak and object payloads retain typing.
        && !arguments.has_crown_python_payloads()
    {
        CrownBatch::run_with(shape, threads, |index| {
            let plan = behave_fire::engine::CrownPlan::new(crate::controls::crown(arguments));
            let input = crate::crown::input(arguments, index)?;
            let surface = crate::crown::surface_values(arguments, index);
            let fuel = crate::crown::fuel(arguments, index)?;
            #[cfg(feature = "allocation-probe")]
            let _scope = crate::allocation_probe::Scope::enter(crate::allocation_probe::KERNEL);
            Ok::<_, String>(plan.run_cell(
                &input,
                &surface,
                &fuel,
                behave_fire::engine::CrownTweaks::default(),
            ))
        })
        .map(|batch| {
            let stats = *batch.statistics();
            (
                batch.into_named_vectors(),
                stats.concatenation_seconds,
                stats.appended_bytes,
            )
        })
    } else {
        return None;
    };
    Some(
        result
            .map(|(fields, seconds, bytes)| {
                (
                    fields
                        .into_iter()
                        .map(|(name, values)| (name.to_owned(), values))
                        .collect(),
                    seconds,
                    bytes,
                )
            })
            .map_err(error),
    )
}

pub fn chunk(
    operation: &str,
    arguments: &Arguments,
    range: Range<usize>,
    single: bool,
) -> Result<BTreeMap<String, Vec<f64>>, String> {
    let count = range.len();
    let mut output: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for index in range {
        for (name, value) in crate::dispatch::evaluate(operation, arguments, index, single)? {
            let values = match output.entry(name) {
                std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    let mut values = Vec::new();
                    values
                        .try_reserve_exact(count)
                        .map_err(|error| format!("MemoryError: {error}"))?;
                    entry.insert(values)
                }
            };
            values.push(value);
        }
    }
    Ok(output)
}

pub fn merge(
    chunks: Vec<BTreeMap<String, Vec<f64>>>,
    count: usize,
) -> Result<BTreeMap<String, Vec<f64>>, String> {
    let mut chunks = chunks.into_iter();
    let mut output = chunks.next().unwrap_or_default();
    if chunks.len() > 0 {
        for values in output.values_mut() {
            values
                .try_reserve_exact(count - values.len())
                .map_err(|error| format!("MemoryError: {error}"))?;
        }
    }
    for chunk in chunks {
        for (name, mut values) in chunk {
            output.entry(name).or_default().append(&mut values);
        }
    }
    Ok(output)
}
/// Preserve native error categories across the Python boundary.
pub fn error<E: std::fmt::Display>(error: behave_fire::ParallelError<E>) -> String {
    match error {
        behave_fire::ParallelError::Work(error) => error.to_string(),
        behave_fire::ParallelError::Allocation(message) => format!("MemoryError: {message}"),
        other => format!("RuntimeError: {other}"),
    }
}
