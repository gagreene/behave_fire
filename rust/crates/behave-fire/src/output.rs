//! Safe borrowed slices into final typed output buffers.
use crate::{effective_worker_count, BehaveError, ParallelError};
use std::ops::Range;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// One owning column moved from an existing model batch without conversion.
#[derive(Clone, Debug, PartialEq)]
pub enum ColumnValues {
    /// Scientific float64 data, with NaN for unavailable quantities.
    Float64(Vec<f64>),
    /// Exact int32 categories and statuses, with -1 for unavailable values.
    Int32(Vec<i32>),
    /// Boolean fields, stored independently of scientific numbers.
    Bool(Vec<bool>),
}

/// Disjoint borrowed output ranges; all indices are local to the callback range.
pub struct OutputShard<'a> {
    /// Requested physical columns in their prepared slot order.
    pub physical: Vec<&'a mut [f64]>,
    /// Requested category columns in their prepared slot order.
    pub codes: Vec<&'a mut [i32]>,
    /// Requested flag columns in their prepared slot order.
    pub flags: Vec<&'a mut [bool]>,
}

fn borrowed<'a, T>(columns: &'a mut [&mut [T]]) -> Result<Vec<&'a mut [T]>, String> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(columns.len())
        .map_err(|error| error.to_string())?;
    result.extend(columns.iter_mut().map(|values| &mut **values));
    Ok(result)
}

fn prefix<'a, T>(columns: &mut [&'a mut [T]], count: usize) -> Result<Vec<&'a mut [T]>, String> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(columns.len())
        .map_err(|error| error.to_string())?;
    for remaining in columns {
        let (head, tail) = std::mem::take(remaining).split_at_mut(count);
        *remaining = tail;
        result.push(head);
    }
    Ok(result)
}

fn validate<T>(cells: usize, columns: &[&mut [T]], kind: &str) -> Result<(), BehaveError> {
    for (slot, values) in columns.iter().enumerate() {
        if values.len() != cells {
            return Err(BehaveError::LengthMismatch {
                field: format!("output.{kind}.{slot}"),
                expected: cells,
                actual: values.len(),
            });
        }
    }
    Ok(())
}

/// Fill final buffers directly using safe scoped workers, without concatenation.
///
/// Shapes are validated before callbacks run. Empty grids call no worker. All
/// workers are joined after errors; without a spawn failure, the earliest
/// logical chunk's error wins. Allocations occur only while preparing shards.
pub fn fill_chunks<E: Send, F>(
    cells: usize,
    requested: usize,
    physical: &mut [&mut [f64]],
    codes: &mut [&mut [i32]],
    flags: &mut [&mut [bool]],
    fill: F,
) -> Result<(), ParallelError<E>>
where
    F: Fn(Range<usize>, &mut OutputShard<'_>) -> Result<(), E> + Sync,
{
    validate(cells, physical, "physical").map_err(ParallelError::InvalidInput)?;
    validate(cells, codes, "codes").map_err(ParallelError::InvalidInput)?;
    validate(cells, flags, "flags").map_err(ParallelError::InvalidInput)?;
    if cells == 0 {
        return Ok(());
    }
    let workers = effective_worker_count(requested, cells);
    let mut physical = borrowed(physical).map_err(ParallelError::Allocation)?;
    let mut codes = borrowed(codes).map_err(ParallelError::Allocation)?;
    let mut flags = borrowed(flags).map_err(ParallelError::Allocation)?;
    if workers == 1 {
        let mut shard = OutputShard {
            physical,
            codes,
            flags,
        };
        return catch_unwind(AssertUnwindSafe(|| fill(0..cells, &mut shard)))
            .map_err(|_| ParallelError::WorkerPanic)?
            .map_err(ParallelError::Work);
    }
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(workers)
            .map_err(|error| ParallelError::Allocation(error.to_string()))?;
        let mut error = None;
        let mut start = 0;
        for worker in 0..workers {
            let count = cells / workers + usize::from(worker < cells % workers);
            let shard = (|| {
                Ok::<_, String>(OutputShard {
                    physical: prefix(&mut physical, count)?,
                    codes: prefix(&mut codes, count)?,
                    flags: prefix(&mut flags, count)?,
                })
            })();
            let mut shard = match shard {
                Ok(shard) => shard,
                Err(message) => {
                    error = Some(ParallelError::Allocation(message));
                    break;
                }
            };
            let range = start..start + count;
            let fill = &fill;
            match std::thread::Builder::new().spawn_scoped(scope, move || fill(range, &mut shard)) {
                Ok(handle) => handles.push(handle),
                Err(failure) => {
                    error = Some(ParallelError::WorkerSpawn(failure.to_string()));
                    break;
                }
            }
            start += count;
        }
        for handle in handles {
            let result = match handle.join() {
                Ok(result) => result.map_err(ParallelError::Work),
                Err(_) => Err(ParallelError::WorkerPanic),
            };
            if error.is_none() {
                if let Err(failure) = result {
                    error = Some(failure);
                }
            }
        }
        match error {
            Some(error) => Err(error),
            None => Ok(()),
        }
    })
}

/// Actual scientific versus final-write work, excluding setup allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunPhase {
    /// Scientific evaluation of one explicitly valid cell.
    Kernel,
    /// Writing one row into the final typed result columns.
    Encoder,
}
/// Optional static observation of real execution scopes without a runtime dependency.
///
/// The default implementation inlines to the scientific/write operation itself.
/// Profiling and allocation tools can provide a scoped guard without copying math.
pub trait RunObserver: Sync {
    /// Execute one real scope, retaining normal return/error/unwind behavior.
    fn observe<T>(&self, phase: RunPhase, work: impl FnOnce() -> T) -> T;
}
/// Ordinary execution with no counters, timers, or shared state.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopObserver;
impl RunObserver for NoopObserver {
    #[inline(always)]
    fn observe<T>(&self, _phase: RunPhase, work: impl FnOnce() -> T) -> T {
        work()
    }
}
