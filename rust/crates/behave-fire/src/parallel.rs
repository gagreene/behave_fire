//! Bounded, ordered chunk execution using scoped native workers.

use std::fmt;
use std::ops::Range;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Failure of allocation, worker creation, or a native calculation.
#[derive(Debug)]
#[non_exhaustive]
pub enum ParallelError<E> {
    /// A native calculation returned an error.
    Work(E),
    /// A scoped native worker panicked.
    WorkerPanic,
    /// The operating system refused a worker.
    WorkerSpawn(String),
    /// Result or handle storage could not be allocated.
    Allocation(String),
}

impl<E: fmt::Display> fmt::Display for ParallelError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Work(error) => error.fmt(formatter),
            Self::WorkerPanic => formatter.write_str("native worker panicked"),
            Self::WorkerSpawn(message) => {
                write!(formatter, "cannot start native worker: {message}")
            }
            Self::Allocation(message) => {
                write!(formatter, "cannot allocate native work storage: {message}")
            }
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ParallelError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Work(error) => Some(error),
            _ => None,
        }
    }
}

/// Process contiguous ranges with bounded scoped threads and ordered results.
///
/// Empty work calls no closure. Zero workers requests available parallelism;
/// positive requests are capped by the cell count. Each result describes one
/// range, allowing callers to build per-field vectors directly. All spawned
/// workers are joined even after an error. Without a resource failure, the
/// earliest logical chunk's error is returned, independent of scheduling.
pub fn map_chunks<T, E, F>(
    cells: usize,
    requested: usize,
    operation: F,
) -> Result<Vec<T>, ParallelError<E>>
where
    T: Send,
    E: Send,
    F: Fn(Range<usize>) -> Result<T, E> + Sync,
{
    if cells == 0 {
        return Ok(Vec::new());
    }
    let workers = crate::effective_worker_count(requested, cells);
    let mut results = Vec::new();
    results
        .try_reserve_exact(workers)
        .map_err(|error| ParallelError::Allocation(error.to_string()))?;
    if workers == 1 {
        let result = catch_unwind(AssertUnwindSafe(|| operation(0..cells)))
            .map_err(|_| ParallelError::WorkerPanic)?
            .map_err(ParallelError::Work)?;
        results.push(result);
        return Ok(results);
    }
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        handles
            .try_reserve_exact(workers)
            .map_err(|error| ParallelError::Allocation(error.to_string()))?;
        let mut first_error = None;
        let mut start = 0;
        for worker in 0..workers {
            let count = cells / workers + usize::from(worker < cells % workers);
            let end = start + count;
            let range = start..end;
            let operation = &operation;
            match std::thread::Builder::new().spawn_scoped(scope, move || operation(range)) {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    first_error = Some(ParallelError::WorkerSpawn(error.to_string()));
                    break;
                }
            }
            start = end;
        }
        for handle in handles {
            let result = match handle.join() {
                Ok(result) => result.map_err(ParallelError::Work),
                Err(_) => Err(ParallelError::WorkerPanic),
            };
            match result {
                Ok(value) if first_error.is_none() => results.push(value),
                Err(error) if first_error.is_none() => first_error = Some(error),
                _ => {}
            }
        }
        match first_error {
            Some(error) => Err(error),
            None => Ok(results),
        }
    })
}
