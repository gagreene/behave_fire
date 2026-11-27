use crate::BehaveError;

/// Checked row-major shape metadata, independent of any array runtime.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GridShape {
    dimensions: Vec<usize>,
    len: usize,
}

impl GridShape {
    /// Borrow the original dimensions; a scalar shape has no dimensions.
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    /// Return whether at least one dimension is zero.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the checked logical cell count; a scalar has one cell.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Construct shape metadata, rejecting nonzero dimension-product overflow.
    ///
    /// Empty dimensions describe one scalar. Any zero dimension describes zero
    /// cells, regardless of the order or size of the other dimensions.
    pub fn new(dimensions: Vec<usize>) -> Result<Self, BehaveError> {
        let len = if dimensions.contains(&0) {
            0
        } else {
            dimensions
                .iter()
                .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
                .ok_or_else(|| BehaveError::ShapeOverflow {
                    dimensions: dimensions.clone(),
                })?
        };
        Ok(Self { dimensions, len })
    }

    /// Validate a named field's logical cell count against this shape.
    pub fn validate_len(&self, field: &str, actual: usize) -> Result<(), BehaveError> {
        if actual == self.len {
            Ok(())
        } else {
            Err(BehaveError::LengthMismatch {
                field: field.to_owned(),
                expected: self.len,
                actual,
            })
        }
    }
}

/// Owned typed per-cell results for small generic batch APIs.
///
/// Large surface and crown APIs will use named per-field vectors instead.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct GridOutput<R> {
    shape: GridShape,
    cells: Vec<R>,
}

impl<R> GridOutput<R> {
    /// Borrow the row-major cell results.
    pub fn cells(&self) -> &[R] {
        &self.cells
    }

    /// Consume the output and return its owned row-major cells.
    pub fn into_cells(self) -> Vec<R> {
        self.cells
    }

    /// Construct output after validating that its cells match the shape.
    pub fn new(shape: GridShape, cells: Vec<R>) -> Result<Self, BehaveError> {
        shape.validate_len("output", cells.len())?;
        Ok(Self { shape, cells })
    }

    /// Borrow the checked output shape.
    pub fn shape(&self) -> &GridShape {
        &self.shape
    }
}

/// A value broadcast to all cells, or one value per logical row-major cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PerCell<'a, T> {
    /// Single value shared by every logical cell.
    Scalar(T),
    /// Borrowed values whose length must match the associated shape.
    Cells(&'a [T]),
}

impl<T> PerCell<'_, T> {
    /// Borrow a value at a logical index.
    ///
    /// Scalars return their value at any index. Callers bound iteration by the
    /// checked shape after calling [`Self::validate`]. Slices return `None` for
    /// indices beyond their length.
    pub fn get(&self, index: usize) -> Option<&T> {
        match self {
            Self::Scalar(value) => Some(value),
            Self::Cells(values) => values.get(index),
        }
    }

    /// Validate a slice's length; scalar broadcasting is valid for every shape.
    pub fn validate(&self, shape: &GridShape, field: &str) -> Result<(), BehaveError> {
        match self {
            Self::Scalar(_) => Ok(()),
            Self::Cells(values) => shape.validate_len(field, values.len()),
        }
    }
}

/// Resolve a worker request without creating threads or a global pool.
///
/// A request of zero uses available parallelism, falling back to one worker
/// when detection fails. Positive requests are capped by the work count. Empty
/// work uses zero workers. Python validates negative and noninteger requests
/// before converting them to this unsigned core API. Serial thresholds require
/// workload measurements and are not imposed by this foundational helper.
pub fn effective_worker_count(requested: usize, cells: usize) -> usize {
    if cells == 0 {
        return 0;
    }
    let workers = if requested == 0 {
        std::thread::available_parallelism().map_or(1, usize::from)
    } else {
        requested
    };
    workers.min(cells)
}
