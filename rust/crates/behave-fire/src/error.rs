use std::error::Error;
use std::fmt;

/// Malformed API input, distinct from a model's scientific no-data result.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BehaveError {
    /// A cell field was absent or used through an incorrect typed accessor.
    InvalidFieldAccess {
        /// Public field name.
        field: String,
        /// Structural access error, distinct from a known NaN.
        reason: &'static str,
    },
    /// A result field does not belong to the requested model.
    InvalidOutputField {
        /// Public field name.
        field: String,
        /// Requested model name.
        model: &'static str,
    },
    /// Native storage could not be reserved.
    Allocation(String),
    /// A field or output contains a different number of cells than its shape.
    LengthMismatch {
        /// Input or output field that failed validation.
        field: String,
        /// Required logical number of cells.
        expected: usize,
        /// Supplied logical number of cells.
        actual: usize,
    },
    /// A nonempty shape cannot be represented by the platform's `usize`.
    ShapeOverflow {
        /// Requested dimensions, preserved for diagnostics.
        dimensions: Vec<usize>,
    },
    /// A requested row-major tile is reversed or extends beyond its raster.
    InvalidRange {
        /// Inclusive logical starting index.
        start: usize,
        /// Exclusive logical ending index.
        end: usize,
        /// Available raster cell count.
        len: usize,
    },
}

impl fmt::Display for BehaveError {
    /// Describe invalid input without changing its structured diagnostic fields.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFieldAccess { field, reason } => {
                write!(formatter, "cannot access {field}: {reason}")
            }
            Self::InvalidOutputField { field, model } => {
                write!(formatter, "{field} is not an output of {model}")
            }
            Self::Allocation(message) => {
                write!(formatter, "cannot allocate native storage: {message}")
            }
            Self::LengthMismatch {
                field,
                expected,
                actual,
            } => {
                write!(formatter, "{field} has {actual} cells; expected {expected}")
            }
            Self::ShapeOverflow { dimensions } => {
                write!(
                    formatter,
                    "shape {dimensions:?} exceeds the platform cell-count limit"
                )
            }
            Self::InvalidRange { start, end, len } => {
                write!(formatter, "tile {start}..{end} exceeds raster length {len}")
            }
        }
    }
}

impl Error for BehaveError {}
