use std::error::Error;
use std::fmt;

/// Malformed API input, distinct from a model's scientific no-data result.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum BehaveError {
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
}

impl fmt::Display for BehaveError {
    /// Describe invalid input without changing its structured diagnostic fields.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
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
        }
    }
}

impl Error for BehaveError {}
