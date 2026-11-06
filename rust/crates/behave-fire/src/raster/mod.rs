//! Borrowed raster columns, explicit stage inputs, and row-major tile execution.
mod crown;
mod inputs;
mod outputs;
mod stages;
mod surface;

pub use crown::CrownRaster;
pub use inputs::{CrownColumns, SurfaceColumns};
pub use outputs::{CrownRasterOutput, SurfaceRasterOutput};
pub use stages::{CrownTweakColumns, FuelbedColumns, SurfaceTweakColumns};
pub use surface::SurfaceRaster;

use crate::{BehaveError, GridShape, ParallelError, PerCell};
use std::ops::Range;

pub(crate) fn value<T: Copy>(column: &PerCell<'_, T>, index: usize) -> T {
    *column.get(index).expect("validated row-major column")
}

pub(crate) fn validate_optional<T>(
    column: &Option<PerCell<'_, T>>,
    shape: &GridShape,
    name: &str,
) -> Result<(), BehaveError> {
    if let Some(column) = column {
        column.validate(shape, name)?;
    }
    Ok(())
}

pub(crate) fn tile_shape(range: &Range<usize>, len: usize) -> Result<GridShape, BehaveError> {
    if range.start > range.end || range.end > len {
        return Err(BehaveError::InvalidRange {
            start: range.start,
            end: range.end,
            len,
        });
    }
    GridShape::new(vec![range.len()])
}

pub(crate) fn validity(
    column: Option<PerCell<'_, bool>>,
    range: Range<usize>,
) -> Result<Option<Vec<bool>>, ParallelError<BehaveError>> {
    let Some(column) = column else {
        return Ok(None);
    };
    let mut values = Vec::new();
    values
        .try_reserve_exact(range.len())
        .map_err(|error| ParallelError::Allocation(error.to_string()))?;
    values.extend(range.map(|index| value(&column, index)));
    Ok(Some(values))
}
