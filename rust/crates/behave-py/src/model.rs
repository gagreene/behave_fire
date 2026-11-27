//! Native model marshalling and scalar entry-point dispatch.
use crate::component_columns::RowWriter;
use crate::dispatch::scalar;
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    out: &mut RowWriter<'_, '_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::ModelCanopyCrownRatio => scalar(
            out,
            behave_fire::canopy_crown_ratio(a.n(Field::Height, i)?, a.n(Field::Base, i)?),
        ),
        _ => return Ok(None),
    }
    Ok(Some(()))
}
