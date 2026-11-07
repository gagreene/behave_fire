//! Native model marshalling and scalar entry-point dispatch.
use crate::column_output::RowWriter;
use crate::dispatch::scalar;
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    _single: bool,
    out: &mut RowWriter<'_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::ModelCanopyCrownRatio => scalar(
            out,
            behave_fire::canopy_crown_ratio(a.n(Field::Height, i)?, a.n(Field::Base, i)?),
        ),
        Operation::ModelRunSurface => {
            crate::structures::surface_output(out, crate::parallel::surface(a, i)?)
        }
        _ => return Ok(None),
    }
    Ok(Some(()))
}
