//! Native model marshalling and scalar entry-point dispatch.
use crate::dispatch::{scalar, Cell};
use crate::prepared::{Field, PreparedArguments};
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &PreparedArguments<'_>,
    i: usize,
    _single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "model.canopy_crown_ratio" => scalar(behave_fire::canopy_crown_ratio(
            a.n(Field::Height, i)?,
            a.n(Field::Base, i)?,
        )),
        _ => return Ok(None),
    }))
}
