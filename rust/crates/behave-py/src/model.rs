//! Native model marshalling and scalar entry-point dispatch.
use crate::arrays::Arguments;
use crate::dispatch::{scalar, Cell};
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &Arguments,
    i: usize,
    _single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "model.canopy_crown_ratio" => scalar(behave_fire::canopy_crown_ratio(
            a.n("height", i)?,
            a.n("base", i)?,
        )),
        _ => return Ok(None),
    }))
}
