//! Native contain marshalling and scalar entry-point dispatch.
use crate::column_output::RowWriter;
use crate::dispatch::{named, widen};
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
use behave_fire::contain;
#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    _single: bool,
    out: &mut RowWriter<'_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::ContainRunContainSimArray => named(
            out,
            contain::run_contain_sim_array(
                a.n(Field::ReportSpreadRate, i)?,
                a.n(Field::LwRatio, i)?,
                a.n(Field::ReportSize, i)?,
                a.n(Field::ProductionRate, i)?,
                a.n(Field::AttackTime, i)?,
                a.n(Field::MaxSteps, i)? as i64,
                a.n(Field::DistStep, i)?,
            )
            .named_values()
            .into_iter()
            .map(|(k, v)| (k, widen(v))),
        ),
        _ => return Ok(None),
    }
    Ok(Some(()))
}
