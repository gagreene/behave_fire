//! Native contain marshalling and scalar entry-point dispatch.
use crate::component_columns::RowWriter;
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
use behave_fire::contain;
#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    out: &mut RowWriter<'_, '_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::ContainRunContainSimArray => {
            let value = contain::run_contain_sim_array(
                a.n(Field::ReportSpreadRate, i)?,
                a.n(Field::LwRatio, i)?,
                a.n(Field::ReportSize, i)?,
                a.n(Field::ProductionRate, i)?,
                a.n(Field::AttackTime, i)?,
                a.code(Field::MaxSteps, i)?,
                a.n(Field::DistStep, i)?,
            );
            out.i32("status", value.status);
            out.f64("contained_time", value.contained_time);
            out.f64("final_perimeter", value.final_perimeter);
            out.f64("final_area", value.final_area);
        }
        _ => return Ok(None),
    }
    Ok(Some(()))
}
