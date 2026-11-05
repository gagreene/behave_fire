//! Native contain marshalling and scalar entry-point dispatch.
use crate::arrays::Arguments;
use crate::dispatch::{named, widen, Cell};
use behave_fire::contain;
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &Arguments,
    i: usize,
    _single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "contain.run_contain_sim_array" => named(
            contain::run_contain_sim_array(
                a.n("report_spread_rate", i)?,
                a.n("lw_ratio", i)?,
                a.n("report_size", i)?,
                a.n("production_rate", i)?,
                a.n("attack_time", i)?,
                a.n("max_steps", i)? as i64,
                a.n("dist_step", i)?,
            )
            .named_values()
            .into_iter()
            .map(|(k, v)| (k, widen(v))),
        ),
        _ => return Ok(None),
    }))
}
