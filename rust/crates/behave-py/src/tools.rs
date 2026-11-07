//! Native tools marshalling and scalar entry-point dispatch.
use crate::column_output::RowWriter;
use crate::dispatch::{named, scalar, widen};
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
use behave_fire::{fine_dead_fuel_moisture, vapor_pressure_deficit};
#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    _single: bool,
    out: &mut RowWriter<'_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::VaporPressureDeficitCalculatorCalculateVpd => named(
            out,
            vapor_pressure_deficit::calculate_vpd(
                a.n(Field::Temperature, i)?,
                a.temperature_units(i)?,
                a.n(Field::RelativeHumidity, i)?,
                a.humidity_units(i)?,
                a.pressure_units(i)?,
            )
            .named_values()
            .into_iter()
            .map(|(k, v)| (k, widen(v))),
        ),
        Operation::FineDeadFuelMoistureToolCalculateFineDeadFuelMoisture => scalar(
            out,
            widen(fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(
                a.n(Field::DryBulbI, i)? as i32,
                a.n(Field::RhI, i)? as i32,
                a.n(Field::SlopeI, i)? as i32,
                a.n(Field::AspectI, i)? as i32,
                a.n(Field::ShadingI, i)? as i32,
                a.n(Field::MonthI, i)? as i32,
                a.n(Field::ElevI, i)? as i32,
                a.n(Field::TimeI, i)? as i32,
            )),
        ),
        _ => return Ok(None),
    }
    Ok(Some(()))
}
