//! Native tools marshalling and scalar entry-point dispatch.
use crate::component_columns::RowWriter;
use crate::dispatch::{named, scalar, widen};
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
use behave_fire::{fine_dead_fuel_moisture, vapor_pressure_deficit};
#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    out: &mut RowWriter<'_, '_>,
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
                a.code(Field::DryBulbI, i)? as i32,
                a.code(Field::RhI, i)? as i32,
                a.code(Field::SlopeI, i)? as i32,
                a.code(Field::AspectI, i)? as i32,
                a.code(Field::ShadingI, i)? as i32,
                a.code(Field::MonthI, i)? as i32,
                a.code(Field::ElevI, i)? as i32,
                a.code(Field::TimeI, i)? as i32,
            )),
        ),
        _ => return Ok(None),
    }
    Ok(Some(()))
}
