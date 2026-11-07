//! Native tools marshalling and scalar entry-point dispatch.
use crate::dispatch::{named, scalar, widen, Cell};
use crate::prepared::{Field, PreparedArguments};
use behave_fire::{fine_dead_fuel_moisture, vapor_pressure_deficit};
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &PreparedArguments<'_>,
    i: usize,
    _single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "vapor_pressure_deficit_calculator.calculate_vpd" => named(
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
        "fine_dead_fuel_moisture_tool.calculate_fine_dead_fuel_moisture" => scalar(widen(
            fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(
                a.n(Field::DryBulbI, i)? as i32,
                a.n(Field::RhI, i)? as i32,
                a.n(Field::SlopeI, i)? as i32,
                a.n(Field::AspectI, i)? as i32,
                a.n(Field::ShadingI, i)? as i32,
                a.n(Field::MonthI, i)? as i32,
                a.n(Field::ElevI, i)? as i32,
                a.n(Field::TimeI, i)? as i32,
            ),
        )),
        _ => return Ok(None),
    }))
}
