//! Native tools marshalling and scalar entry-point dispatch.
use crate::arrays::Arguments;
use crate::dispatch::{named, scalar, widen, Cell};
use behave_fire::units::{FractionUnit, PressureUnit, TemperatureUnit};
use behave_fire::{fine_dead_fuel_moisture, vapor_pressure_deficit};
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &Arguments,
    i: usize,
    _single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "vapor_pressure_deficit_calculator.calculate_vpd" => named(
            vapor_pressure_deficit::calculate_vpd(
                a.n("temperature", i)?,
                TemperatureUnit::try_from(a.n("temp_units", i)? as i32)
                    .map_err(|e| e.to_string())?,
                a.n("relative_humidity", i)?,
                FractionUnit::try_from(a.n("rh_units", i)? as i32).map_err(|e| e.to_string())?,
                PressureUnit::try_from(a.n("output_units", i)? as i32)
                    .map_err(|e| e.to_string())?,
            )
            .named_values()
            .into_iter()
            .map(|(k, v)| (k, widen(v))),
        ),
        "fine_dead_fuel_moisture_tool.calculate_fine_dead_fuel_moisture" => scalar(widen(
            fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(
                a.n("dry_bulb_i", i)? as i32,
                a.n("rh_i", i)? as i32,
                a.n("slope_i", i)? as i32,
                a.n("aspect_i", i)? as i32,
                a.n("shading_i", i)? as i32,
                a.n("month_i", i)? as i32,
                a.n("elev_i", i)? as i32,
                a.n("time_i", i)? as i32,
            ),
        )),
        _ => return Ok(None),
    }))
}
