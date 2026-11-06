//! Pass per-argument data, masks, array kind, and precision into native stages.
use crate::{arrays::Arguments, dispatch::Cell};
use behave_fire::surface::masked::*;
use behave_fire::units::*;
pub fn output(value: MaskedValue, key: &str) -> Result<Cell, String> {
    let value = value
        .validate_output()
        .map_err(crate::arrays::masked_error)?;
    let mut result = Cell::from([
        (key.to_owned(), value.value),
        (format!("__mask.{key}"), u8::from(value.mask) as f64),
        (
            format!("__masked.{key}"),
            u8::from(value.is_masked_array) as f64,
        ),
        (
            format!("__single.{key}"),
            u8::from(value.precision == Precision::F32) as f64,
        ),
        (
            format!("__fill_present.{key}"),
            u8::from(value.fill_value.is_some()) as f64,
        ),
        (format!("__fill.{key}"), value.fill_value.unwrap_or(0.0)),
    ]);
    result.insert(
        format!("__complex.{key}"),
        u8::from(value.complex.is_some()) as f64,
    );
    result.insert(
        format!("__imag.{key}"),
        value.complex.map_or(0.0, |v| v.imag),
    );
    result.insert(
        format!("__object.{key}"),
        u8::from(value.object_result) as f64,
    );
    result.insert(
        format!("__object_numpy.{key}"),
        u8::from(value.object_numpy_scalar) as f64,
    );
    result.insert(
        format!("__integer_type.{key}"),
        value.numpy_integer_type.map_or(-1.0, f64::from),
    );
    result.insert(
        format!("__integer_dtype.{key}"),
        value.integer.map_or(-1.0, |v| v.kind.code() as f64),
    );
    let bits = value.integer.map_or(0, |integer| integer.raw_bits);
    result.insert(format!("__integer_low.{key}"), (bits as u32) as f64);
    result.insert(
        format!("__integer_high.{key}"),
        ((bits >> 32) as u32) as f64,
    );
    Ok(result)
}
pub fn outputs<'a>(
    fields: impl IntoIterator<Item = (&'a str, MaskedValue)>,
) -> Result<Cell, String> {
    let fields: Vec<_> = fields.into_iter().collect();
    MaskedValue::validate_outputs(&fields.iter().map(|(_, value)| *value).collect::<Vec<_>>())
        .map_err(crate::arrays::masked_error)?;
    let mut result = Cell::new();
    for (key, value) in fields {
        result.extend(output(value, key)?);
    }
    Ok(result)
}
pub fn evaluate(operation: &str, a: &Arguments, i: usize) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "surface.calculate_backing_spread_rate" => output(
            calculate_backing_spread_rate(
                a.masked("spread_rate", i)?,
                a.masked("eccentricity", i)?,
            ),
            "",
        )?,
        "surface.calculate_direction_of_max_spread" => output(
            calculate_direction_of_max_spread(
                a.masked("x_component", i)?,
                a.masked("y_component", i)?,
                a.masked("rate_vector", i)?,
                crate::controls::orientation(a.s("wind_orientation_mode")?),
                a.masked("aspect", i)?,
            ),
            "",
        )?,
        "surface.calculate_eccentricity" => output(
            calculate_eccentricity(a.masked("fire_length_to_width_ratio", i)?),
            "",
        )?,
        "surface.calculate_effective_wind_speed" => {
            validate_condition_and(&[
                a.masked("wind_b", i)?,
                a.masked("wind_c", i)?,
                a.masked("relative_packing_ratio", i)?,
            ])
            .map_err(|_| "TypeError: invalid masked scalar boolean operation".to_owned())?;
            output(
                calculate_effective_wind_speed(
                    a.masked("spread_rate", i)?,
                    a.masked("no_wind_no_slope_spread_rate", i)?,
                    a.masked("wind_b", i)?,
                    a.masked("wind_c", i)?,
                    a.masked("relative_packing_ratio", i)?,
                    a.masked("wind_e", i)?,
                ),
                "",
            )?
        }
        "surface.calculate_fire_area" => output(
            calculate_fire_area(
                a.masked("forward_ros", i)?,
                a.masked("backing_ros", i)?,
                a.masked("lwr", i)?,
                a.masked("elapsed_min", i)?,
                a.n("is_crown", i)? != 0.0,
            ),
            "",
        )?,
        "surface.calculate_fire_length" => output(
            calculate_fire_length(
                a.masked("forward_ros", i)?,
                a.masked("backing_ros", i)?,
                a.masked("elapsed_min", i)?,
            ),
            "",
        )?,
        "surface.calculate_fire_length_to_width_ratio" => output(
            calculate_fire_length_to_width_ratio(a.masked("effective_wind_speed", i)?),
            "",
        )?,
        "surface.calculate_fire_perimeter" => output(
            calculate_fire_perimeter(
                a.masked("forward_ros", i)?,
                a.masked("backing_ros", i)?,
                a.masked("lwr", i)?,
                a.masked("elapsed_min", i)?,
                a.n("is_crown", i)? != 0.0,
            ),
            "",
        )?,
        "surface.calculate_fire_width" => output(
            calculate_fire_width(
                a.masked("forward_ros", i)?,
                a.masked("backing_ros", i)?,
                a.masked("lwr", i)?,
                a.masked("elapsed_min", i)?,
            ),
            "",
        )?,
        "surface.calculate_flame_length" => output(
            calculate_flame_length(a.masked("fireline_intensity", i)?),
            "",
        )?,
        "surface.calculate_flanking_spread_rate" => output(
            calculate_flanking_spread_rate(
                a.masked("spread_rate", i)?,
                a.masked("backing_spread_rate", i)?,
                a.masked("fire_length_to_width_ratio", i)?,
            ),
            "",
        )?,
        "surface.calculate_no_wind_no_slope_spread_rate" => output(
            calculate_no_wind_no_slope_spread_rate(
                a.masked("ri", i)?,
                a.masked("propagating_flux", i)?,
                a.masked("heat_sink", i)?,
            ),
            "",
        )?,
        "surface.calculate_residence_time" => {
            output(calculate_residence_time(a.masked("sigma", i)?), "")?
        }
        "surface.calculate_slope_phi" => {
            validate_condition_and(&[a.masked("packing_ratio", i)?])
                .map_err(|_| "TypeError: invalid masked scalar boolean operation".to_owned())?;
            output(
                calculate_slope_phi(
                    a.masked("slope", i)?,
                    SlopeUnit::try_from(a.n("slope_units", i)? as i32)
                        .map_err(|e| e.to_string())?,
                    a.masked("packing_ratio", i)?,
                    a.masked("reaction_intensity", i)?,
                ),
                "",
            )?
        }
        "surface.calculate_surface_rate_vector" => {
            let r = calculate_surface_rate_vector(
                a.masked("no_wind_no_slope_spread_rate", i)?,
                a.masked("wind_phi", i)?,
                a.masked("slope_phi", i)?,
                a.masked("wind_direction", i)?,
                crate::controls::orientation(a.s("wind_orientation_mode")?),
                a.masked("aspect", i)?,
            );
            outputs([
                ("x_component", r.x_component),
                ("y_component", r.y_component),
                ("rate_vector", r.rate_vector),
            ])?
        }
        "surface.calculate_surface_wind_coefficients" => {
            outputs(calculate_surface_wind_coefficients(a.masked("sigma", i)?).named_values())?
        }
        "surface.calculate_wind_phi" => {
            validate_condition_and(&[a.masked("sigma", i)?, a.masked("midflame_wind_speed", i)?])
                .map_err(|_| "TypeError: invalid masked scalar boolean operation".to_owned())?;
            output(
                calculate_wind_phi(
                    a.masked("sigma", i)?,
                    a.masked("midflame_wind_speed", i)?,
                    a.masked("wind_b", i)?,
                    a.masked("wind_c", i)?,
                    a.masked("relative_packing_ratio", i)?,
                    a.masked("wind_e", i)?,
                ),
                "",
            )?
        }
        "surface.run_surface_fire" => crate::structures::surface_output(run_surface_fire(
            a.masked("ri", i)?,
            &crate::structures::ib(a, i)?,
            &crate::structures::input(a, i)?,
            &crate::controls::surface(a)?,
        )),
        _ => return Ok(None),
    }))
}
