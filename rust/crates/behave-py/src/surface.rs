//! Native surface marshalling and scalar entry-point dispatch.
use crate::arrays::Arguments;
use crate::dispatch::{named, scalar, widen, Cell};
use behave_fire::surface;
use behave_fire::units::{SlopeUnit, SpeedUnit};
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &Arguments,
    i: usize,
    single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "surface.calculate_backing_spread_rate" => {
            if single {
                scalar(widen(surface::single::calculate_backing_spread_rate(
                    a.n("spread_rate", i)? as f32,
                    a.n("eccentricity", i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_backing_spread_rate(
                    a.n("spread_rate", i)?,
                    a.n("eccentricity", i)?,
                )))
            }
        }
        "surface.calculate_direction_of_max_spread" => {
            if single {
                scalar(widen(surface::single::calculate_direction_of_max_spread(
                    a.n("x_component", i)? as f32,
                    a.n("y_component", i)? as f32,
                    a.n("rate_vector", i)? as f32,
                    crate::controls::orientation(a.s("wind_orientation_mode")?),
                    a.n("aspect", i)?,
                )))
            } else {
                scalar(widen(surface::calculate_direction_of_max_spread(
                    a.n("x_component", i)?,
                    a.n("y_component", i)?,
                    a.n("rate_vector", i)?,
                    crate::controls::orientation(a.s("wind_orientation_mode")?),
                    a.n("aspect", i)?,
                )))
            }
        }
        "surface.calculate_eccentricity" => {
            if single {
                scalar(widen(surface::single::calculate_eccentricity(
                    a.n("fire_length_to_width_ratio", i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_eccentricity(
                    a.n("fire_length_to_width_ratio", i)?,
                )))
            }
        }
        "surface.calculate_effective_wind_speed" => {
            if single {
                scalar(widen(surface::single::calculate_effective_wind_speed(
                    a.n("spread_rate", i)? as f32,
                    a.n("no_wind_no_slope_spread_rate", i)? as f32,
                    a.n("wind_b", i)? as f32,
                    a.n("wind_c", i)? as f32,
                    a.n("relative_packing_ratio", i)? as f32,
                    a.n("wind_e", i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_effective_wind_speed(
                    a.n("spread_rate", i)?,
                    a.n("no_wind_no_slope_spread_rate", i)?,
                    a.n("wind_b", i)?,
                    a.n("wind_c", i)?,
                    a.n("relative_packing_ratio", i)?,
                    a.n("wind_e", i)?,
                )))
            }
        }
        "surface.calculate_fire_area" => scalar(widen(surface::calculate_fire_area(
            a.n("forward_ros", i)?,
            a.n("backing_ros", i)?,
            a.n("lwr", i)?,
            a.n("elapsed_min", i)?,
            a.n("is_crown", i)? != 0.0,
        ))),
        "surface.calculate_fire_length" => {
            if a.optional("_bool", i).is_some() {
                scalar(surface::calculate_fire_length_bool(
                    a.n("forward_ros", i)? != 0.0,
                    a.n("backing_ros", i)? != 0.0,
                    a.n("elapsed_min", i)?,
                ))
            } else {
                if single {
                    scalar(widen(surface::single::calculate_fire_length(
                        a.n("forward_ros", i)? as f32,
                        a.n("backing_ros", i)? as f32,
                        a.n("elapsed_min", i)? as f32,
                    )))
                } else {
                    scalar(widen(surface::calculate_fire_length(
                        a.n("forward_ros", i)?,
                        a.n("backing_ros", i)?,
                        a.n("elapsed_min", i)?,
                    )))
                }
            }
        }
        "surface.calculate_fire_length_to_width_ratio" => {
            if single {
                scalar(widen(
                    surface::single::calculate_fire_length_to_width_ratio(
                        a.n("effective_wind_speed", i)? as f32,
                    ),
                ))
            } else {
                scalar(widen(surface::calculate_fire_length_to_width_ratio(
                    a.n("effective_wind_speed", i)?,
                )))
            }
        }
        "surface.calculate_fire_perimeter" => scalar(widen(surface::calculate_fire_perimeter(
            a.n("forward_ros", i)?,
            a.n("backing_ros", i)?,
            a.n("lwr", i)?,
            a.n("elapsed_min", i)?,
            a.n("is_crown", i)? != 0.0,
        ))),
        "surface.calculate_fire_width" => scalar(widen(surface::calculate_fire_width(
            a.n("forward_ros", i)?,
            a.n("backing_ros", i)?,
            a.n("lwr", i)?,
            a.n("elapsed_min", i)?,
        ))),
        "surface.calculate_fireline_intensity" => {
            scalar(widen(surface::calculate_fireline_intensity(
                a.n("spread_rate", i)?,
                a.n("heat_per_unit_area", i)?,
            )))
        }
        "surface.calculate_flame_length" => {
            if single {
                scalar(widen(surface::single::calculate_flame_length(
                    a.n("fireline_intensity", i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_flame_length(
                    a.n("fireline_intensity", i)?,
                )))
            }
        }
        "surface.calculate_flanking_spread_rate" => {
            if a.optional("_bool", i).is_some() {
                scalar(surface::calculate_flanking_spread_rate_bool(
                    a.n("spread_rate", i)? != 0.0,
                    a.n("backing_spread_rate", i)? != 0.0,
                    a.n("fire_length_to_width_ratio", i)?,
                ))
            } else {
                if single {
                    scalar(widen(surface::single::calculate_flanking_spread_rate(
                        a.n("spread_rate", i)? as f32,
                        a.n("backing_spread_rate", i)? as f32,
                        a.n("fire_length_to_width_ratio", i)? as f32,
                    )))
                } else {
                    scalar(widen(surface::calculate_flanking_spread_rate(
                        a.n("spread_rate", i)?,
                        a.n("backing_spread_rate", i)?,
                        a.n("fire_length_to_width_ratio", i)?,
                    )))
                }
            }
        }
        "surface.calculate_forward_spread_rate" => {
            scalar(widen(surface::calculate_forward_spread_rate(
                a.n("no_wind_no_slope_spread_rate", i)?,
                a.n("rate_vector", i)?,
            )))
        }
        "surface.calculate_heat_per_unit_area" => {
            scalar(widen(surface::calculate_heat_per_unit_area(
                a.n("reaction_intensity", i)?,
                a.n("residence_time", i)?,
            )))
        }
        "surface.calculate_midflame_wind_speed" => {
            scalar(widen(surface::calculate_midflame_wind_speed(
                a.n("waf", i)?,
                a.n("twenty_foot_wind_speed", i)?,
            )))
        }
        "surface.calculate_no_wind_no_slope_spread_rate" => {
            if single {
                scalar(widen(
                    surface::single::calculate_no_wind_no_slope_spread_rate(
                        a.n("ri", i)? as f32,
                        a.n("propagating_flux", i)? as f32,
                        a.n("heat_sink", i)? as f32,
                    ),
                ))
            } else {
                scalar(widen(surface::calculate_no_wind_no_slope_spread_rate(
                    a.n("ri", i)?,
                    a.n("propagating_flux", i)?,
                    a.n("heat_sink", i)?,
                )))
            }
        }
        "surface.calculate_reaction_intensity_output" => scalar(widen(
            surface::calculate_reaction_intensity_output(a.n("reaction_intensity", i)?),
        )),
        "surface.calculate_residence_time" => {
            if single {
                scalar(widen(surface::single::calculate_residence_time(
                    a.n("sigma", i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_residence_time(a.n("sigma", i)?)))
            }
        }
        "surface.calculate_slope_phi" => {
            if single {
                scalar(widen(surface::single::calculate_slope_phi(
                    a.n("slope", i)?,
                    SlopeUnit::try_from(a.n("slope_units", i)? as i32)
                        .map_err(|e| e.to_string())?,
                    a.n("packing_ratio", i)? as f32,
                    a.n("reaction_intensity", i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_slope_phi(
                    a.n("slope", i)?,
                    SlopeUnit::try_from(a.n("slope_units", i)? as i32)
                        .map_err(|e| e.to_string())?,
                    a.n("packing_ratio", i)?,
                    a.n("reaction_intensity", i)?,
                )))
            }
        }
        "surface.calculate_surface_rate_vector" => {
            if single {
                named(
                    surface::single::calculate_surface_rate_vector(
                        a.n("no_wind_no_slope_spread_rate", i)? as f32,
                        a.n("wind_phi", i)? as f32,
                        a.n("slope_phi", i)? as f32,
                        a.n("wind_direction", i)?,
                        crate::controls::orientation(a.s("wind_orientation_mode")?),
                        a.n("aspect", i)?,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            } else {
                named(
                    surface::calculate_surface_rate_vector(
                        a.n("no_wind_no_slope_spread_rate", i)?,
                        a.n("wind_phi", i)?,
                        a.n("slope_phi", i)?,
                        a.n("wind_direction", i)?,
                        crate::controls::orientation(a.s("wind_orientation_mode")?),
                        a.n("aspect", i)?,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            }
        }
        "surface.calculate_surface_wind_adjustment_factor" => {
            scalar(widen(surface::calculate_surface_wind_adjustment_factor(
                a.n("canopy_cover", i)?,
                a.n("canopy_height", i)?,
                a.n("crown_ratio", i)?,
                a.n("depth", i)?,
                crate::controls::waf(a.s("waf_method")?),
                a.optional("user_waf", i),
            )))
        }
        "surface.calculate_surface_wind_coefficients" => {
            if single {
                named(
                    surface::single::calculate_surface_wind_coefficients(a.n("sigma", i)? as f32)
                        .named_values()
                        .into_iter()
                        .map(|(k, v)| (k, widen(v))),
                )
            } else {
                named(
                    surface::calculate_surface_wind_coefficients(a.n("sigma", i)?)
                        .named_values()
                        .into_iter()
                        .map(|(k, v)| (k, widen(v))),
                )
            }
        }
        "surface.calculate_twenty_foot_wind_speed" => {
            scalar(widen(surface::calculate_twenty_foot_wind_speed(
                a.n("wind_speed", i)?,
                SpeedUnit::try_from(a.n("wind_speed_units", i)? as i32)
                    .map_err(|e| e.to_string())?,
                crate::controls::height(a.s("wind_height_mode")?),
            )))
        }
        "surface.calculate_wind_adjustment_factor" => {
            scalar(widen(surface::calculate_wind_adjustment_factor(
                a.n("canopy_cover", i)?,
                a.n("canopy_height", i)?,
                a.n("crown_ratio", i)?,
                a.n("depth", i)?,
            )))
        }
        "surface.calculate_wind_phi" => {
            if single {
                scalar(widen(surface::single::calculate_wind_phi(
                    a.n("sigma", i)? as f32,
                    a.n("midflame_wind_speed", i)? as f32,
                    a.n("wind_b", i)? as f32,
                    a.n("wind_c", i)? as f32,
                    a.n("relative_packing_ratio", i)? as f32,
                    a.n("wind_e", i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_wind_phi(
                    a.n("sigma", i)?,
                    a.n("midflame_wind_speed", i)?,
                    a.n("wind_b", i)?,
                    a.n("wind_c", i)?,
                    a.n("relative_packing_ratio", i)?,
                    a.n("wind_e", i)?,
                )))
            }
        }
        "surface.build_particle_arrays" => {
            crate::structures::particles_output(surface::build_particle_arrays_from_prepared(
                &crate::structures::fuel(a, i)?,
                a.n("m1h", i)?,
                a.n("m10h", i)?,
                a.n("m100h", i)?,
                a.n("mlh", i)?,
                a.n("mlw", i)?,
            ))
        }
        "surface.calculate_fuelbed_intermediates" => crate::structures::fuelbed_output(if single {
            surface::single::calculate_fuelbed_intermediates(&crate::structures::p(a, i)?)
        } else {
            surface::calculate_fuelbed_intermediates(&crate::structures::p(a, i)?)
        }),
        "surface.calculate_reaction_intensity" => scalar(if single {
            surface::single::calculate_reaction_intensity(&crate::structures::ib(a, i)?) as f64
        } else {
            surface::calculate_reaction_intensity(&crate::structures::ib(a, i)?)
        }),
        "surface.run_surface_fire" => {
            named(crate::structures::run_surface(a, i, single)?.named_values())
        }
        _ => return Ok(None),
    }))
}
