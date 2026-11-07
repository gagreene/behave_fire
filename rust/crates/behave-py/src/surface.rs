//! Native surface marshalling and scalar entry-point dispatch.
use crate::dispatch::{named, scalar, widen, Cell};
use crate::prepared::{Field, PreparedArguments};
use behave_fire::surface;
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &PreparedArguments<'_>,
    i: usize,
    single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "surface.calculate_backing_spread_rate" => {
            if single {
                scalar(widen(surface::single::calculate_backing_spread_rate(
                    a.n(Field::SpreadRate, i)? as f32,
                    a.n(Field::Eccentricity, i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_backing_spread_rate(
                    a.n(Field::SpreadRate, i)?,
                    a.n(Field::Eccentricity, i)?,
                )))
            }
        }
        "surface.calculate_direction_of_max_spread" => {
            if single {
                scalar(widen(surface::single::calculate_direction_of_max_spread(
                    a.n(Field::XComponent, i)? as f32,
                    a.n(Field::YComponent, i)? as f32,
                    a.n(Field::RateVector, i)? as f32,
                    a.orientation()?,
                    a.n(Field::Aspect, i)?,
                )))
            } else {
                scalar(widen(surface::calculate_direction_of_max_spread(
                    a.n(Field::XComponent, i)?,
                    a.n(Field::YComponent, i)?,
                    a.n(Field::RateVector, i)?,
                    a.orientation()?,
                    a.n(Field::Aspect, i)?,
                )))
            }
        }
        "surface.calculate_eccentricity" => {
            if single {
                scalar(widen(surface::single::calculate_eccentricity(
                    a.n(Field::FireLengthToWidthRatio, i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_eccentricity(
                    a.n(Field::FireLengthToWidthRatio, i)?,
                )))
            }
        }
        "surface.calculate_effective_wind_speed" => {
            if single {
                scalar(widen(surface::single::calculate_effective_wind_speed(
                    a.n(Field::SpreadRate, i)? as f32,
                    a.n(Field::NoWindNoSlopeSpreadRate, i)? as f32,
                    a.n(Field::WindB, i)? as f32,
                    a.n(Field::WindC, i)? as f32,
                    a.n(Field::RelativePackingRatio, i)? as f32,
                    a.n(Field::WindE, i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_effective_wind_speed(
                    a.n(Field::SpreadRate, i)?,
                    a.n(Field::NoWindNoSlopeSpreadRate, i)?,
                    a.n(Field::WindB, i)?,
                    a.n(Field::WindC, i)?,
                    a.n(Field::RelativePackingRatio, i)?,
                    a.n(Field::WindE, i)?,
                )))
            }
        }
        "surface.calculate_fire_area" => scalar(widen(surface::calculate_fire_area(
            a.n(Field::ForwardRos, i)?,
            a.n(Field::BackingRos, i)?,
            a.n(Field::Lwr, i)?,
            a.n(Field::ElapsedMin, i)?,
            a.n(Field::IsCrown, i)? != 0.0,
        ))),
        "surface.calculate_fire_length" => {
            if a.optional(Field::GlobalBool, i).is_some() {
                scalar(surface::calculate_fire_length_bool(
                    a.n(Field::ForwardRos, i)? != 0.0,
                    a.n(Field::BackingRos, i)? != 0.0,
                    a.n(Field::ElapsedMin, i)?,
                ))
            } else {
                if single {
                    scalar(widen(surface::single::calculate_fire_length(
                        a.n(Field::ForwardRos, i)? as f32,
                        a.n(Field::BackingRos, i)? as f32,
                        a.n(Field::ElapsedMin, i)? as f32,
                    )))
                } else {
                    scalar(widen(surface::calculate_fire_length(
                        a.n(Field::ForwardRos, i)?,
                        a.n(Field::BackingRos, i)?,
                        a.n(Field::ElapsedMin, i)?,
                    )))
                }
            }
        }
        "surface.calculate_fire_length_to_width_ratio" => {
            if single {
                scalar(widen(
                    surface::single::calculate_fire_length_to_width_ratio(
                        a.n(Field::EffectiveWindSpeed, i)? as f32,
                    ),
                ))
            } else {
                scalar(widen(surface::calculate_fire_length_to_width_ratio(
                    a.n(Field::EffectiveWindSpeed, i)?,
                )))
            }
        }
        "surface.calculate_fire_perimeter" => scalar(widen(surface::calculate_fire_perimeter(
            a.n(Field::ForwardRos, i)?,
            a.n(Field::BackingRos, i)?,
            a.n(Field::Lwr, i)?,
            a.n(Field::ElapsedMin, i)?,
            a.n(Field::IsCrown, i)? != 0.0,
        ))),
        "surface.calculate_fire_width" => scalar(widen(surface::calculate_fire_width(
            a.n(Field::ForwardRos, i)?,
            a.n(Field::BackingRos, i)?,
            a.n(Field::Lwr, i)?,
            a.n(Field::ElapsedMin, i)?,
        ))),
        "surface.calculate_fireline_intensity" => {
            scalar(widen(surface::calculate_fireline_intensity(
                a.n(Field::SpreadRate, i)?,
                a.n(Field::HeatPerUnitArea, i)?,
            )))
        }
        "surface.calculate_flame_length" => {
            if single {
                scalar(widen(surface::single::calculate_flame_length(
                    a.n(Field::FirelineIntensity, i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_flame_length(
                    a.n(Field::FirelineIntensity, i)?,
                )))
            }
        }
        "surface.calculate_flanking_spread_rate" => {
            if a.optional(Field::GlobalBool, i).is_some() {
                scalar(surface::calculate_flanking_spread_rate_bool(
                    a.n(Field::SpreadRate, i)? != 0.0,
                    a.n(Field::BackingSpreadRate, i)? != 0.0,
                    a.n(Field::FireLengthToWidthRatio, i)?,
                ))
            } else {
                if single {
                    scalar(widen(surface::single::calculate_flanking_spread_rate(
                        a.n(Field::SpreadRate, i)? as f32,
                        a.n(Field::BackingSpreadRate, i)? as f32,
                        a.n(Field::FireLengthToWidthRatio, i)? as f32,
                    )))
                } else {
                    scalar(widen(surface::calculate_flanking_spread_rate(
                        a.n(Field::SpreadRate, i)?,
                        a.n(Field::BackingSpreadRate, i)?,
                        a.n(Field::FireLengthToWidthRatio, i)?,
                    )))
                }
            }
        }
        "surface.calculate_forward_spread_rate" => {
            scalar(widen(surface::calculate_forward_spread_rate(
                a.n(Field::NoWindNoSlopeSpreadRate, i)?,
                a.n(Field::RateVector, i)?,
            )))
        }
        "surface.calculate_heat_per_unit_area" => {
            scalar(widen(surface::calculate_heat_per_unit_area(
                a.n(Field::ReactionIntensity, i)?,
                a.n(Field::ResidenceTime, i)?,
            )))
        }
        "surface.calculate_midflame_wind_speed" => {
            scalar(widen(surface::calculate_midflame_wind_speed(
                a.n(Field::Waf, i)?,
                a.n(Field::TwentyFootWindSpeed, i)?,
            )))
        }
        "surface.calculate_no_wind_no_slope_spread_rate" => {
            if single {
                scalar(widen(
                    surface::single::calculate_no_wind_no_slope_spread_rate(
                        a.n(Field::Ri, i)? as f32,
                        a.n(Field::PropagatingFlux, i)? as f32,
                        a.n(Field::HeatSink, i)? as f32,
                    ),
                ))
            } else {
                scalar(widen(surface::calculate_no_wind_no_slope_spread_rate(
                    a.n(Field::Ri, i)?,
                    a.n(Field::PropagatingFlux, i)?,
                    a.n(Field::HeatSink, i)?,
                )))
            }
        }
        "surface.calculate_reaction_intensity_output" => scalar(widen(
            surface::calculate_reaction_intensity_output(a.n(Field::ReactionIntensity, i)?),
        )),
        "surface.calculate_residence_time" => {
            if single {
                scalar(widen(surface::single::calculate_residence_time(
                    a.n(Field::Sigma, i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_residence_time(
                    a.n(Field::Sigma, i)?,
                )))
            }
        }
        "surface.calculate_slope_phi" => {
            if single {
                scalar(widen(surface::single::calculate_slope_phi(
                    a.n(Field::Slope, i)?,
                    a.slope_units(i)?,
                    a.n(Field::PackingRatio, i)? as f32,
                    a.n(Field::ReactionIntensity, i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_slope_phi(
                    a.n(Field::Slope, i)?,
                    a.slope_units(i)?,
                    a.n(Field::PackingRatio, i)?,
                    a.n(Field::ReactionIntensity, i)?,
                )))
            }
        }
        "surface.calculate_surface_rate_vector" => {
            if single {
                named(
                    surface::single::calculate_surface_rate_vector(
                        a.n(Field::NoWindNoSlopeSpreadRate, i)? as f32,
                        a.n(Field::WindPhi, i)? as f32,
                        a.n(Field::SlopePhi, i)? as f32,
                        a.n(Field::WindDirection, i)?,
                        a.orientation()?,
                        a.n(Field::Aspect, i)?,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            } else {
                named(
                    surface::calculate_surface_rate_vector(
                        a.n(Field::NoWindNoSlopeSpreadRate, i)?,
                        a.n(Field::WindPhi, i)?,
                        a.n(Field::SlopePhi, i)?,
                        a.n(Field::WindDirection, i)?,
                        a.orientation()?,
                        a.n(Field::Aspect, i)?,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            }
        }
        "surface.calculate_surface_wind_adjustment_factor" => {
            scalar(widen(surface::calculate_surface_wind_adjustment_factor(
                a.n(Field::CanopyCover, i)?,
                a.n(Field::CanopyHeight, i)?,
                a.n(Field::CrownRatio, i)?,
                a.n(Field::Depth, i)?,
                a.waf()?,
                a.optional(Field::UserWaf, i),
            )))
        }
        "surface.calculate_surface_wind_coefficients" => {
            if single {
                named(
                    surface::single::calculate_surface_wind_coefficients(
                        a.n(Field::Sigma, i)? as f32
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            } else {
                named(
                    surface::calculate_surface_wind_coefficients(a.n(Field::Sigma, i)?)
                        .named_values()
                        .into_iter()
                        .map(|(k, v)| (k, widen(v))),
                )
            }
        }
        "surface.calculate_twenty_foot_wind_speed" => {
            scalar(widen(surface::calculate_twenty_foot_wind_speed(
                a.n(Field::WindSpeed, i)?,
                a.speed_units(i)?,
                a.height()?,
            )))
        }
        "surface.calculate_wind_adjustment_factor" => {
            scalar(widen(surface::calculate_wind_adjustment_factor(
                a.n(Field::CanopyCover, i)?,
                a.n(Field::CanopyHeight, i)?,
                a.n(Field::CrownRatio, i)?,
                a.n(Field::Depth, i)?,
            )))
        }
        "surface.calculate_wind_phi" => {
            if single {
                scalar(widen(surface::single::calculate_wind_phi(
                    a.n(Field::Sigma, i)? as f32,
                    a.n(Field::MidflameWindSpeed, i)? as f32,
                    a.n(Field::WindB, i)? as f32,
                    a.n(Field::WindC, i)? as f32,
                    a.n(Field::RelativePackingRatio, i)? as f32,
                    a.n(Field::WindE, i)? as f32,
                )))
            } else {
                scalar(widen(surface::calculate_wind_phi(
                    a.n(Field::Sigma, i)?,
                    a.n(Field::MidflameWindSpeed, i)?,
                    a.n(Field::WindB, i)?,
                    a.n(Field::WindC, i)?,
                    a.n(Field::RelativePackingRatio, i)?,
                    a.n(Field::WindE, i)?,
                )))
            }
        }
        "surface.build_particle_arrays" => {
            crate::structures::particles_output(surface::build_particle_arrays_from_prepared(
                &crate::structures::fuel(a, i)?,
                a.n(Field::M1h, i)?,
                a.n(Field::M10h, i)?,
                a.n(Field::M100h, i)?,
                a.n(Field::Mlh, i)?,
                a.n(Field::Mlw, i)?,
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
