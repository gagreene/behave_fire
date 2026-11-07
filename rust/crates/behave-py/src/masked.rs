//! Pass per-argument data, masks, array kind, and precision into native stages.
use crate::column_output::RowWriter;
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
use behave_fire::surface::masked::*;
pub fn output(
    out: &mut RowWriter<'_>,
    value: MaskedValue,
    key: &'static str,
) -> Result<(), String> {
    out.checked(key, value)
}
pub fn outputs<const N: usize>(
    out: &mut RowWriter<'_>,
    fields: [(&'static str, MaskedValue); N],
) -> Result<(), String> {
    MaskedValue::validate_outputs(&fields.map(|(_, value)| value))
        .map_err(crate::arrays::masked_error)?;
    for (name, value) in fields {
        out.checked(name, value)?;
    }
    Ok(())
}
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    out: &mut RowWriter<'_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::SurfaceCalculateBackingSpreadRate => output(
            out,
            calculate_backing_spread_rate(
                a.masked(Field::SpreadRate, i)?,
                a.masked(Field::Eccentricity, i)?,
            ),
            "",
        )?,
        Operation::SurfaceCalculateDirectionOfMaxSpread => output(
            out,
            calculate_direction_of_max_spread(
                a.masked(Field::XComponent, i)?,
                a.masked(Field::YComponent, i)?,
                a.masked(Field::RateVector, i)?,
                a.orientation()?,
                a.masked(Field::Aspect, i)?,
            ),
            "",
        )?,
        Operation::SurfaceCalculateEccentricity => output(
            out,
            calculate_eccentricity(a.masked(Field::FireLengthToWidthRatio, i)?),
            "",
        )?,
        Operation::SurfaceCalculateEffectiveWindSpeed => {
            validate_condition_and(&[
                a.masked(Field::WindB, i)?,
                a.masked(Field::WindC, i)?,
                a.masked(Field::RelativePackingRatio, i)?,
            ])
            .map_err(|_| "TypeError: invalid masked scalar boolean operation".to_owned())?;
            output(
                out,
                calculate_effective_wind_speed(
                    a.masked(Field::SpreadRate, i)?,
                    a.masked(Field::NoWindNoSlopeSpreadRate, i)?,
                    a.masked(Field::WindB, i)?,
                    a.masked(Field::WindC, i)?,
                    a.masked(Field::RelativePackingRatio, i)?,
                    a.masked(Field::WindE, i)?,
                ),
                "",
            )?
        }
        Operation::SurfaceCalculateFireArea => output(
            out,
            calculate_fire_area(
                a.masked(Field::ForwardRos, i)?,
                a.masked(Field::BackingRos, i)?,
                a.masked(Field::Lwr, i)?,
                a.masked(Field::ElapsedMin, i)?,
                a.n(Field::IsCrown, i)? != 0.0,
            ),
            "",
        )?,
        Operation::SurfaceCalculateFireLength => output(
            out,
            calculate_fire_length(
                a.masked(Field::ForwardRos, i)?,
                a.masked(Field::BackingRos, i)?,
                a.masked(Field::ElapsedMin, i)?,
            ),
            "",
        )?,
        Operation::SurfaceCalculateFireLengthToWidthRatio => output(
            out,
            calculate_fire_length_to_width_ratio(a.masked(Field::EffectiveWindSpeed, i)?),
            "",
        )?,
        Operation::SurfaceCalculateFirePerimeter => output(
            out,
            calculate_fire_perimeter(
                a.masked(Field::ForwardRos, i)?,
                a.masked(Field::BackingRos, i)?,
                a.masked(Field::Lwr, i)?,
                a.masked(Field::ElapsedMin, i)?,
                a.n(Field::IsCrown, i)? != 0.0,
            ),
            "",
        )?,
        Operation::SurfaceCalculateFireWidth => output(
            out,
            calculate_fire_width(
                a.masked(Field::ForwardRos, i)?,
                a.masked(Field::BackingRos, i)?,
                a.masked(Field::Lwr, i)?,
                a.masked(Field::ElapsedMin, i)?,
            ),
            "",
        )?,
        Operation::SurfaceCalculateFlameLength => output(
            out,
            calculate_flame_length(a.masked(Field::FirelineIntensity, i)?),
            "",
        )?,
        Operation::SurfaceCalculateFlankingSpreadRate => output(
            out,
            calculate_flanking_spread_rate(
                a.masked(Field::SpreadRate, i)?,
                a.masked(Field::BackingSpreadRate, i)?,
                a.masked(Field::FireLengthToWidthRatio, i)?,
            ),
            "",
        )?,
        Operation::SurfaceCalculateNoWindNoSlopeSpreadRate => output(
            out,
            calculate_no_wind_no_slope_spread_rate(
                a.masked(Field::Ri, i)?,
                a.masked(Field::PropagatingFlux, i)?,
                a.masked(Field::HeatSink, i)?,
            ),
            "",
        )?,
        Operation::SurfaceCalculateResidenceTime => output(
            out,
            calculate_residence_time(a.masked(Field::Sigma, i)?),
            "",
        )?,
        Operation::SurfaceCalculateSlopePhi => {
            validate_condition_and(&[a.masked(Field::PackingRatio, i)?])
                .map_err(|_| "TypeError: invalid masked scalar boolean operation".to_owned())?;
            output(
                out,
                calculate_slope_phi(
                    a.masked(Field::Slope, i)?,
                    a.slope_units(i)?,
                    a.masked(Field::PackingRatio, i)?,
                    a.masked(Field::ReactionIntensity, i)?,
                ),
                "",
            )?
        }
        Operation::SurfaceCalculateSurfaceRateVector => {
            let r = calculate_surface_rate_vector(
                a.masked(Field::NoWindNoSlopeSpreadRate, i)?,
                a.masked(Field::WindPhi, i)?,
                a.masked(Field::SlopePhi, i)?,
                a.masked(Field::WindDirection, i)?,
                a.orientation()?,
                a.masked(Field::Aspect, i)?,
            );
            outputs(
                out,
                [
                    ("x_component", r.x_component),
                    ("y_component", r.y_component),
                    ("rate_vector", r.rate_vector),
                ],
            )?
        }
        Operation::SurfaceCalculateSurfaceWindCoefficients => outputs(
            out,
            calculate_surface_wind_coefficients(a.masked(Field::Sigma, i)?).named_values(),
        )?,
        Operation::SurfaceCalculateWindPhi => {
            validate_condition_and(&[
                a.masked(Field::Sigma, i)?,
                a.masked(Field::MidflameWindSpeed, i)?,
            ])
            .map_err(|_| "TypeError: invalid masked scalar boolean operation".to_owned())?;
            output(
                out,
                calculate_wind_phi(
                    a.masked(Field::Sigma, i)?,
                    a.masked(Field::MidflameWindSpeed, i)?,
                    a.masked(Field::WindB, i)?,
                    a.masked(Field::WindC, i)?,
                    a.masked(Field::RelativePackingRatio, i)?,
                    a.masked(Field::WindE, i)?,
                ),
                "",
            )?
        }
        Operation::SurfaceRunSurfaceFire => crate::structures::surface_output(
            out,
            run_surface_fire(
                a.masked(Field::Ri, i)?,
                &crate::structures::ib(a, i)?,
                &crate::structures::input(a, i)?,
                &crate::prepared::controls::surface(a)?,
            ),
        ),
        _ => return Ok(None),
    }
    Ok(Some(()))
}
