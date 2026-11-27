//! Native surface marshalling and scalar entry-point dispatch.
use crate::component_columns::RowWriter;
use crate::dispatch::{named, scalar, widen};
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
use behave_fire::surface;
#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    out: &mut RowWriter<'_, '_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::SurfaceCalculateBackingSpreadRate => scalar(
            out,
            widen(surface::calculate_backing_spread_rate(
                a.n(Field::SpreadRate, i)?,
                a.n(Field::Eccentricity, i)?,
            )),
        ),
        Operation::SurfaceCalculateDirectionOfMaxSpread => scalar(
            out,
            widen(surface::calculate_direction_of_max_spread(
                a.n(Field::XComponent, i)?,
                a.n(Field::YComponent, i)?,
                a.n(Field::RateVector, i)?,
                a.orientation()?,
                a.optional(Field::Aspect, i).unwrap_or(f64::NAN),
            )),
        ),
        Operation::SurfaceCalculateEccentricity => scalar(
            out,
            widen(surface::calculate_eccentricity(
                a.n(Field::FireLengthToWidthRatio, i)?,
            )),
        ),
        Operation::SurfaceCalculateEffectiveWindSpeed => scalar(
            out,
            widen(surface::calculate_effective_wind_speed(
                a.n(Field::SpreadRate, i)?,
                a.n(Field::NoWindNoSlopeSpreadRate, i)?,
                a.n(Field::WindB, i)?,
                a.n(Field::WindC, i)?,
                a.n(Field::RelativePackingRatio, i)?,
                a.n(Field::WindE, i)?,
            )),
        ),
        Operation::SurfaceCalculateFireArea => scalar(
            out,
            widen(surface::calculate_fire_area(
                a.n(Field::ForwardRos, i)?,
                a.n(Field::BackingRos, i)?,
                a.n(Field::Lwr, i)?,
                a.n(Field::ElapsedMin, i)?,
                a.flag(Field::IsCrown, i)?,
            )),
        ),
        Operation::SurfaceCalculateFireLength => scalar(
            out,
            widen(surface::calculate_fire_length(
                a.n(Field::ForwardRos, i)?,
                a.n(Field::BackingRos, i)?,
                a.n(Field::ElapsedMin, i)?,
            )),
        ),
        Operation::SurfaceCalculateFireLengthToWidthRatio => scalar(
            out,
            widen(surface::calculate_fire_length_to_width_ratio(
                a.n(Field::EffectiveWindSpeed, i)?,
            )),
        ),
        Operation::SurfaceCalculateFirePerimeter => scalar(
            out,
            widen(surface::calculate_fire_perimeter(
                a.n(Field::ForwardRos, i)?,
                a.n(Field::BackingRos, i)?,
                a.n(Field::Lwr, i)?,
                a.n(Field::ElapsedMin, i)?,
                a.flag(Field::IsCrown, i)?,
            )),
        ),
        Operation::SurfaceCalculateFireWidth => scalar(
            out,
            widen(surface::calculate_fire_width(
                a.n(Field::ForwardRos, i)?,
                a.n(Field::BackingRos, i)?,
                a.n(Field::Lwr, i)?,
                a.n(Field::ElapsedMin, i)?,
            )),
        ),
        Operation::SurfaceCalculateFirelineIntensity => scalar(
            out,
            widen(surface::calculate_fireline_intensity(
                a.n(Field::SpreadRate, i)?,
                a.n(Field::HeatPerUnitArea, i)?,
            )),
        ),
        Operation::SurfaceCalculateFlameLength => scalar(
            out,
            widen(surface::calculate_flame_length(
                a.n(Field::FirelineIntensity, i)?,
            )),
        ),
        Operation::SurfaceCalculateFlankingSpreadRate => scalar(
            out,
            widen(surface::calculate_flanking_spread_rate(
                a.n(Field::SpreadRate, i)?,
                a.n(Field::BackingSpreadRate, i)?,
                a.n(Field::FireLengthToWidthRatio, i)?,
            )),
        ),
        Operation::SurfaceCalculateForwardSpreadRate => scalar(
            out,
            widen(surface::calculate_forward_spread_rate(
                a.n(Field::NoWindNoSlopeSpreadRate, i)?,
                a.n(Field::RateVector, i)?,
            )),
        ),
        Operation::SurfaceCalculateHeatPerUnitArea => scalar(
            out,
            widen(surface::calculate_heat_per_unit_area(
                a.n(Field::ReactionIntensity, i)?,
                a.n(Field::ResidenceTime, i)?,
            )),
        ),
        Operation::SurfaceCalculateMidflameWindSpeed => scalar(
            out,
            widen(surface::calculate_midflame_wind_speed(
                a.n(Field::Waf, i)?,
                a.n(Field::TwentyFootWindSpeed, i)?,
            )),
        ),
        Operation::SurfaceCalculateNoWindNoSlopeSpreadRate => scalar(
            out,
            widen(surface::calculate_no_wind_no_slope_spread_rate(
                a.n(Field::Ri, i)?,
                a.n(Field::PropagatingFlux, i)?,
                a.n(Field::HeatSink, i)?,
            )),
        ),
        Operation::SurfaceCalculateReactionIntensityOutput => scalar(
            out,
            widen(surface::calculate_reaction_intensity_output(
                a.n(Field::ReactionIntensity, i)?,
            )),
        ),
        Operation::SurfaceCalculateResidenceTime => scalar(
            out,
            widen(surface::calculate_residence_time(a.n(Field::Sigma, i)?)),
        ),
        Operation::SurfaceCalculateSlopePhi => scalar(
            out,
            widen(surface::calculate_slope_phi(
                a.n(Field::Slope, i)?,
                a.slope_units(i)?,
                a.n(Field::PackingRatio, i)?,
                a.n(Field::ReactionIntensity, i)?,
            )),
        ),
        Operation::SurfaceCalculateSurfaceRateVector => named(
            out,
            surface::calculate_surface_rate_vector(
                a.n(Field::NoWindNoSlopeSpreadRate, i)?,
                a.n(Field::WindPhi, i)?,
                a.n(Field::SlopePhi, i)?,
                a.n(Field::WindDirection, i)?,
                a.orientation()?,
                a.optional(Field::Aspect, i).unwrap_or(f64::NAN),
            )
            .named_values()
            .into_iter()
            .map(|(k, v)| (k, widen(v))),
        ),
        Operation::SurfaceCalculateSurfaceWindAdjustmentFactor
            if a.waf()? == behave_fire::engine::WafMethod::UserInput && a.has(Field::UserWaf) =>
        {
            scalar(out, a.n(Field::UserWaf, i)?)
        }
        Operation::SurfaceCalculateSurfaceWindAdjustmentFactor => scalar(
            out,
            widen(surface::calculate_surface_wind_adjustment_factor(
                a.n(Field::CanopyCover, i)?,
                a.n(Field::CanopyHeight, i)?,
                a.n(Field::CrownRatio, i)?,
                a.n(Field::Depth, i)?,
                a.waf()?,
                a.optional(Field::UserWaf, i),
            )),
        ),
        Operation::SurfaceCalculateSurfaceWindCoefficients => named(
            out,
            surface::calculate_surface_wind_coefficients(a.n(Field::Sigma, i)?)
                .named_values()
                .into_iter()
                .map(|(k, v)| (k, widen(v))),
        ),
        Operation::SurfaceCalculateTwentyFootWindSpeed => scalar(
            out,
            widen(surface::calculate_twenty_foot_wind_speed(
                a.n(Field::WindSpeed, i)?,
                a.speed_units(i)?,
                a.height()?,
            )),
        ),
        Operation::SurfaceCalculateWindAdjustmentFactor => scalar(
            out,
            widen(surface::calculate_wind_adjustment_factor(
                a.n(Field::CanopyCover, i)?,
                a.n(Field::CanopyHeight, i)?,
                a.n(Field::CrownRatio, i)?,
                a.n(Field::Depth, i)?,
            )),
        ),
        Operation::SurfaceCalculateWindPhi => scalar(
            out,
            widen(surface::calculate_wind_phi(
                a.n(Field::Sigma, i)?,
                a.n(Field::MidflameWindSpeed, i)?,
                a.n(Field::WindB, i)?,
                a.n(Field::WindC, i)?,
                a.n(Field::RelativePackingRatio, i)?,
                a.n(Field::WindE, i)?,
            )),
        ),
        Operation::SurfaceBuildParticleArrays => crate::structures::particles_output(
            out,
            surface::build_particle_arrays_from_prepared(
                &crate::structures::fuel(a, i)?,
                a.n(Field::M1h, i)?,
                a.n(Field::M10h, i)?,
                a.n(Field::M100h, i)?,
                a.n(Field::Mlh, i)?,
                a.n(Field::Mlw, i)?,
            ),
        ),
        Operation::SurfaceCalculateFuelbedIntermediates => {
            crate::structures::fuelbed_output(out, {
                surface::calculate_fuelbed_intermediates(&crate::structures::p(a, i)?)
            })
        }
        Operation::SurfaceCalculateReactionIntensity => scalar(out, {
            surface::calculate_reaction_intensity(&crate::structures::ib(a, i)?)
        }),
        Operation::SurfaceRunSurfaceFire => {
            named(out, crate::structures::run_surface(a, i)?.named_values())
        }
        _ => return Ok(None),
    }
    Ok(Some(()))
}
