//! Native effects marshalling and scalar entry-point dispatch.
use crate::component_columns::RowWriter;
use crate::dispatch::{named, scalar, widen};
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
use behave_fire::{ignite, mortality, safety, spot};
#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    out: &mut RowWriter<'_, '_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::SafetyCalculateSafetyZone => named(
            out,
            safety::calculate_safety_zone(
                a.n(Field::FlameHeightFt, i)?,
                a.n(Field::NumberOfPersonnel, i)?,
                a.n(Field::AreaPerPersonSqft, i)?,
                a.n(Field::NumberOfEquipment, i)?,
                a.n(Field::AreaPerEquipmentSqft, i)?,
            )
            .named_values()
            .into_iter()
            .map(|(k, v)| (k, widen(v))),
        ),
        Operation::SpotCalculateSpottingFromSurfaceFire => scalar(
            out,
            widen(spot::calculate_spotting_from_surface_fire(
                a.n(Field::FlameLengthFt, i)?,
                a.n(Field::WindMph, i)?,
                a.n(Field::CoverHeightFt, i)?,
            )),
        ),
        Operation::SpotCalculateSpottingFromBurningPile => scalar(
            out,
            widen(spot::calculate_spotting_from_burning_pile(
                a.n(Field::FlameHeightFt, i)?,
                a.n(Field::WindMph, i)?,
                a.n(Field::CoverHeightFt, i)?,
            )),
        ),
        Operation::SpotCalculateSpottingFromTorchingTrees => scalar(
            out,
            widen(spot::calculate_spotting_from_torching_trees(
                a.n(Field::DbhIn, i)?,
                a.n(Field::HeightFt, i)?,
                a.n(Field::Count, i)?,
                a.n(Field::WindMph, i)?,
                a.n(Field::CoverHeightFt, i)?,
            )),
        ),
        Operation::IgniteCalculateFuelTemperature => scalar(
            out,
            widen(ignite::calculate_fuel_temperature(
                a.n(Field::AirTempF, i)?,
                a.n(Field::SunShadeFraction, i)?,
            )),
        ),
        Operation::IgniteCalculateFirebrandIgnitionProbability => scalar(
            out,
            widen(ignite::calculate_firebrand_ignition_probability(
                a.n(Field::AirTempF, i)?,
                a.n(Field::SunShadeFraction, i)?,
                a.n(Field::Moisture1hFraction, i)?,
            )),
        ),
        Operation::IgniteCalculateLightningIgnitionProbability => scalar(
            out,
            widen(ignite::calculate_lightning_ignition_probability(
                a.code(Field::FuelBedTypeGrid, i)? as i32,
                a.n(Field::Moisture100hFraction, i)?,
                a.n(Field::DuffDepthInches, i)?,
                a.code(Field::ChargeType, i)? as i32,
            )),
        ),
        Operation::MortalityCalculateScorchHeight => scalar(
            out,
            widen(mortality::calculate_scorch_height(
                a.n(Field::FirelineIntensityBtuFtS, i)?,
                a.n(Field::MidflameWindMph, i)?,
                a.n(Field::AirTempF, i)?,
            )),
        ),
        Operation::MortalityCalculateCrownScorchMortality => named(
            out,
            mortality::calculate_crown_scorch_mortality_with_coefficients(
                a.n(Field::ScorchHeightFt, i)?,
                a.n(Field::TreeHeightFt, i)?,
                a.n(Field::CrownRatio, i)?,
                a.n(Field::DbhInches, i)?,
                [
                    a.n(Field::Coefficients0, i)?,
                    a.n(Field::Coefficients1, i)?,
                    a.n(Field::Coefficients2, i)?,
                ],
            )
            .named_values(),
        ),
        _ => return Ok(None),
    }
    Ok(Some(()))
}
