//! Native effects marshalling and scalar entry-point dispatch.
use crate::dispatch::{named, scalar, widen, Cell};
use crate::prepared::{Field, PreparedArguments};
use behave_fire::{ignite, mortality, safety, spot};
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &PreparedArguments<'_>,
    i: usize,
    _single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "safety.calculate_safety_zone" => named(
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
        "spot.calculate_spotting_from_surface_fire" => {
            scalar(widen(spot::calculate_spotting_from_surface_fire(
                a.n(Field::FlameLengthFt, i)?,
                a.n(Field::WindMph, i)?,
                a.n(Field::CoverHeightFt, i)?,
            )))
        }
        "spot.calculate_spotting_from_burning_pile" => {
            scalar(widen(spot::calculate_spotting_from_burning_pile(
                a.n(Field::FlameHeightFt, i)?,
                a.n(Field::WindMph, i)?,
                a.n(Field::CoverHeightFt, i)?,
            )))
        }
        "spot.calculate_spotting_from_torching_trees" => {
            scalar(widen(spot::calculate_spotting_from_torching_trees(
                a.n(Field::DbhIn, i)?,
                a.n(Field::HeightFt, i)?,
                a.n(Field::Count, i)?,
                a.n(Field::WindMph, i)?,
                a.n(Field::CoverHeightFt, i)?,
            )))
        }
        "ignite.calculate_fuel_temperature" => scalar(widen(ignite::calculate_fuel_temperature(
            a.n(Field::AirTempF, i)?,
            a.n(Field::SunShadeFraction, i)?,
        ))),
        "ignite.calculate_firebrand_ignition_probability" => {
            scalar(widen(ignite::calculate_firebrand_ignition_probability(
                a.n(Field::AirTempF, i)?,
                a.n(Field::SunShadeFraction, i)?,
                a.n(Field::Moisture1hFraction, i)?,
            )))
        }
        "ignite.calculate_lightning_ignition_probability" => {
            scalar(widen(ignite::calculate_lightning_ignition_probability(
                a.n(Field::FuelBedTypeGrid, i)? as i32,
                a.n(Field::Moisture100hFraction, i)?,
                a.n(Field::DuffDepthInches, i)?,
                a.n(Field::ChargeType, i)? as i32,
            )))
        }
        "mortality.calculate_scorch_height" => scalar(widen(mortality::calculate_scorch_height(
            a.n(Field::FirelineIntensityBtuFtS, i)?,
            a.n(Field::MidflameWindMph, i)?,
            a.n(Field::AirTempF, i)?,
        ))),
        "mortality.calculate_crown_scorch_mortality" => named(
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
    }))
}
