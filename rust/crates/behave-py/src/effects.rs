//! Native effects marshalling and scalar entry-point dispatch.
use crate::arrays::Arguments;
use crate::dispatch::{named, scalar, widen, Cell};
use behave_fire::{ignite, mortality, safety, spot};
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &Arguments,
    i: usize,
    _single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "safety.calculate_safety_zone" => named(
            safety::calculate_safety_zone(
                a.n("flame_height_ft", i)?,
                a.n("number_of_personnel", i)?,
                a.n("area_per_person_sqft", i)?,
                a.n("number_of_equipment", i)?,
                a.n("area_per_equipment_sqft", i)?,
            )
            .named_values()
            .into_iter()
            .map(|(k, v)| (k, widen(v))),
        ),
        "spot.calculate_spotting_from_surface_fire" => {
            scalar(widen(spot::calculate_spotting_from_surface_fire(
                a.n("flame_length_ft", i)?,
                a.n("wind_mph", i)?,
                a.n("cover_height_ft", i)?,
            )))
        }
        "spot.calculate_spotting_from_burning_pile" => {
            scalar(widen(spot::calculate_spotting_from_burning_pile(
                a.n("flame_height_ft", i)?,
                a.n("wind_mph", i)?,
                a.n("cover_height_ft", i)?,
            )))
        }
        "spot.calculate_spotting_from_torching_trees" => {
            scalar(widen(spot::calculate_spotting_from_torching_trees(
                a.n("dbh_in", i)?,
                a.n("height_ft", i)?,
                a.n("count", i)?,
                a.n("wind_mph", i)?,
                a.n("cover_height_ft", i)?,
            )))
        }
        "ignite.calculate_fuel_temperature" => scalar(widen(ignite::calculate_fuel_temperature(
            a.n("air_temp_f", i)?,
            a.n("sun_shade_fraction", i)?,
        ))),
        "ignite.calculate_firebrand_ignition_probability" => {
            scalar(widen(ignite::calculate_firebrand_ignition_probability(
                a.n("air_temp_f", i)?,
                a.n("sun_shade_fraction", i)?,
                a.n("moisture_1h_fraction", i)?,
            )))
        }
        "ignite.calculate_lightning_ignition_probability" => {
            scalar(widen(ignite::calculate_lightning_ignition_probability(
                a.n("fuel_bed_type_grid", i)? as i32,
                a.n("moisture_100h_fraction", i)?,
                a.n("duff_depth_inches", i)?,
                a.n("charge_type", i)? as i32,
            )))
        }
        "mortality.calculate_scorch_height" => scalar(widen(mortality::calculate_scorch_height(
            a.n("fireline_intensity_btu_ft_s", i)?,
            a.n("midflame_wind_mph", i)?,
            a.n("air_temp_f", i)?,
        ))),
        "mortality.calculate_crown_scorch_mortality" => named(
            mortality::calculate_crown_scorch_mortality_with_coefficients(
                a.n("scorch_height_ft", i)?,
                a.n("tree_height_ft", i)?,
                a.n("crown_ratio", i)?,
                a.n("dbh_inches", i)?,
                [
                    a.n("coefficients.0", i)?,
                    a.n("coefficients.1", i)?,
                    a.n("coefficients.2", i)?,
                ],
            )
            .named_values(),
        ),
        _ => return Ok(None),
    }))
}
