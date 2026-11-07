//! Scalar dispatch; all numerical work belongs to the standalone core.
use crate::prepared::PreparedArguments;
use behave_fire::crown;
use std::collections::BTreeMap;
pub type Cell = BTreeMap<String, f64>;
pub(crate) fn widen<T: Into<f64>>(value: T) -> f64 {
    value.into()
}
pub(crate) fn scalar(value: f64) -> Cell {
    BTreeMap::from([(String::new(), value)])
}
pub(crate) fn named<'a>(values: impl IntoIterator<Item = (&'a str, f64)>) -> Cell {
    values.into_iter().map(|(k, v)| (k.to_owned(), v)).collect()
}
pub(crate) fn classification(r: crown::FireClassification) -> Cell {
    named([
        ("fire_type", r.fire_type as f64),
        ("is_surface_fire", u8::from(r.is_surface_fire) as f64),
        ("is_passive_crown", u8::from(r.is_passive_crown) as f64),
        ("is_active_crown", u8::from(r.is_active_crown) as f64),
    ])
}
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &PreparedArguments<'_>,
    i: usize,
    single: bool,
) -> Result<Cell, String> {
    if a.has_masks() || a.has_single() || a.has_integers() || a.has_python_payloads() {
        if let Some(result) = crate::crown_masked::evaluate(operation, a, i)? {
            return Ok(result);
        }
        if let Some(result) = crate::masked_records::evaluate(operation, a, i)? {
            return Ok(result);
        }
        if let Some(result) = crate::masked::evaluate(operation, a, i)? {
            return Ok(result);
        }
    }
    let result = match operation.split('.').next().unwrap_or("") {
        "surface" => crate::surface::evaluate(operation, a, i, single)?,
        "crown" => crate::crown_dispatch::evaluate(operation, a, i, single)?,
        "safety" => crate::effects::evaluate(operation, a, i, single)?,
        "spot" => crate::effects::evaluate(operation, a, i, single)?,
        "ignite" => crate::effects::evaluate(operation, a, i, single)?,
        "mortality" => crate::effects::evaluate(operation, a, i, single)?,
        "vapor_pressure_deficit_calculator" => crate::tools::evaluate(operation, a, i, single)?,
        "fine_dead_fuel_moisture_tool" => crate::tools::evaluate(operation, a, i, single)?,
        "contain" => crate::contain::evaluate(operation, a, i, single)?,
        "units" => crate::units::evaluate(operation, a, i, single)?,
        "model" => crate::model::evaluate(operation, a, i, single)?,
        _ => None,
    };
    result.ok_or_else(|| format!("Native operation is unavailable: {operation}"))
}

pub fn supported(operation: &str) -> bool {
    matches!(
        operation,
        "contain.run_contain_sim_array"
            | "crown.assign_final_crown_fire_behavior"
            | "crown.calculate_critical_crown_fire_spread_rate"
            | "crown.calculate_critical_surface_fireline_intensity"
            | "crown.calculate_crown_fire_ratios"
            | "crown.calculate_crown_fraction_burned"
            | "crown.calculate_crown_heat_and_intensity"
            | "crown.calculate_crown_surface_fire"
            | "crown.calculate_crowning_surface_fire_spread_rate"
            | "crown.calculate_passive_crown_fire_behavior"
            | "crown.calculate_surface_fire_critical_spread_rate"
            | "crown.classify_crown_fire_type"
            | "crown.run_crown_fire"
            | "fine_dead_fuel_moisture_tool.calculate_fine_dead_fuel_moisture"
            | "ignite.calculate_firebrand_ignition_probability"
            | "ignite.calculate_fuel_temperature"
            | "ignite.calculate_lightning_ignition_probability"
            | "model.canopy_crown_ratio"
            | "model.run_surface"
            | "mortality.calculate_crown_scorch_mortality"
            | "mortality.calculate_scorch_height"
            | "safety.calculate_safety_zone"
            | "spot.calculate_spotting_from_burning_pile"
            | "spot.calculate_spotting_from_surface_fire"
            | "spot.calculate_spotting_from_torching_trees"
            | "surface.build_particle_arrays"
            | "surface.calculate_backing_spread_rate"
            | "surface.calculate_direction_of_max_spread"
            | "surface.calculate_eccentricity"
            | "surface.calculate_effective_wind_speed"
            | "surface.calculate_fire_area"
            | "surface.calculate_fire_length"
            | "surface.calculate_fire_length_to_width_ratio"
            | "surface.calculate_fire_perimeter"
            | "surface.calculate_fire_width"
            | "surface.calculate_fireline_intensity"
            | "surface.calculate_flame_length"
            | "surface.calculate_flanking_spread_rate"
            | "surface.calculate_forward_spread_rate"
            | "surface.calculate_fuelbed_intermediates"
            | "surface.calculate_heat_per_unit_area"
            | "surface.calculate_midflame_wind_speed"
            | "surface.calculate_no_wind_no_slope_spread_rate"
            | "surface.calculate_reaction_intensity"
            | "surface.calculate_reaction_intensity_output"
            | "surface.calculate_residence_time"
            | "surface.calculate_slope_phi"
            | "surface.calculate_surface_rate_vector"
            | "surface.calculate_surface_wind_adjustment_factor"
            | "surface.calculate_surface_wind_coefficients"
            | "surface.calculate_twenty_foot_wind_speed"
            | "surface.calculate_wind_adjustment_factor"
            | "surface.calculate_wind_phi"
            | "surface.run_surface_fire"
            | "units.Area.from_base"
            | "units.Area.to_base"
            | "units.BasalArea.from_base"
            | "units.BasalArea.to_base"
            | "units.Density.from_base"
            | "units.Density.to_base"
            | "units.FirelineIntensity.from_base"
            | "units.FirelineIntensity.to_base"
            | "units.Fraction.from_base"
            | "units.Fraction.to_base"
            | "units.HeatOfCombustion.from_base"
            | "units.HeatOfCombustion.to_base"
            | "units.HeatPerUnitArea.from_base"
            | "units.HeatPerUnitArea.to_base"
            | "units.HeatSink.from_base"
            | "units.HeatSink.to_base"
            | "units.HeatSourceAndReactionIntensity.from_base"
            | "units.HeatSourceAndReactionIntensity.to_base"
            | "units.Length.from_base"
            | "units.Length.to_base"
            | "units.Loading.from_base"
            | "units.Loading.to_base"
            | "units.Pressure.from_base"
            | "units.Pressure.to_base"
            | "units.Slope.from_base"
            | "units.Slope.to_base"
            | "units.Speed.from_base"
            | "units.Speed.to_base"
            | "units.SurfaceAreaToVolume.from_base"
            | "units.SurfaceAreaToVolume.to_base"
            | "units.Temperature.from_base"
            | "units.Temperature.to_base"
            | "units.Time.from_base"
            | "units.Time.to_base"
            | "vapor_pressure_deficit_calculator.calculate_vpd"
    )
}
