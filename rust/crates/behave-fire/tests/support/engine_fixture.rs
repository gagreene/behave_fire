//! Decode frozen Python fixture inputs into the new base-unit Rust interface.
// Individual test modules exercise different subsets of these fixture readers.
#![allow(dead_code, clippy::field_reassign_with_default)]

use behave_fire::crown::SurfaceFireInputs;
use behave_fire::engine::{
    CrownCell, SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceTweaks, WafMethod, WindHeight,
    WindOrientation,
};
use behave_fire::surface::{FuelbedIntermediates, SurfaceResult};
use behave_fire::units::{SlopeUnit, SpeedUnit};
use serde_json::Value;

pub fn number(value: &Value, index: usize) -> f64 {
    if let Some(values) = value.get("values").and_then(Value::as_array) {
        return number(&values[index % values.len()], 0);
    }
    if let Some(values) = value.as_array() {
        return number(&values[index % values.len()], 0);
    }
    if let Some(value) = value.get("float") {
        return number(value, 0);
    }
    if let Some(value) = value.as_f64() {
        return value;
    }
    if let Some(value) = value.as_bool() {
        return f64::from(u8::from(value));
    }
    match value.as_str().unwrap_or("") {
        "NaN" | "nan" => f64::NAN,
        "Infinity" | "inf" | "+inf" => f64::INFINITY,
        "-Infinity" | "-inf" => f64::NEG_INFINITY,
        _ => panic!("fixture is not numerical: {value}"),
    }
}

pub fn optional(values: &Value, name: &str, index: usize, default: f64) -> f64 {
    values
        .get(name)
        .filter(|value| !value.is_null())
        .map_or(default, |value| number(value, index))
}

pub fn orientation(value: &str) -> WindOrientation {
    if value.to_ascii_lowercase().contains("north") {
        WindOrientation::North
    } else {
        WindOrientation::Upslope
    }
}

pub fn height(value: &str) -> WindHeight {
    let value = value.to_ascii_lowercase();
    if value.contains("10") || value.contains("ten") {
        WindHeight::TenMetre
    } else {
        WindHeight::TwentyFoot
    }
}

pub fn waf(value: &str) -> WafMethod {
    if value.replace('_', "").eq_ignore_ascii_case("userinput") {
        WafMethod::UserInput
    } else {
        WafMethod::CrownRatio
    }
}

pub fn surface_options(values: &Value) -> SurfaceOptions {
    SurfaceOptions {
        wind_orientation: orientation(
            values["wind_orientation_mode"]
                .as_str()
                .unwrap_or("RelativeToUpslope"),
        ),
        wind_height: height(values["wind_height_mode"].as_str().unwrap_or("TwentyFoot")),
        waf_method: waf(values["waf_method"].as_str().unwrap_or("UseCrownRatio")),
    }
}

pub fn surface_cell(values: &Value, index: usize) -> SurfaceCell {
    let mut cell = SurfaceCell::new(optional(values, "fuel_model_grid", index, 1.0) as i32);
    cell.m1h = optional(values, "m1h", index, cell.m1h);
    cell.m10h = optional(values, "m10h", index, cell.m10h);
    cell.m100h = optional(values, "m100h", index, cell.m100h);
    cell.mlh = optional(values, "mlh", index, cell.mlh);
    cell.mlw = optional(values, "mlw", index, cell.mlw);
    cell.wind_speed = SpeedUnit::try_from(optional(values, "wind_speed_units", index, 5.0) as i32)
        .unwrap()
        .to_base(optional(values, "wind_speed", index, 0.0));
    cell.wind_direction = optional(values, "wind_direction", index, 0.0);
    cell.slope = SlopeUnit::try_from(optional(values, "slope_units", index, 0.0) as i32)
        .unwrap()
        .to_base(optional(
            values,
            "slope",
            index,
            optional(values, "slope_deg", index, 0.0),
        ));
    cell.aspect = optional(values, "aspect", index, 0.0);
    cell.canopy_cover = optional(values, "canopy_cover", index, 0.0);
    cell.canopy_height = optional(values, "canopy_height", index, 0.0);
    cell.crown_ratio = optional(values, "crown_ratio", index, 0.0);
    cell.user_waf = values
        .get("user_waf")
        .filter(|value| !value.is_null())
        .map(|value| number(value, index));
    cell
}

pub fn crown_cell(values: &Value, index: usize) -> CrownCell {
    let surface = surface_cell(values, index);
    let mut cell = CrownCell::default();
    cell.m1h = surface.m1h;
    cell.m10h = surface.m10h;
    cell.m100h = surface.m100h;
    cell.mlh = surface.mlh;
    cell.mlw = surface.mlw;
    cell.wind_speed = surface.wind_speed;
    cell.wind_direction = surface.wind_direction;
    cell.slope = surface.slope;
    cell.aspect = surface.aspect;
    cell.canopy_base_height = optional(values, "canopy_base_height", index, 0.0);
    cell.canopy_height = optional(values, "canopy_height", index, 0.0);
    cell.canopy_bulk_density = optional(values, "canopy_bulk_density", index, 0.0);
    cell.moisture_foliar = optional(values, "moisture_foliar", index, 100.0);
    cell
}

pub fn surface_values(values: &Value, index: usize) -> SurfaceFireInputs<f64> {
    let mut result = SurfaceFireInputs::<f64>::new();
    result.surface_ros = optional(values, "spread_rate", index, 0.0);
    result.surface_fli = optional(values, "fireline_intensity", index, 0.0);
    result.surface_hpua = optional(values, "heat_per_unit_area", index, 0.0);
    result
}

pub fn surface_with_reaction(
    reaction: f64,
    fuelbed: &FuelbedIntermediates,
    cell: &SurfaceCell,
    options: SurfaceOptions,
    single: bool,
) -> SurfaceResult {
    let mut tweaks = SurfaceTweaks {
        reaction_intensity: Some(reaction),
        ..SurfaceTweaks::default()
    };
    if single {
        let reaction = reaction as f32;
        tweaks.reaction_intensity = Some(reaction as f64);
        tweaks.slope_phi = Some(behave_fire::math::np_min(
            behave_fire::surface::calculate_slope_phi(
                cell.slope,
                SlopeUnit::Degrees,
                fuelbed.packing_ratio,
                f64::INFINITY,
            ),
            (0.9_f32 * reaction) as f64,
        ));
    }
    SurfacePlan::new(options).run_from_fuelbed(cell, fuelbed, tweaks)
}
