//! Frozen scientific evidence for the replacement numerical engine.
#![allow(clippy::field_reassign_with_default)]

use behave_fire::crown::SurfaceFireInputs;
use behave_fire::engine::{
    CrownCell, CrownPlan, CrownTweaks, PreparedFuel, PreparedFuels, SurfaceCell, SurfaceOptions,
    SurfacePlan, SurfaceTweaks, WafMethod, WindHeight, WindOrientation,
};
use behave_fire::fuel_models::{CustomFuelModel, FuelModel, FuelModels};
use behave_fire::units::{SlopeUnit, SpeedUnit};
use serde_json::Value;

fn close(actual: f64, expected: f64, label: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{label}: {actual} should be NaN");
    } else if expected.is_infinite() {
        assert_eq!(actual, expected, "{label}");
    } else {
        assert!(
            (actual - expected).abs() <= 1e-12 + 1e-9 * expected.abs(),
            "{label}: {actual} != {expected}"
        );
        if actual == 0.0 && expected == 0.0 {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{label}: zero sign");
        }
    }
}

fn number(value: &Value) -> f64 {
    if let Some(number) = value.as_f64() {
        return number;
    }
    if let Some(values) = value.as_array() {
        return number(&values[0]);
    }
    if let Some(values) = value.get("values") {
        return number(&values[0]);
    }
    match value
        .as_str()
        .or_else(|| value.get("float").and_then(Value::as_str))
        .unwrap_or("")
    {
        "nan" | "NaN" => f64::NAN,
        "inf" | "Infinity" => f64::INFINITY,
        "-inf" | "-Infinity" => f64::NEG_INFINITY,
        _ => panic!("not a fixture number: {value}"),
    }
}

fn options(value: &Value) -> SurfaceOptions {
    let mut result = SurfaceOptions::default();
    if value["wind_orientation_mode"]
        .as_str()
        .unwrap_or("")
        .to_ascii_lowercase()
        .contains("north")
    {
        result.wind_orientation = WindOrientation::North;
    }
    let height = value["wind_height_mode"]
        .as_str()
        .unwrap_or("")
        .to_ascii_lowercase();
    if height.contains("10") || height.contains("ten") {
        result.wind_height = WindHeight::TenMetre;
    }
    if value["waf_method"]
        .as_str()
        .unwrap_or("")
        .replace('_', "")
        .eq_ignore_ascii_case("userinput")
    {
        result.waf_method = WafMethod::UserInput;
    }
    result
}

#[test]
fn crown_custom_model_ten_and_slope_match_pinned_python() {
    let fixtures: Value =
        serde_json::from_str(include_str!("data/port_crown_crown_coupling.json")).unwrap();
    for case in fixtures.as_array().unwrap() {
        let mut fuels = FuelModels::default();
        if let Some(value) = case["model10_changes"].get("dead_1h") {
            fuels.get_mut(10).unwrap().dead_1h = number(value);
        }
        if let Some(value) = case["model10_changes"].get("is_defined") {
            fuels.get_mut(10).unwrap().is_defined = value.as_bool().unwrap();
        }
        let input = &case["inputs"];
        let mut cell = CrownCell::default();
        cell.m1h = number(&input["m1h"]);
        cell.m10h = number(&input["m10h"]);
        cell.m100h = number(&input["m100h"]);
        cell.mlh = number(&input["mlh"]);
        cell.mlw = number(&input["mlw"]);
        cell.wind_speed = SpeedUnit::try_from(number(&input["wind_speed_units"]) as i32)
            .unwrap()
            .to_base(number(&input["wind_speed"]));
        cell.wind_direction = number(&input["wind_direction"]);
        cell.slope = SlopeUnit::try_from(number(&input["slope_units"]) as i32)
            .unwrap()
            .to_base(number(&input["slope"]));
        cell.aspect = number(&input["aspect"]);
        cell.canopy_base_height = number(&input["canopy_base_height"]);
        cell.canopy_height = number(&input["canopy_height"]);
        cell.canopy_bulk_density = number(&input["canopy_bulk_density"]);
        cell.moisture_foliar = number(&input["moisture_foliar"]);
        let mut supplied = SurfaceFireInputs::<f64>::new();
        supplied.surface_ros = number(&input["surface_results"]["spread_rate"]);
        supplied.surface_fli = number(&input["surface_results"]["fireline_intensity"]);
        supplied.surface_hpua = number(&input["surface_results"]["heat_per_unit_area"]);
        let prepared = PreparedFuels::from_registry(&fuels);
        let result = CrownPlan::new(options(input)).run_cell(
            &cell,
            &supplied,
            prepared.for_grid(10),
            CrownTweaks::default(),
        );
        assert_eq!(
            result.fire_type,
            number(&case["output"]["fire_type"]) as i32
        );
        for (name, actual) in result.named_values() {
            close(actual, number(&case["output"][name]), name);
        }
    }
}

#[test]
fn engine_cells_and_tweaks_have_no_heap_owners() {
    fn copy_without_drop<T: Copy>() {
        assert!(!std::mem::needs_drop::<T>());
    }
    copy_without_drop::<CrownCell>();
    copy_without_drop::<CrownTweaks>();
    copy_without_drop::<PreparedFuel>();
    copy_without_drop::<SurfaceCell>();
    copy_without_drop::<SurfaceOptions>();
    copy_without_drop::<SurfaceTweaks>();
    assert!(std::mem::size_of::<SurfaceCell>() <= 128);
    assert!(std::mem::size_of::<CrownCell>() <= 128);
}

#[test]
fn prepared_fuels_snapshot_registry_and_preserve_dynamic_curing() {
    let mut fuels = FuelModels::default();
    let prepared = PreparedFuels::from_registry(&fuels);
    let plan = SurfacePlan::new(SurfaceOptions::default());
    let mut cell = SurfaceCell::new(101);
    cell.mlh = 0.20;
    let cured = plan.prepare_fuelbed(&cell, prepared.for_grid(101));
    cell.mlh = 1.30;
    let live = plan.prepare_fuelbed(&cell, prepared.for_grid(101));
    assert_eq!(cured.wfl_live, 0.0);
    assert!(live.wfl_live > 0.0);
    let before = plan.run_cell(&cell, prepared.for_grid(101), SurfaceTweaks::default());
    fuels.get_mut(101).unwrap().dead_1h += 0.1;
    assert_eq!(
        before,
        plan.run_cell(&cell, prepared.for_grid(101), SurfaceTweaks::default())
    );
    let changed = PreparedFuels::from_registry(&fuels);
    assert_ne!(
        before,
        plan.run_cell(&cell, changed.for_grid(101), SurfaceTweaks::default())
    );
    assert_eq!(prepared.for_grid(-1), prepared.for_grid(0));
    assert_eq!(prepared.for_grid(261), prepared.for_grid(260));
}

#[test]
fn surface_all_grid_models_and_custom_science_match_pinned_python() {
    let fixtures: Value =
        serde_json::from_str(include_str!("data/port_surface_composed.json")).unwrap();
    let mut fuels = FuelModels::default();
    for (id, record) in fixtures["custom"].as_object().unwrap() {
        let mut model = FuelModel::default();
        model.code = record["code"].as_str().unwrap().into();
        model.name = record["name"].as_str().unwrap().into();
        model.fuel_bed_depth = number(&record["fuel_bed_depth"]);
        model.moisture_of_extinction_dead = number(&record["moisture_of_extinction_dead"]);
        model.heat_of_combustion_dead = number(&record["heat_of_combustion_dead"]);
        model.heat_of_combustion_live = number(&record["heat_of_combustion_live"]);
        model.dead_1h = number(&record["dead_1h"]);
        model.dead_10h = number(&record["dead_10h"]);
        model.dead_100h = number(&record["dead_100h"]);
        model.live_herb = number(&record["live_herb"]);
        model.live_woody = number(&record["live_woody"]);
        model.savr_1h = number(&record["savr_1h"]);
        model.savr_live_herb = number(&record["savr_live_herb"]);
        model.savr_live_woody = number(&record["savr_live_woody"]);
        model.is_dynamic = true;
        model.is_defined = true;
        assert!(fuels.set_custom_fuel_model(
            id.parse().unwrap(),
            CustomFuelModel {
                model,
                ..CustomFuelModel::default()
            }
        ));
    }
    let prepared = PreparedFuels::from_registry(&fuels);
    for case in fixtures["cases"].as_array().unwrap() {
        let settings = &case["settings"];
        let mut cell = SurfaceCell::new(case["model"].as_i64().unwrap() as i32);
        cell.m1h = number(&case["moisture"][0]);
        cell.m10h = number(&case["moisture"][1]);
        cell.m100h = number(&case["moisture"][2]);
        cell.mlh = number(&case["moisture"][3]);
        cell.mlw = number(&case["moisture"][4]);
        cell.wind_speed = SpeedUnit::try_from(number(&settings["wind_speed_units"]) as i32)
            .unwrap()
            .to_base(number(&settings["wind_speed"]));
        cell.wind_direction = number(&settings["wind_direction"]);
        cell.slope = SlopeUnit::try_from(number(&settings["slope_units"]) as i32)
            .unwrap()
            .to_base(number(&settings["slope"]));
        cell.aspect = number(&settings["aspect"]);
        cell.canopy_cover = number(&settings["canopy_cover"]);
        cell.canopy_height = number(&settings["canopy_height"]);
        cell.crown_ratio = number(&settings["crown_ratio"]);
        cell.user_waf = (!settings["user_waf"].is_null()).then(|| number(&settings["user_waf"]));
        let result = SurfacePlan::new(options(settings)).run_cell(
            &cell,
            prepared.for_grid(cell.fuel_model_grid),
            SurfaceTweaks::default(),
        );
        for (name, actual) in result.named_values() {
            let mut expected = number(&case["output"][name]);
            if (name == "flame_length" && number(&case["output"]["fireline_intensity"]).is_nan())
                || (name == "direction_of_max_spread"
                    && number(&case["output"]["spread_rate"]).is_nan())
            {
                expected = f64::NAN;
            }
            if number(&case["intermediates"]["heat_sink"]).is_nan()
                && matches!(
                    name,
                    "no_wind_no_slope_spread_rate"
                        | "spread_rate"
                        | "backing_spread_rate"
                        | "flanking_spread_rate"
                        | "fireline_intensity"
                        | "flame_length"
                        | "direction_of_max_spread"
                )
            {
                expected = f64::NAN;
            }
            if number(&case["intermediates"]["heat_sink"]).is_nan()
                && (number(&case["intermediates"]["relative_packing_ratio"]).is_nan()
                    || number(&case["intermediates"]["relative_packing_ratio"]) > 1e-7)
                && matches!(
                    name,
                    "effective_wind_speed" | "fire_length_to_width_ratio" | "eccentricity"
                )
            {
                expected = f64::NAN;
            }
            close(
                actual,
                expected,
                &format!(
                    "model={:?}, settings={:?}, {name}",
                    case["model"], case["settings"]
                ),
            );
        }
    }
}
