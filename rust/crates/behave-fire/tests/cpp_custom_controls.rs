//! Explicit percent-unit controls preserve the original numbered observations.

#[path = "support/cpp_hash.rs"]
mod cpp_hash;
#[path = "support/cpp_policy.rs"]
mod cpp_policy;

use behave_fire::engine::{
    PreparedFuel, SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceTweaks, WindOrientation,
};
use behave_fire::fuel_models::{CustomFuelModel, FuelModels};
use behave_fire::units::{FractionUnit, SpeedUnit};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

fn number(case: &Value, key: &str) -> f64 {
    case[key].as_f64().unwrap()
}

fn rows(source: &str) -> BTreeMap<String, Value> {
    let mut result = BTreeMap::new();
    for line in source.lines() {
        let row: Value = serde_json::from_str(line).unwrap();
        let id = row["id"].as_str().unwrap().to_string();
        assert!(result.insert(id, row).is_none(), "duplicate custom case");
    }
    assert!(!result.is_empty());
    result
}

// ---- custom fuel controls ----
#[test]
fn explicit_percent_controls_match_numbered_physics_and_consume_all_cpp_fields() {
    let inputs = include_bytes!("data/cpp_reference/surface_custom/cases.jsonl");
    let outputs = include_bytes!("data/cpp_reference/surface_custom/results.jsonl");
    let policy_bytes = include_bytes!("data/cpp_reference/surface/tolerance-policy.json");
    let manifest: Value = serde_json::from_str(include_str!(
        "data/cpp_reference/surface_custom/manifest.json"
    ))
    .unwrap();
    let files = manifest["files"].as_object().unwrap();
    assert_eq!(files.len(), 2);
    for (name, bytes) in [
        ("cases.jsonl", inputs.as_slice()),
        ("results.jsonl", outputs.as_slice()),
    ] {
        assert_eq!(files[name], cpp_hash::sha256(bytes));
    }
    assert_eq!(
        manifest["shared_policy_sha256"],
        cpp_hash::sha256(policy_bytes)
    );
    assert_eq!(
        manifest["upstream_sha"],
        "f3b2943e69abca942e795c5b1b45276f6338abb5"
    );
    let cases = rows(std::str::from_utf8(inputs).unwrap());
    let results = rows(std::str::from_utf8(outputs).unwrap());
    let policy: Value = serde_json::from_slice(policy_bytes).unwrap();
    assert_eq!(cases.len(), 139);
    assert_eq!(
        cases.keys().collect::<Vec<_>>(),
        results.keys().collect::<Vec<_>>()
    );
    if policy["approved"].as_bool() == Some(true) {
        cpp_policy::require_qualification(&policy).unwrap();
    }
    let mut source_models = BTreeSet::new();
    let mut compared = 0;
    for (id, case) in &cases {
        let source = case["fuel_model"].as_i64().unwrap();
        assert!(cpp_policy::unit_record(source));
        source_models.insert(source);
        assert_eq!(case["fuel_mode"], "percent_unit_custom");
        let mut fuels = FuelModels::default();
        let source_fuel = fuels.for_grid(source as i32).clone();
        assert_eq!(
            source_fuel.moisture_of_extinction_dead.to_bits(),
            FractionUnit::Percent
                .to_base(number(case, "moe_percent"))
                .to_bits()
        );
        let mut custom = CustomFuelModel {
            model: source_fuel.clone(),
            moisture_units: FractionUnit::Percent,
            ..CustomFuelModel::default()
        };
        custom.model.moisture_of_extinction_dead = number(case, "moe_percent");
        assert!(fuels.set_custom_fuel_model(250, custom));
        let cell = SurfaceCell {
            fuel_model_grid: source as i32,
            m1h: number(case, "m1h") / 100.,
            m10h: number(case, "m10h") / 100.,
            m100h: number(case, "m100h") / 100.,
            mlh: number(case, "mlh") / 100.,
            mlw: number(case, "mlw") / 100.,
            wind_speed: number(case, "wind_speed") * 88.,
            wind_direction: number(case, "wind_direction"),
            slope: (number(case, "slope") / 100.).atan().to_degrees(),
            aspect: number(case, "aspect"),
            canopy_cover: number(case, "canopy_cover") / 100.,
            canopy_height: number(case, "canopy_height"),
            crown_ratio: number(case, "crown_ratio"),
            user_waf: None,
        };
        for (key, default) in [
            ("wind_height", "TwentyFoot"),
            ("wind_orientation", "North"),
            ("waf_method", "CrownRatio"),
            ("moisture_mode", "BySizeClass"),
        ] {
            assert_eq!(
                case.get(key).and_then(Value::as_str).unwrap_or(default),
                default
            );
        }
        let plan = SurfacePlan::new(SurfaceOptions {
            wind_orientation: WindOrientation::North,
            ..SurfaceOptions::default()
        });
        let original = plan.run_cell(
            &cell,
            &PreparedFuel::from_model(&source_fuel),
            SurfaceTweaks::default(),
        );
        let actual = plan.run_cell(
            &SurfaceCell {
                fuel_model_grid: 250,
                ..cell
            },
            &PreparedFuel::from_model(fuels.for_grid(250)),
            SurfaceTweaks::default(),
        );
        assert_eq!(original, actual, "numbered/custom physics changed: {id}");
        let values = [
            actual.fireline_intensity,
            actual.flame_length,
            actual.heat_per_unit_area,
            actual.fire_length_to_width_ratio,
            SpeedUnit::ChainsPerHour.from_base(actual.spread_rate),
        ];
        let original_values = [
            original.fireline_intensity,
            original.flame_length,
            original.heat_per_unit_area,
            original.fire_length_to_width_ratio,
            SpeedUnit::ChainsPerHour.from_base(original.spread_rate),
        ];
        for (numbered, custom) in original_values.into_iter().zip(values) {
            assert_eq!(
                numbered.to_bits(),
                custom.to_bits(),
                "numbered/custom bits changed: {id}"
            );
        }
        let golden = &results[id];
        assert_eq!(golden["status"], "ok");
        assert_eq!(golden["upstream_entries"], 1);
        assert_eq!(golden["geometry_entries"], 1);
        assert_eq!(golden["output"].as_object().unwrap().len(), 5);
        for (field, rust) in cpp_policy::FIELDS.into_iter().zip(values) {
            let cpp = number(&golden["output"], field);
            assert!(rust.is_finite() && cpp.is_finite());
            if policy["approved"].as_bool() == Some(true) {
                cpp_policy::compare(rust, cpp, &policy, field).unwrap();
            }
            println!("CPP_PERCENT_CONTROL id={id} field={field} cpp={cpp:.17e} rust={rust:.17e}");
            compared += 1;
        }
    }
    assert_eq!(source_models, cpp_policy::UNIT_MODELS.into_iter().collect());
    assert_eq!(compared, 695);
}

// ---- qualification inventory ----
#[test]
fn qualification_rejects_missing_core_source_inventory() {
    let mut policy: Value = serde_json::from_str(include_str!(
        "data/cpp_reference/surface/tolerance-policy.json"
    ))
    .unwrap();
    policy["approved"] = true.into();
    let mut artifacts = serde_json::Map::new();
    for (name, payload) in [
        (
            "surface/cases.jsonl",
            include_bytes!("data/cpp_reference/surface/cases.jsonl").as_slice(),
        ),
        (
            "surface/results.jsonl",
            include_bytes!("data/cpp_reference/surface/results.jsonl").as_slice(),
        ),
        (
            "surface_custom/cases.jsonl",
            include_bytes!("data/cpp_reference/surface_custom/cases.jsonl").as_slice(),
        ),
        (
            "surface_custom/results.jsonl",
            include_bytes!("data/cpp_reference/surface_custom/results.jsonl").as_slice(),
        ),
    ] {
        artifacts.insert(
            format!("tests/data/cpp_reference/{name}"),
            cpp_hash::sha256(payload).into(),
        );
    }
    let mut reports = serde_json::Map::new();
    for key in [
        "cpp-GNU",
        "cpp-MSVC",
        "ports-linux-x86_64",
        "ports-linux-aarch64",
        "ports-linux-musl-x86_64",
        "ports-macos-x86_64",
        "ports-macos-arm64",
        "ports-windows-x86_64",
        "ports-windows-arm64",
    ] {
        let summary = if key.starts_with("cpp-") {
            serde_json::json!({"cases":773,"available":722,"fields":3610,"unavailable":51})
        } else {
            serde_json::json!({"cases_per_backend":583,"fields_per_backend":2915,"paired_cases_per_backend":139})
        };
        reports.insert(
            key.into(),
            serde_json::json!({"sha256":"1".repeat(64),"summary":summary}),
        );
    }
    policy["qualification"] = serde_json::json!({"calibration_complete":true,
        "upstream_sha":"f3b2943e69abca942e795c5b1b45276f6338abb5",
        "fields":policy["fields"],"artifacts":artifacts,"reports":reports,"source_files":{}});
    assert!(cpp_policy::require_qualification(&policy).is_err());
}
