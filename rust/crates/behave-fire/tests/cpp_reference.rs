//! Independent C++ Surface observations with a fail-closed calibrated policy.
//!
//! Passing these tests validates fixture structure, real core evaluation, and a
//! calibrated gate, while preserving documented original-unit differences. Run with
//! `--test cpp_reference -- --nocapture` to retain every raw numeric difference.

use behave_fire::engine::{
    PreparedFuel, SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceTweaks, WafMethod, WindHeight,
    WindOrientation,
};
use behave_fire::fuel_models::FuelModels;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[path = "support/cpp_hash.rs"]
mod cpp_hash;
use cpp_hash::sha256;
#[path = "support/cpp_policy.rs"]
mod cpp_policy;

const FIELDS: [&str; 5] = [
    "fireline_intensity_btu_ft_s",
    "flame_length_ft",
    "heat_per_unit_area_btu_ft2",
    "length_to_width_ratio",
    "spread_chains_per_hour",
];

#[derive(Debug, Eq, PartialEq)]
enum ReviewRequired {
    MissingCalibration,
    NumericalMismatch,
    IncompleteCoverage,
    RequiresOwnerReview,
}

fn compare_core_to_cpp(case: &Value, golden: &Value, fuels: &FuelModels) -> Vec<Value> {
    assert_eq!(case["operation"], "surface");
    let cell = surface_cell(case);
    let plan = SurfacePlan::new(surface_options(case));
    let actual = plan.run_cell(
        &cell,
        &PreparedFuel::from_model(fuels.for_grid(cell.fuel_model_grid)),
        SurfaceTweaks::default(),
    );
    let actual_values = [
        actual.fireline_intensity,
        actual.flame_length,
        actual.heat_per_unit_area,
        actual.fire_length_to_width_ratio,
        actual.spread_rate * 60.0 / 66.0,
    ];
    FIELDS
        .iter()
        .zip(actual_values)
        .map(|(field, rust)| {
            let cpp = number(&golden["output"], field);
            let absolute = (rust - cpp).abs();
            let relative = if cpp == 0.0 {
                None
            } else {
                Some((absolute / cpp.abs()).to_string())
            };
            json!({
                "id": case["id"],
                "source_model": case["fuel_model"],
                "field": field,
                "cpp": cpp.to_string(),
                "rust": rust.to_string(),
                "signed_difference": (rust - cpp).to_string(),
                "absolute_difference": absolute.to_string(),
                "relative_difference": relative,
                "ulp_distance": ulp_distance(cpp, rust),
                "exact_bits": cpp.to_bits() == rust.to_bits(),
                "cpp_finite": cpp.is_finite(),
                "rust_finite": rust.is_finite(),
                "comparison_status": "raw_observation",
            })
        })
        .collect()
}

fn comparison_gate(policy: &Value, observations: &[Value]) -> Result<(), ReviewRequired> {
    assert!(!observations.is_empty(), "no independent observations");
    if policy["approved"].as_bool() != Some(true) {
        return Err(ReviewRequired::RequiresOwnerReview);
    }
    cpp_policy::require_qualification(policy).map_err(|_| ReviewRequired::MissingCalibration)?;
    let mut compared = 0;
    let mut documented = 0;
    for row in observations {
        let model = row["source_model"]
            .as_i64()
            .ok_or(ReviewRequired::IncompleteCoverage)?;
        if cpp_policy::unit_record(model) {
            documented += 1;
            continue;
        }
        let left = row["rust"]
            .as_str()
            .and_then(|value| value.parse::<f64>().ok())
            .ok_or(ReviewRequired::NumericalMismatch)?;
        let right = row["cpp"]
            .as_str()
            .and_then(|value| value.parse::<f64>().ok())
            .ok_or(ReviewRequired::NumericalMismatch)?;
        let field = row["field"]
            .as_str()
            .ok_or(ReviewRequired::IncompleteCoverage)?;
        cpp_policy::compare(left, right, policy, field)
            .map_err(|_| ReviewRequired::NumericalMismatch)?;
        compared += 1;
    }
    if compared != 2220 || documented != 695 {
        return Err(ReviewRequired::IncompleteCoverage);
    }
    Ok(())
}

#[test]
fn cpp_surface_observations_follow_the_qualified_policy() {
    verify_manifest();
    let cases = parse_rows(include_str!("data/cpp_reference/surface/cases.jsonl"));
    let results = parse_rows(include_str!("data/cpp_reference/surface/results.jsonl"));
    let policy: Value = serde_json::from_str(include_str!(
        "data/cpp_reference/surface/tolerance-policy.json"
    ))
    .unwrap();
    assert_eq!(policy["conditional_owner_approval"], true);
    let expected_fields: BTreeSet<_> = FIELDS.into_iter().collect();
    let policy_fields: BTreeSet<_> = policy["fields"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(policy_fields, expected_fields);
    for field in FIELDS {
        assert_eq!(policy["fields"][field]["kind"], "relative_absolute");
        assert_eq!(policy["fields"][field]["rtol"].as_f64(), Some(1e-12));
        assert_eq!(policy["fields"][field]["atol"].as_f64(), Some(1e-12));
    }
    assert!(!cases.is_empty());
    assert_eq!(
        cases.keys().collect::<Vec<_>>(),
        results.keys().collect::<Vec<_>>()
    );
    let fuels = FuelModels::default();
    let mut observations = Vec::new();
    let mut unavailable = 0;
    for (id, case) in &cases {
        let golden = &results[id];
        match golden["status"].as_str().unwrap() {
            "ok" => {
                assert_eq!(golden["upstream_entries"], 1);
                assert_eq!(golden["geometry_entries"], 1);
                let fields: BTreeSet<_> = golden["output"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .map(String::as_str)
                    .collect();
                assert_eq!(fields, expected_fields, "case {id}");
                observations.extend(compare_core_to_cpp(case, golden, &fuels));
            }
            "unavailable" => {
                assert_eq!(golden["finding"], "F-013");
                assert!(
                    golden.get("output").is_none(),
                    "unavailable golden has outputs"
                );
                unavailable += 1;
            }
            status => panic!("unsupported C++ status for {id}: {status}"),
        }
    }
    assert_eq!(
        observations.len(),
        (cases.len() - unavailable) * FIELDS.len()
    );
    for observation in &observations {
        println!("CPP_SURFACE_RAW {observation}");
    }
    if policy["approved"].as_bool() == Some(true) {
        assert_eq!(comparison_gate(&policy, &observations), Ok(()));
    } else {
        assert_eq!(
            comparison_gate(&policy, &observations),
            Err(ReviewRequired::RequiresOwnerReview)
        );
    }
    println!(
        "CPP_SURFACE_POLICY cases={} observations={} unavailable={} qualified={}",
        cases.len(),
        observations.len(),
        unavailable,
        policy["approved"],
    );
}

fn number(record: &Value, key: &str) -> f64 {
    record[key]
        .as_f64()
        .unwrap_or_else(|| panic!("missing numeric {key}: {record}"))
}

fn parse_rows(source: &str) -> BTreeMap<String, Value> {
    let mut rows = BTreeMap::new();
    for line in source.lines() {
        assert!(!line.trim().is_empty(), "blank JSONL fixture row");
        let row: Value = serde_json::from_str(line).unwrap();
        let id = row["id"].as_str().unwrap().to_owned();
        assert!(!id.is_empty());
        assert!(
            rows.insert(id.clone(), row).is_none(),
            "duplicate fixture id: {id}"
        );
    }
    rows
}

#[test]
fn pending_policy_cannot_certify_even_exact_values() {
    let observations = [json!({"cpp": "1", "rust": "1", "exact_bits": true})];
    assert_eq!(
        comparison_gate(&json!({"approved": false}), &observations),
        Err(ReviewRequired::RequiresOwnerReview),
    );
    assert_eq!(
        comparison_gate(&json!({"approved": true}), &observations),
        Err(ReviewRequired::MissingCalibration),
    );
}

#[test]
fn sha256_matches_standard_vectors_and_detects_fixture_mutation() {
    assert_eq!(
        sha256(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
    );
    let fixture = include_bytes!("data/cpp_reference/surface/results.jsonl");
    let mut mutated = fixture.to_vec();
    mutated[0] ^= 1;
    assert_ne!(sha256(fixture), sha256(&mutated));
}

fn surface_cell(case: &Value) -> SurfaceCell {
    SurfaceCell {
        fuel_model_grid: i32::try_from(case["fuel_model"].as_i64().unwrap()).unwrap(),
        m1h: number(case, "m1h") / 100.0,
        m10h: number(case, "m10h") / 100.0,
        m100h: number(case, "m100h") / 100.0,
        mlh: number(case, "mlh") / 100.0,
        mlw: number(case, "mlw") / 100.0,
        wind_speed: number(case, "wind_speed") * 88.0,
        wind_direction: number(case, "wind_direction"),
        slope: (number(case, "slope") / 100.0).atan().to_degrees(),
        aspect: number(case, "aspect"),
        canopy_cover: number(case, "canopy_cover") / 100.0,
        canopy_height: number(case, "canopy_height"),
        crown_ratio: number(case, "crown_ratio"),
        user_waf: case.get("user_waf").map(|value| value.as_f64().unwrap()),
    }
}

#[test]
fn surface_controls_preserve_defaults_and_honor_optional_modes() {
    let defaults = surface_options(&json!({}));
    assert_eq!(defaults.wind_orientation, WindOrientation::North);
    assert_eq!(defaults.wind_height, WindHeight::TwentyFoot);
    assert_eq!(defaults.waf_method, WafMethod::CrownRatio);
    let source = include_str!("data/cpp_reference/surface/cases.jsonl");
    let mut case: Value = serde_json::from_str(source.lines().next().unwrap()).unwrap();
    assert_eq!(surface_cell(&case).user_waf, None);
    case["moisture_mode"] = json!("BySizeClass");
    for (height, expected_height) in [
        ("TwentyFoot", WindHeight::TwentyFoot),
        ("TenMeter", WindHeight::TenMetre),
    ] {
        for (orientation, expected_orientation) in [
            ("North", WindOrientation::North),
            ("Upslope", WindOrientation::Upslope),
        ] {
            case["wind_height"] = json!(height);
            case["wind_orientation"] = json!(orientation);
            for (method, user_waf) in [
                ("CrownRatio", None),
                ("UserInput", Some(0.0)),
                ("UserInput", Some(0.4)),
                ("UserInput", Some(1.0)),
            ] {
                case["waf_method"] = json!(method);
                if let Some(value) = user_waf {
                    case["user_waf"] = json!(value);
                } else {
                    case.as_object_mut().unwrap().remove("user_waf");
                }
                let actual = surface_options(&case);
                assert_eq!(actual.wind_height, expected_height);
                assert_eq!(actual.wind_orientation, expected_orientation);
                assert_eq!(
                    actual.waf_method,
                    if method == "UserInput" {
                        WafMethod::UserInput
                    } else {
                        WafMethod::CrownRatio
                    }
                );
                assert_eq!(surface_cell(&case).user_waf, user_waf);
            }
        }
    }
}

#[test]
fn surface_controls_reject_unregistered_modes() {
    for case in [
        json!({"wind_height": "unsupported"}),
        json!({"wind_orientation": "unsupported"}),
        json!({"waf_method": "unsupported"}),
        json!({"moisture_mode": "AllAggregate"}),
        json!({"wind_height": 20}),
    ] {
        assert!(std::panic::catch_unwind(|| surface_options(&case)).is_err());
    }
}

fn surface_options(case: &Value) -> SurfaceOptions {
    if let Some(mode) = case.get("moisture_mode") {
        assert_eq!(mode.as_str().unwrap(), "BySizeClass");
    }
    let wind_orientation = case
        .get("wind_orientation")
        .map(|value| value.as_str().unwrap())
        .unwrap_or("North");
    let wind_height = case
        .get("wind_height")
        .map(|value| value.as_str().unwrap())
        .unwrap_or("TwentyFoot");
    let waf_method = case
        .get("waf_method")
        .map(|value| value.as_str().unwrap())
        .unwrap_or("CrownRatio");
    SurfaceOptions {
        wind_orientation: match wind_orientation {
            "North" => WindOrientation::North,
            "Upslope" => WindOrientation::Upslope,
            unknown => panic!("unsupported wind_orientation: {unknown}"),
        },
        wind_height: match wind_height {
            "TwentyFoot" => WindHeight::TwentyFoot,
            "TenMeter" => WindHeight::TenMetre,
            unknown => panic!("unsupported wind_height: {unknown}"),
        },
        waf_method: match waf_method {
            "CrownRatio" => WafMethod::CrownRatio,
            "UserInput" => WafMethod::UserInput,
            unknown => panic!("unsupported waf_method: {unknown}"),
        },
    }
}

fn ulp_distance(left: f64, right: f64) -> Option<u64> {
    if !left.is_finite() || !right.is_finite() {
        return None;
    }
    let ordered = |value: f64| {
        let bits = value.to_bits();
        if bits & (1 << 63) != 0 {
            !bits
        } else {
            bits | (1 << 63)
        }
    };
    Some(ordered(left).abs_diff(ordered(right)))
}

fn verify_manifest() {
    let manifest: Value =
        serde_json::from_str(include_str!("data/cpp_reference/surface/manifest.json")).unwrap();
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(
        manifest["upstream_sha"],
        "f3b2943e69abca942e795c5b1b45276f6338abb5"
    );
    let payloads: [(&str, &[u8]); 3] = [
        (
            "cases.jsonl",
            include_bytes!("data/cpp_reference/surface/cases.jsonl"),
        ),
        (
            "results.jsonl",
            include_bytes!("data/cpp_reference/surface/results.jsonl"),
        ),
        (
            "tolerance-policy.json",
            include_bytes!("data/cpp_reference/surface/tolerance-policy.json"),
        ),
    ];
    let expected_names: BTreeSet<_> = payloads.iter().map(|(name, _)| *name).collect();
    let actual_names: BTreeSet<_> = manifest["files"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(actual_names, expected_names);
    for (name, payload) in payloads {
        assert_eq!(
            manifest["files"][name].as_str().unwrap(),
            sha256(payload),
            "crate-local fixture hash mismatch: {name}"
        );
    }
}
