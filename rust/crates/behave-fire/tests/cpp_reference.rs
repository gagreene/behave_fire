//! Independent C++ Surface observations, pending owner-approved numeric policy.
//!
//! Passing these tests validates fixture structure, real core evaluation, and a
//! fail-closed review gate. The gate does not certify scientific parity. Run with
//! `--test cpp_reference -- --nocapture` to retain every raw numeric difference.

use behave_fire::engine::{
    PreparedFuel, SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceTweaks, WafMethod, WindHeight,
    WindOrientation,
};
use behave_fire::fuel_models::FuelModels;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

const FIELDS: [&str; 5] = [
    "fireline_intensity_btu_ft_s",
    "flame_length_ft",
    "heat_per_unit_area_btu_ft2",
    "length_to_width_ratio",
    "spread_chains_per_hour",
];

#[derive(Debug, Eq, PartialEq)]
enum ReviewRequired {
    ApprovedPolicyNotImplemented,
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
                "comparison_status": "requires_owner_review",
            })
        })
        .collect()
}

fn comparison_gate(policy: &Value, observations: &[Value]) -> Result<(), ReviewRequired> {
    assert!(!observations.is_empty(), "no independent observations");
    if policy["approved"].as_bool() != Some(true) {
        return Err(ReviewRequired::RequiresOwnerReview);
    }
    // An approval bit alone cannot implement or authorize scientific thresholds.
    Err(ReviewRequired::ApprovedPolicyNotImplemented)
}

#[test]
fn cpp_surface_raw_differences_require_owner_review() {
    verify_manifest();
    let cases = parse_rows(include_str!("data/cpp_reference/surface/cases.jsonl"));
    let results = parse_rows(include_str!("data/cpp_reference/surface/results.jsonl"));
    let policy: Value = serde_json::from_str(include_str!(
        "data/cpp_reference/surface/tolerance-policy.json"
    ))
    .unwrap();
    assert_eq!(policy["approved"], false);
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
        assert!(policy["fields"][field]["proposal"].is_null());
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
    assert_eq!(
        comparison_gate(&policy, &observations),
        Err(ReviewRequired::RequiresOwnerReview),
    );
    println!(
        "CPP_SURFACE_REVIEW_REQUIRED cases={} observations={} unavailable={} parity_certified=false",
        cases.len(), observations.len(), unavailable,
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
        Err(ReviewRequired::ApprovedPolicyNotImplemented),
    );
}

fn sha256(source: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut padded = source.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&(source.len() as u64).wrapping_mul(8).to_be_bytes());
    for block in padded.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (index, word) in block.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for index in 16..64 {
            let left = words[index - 15];
            let right = words[index - 2];
            let s0 = left.rotate_right(7) ^ left.rotate_right(18) ^ (left >> 3);
            let s1 = right.rotate_right(17) ^ right.rotate_right(19) ^ (right >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ (!e & g);
            let first = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let second = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(first);
            d = c;
            c = b;
            b = a;
            a = first.wrapping_add(second);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    state.iter().map(|word| format!("{word:08x}")).collect()
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
