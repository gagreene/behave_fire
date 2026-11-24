//! Exact standalone FireSize comparisons against pinned Behave C++ observations.

use behave_fire::fire_size::FireSize;
use behave_fire::units::{AreaUnit, LengthUnit, SpeedUnit, TimeUnit};
use serde_json::{json, Value};

#[path = "support/cpp_hash.rs"]
mod cpp_hash;
use cpp_hash::sha256;
// The shared qualification code also includes Surface-specific unit-record checks.
#[allow(dead_code)]
#[path = "support/cpp_policy.rs"]
mod cpp_policy;

const CASES: &str = include_str!("data/cpp_reference/fire_size/cases.jsonl");
const RESULTS: &str = include_str!("data/cpp_reference/fire_size/results.jsonl");
const MANIFEST: &str = include_str!("data/cpp_reference/fire_size/manifest.json");
const POLICY: &str = include_str!("data/cpp_reference/fire_size/tolerance-policy.json");
const SURFACE_POLICY: &str = include_str!("data/cpp_reference/surface/tolerance-policy.json");
const UPSTREAM_SHA: &str = "3ca33d5b919d4253b411bfb6464b4c872dda6fc9";
const CI_RUN_PREFIX: &str = "https://github.com/gagreene/behave_fire/actions/runs/";
const APPROVED_CASES: [(&str, &str, i32, i32, i32); 3] = [
    ("calm-crown", "Crown", 0, 4, 2),
    ("surface-5-mph", "Surface", 5, 10, 10),
    ("zero-elapsed", "Surface", 5, 10, 0),
];
const FIELDS: [&str; 3] = ["area_ft2", "length_to_width_ratio", "perimeter_ft"];
const QUALIFICATION_REQUIRED: &str = "FireSize comparison requires complete qualification";

struct Observation {
    case: &'static str,
    field: &'static str,
    cpp: f64,
    rust: f64,
}

/// Require approval and qualification before accepting the measured observations.
fn comparison_gate(policy: &Value, observations: &[Observation]) -> Result<(), String> {
    if policy["fields"]
        != json!({
            "area_ft2": {"kind": "exact_binary64"},
            "length_to_width_ratio": {"kind": "exact_binary64"},
            "perimeter_ft": {"kind": "exact_binary64"},
        })
    {
        return Err("FireSize exact comparison fields changed".into());
    }
    match policy["approved"].as_bool() {
        Some(true) if has_exact_keys(policy, &["approved", "fields", "qualification"]) => {}
        Some(false) if has_exact_keys(policy, &["approved", "fields"]) => {
            return Err(QUALIFICATION_REQUIRED.into());
        }
        _ => return Err("invalid FireSize policy keys".into()),
    }
    require_qualification(&policy["qualification"])?;
    require_source_qualification(policy)?;
    for observation in observations {
        let Observation {
            case,
            field,
            cpp,
            rust,
        } = observation;
        if !same_binary64(*cpp, *rust) {
            return Err(format!(
                "{case} {field}: C++ {cpp} ({:#018x}), Rust {rust} ({:#018x})",
                cpp.to_bits(),
                rust.to_bits()
            ));
        }
    }
    Ok(())
}

/// Require every declared JSON object key and reject extra keys.
fn has_exact_keys(value: &Value, expected: &[&str]) -> bool {
    value.as_object().is_some_and(|object| {
        object.len() == expected.len() && expected.iter().all(|key| object.contains_key(*key))
    })
}

/// Check the syntax of a lowercase hexadecimal artifact or commit identity.
fn is_hex_identity(value: Option<&Value>, length: usize) -> bool {
    value.and_then(Value::as_str).is_some_and(|value| {
        value.len() == length
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

/// Read JSONL in fixture order, rejecting blank or malformed rows.
fn parse_rows(input: &str) -> Vec<Value> {
    input
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// Check complete CI evidence and bind it to the packaged FireSize observations.
fn require_qualification(qualification: &Value) -> Result<(), String> {
    if !has_exact_keys(
        qualification,
        &[
            "artifacts",
            "ci_run_url",
            "comparisons",
            "maxima",
            "source_commit",
        ],
    ) {
        return Err("invalid FireSize qualification keys".into());
    }
    let run = qualification["ci_run_url"]
        .as_str()
        .and_then(|url| url.strip_prefix(CI_RUN_PREFIX));
    if !is_hex_identity(qualification.get("source_commit"), 40)
        || !run.is_some_and(|id| !id.is_empty() && id.bytes().all(|byte| byte.is_ascii_digit()))
        || qualification["comparisons"].as_u64() != Some(144)
    {
        return Err("missing complete FireSize CI qualification".into());
    }
    let artifacts = qualification["artifacts"]
        .as_object()
        .ok_or("missing qualified FireSize artifact identities")?;
    if artifacts.len() != 4
        || artifacts.get("cases.jsonl").and_then(Value::as_str)
            != Some(sha256(CASES.as_bytes()).as_str())
        || artifacts.get("results.jsonl").and_then(Value::as_str)
            != Some(sha256(RESULTS.as_bytes()).as_str())
        || !is_hex_identity(artifacts.get("manifest.json"), 64)
        || !is_hex_identity(artifacts.get("schema.json"), 64)
    {
        return Err("qualified FireSize artifact identities changed".into());
    }
    let maxima = qualification["maxima"]
        .as_object()
        .ok_or("missing FireSize maxima")?;
    let exact = json!({"absolute": 0.0, "relative": 0.0, "ulp": 0, "nonmatching": 0});
    if maxima.len() != FIELDS.len()
        || FIELDS
            .iter()
            .any(|field| maxima.get(*field) != Some(&exact))
    {
        return Err("FireSize qualification includes inconsistent maxima".into());
    }
    Ok(())
}

/// Bind the compact policy to the shared qualification and current Rust core sources.
fn require_source_qualification(policy: &Value) -> Result<(), String> {
    let surface: Value = serde_json::from_str(SURFACE_POLICY).map_err(|error| error.to_string())?;
    let qualification = &surface["qualification"];
    let proof = &policy["qualification"];
    if surface["fire_size"] != *policy
        || qualification["fire_size"] != *proof
        || qualification["source_commit"] != proof["source_commit"]
        || qualification["ci_run_url"] != proof["ci_run_url"]
    {
        return Err("FireSize policy differs from the shared qualification".into());
    }
    cpp_policy::require_qualification(&surface)
        .map_err(|error| format!("FireSize source qualification: {error}"))
}

/// Compare finite FireSize geometry values.
fn same_binary64(cpp: f64, rust: f64) -> bool {
    cpp.is_finite() && rust.is_finite() && cpp.to_bits() == rust.to_bits()
}

/// Verify the pin and all crate-local payload hashes.
fn verify_manifest() {
    let manifest: Value = serde_json::from_str(MANIFEST).unwrap();
    assert_eq!(manifest["schema_version"], 1);
    assert_eq!(manifest["upstream_sha"], UPSTREAM_SHA);
    let payloads = [
        ("cases.jsonl", CASES),
        ("results.jsonl", RESULTS),
        ("tolerance-policy.json", POLICY),
    ];
    let files = manifest["files"].as_object().unwrap();
    assert_eq!(files.len(), payloads.len());
    for (name, source) in payloads {
        assert_eq!(
            files[name].as_str().unwrap(),
            sha256(source.as_bytes()),
            "crate-local FireSize fixture hash mismatch: {name}"
        );
    }
}

// ---- comparison policy ----
#[test]
fn cpp_fire_size_observations_follow_the_qualified_policy() {
    verify_manifest();
    let cases = parse_rows(CASES);
    let results = parse_rows(RESULTS);
    assert_eq!(cases.len(), APPROVED_CASES.len());
    assert_eq!(results.len(), APPROVED_CASES.len());
    let mut observations = Vec::new();
    for (((id, mode, wind, spread, elapsed), case), result) in
        APPROVED_CASES.into_iter().zip(cases).zip(results)
    {
        assert_eq!(
            case,
            json!({
                "id": id, "operation": "fire_size", "mode": mode,
                "wind_mph": wind, "spread_ft_min": spread, "elapsed_min": elapsed,
            })
        );
        assert_eq!(result.as_object().unwrap().len(), 5, "case {id}");
        assert_eq!(result["id"], id);
        assert_eq!(result["status"], "ok", "case {id}");
        assert_eq!(result["upstream_entries"].as_u64(), Some(1), "case {id}");
        assert_eq!(result["geometry_entries"].as_u64(), Some(1), "case {id}");
        let output = result["output"].as_object().unwrap();
        assert_eq!(output.len(), FIELDS.len(), "case {id}");
        let mut size = FireSize::new();
        let crown = mode == "Crown";
        let elapsed = f64::from(elapsed);
        size.calculate_fire_basic_dimensions(
            crown,
            f64::from(wind),
            SpeedUnit::MilesPerHour,
            f64::from(spread),
            SpeedUnit::FeetPerMinute,
        );
        let actual = [
            size.get_fire_area(crown, AreaUnit::SquareFeet, elapsed, TimeUnit::Minutes),
            size.get_fire_length_to_width_ratio(),
            size.get_fire_perimeter(crown, LengthUnit::Feet, elapsed, TimeUnit::Minutes),
        ];
        for (field, rust) in FIELDS.into_iter().zip(actual) {
            let cpp = output[field].as_f64().unwrap();
            assert!(cpp.is_finite() && cpp >= 0.0, "case {id} {field}: {cpp}");
            assert!(rust.is_finite(), "case {id} {field}: {rust}");
            observations.push(Observation {
                case: id,
                field,
                cpp,
                rust,
            });
        }
    }
    assert_eq!(observations.len(), 9);
    let policy: Value = serde_json::from_str(POLICY).unwrap();
    if policy["approved"].as_bool().unwrap() {
        assert_eq!(comparison_gate(&policy, &observations), Ok(()));
    } else {
        assert_eq!(
            comparison_gate(&policy, &observations),
            Err(QUALIFICATION_REQUIRED.into())
        );
    }
}

#[test]
fn opposite_signed_zeros_fail_exact_comparison() {
    for literal in ["-0", "-0.0"] {
        let cpp: Value = serde_json::from_str(literal).unwrap();
        let cpp = cpp.as_f64().unwrap();
        assert!(same_binary64(cpp, -0.0));
        assert!(!same_binary64(cpp, 0.0));
    }
}

#[test]
fn policy_rejects_unknown_keys_and_missing_qualification() {
    let mut policy: Value = serde_json::from_str(POLICY).unwrap();
    policy.as_object_mut().unwrap().remove("qualification");
    let observations = [Observation {
        case: "calm-crown",
        field: "length_to_width_ratio",
        cpp: 1.0,
        rust: 1.0,
    }];
    policy["approved"] = json!(false);
    assert_eq!(
        comparison_gate(&policy, &observations),
        Err(QUALIFICATION_REQUIRED.into())
    );
    policy["unexpected"] = Value::Null;
    assert_eq!(
        comparison_gate(&policy, &observations),
        Err("invalid FireSize policy keys".into())
    );
    policy.as_object_mut().unwrap().remove("unexpected");
    policy["approved"] = json!(true);
    assert_eq!(
        comparison_gate(&policy, &observations),
        Err("invalid FireSize policy keys".into())
    );
    policy["qualification"] = json!({
        "artifacts": null,
        "ci_run_url": null,
        "comparisons": null,
        "maxima": null,
        "source_commit": null,
        "unexpected": null,
    });
    assert_eq!(
        comparison_gate(&policy, &observations),
        Err("invalid FireSize qualification keys".into())
    );
}
