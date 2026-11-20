//! Exact standalone scorch-height comparisons against pinned Behave C++ observations.

use behave_fire::mortality::calculate_scorch_height;
use serde_json::{json, Value};

#[path = "support/cpp_hash.rs"]
mod cpp_hash;
use cpp_hash::sha256;
// The shared qualification code also includes Surface-specific unit-record checks.
#[allow(dead_code)]
#[path = "support/cpp_policy.rs"]
mod cpp_policy;

const CASES: &str = include_str!("data/cpp_reference/mortality/cases.jsonl");
const RESULTS: &str = include_str!("data/cpp_reference/mortality/results.jsonl");
const MANIFEST: &str = include_str!("data/cpp_reference/mortality/manifest.json");
const POLICY: &str = include_str!("data/cpp_reference/mortality/tolerance-policy.json");
const SURFACE_POLICY: &str = include_str!("data/cpp_reference/surface/tolerance-policy.json");
const UPSTREAM_SHA: &str = "f3b2943e69abca942e795c5b1b45276f6338abb5";
const CI_RUN_PREFIX: &str = "https://github.com/gagreene/behave_fire/actions/runs/";
const APPROVED_CASES: [(&str, i32, i32, i32); 6] = [
    ("scorch-zero-calm-77", 0, 0, 77),
    ("scorch-zero-wind-77", 0, 2, 77),
    ("scorch-zero-calm-14", 0, 0, 14),
    ("scorch-one-calm-77", 1, 0, 77),
    ("scorch-one-wind-77", 1, 2, 77),
    ("scorch-one-calm-14", 1, 0, 14),
];
const FIELD: &str = "scorch_height_ft";
const QUALIFICATION_REQUIRED: &str = "Mortality comparison requires complete qualification";

struct Observation {
    case: &'static str,
    cpp: f64,
    rust: f64,
}

/// Require qualification before accepting the measured observations.
fn comparison_gate(
    policy: &Value,
    surface: &Value,
    observations: &[Observation],
) -> Result<(), String> {
    if !has_exact_keys(policy, &["approved", "fields"]) {
        return Err("invalid Mortality policy keys".into());
    }
    if policy["fields"] != json!({FIELD: {"kind": "exact_binary64"}}) {
        return Err("Mortality exact comparison field changed".into());
    }
    match policy["approved"].as_bool() {
        Some(true) => {}
        Some(false) => return Err(QUALIFICATION_REQUIRED.into()),
        None => return Err("invalid Mortality approval flag".into()),
    }
    if surface["mortality"] != *policy {
        return Err("Mortality policy differs from the shared qualification".into());
    }
    let qualification = &surface["qualification"];
    let run = qualification["ci_run_url"]
        .as_str()
        .and_then(|url| url.strip_prefix(CI_RUN_PREFIX));
    if !is_lower_hex(&qualification["source_commit"], 40)
        || !run.is_some_and(|id| !id.is_empty() && id.bytes().all(|byte| byte.is_ascii_digit()))
    {
        return Err("invalid shared Mortality CI qualification identity".into());
    }
    require_mortality_proof(&qualification["mortality"])?;
    cpp_policy::require_qualification(surface)
        .map_err(|error| format!("Mortality source qualification: {error}"))?;
    for Observation { case, cpp, rust } in observations {
        if !same_binary64(*cpp, *rust) {
            return Err(format!(
                "{case} {FIELD}: C++ {cpp} ({:#018x}), Rust {rust} ({:#018x})",
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

/// Check the syntax of a lowercase hexadecimal commit or artifact digest.
fn is_lower_hex(value: &Value, length: usize) -> bool {
    value.as_str().is_some_and(|hex| {
        hex.len() == length
            && hex
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

/// Require complete exact comparisons bound to the packaged Mortality observations.
fn require_mortality_proof(proof: &Value) -> Result<(), String> {
    if !has_exact_keys(proof, &["artifacts", "comparisons", "nonmatching"])
        || proof["comparisons"].as_u64() != Some(96)
        || proof["nonmatching"].as_u64() != Some(0)
    {
        return Err("missing complete Mortality CI qualification".into());
    }
    let artifacts = &proof["artifacts"];
    if !has_exact_keys(
        artifacts,
        &[
            "cases.jsonl",
            "manifest.json",
            "results.jsonl",
            "schema.json",
        ],
    ) || artifacts["cases.jsonl"].as_str() != Some(sha256(CASES.as_bytes()).as_str())
        || artifacts["results.jsonl"].as_str() != Some(sha256(RESULTS.as_bytes()).as_str())
        || !is_lower_hex(&artifacts["manifest.json"], 64)
        || !is_lower_hex(&artifacts["schema.json"], 64)
    {
        return Err("qualified Mortality artifact identities changed".into());
    }
    Ok(())
}

/// Compare finite scorch heights.
fn same_binary64(cpp: f64, rust: f64) -> bool {
    cpp.is_finite() && rust.is_finite() && cpp.to_bits() == rust.to_bits()
}

/// Verify the pin and all crate-local payload hashes.
fn verify_manifest() {
    let manifest: Value = serde_json::from_str(MANIFEST).unwrap();
    assert!(has_exact_keys(
        &manifest,
        &["files", "schema_version", "upstream_sha"]
    ));
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
            "crate-local Mortality fixture hash mismatch: {name}"
        );
    }
}

// ---- comparison policy ----
#[test]
fn cpp_mortality_observations_follow_the_qualified_policy() {
    verify_manifest();
    let cases = parse_rows(CASES);
    let results = parse_rows(RESULTS);
    assert_eq!(cases.len(), APPROVED_CASES.len());
    assert_eq!(results.len(), APPROVED_CASES.len());
    let mut observations = Vec::new();
    for (((id, intensity, wind, air), case), result) in
        APPROVED_CASES.into_iter().zip(cases).zip(results)
    {
        assert_eq!(
            case,
            json!({
                "id": id, "operation": "mortality", "intensity_btu_ft_s": intensity,
                "wind_mph": wind, "air_temperature_f": air,
            }),
            "case {id}"
        );
        assert!(has_exact_keys(
            &result,
            &[
                "id",
                "output",
                "output_entries",
                "status",
                "upstream_entries"
            ],
        ));
        assert_eq!(result["id"], id);
        assert_eq!(result["status"], "ok", "case {id}");
        assert_eq!(result["upstream_entries"].as_u64(), Some(1), "case {id}");
        assert_eq!(result["output_entries"].as_u64(), Some(1), "case {id}");
        let output = &result["output"];
        assert!(has_exact_keys(output, &[FIELD]), "case {id}");
        let cpp = output[FIELD].as_f64().unwrap();
        let rust = calculate_scorch_height(f64::from(intensity), f64::from(wind), f64::from(air));
        assert!(cpp.is_finite() && cpp >= 0.0, "case {id} {FIELD}: {cpp}");
        assert!(rust.is_finite(), "case {id} {FIELD}: {rust}");
        observations.push(Observation {
            case: id,
            cpp,
            rust,
        });
    }
    assert_eq!(observations.len(), 6);
    let policy: Value = serde_json::from_str(POLICY).unwrap();
    let surface: Value = serde_json::from_str(SURFACE_POLICY).unwrap();
    let decision = comparison_gate(&policy, &surface, &observations);
    if policy["approved"].as_bool().unwrap() {
        assert_eq!(decision, Ok(()));
    } else {
        assert_eq!(decision, Err(QUALIFICATION_REQUIRED.into()));
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
fn policy_rejects_unknown_keys_and_malformed_proof() {
    let mut policy: Value = serde_json::from_str(POLICY).unwrap();
    let observations = [Observation {
        case: "scorch-one-calm-77",
        cpp: 1.0,
        rust: 1.0,
    }];
    policy["approved"] = json!(false);
    assert_eq!(
        comparison_gate(&policy, &Value::Null, &observations),
        Err(QUALIFICATION_REQUIRED.into())
    );
    policy["unexpected"] = Value::Null;
    assert_eq!(
        comparison_gate(&policy, &Value::Null, &observations),
        Err("invalid Mortality policy keys".into())
    );
    policy.as_object_mut().unwrap().remove("unexpected");
    policy["approved"] = json!(true);
    assert_eq!(
        comparison_gate(&policy, &Value::Null, &observations),
        Err("Mortality policy differs from the shared qualification".into())
    );
    let mut surface = json!({
        "mortality": policy,
        "qualification": {"mortality": {"artifacts": {}, "comparisons": 96, "nonmatching": 0, "unexpected": null}},
    });
    assert_eq!(
        comparison_gate(&policy, &surface, &observations),
        Err("invalid shared Mortality CI qualification identity".into())
    );
    surface["qualification"]["source_commit"] = json!("a".repeat(40));
    surface["qualification"]["ci_run_url"] =
        json!("https://github.com/gagreene/behave_fire/actions/runs/12?x=1");
    assert_eq!(
        comparison_gate(&policy, &surface, &observations),
        Err("invalid shared Mortality CI qualification identity".into())
    );
    surface["qualification"]["ci_run_url"] =
        json!("https://github.com/gagreene/behave_fire/actions/runs/12");
    assert_eq!(
        comparison_gate(&policy, &surface, &observations),
        Err("missing complete Mortality CI qualification".into())
    );
}
