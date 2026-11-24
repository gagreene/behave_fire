//! Standalone checks for the qualified Surface bounds and declared load differences.

use super::cpp_hash::sha256;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

pub const FIELDS: [&str; 5] = [
    "fireline_intensity_btu_ft_s",
    "flame_length_ft",
    "heat_per_unit_area_btu_ft2",
    "length_to_width_ratio",
    "spread_chains_per_hour",
];
pub const LOAD_MODELS: [i64; 2] = [152, 153];

fn source_files(
    directory: &Path,
    base: &Path,
    files: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    for entry in std::fs::read_dir(directory).map_err(|error| error.to_string())? {
        let path = entry.map_err(|error| error.to_string())?.path();
        if path.is_dir() {
            source_files(&path, base, files)?;
        } else if path.extension().and_then(|value| value.to_str()) == Some("rs") {
            let relative = path
                .strip_prefix(base)
                .map_err(|error| error.to_string())?
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(&path).map_err(|error| error.to_string())?;
            files.insert(
                format!("rust/crates/behave-fire/{relative}"),
                sha256(text.replace("\r\n", "\n").as_bytes()),
            );
        }
    }
    Ok(())
}

/// Apply the owner's unchanged numerical caps, preserving finite and zero rules.
pub fn compare(left: f64, right: f64, policy: &Value, field: &str) -> Result<(), String> {
    let cap = &policy["fields"][field];
    if cap["kind"] != "relative_absolute"
        || cap["rtol"].as_f64() != Some(1e-12)
        || cap["atol"].as_f64() != Some(1e-12)
        || !left.is_finite()
        || !right.is_finite()
    {
        return Err(format!("invalid Surface cap or nonfinite value: {field}"));
    }
    if left == 0. && right == 0. && left.is_sign_negative() != right.is_sign_negative() {
        return Err(format!("signed-zero mismatch: {field}"));
    }
    let allowed = 1e-12_f64.max(1e-12 * left.abs().max(right.abs()));
    if (left - right).abs() > allowed {
        return Err(format!("{field}: {left} versus {right}"));
    }
    Ok(())
}

/// Require complete calibration metadata bound to the shipped numeric datasets.
pub fn require_qualification(policy: &Value) -> Result<(), String> {
    if policy["approved"].as_bool() != Some(true)
        || policy["conditional_owner_approval"].as_bool() != Some(true)
    {
        return Err("Surface comparison policy is inactive".into());
    }
    let qualification = &policy["qualification"];
    if qualification["calibration_complete"].as_bool() != Some(true)
        || qualification["upstream_sha"] != "3ca33d5b919d4253b411bfb6464b4c872dda6fc9"
        || qualification["fields"] != policy["fields"]
    {
        return Err("missing Surface calibration qualification".into());
    }
    for field in FIELDS {
        compare(1., 1., policy, field)?;
    }
    let actual_models: Vec<_> = policy["documented_load_source_models"]
        .as_array()
        .ok_or("missing load-record disposition")?
        .iter()
        .map(|value| value.as_i64())
        .collect();
    if actual_models != LOAD_MODELS.into_iter().map(Some).collect::<Vec<_>>() {
        return Err("load-record disposition changed".into());
    }
    let expected: BTreeSet<_> = [
        "cpp-GNU",
        "cpp-MSVC",
        "ports-linux-x86_64",
        "ports-linux-aarch64",
        "ports-linux-musl-x86_64",
        "ports-macos-x86_64",
        "ports-macos-arm64",
        "ports-windows-x86_64",
        "ports-windows-arm64",
    ]
    .into_iter()
    .collect();
    let reports = qualification["reports"]
        .as_object()
        .ok_or("missing calibration reports")?;
    if reports.keys().map(String::as_str).collect::<BTreeSet<_>>() != expected {
        return Err("incomplete compiler/platform calibration".into());
    }
    for (key, report) in reports {
        let digest = report["sha256"].as_str().ok_or("missing report identity")?;
        if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err("invalid calibration report identity".into());
        }
        let summary = &report["summary"];
        if key.starts_with("cpp-") {
            if summary["cases"].as_u64() != Some(773)
                || summary["available"].as_u64() != Some(722)
                || summary["fields"].as_u64() != Some(3610)
                || summary["unavailable"].as_u64() != Some(51)
            {
                return Err("compiler coverage changed".into());
            }
        } else if summary["cases_per_backend"].as_u64() != Some(710)
            || summary["fields_per_backend"].as_u64() != Some(3550)
            || summary["paired_cases_per_backend"].as_u64() != Some(139)
        {
            return Err("native-platform coverage changed".into());
        }
    }
    let payloads: [(&str, &[u8]); 4] = [
        (
            "surface/cases.jsonl",
            include_bytes!("../data/cpp_reference/surface/cases.jsonl"),
        ),
        (
            "surface/results.jsonl",
            include_bytes!("../data/cpp_reference/surface/results.jsonl"),
        ),
        (
            "surface_custom/cases.jsonl",
            include_bytes!("../data/cpp_reference/surface_custom/cases.jsonl"),
        ),
        (
            "surface_custom/results.jsonl",
            include_bytes!("../data/cpp_reference/surface_custom/results.jsonl"),
        ),
    ];
    let artifacts = qualification["artifacts"]
        .as_object()
        .ok_or("missing qualified input identities")?;
    if artifacts.len() != 4 {
        return Err("qualified input inventory changed".into());
    }
    for (name, bytes) in payloads {
        let name = "tests/data/cpp_reference/".to_string() + name;
        if artifacts[&name].as_str() != Some(sha256(bytes).as_str()) {
            return Err("qualified numeric inputs changed".into());
        }
    }
    let base = Path::new(env!("CARGO_MANIFEST_DIR"));
    let sources = qualification["source_files"]
        .as_object()
        .ok_or("missing qualified core sources")?;
    let expected: BTreeMap<String, String> = sources
        .iter()
        .filter(|(name, _)| {
            name.starts_with("rust/crates/behave-fire/src/") && name.ends_with(".rs")
        })
        .map(|(name, digest)| {
            Ok((
                name.clone(),
                digest.as_str().ok_or("invalid source hash")?.to_string(),
            ))
        })
        .collect::<Result<_, String>>()?;
    let mut actual = BTreeMap::new();
    source_files(&base.join("src"), base, &mut actual)?;
    if actual.is_empty() || actual != expected {
        return Err("qualified core source inventory changed".into());
    }
    let original_manifest = base.join("Cargo.toml.orig");
    let manifest = if original_manifest.is_file() {
        original_manifest
    } else {
        base.join("Cargo.toml")
    };
    let text = std::fs::read_to_string(manifest).map_err(|error| error.to_string())?;
    if sources["rust/crates/behave-fire/Cargo.toml"].as_str()
        != Some(sha256(text.replace("\r\n", "\n").as_bytes()).as_str())
    {
        return Err("qualified core manifest changed".into());
    }
    Ok(())
}
