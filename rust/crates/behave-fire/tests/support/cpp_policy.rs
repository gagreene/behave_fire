//! Standalone checks for Surface bounds and full compiler/platform qualification.

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

/// Check a version against SemVer 2.0.0, matching the Python qualification writer.
fn valid_semver(version: &str) -> bool {
    fn number(part: &str) -> bool {
        !part.is_empty()
            && part.bytes().all(|b| b.is_ascii_digit())
            && (part == "0" || !part.starts_with('0'))
    }
    fn identifier(part: &str, prerelease: bool) -> bool {
        !part.is_empty()
            && part.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-')
            && (!prerelease || !part.bytes().all(|b| b.is_ascii_digit()) || number(part))
    }
    let (rest, build) = match version.split_once('+') {
        Some((rest, build)) => (rest, Some(build)),
        None => (version, None),
    };
    let (core, prerelease) = match rest.split_once('-') {
        Some((core, prerelease)) => (core, Some(prerelease)),
        None => (rest, None),
    };
    let parts: Vec<&str> = core.split('.').collect();
    let prerelease_ok = match prerelease {
        Some(text) => text.split('.').all(|part| identifier(part, true)),
        None => true,
    };
    let build_ok = match build {
        Some(text) => text.split('.').all(|part| identifier(part, false)),
        None => true,
    };
    parts.len() == 3 && parts.iter().all(|part| number(part)) && prerelease_ok && build_ok
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
    if policy.get("documented_load_source_models").is_some() {
        return Err("Surface qualification cannot exclude numbered models".into());
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
        } else if summary["cases_per_backend"].as_u64() != Some(722)
            || summary["fields_per_backend"].as_u64() != Some(3610)
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
    let neutral = version_neutral_core_manifest(&text.replace("\r\n", "\n"))?;
    if sources["rust/crates/behave-fire/Cargo.toml"].as_str()
        != Some(sha256(neutral.as_bytes()).as_str())
    {
        return Err("qualified core manifest changed".into());
    }
    Ok(())
}

/// Blank the core `[package]` version exactly as `tools/cpp_calibrate.py` does before hashing.
///
/// Every other byte is kept. A missing, duplicated, or invalid version is an error, so a
/// version-only release keeps the qualified digest while any other manifest edit changes it.
pub fn version_neutral_core_manifest(text: &str) -> Result<String, String> {
    let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
    let mut section = String::new();
    let mut found = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_end();
        let inner = trimmed
            .strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'));
        if let Some(name) = inner.filter(|name| !name.contains(['[', ']']) && !name.is_empty()) {
            section = name.trim().to_string();
        } else if section == "package" {
            if let Some(value) = line
                .strip_prefix("version = \"")
                .and_then(|rest| rest.strip_suffix('"'))
            {
                found.push((index, value.to_string()));
            }
        }
    }
    let [(index, value)] = found.as_slice() else {
        return Err("core manifest needs exactly one [package] version line".into());
    };
    if !valid_semver(value) {
        return Err(format!(
            "core manifest version {value:?} is not valid SemVer"
        ));
    }
    lines[*index] = "version = \"0.0.0-version-neutral\"".to_string();
    Ok(lines.join("\n"))
}
