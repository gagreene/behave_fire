"""Verify scoped allocation evidence before attaching it to acceptance reports.

Both builds must use identical source trees. The only permitted build difference
is the private allocation-probe feature. Embedded evidence keeps reports portable
while retaining independently checkable canonical SHA-256 digests.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path


def _hash(value):
    """Hash canonical JSON evidence.

    :param value: JSON-compatible record.
    :return: SHA-256 digest of the canonical record.
    """
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _manifest(evidence):
    """Validate the relationship between timed and instrumented artifacts.

    :param evidence: Embedded manifest and raw records with digests.
    :return: Validated timed and instrumented build records.
    """
    manifest = evidence.get("manifest")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Missing allocation provenance manifest")
    if evidence.get("manifest_sha256") != _hash(manifest):
        raise ValueError("Allocation manifest digest mismatch")
    timed, instrumented = manifest.get("timed"), manifest.get("instrumented")
    for build in (timed, instrumented):
        if not isinstance(build, dict):
            raise ValueError("Missing build identity")
        for field in ("source_hash", "extension_hash"):
            if not _sha256(build.get(field)):
                raise ValueError(f"Invalid {field}")
        files = build.get("source_files")
        if not isinstance(files, dict) or not files or any(not _sha256(v) for v in files.values()):
            raise ValueError("Missing source inventory")
        if any(not isinstance(p, str) or p.startswith(("/", "\\")) or ":" in p
               or ".." in p.split("/") for p in files):
            raise ValueError("Source inventory paths must be repository relative")
        for field in ("compiler", "target", "profile"):
            if not isinstance(build.get(field), str) or not build[field]:
                raise ValueError(f"Missing {field}")
        if not isinstance(build.get("build_environment"), dict):
            raise ValueError("Missing build environment")
        features = build.get("features")
        if not isinstance(features, list) or any(not isinstance(f, str) for f in features) or len(set(features)) != len(features):
            raise ValueError("Invalid feature inventory")
    for field in ("source_hash", "source_files", "compiler", "target", "profile", "build_environment"):
        if timed[field] != instrumented[field]:
            raise ValueError(f"Timed/instrumented {field} differ")
    if timed["extension_hash"] == instrumented["extension_hash"]:
        raise ValueError("Instrumented and timed binaries must be distinct")
    if ("allocation-probe" in timed["features"]
            or set(instrumented["features"]) != set(timed["features"]) | {"allocation-probe"}):
        raise ValueError("Only allocation-probe may differ between build features")
    if manifest.get("source_differences") != [] or manifest.get("patch_sha256") != hashlib.sha256(b"").hexdigest():
        raise ValueError("Instrumentation must use identical sources and an empty patch")
    return timed, instrumented


def _scope(row, name):
    """Read a genuinely measured loop scope and reject partial coverage.

    :param row: One fresh instrumented process observation.
    :param name: Kernel or encoder scope identifier.
    :return: Validated allocation counters.
    """
    scope = row.get("scopes", {}).get(name)
    if not isinstance(scope, dict) or scope.get("definition") != name + "-loop-v1":
        raise ValueError(f"Unmeasured or incorrectly labeled {name} scope")
    for key in ("requests", "bytes", "calls"):
        if type(scope.get(key)) is not int or scope[key] < 0:
            raise ValueError(f"Invalid {name} {key}")
    if scope["calls"] != row["cells"]:
        raise ValueError(f"Incomplete {name} loop coverage")
    workers = scope.get("workers")
    if type(workers) is not int or not 1 <= workers <= row["workers"]:
        raise ValueError(f"Invalid actual {name} worker count")
    if name == "kernel" and row["cells"] == 10000 and row["workers"] == 4 and workers != 4:
        raise ValueError("Four-worker check did not exercise four kernel workers")
    return scope


def _sha256(value):
    """Recognize an exact lowercase SHA-256 digest.

    :param value: Candidate digest.
    :return: Whether the candidate is a hexadecimal SHA-256 digest.
    """
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def assemble(evidence, requirements):
    """Validate raw records and emit one certification row per required case.

    :param evidence: Canonically hashed manifest and raw scoped observations.
    :param requirements: Required datasets and operations.
    :return: Portable allocation rows; raises ValueError on incomplete evidence.
    """
    try:
        timed, instrumented = _manifest(evidence)
        raw = evidence.get("raw")
        if not isinstance(raw, list) or evidence.get("raw_sha256") != _hash(raw):
            raise ValueError("Allocation raw-evidence digest mismatch")
        expected_cases = set(itertools.product(requirements["datasets"], requirements["operations"]))
        expected_samples = set(itertools.product((1000, 10000), (1, 4), range(3)))
        groups = {case: {} for case in expected_cases}
        processes = set()
        for row in raw:
            case = row["dataset"], row["operation"]
            sample = row["cells"], row["workers"], row["repeat"]
            if any(type(row[key]) is not int for key in ("cells", "workers", "repeat")):
                raise ValueError("Invalid sample identity")
            if case not in groups or sample not in expected_samples or sample in groups[case]:
                raise ValueError("Unexpected or duplicate allocation observation")
            if row.get("manifest_sha256") != evidence["manifest_sha256"]:
                raise ValueError("Raw observation manifest mismatch")
            for key in ("source_hash", "extension_hash"):
                if row.get("instrumented_" + key) != instrumented[key]:
                    raise ValueError("Raw instrumented identity mismatch")
            process = row.get("process_id")
            if type(process) is not int or process <= 0 or process in processes:
                raise ValueError("Each observation requires a fresh process")
            processes.add(process)
            if row.get("passed") is not True:
                raise ValueError("Missing independent full-output validation")
            for name in ("kernel", "encoder"):
                scope = _scope(row, name)
                if scope["requests"] != 0 or scope["bytes"] != 0:
                    raise ValueError(f"Nonzero ordinary {name} loop allocation")
            outside = row.get("outside_scopes")
            if not isinstance(outside, dict) or any(type(outside.get(k)) is not int or outside[k] < 0
                                                    for k in ("requests", "bytes")):
                raise ValueError("Missing setup/output allocation counts")
            groups[case][sample] = row
        result = []
        for (dataset, operation), rows in sorted(groups.items()):
            if set(rows) != expected_samples:
                raise ValueError(f"{dataset}/{operation}: missing size/worker/repeat samples")
            slopes = {name: [(rows[(10000, workers, repeat)]["scopes"][name]["requests"]
                             - rows[(1000, workers, repeat)]["scopes"][name]["requests"]) / 9000
                            for workers in (1, 4) for repeat in range(3)] for name in ("kernel", "encoder")}
            result.append({"dataset": dataset, "operation": operation, "source_hash": timed["source_hash"],
                           "extension_hash": timed["extension_hash"],
                           "instrumented_source_hash": instrumented["source_hash"],
                           "instrumented_extension_hash": instrumented["extension_hash"],
                           "kernel_per_cell": max(slopes["kernel"]), "encoder_per_cell": max(slopes["encoder"]),
                           "slopes": slopes, "evidence": evidence})
        return {"allocations": result}
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError(f"Malformed allocation evidence: {error}") from error


def main():
    """Assemble a portable allocation report from a producer evidence file.

    :return: None; writes validated evidence, or exits nonzero without certification.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--requirements", type=Path, default=Path(__file__).with_name("performance_requirements.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = assemble(json.loads(args.evidence.read_text(encoding="utf-8")),
                          json.loads(args.requirements.read_text(encoding="utf-8")))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Allocation evidence rejected: {error}\n")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()