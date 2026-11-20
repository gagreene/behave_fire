"""Validate the committed hybrid C++ coverage inventory without a C++ checkout."""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

CPP_SHA = "f3b2943e69abca942e795c5b1b45276f6338abb5"
PUBLIC_IDENTITY_SHA256 = "213febf45cf781f69925699f2017a357a7d35cbb826e64ca2aed62e71c879442"
EXPECTED_COUNTS = {
    "public_functions": 1394,
    "scientific_candidates": 266,
    "state_lifecycle_methods": 508,
    "no_counterpart": 486,
    "deferred": 134,
    "phase0_routes": 45,
}
FUNCTION_FIELDS = {
    "id", "file", "line", "owner", "name", "qualified_name", "signature",
    "lexically_qualified_signature", "overload_count", "kind", "access", "deleted", "pure_virtual",
    "oracle_access", "python_targets", "rust_targets", "mapping_status", "mapping_note",
    "review_category", "forwarded_cpp_ids", "definition_candidates",
    "literal_source_and_header_dependencies", "proposed_disposition", "route_ids", "function_status",
    "scientific_candidate", "dataset", "execution_evidence", "owner_approved_waiver", "approval",
}
STATUSES = {"pending_pilot", "not_in_pilot", "tested", "waived", "no_counterpart", "deferred", "unreachable_oracle"}


def _check_execution_evidence(evidence, dataset, root):
    """Require hashed case, result, and manifest files for an execution claim.

    :param evidence: Relative artifact paths and expected SHA-256 values.
    :param dataset: Dataset directory within the committed reference artifacts.
    :param root: Selected reference directory, never an external fallback.
    :return: None; raise ValueError when any execution artifact is absent or changed.
    """
    if not isinstance(dataset, str) or not re.fullmatch(r"[a-z0-9_]+", dataset):
        raise ValueError("Invalid execution evidence dataset")
    _check_metadata(evidence, "execution evidence")
    expected = {f"{dataset}/{name}" for name in ("cases.jsonl", "results.jsonl", "manifest.json")}
    if {row["file"] for row in evidence} != expected:
        raise ValueError("Execution evidence must include the dataset cases, results, and manifest")
    root = Path(root).resolve()
    for row in evidence:
        path = (root / row["file"]).resolve()
        if (not path.is_relative_to(root) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]):
            raise ValueError(f"Missing or changed execution evidence: {row['file']}")


def _check_metadata(rows, label):
    """Check recorded source hashes without opening upstream source files.

    :param rows: Committed path and SHA-256 records.
    :param label: Metadata group used in error messages.
    :return: None; raise ValueError for absent, duplicate, or invalid hash records.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Missing {label} source hash metadata")
    paths = set()
    for row in rows:
        _require_fields(row, {"file", "sha256"}, label)
        path = row["file"]
        if not isinstance(path, str) or not path or path in paths:
            raise ValueError(f"Duplicate or empty {label} source path")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError(f"Invalid {label} source hash: {path}")
        paths.add(path)


def _check_reverse_joins(inventory, functions):
    """Check exact forward and reverse C++ to Python and Rust candidate edges.

    :param inventory: Committed inventory mapping.
    :param functions: Public and nonpublic declarations keyed by C++ identity.
    :return: None; raise ValueError for missing, duplicate, or inconsistent edges.
    """
    for language in ("python", "rust"):
        reverse = inventory[f"{language}_reverse_join"]
        if not isinstance(reverse, list) or not reverse:
            raise ValueError(f"Missing {language} reverse join")
        forward_edges = set()
        for identity, function in functions.items():
            for target in function[f"{language}_targets"]:
                _require_fields(target, {"target", "file", "line", "signature", "relation", "evidence"}, language)
                forward_edges.add((identity, target["target"]))
        reverse_edges = set()
        targets = set()
        for row in reverse:
            _require_fields(row, {"target", "file", "line", "signature", "cpp_ids", "mapping_status"}, language)
            if row["target"] in targets or len(row["cpp_ids"]) != len(set(row["cpp_ids"])):
                raise ValueError(f"Duplicate {language} reverse join")
            targets.add(row["target"])
            for identity in row["cpp_ids"]:
                if identity not in functions:
                    raise ValueError(f"Unknown C++ identity in {language} reverse join: {identity}")
                reverse_edges.add((identity, row["target"]))
        if forward_edges != reverse_edges:
            raise ValueError(f"Inconsistent {language} forward/reverse edges")


def _require_fields(row, fields, label):
    """Reject missing fields before inspecting a schema record.

    :param row: Mapping expected to contain the fields.
    :param fields: Required field names.
    :param label: Record description used in errors.
    :return: None; raise ValueError when the record is incomplete.
    """
    if not isinstance(row, dict) or not fields <= row.keys():
        raise ValueError(f"Missing required {label} field")


def main():
    """Validate the committed snapshot from the command line.

    :return: None; print the structural summary or terminate on invalid metadata.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate without writing any file.")
    parser.add_argument("--require-ready", action="store_true", help="Require executable scientific coverage.")
    parser.add_argument("--inventory", type=Path, default=Path(__file__).resolve().parents[1]
                        / "tests/data/cpp_reference/inventory.json")
    arguments = parser.parse_args()
    inventory = json.loads(arguments.inventory.read_text(encoding="utf-8"))
    print(json.dumps(validate_inventory(inventory, require_ready=arguments.require_ready,
                                         reference_root=arguments.inventory.resolve().parent), sort_keys=True))


def validate_inventory(inventory, require_ready=False, reference_root=None):
    """Validate hybrid coverage and distinguish preparation from qualified science.

    :param inventory: Committed source-derived snapshot, read without a live checkout.
    :param require_ready: Require executable scientific coverage and active Surface and FireSize qualification.
    :param reference_root: Explicit artifact directory; defaults to this tool's checkout for direct callers.
    :return: Counts of structurally valid declarations, bundles, and tested functions.
    """
    reference_root = (Path(__file__).resolve().parents[1] / 'tests/data/cpp_reference'
                      if reference_root is None else Path(reference_root))
    _require_fields(inventory, {"schema_version", "cpp_sha", "port_sha", "phase", "coverage_unit",
                              "expected_counts", "snapshot_inputs", "source_inputs", "public_functions",
                              "nonpublic_functions", "routes", "state_unit_contracts", "guards", "enums",
                              "python_reverse_join", "rust_reverse_join", "public_identity_sha256"}, "inventory")
    if inventory["schema_version"] != 1 or inventory["cpp_sha"] != CPP_SHA:
        raise ValueError("Unexpected inventory schema or pinned C++ SHA")
    if inventory["expected_counts"] != EXPECTED_COUNTS:
        raise ValueError("Expected inventory counts changed without reviewed inventory update")
    _check_metadata(inventory["snapshot_inputs"], "snapshot")
    _check_metadata(inventory["source_inputs"], "upstream")
    rows = inventory["public_functions"]
    if not isinstance(rows, list) or len(rows) != EXPECTED_COUNTS["public_functions"]:
        raise ValueError("Dropped or added public function row")
    functions = {}
    statuses = Counter()
    mapping_counts = Counter()
    for row in rows:
        _require_fields(row, FUNCTION_FIELDS, "public function")
        if row["id"] in functions or row["access"] != "public":
            raise ValueError("Duplicate C++ identity or nonpublic declaration in public inventory")
        if not row["signature"] or not row["lexically_qualified_signature"] or not row["qualified_name"]:
            raise ValueError("Missing function signature")
        status = row["function_status"]
        if status not in STATUSES:
            raise ValueError(f"Unknown function status: {status}")
        if status == "tested" and (not row["dataset"] or not row["execution_evidence"]):
            raise ValueError("Tested function lacks dataset and execution evidence")
        if status == "tested":
            _check_execution_evidence(row["execution_evidence"], row["dataset"], reference_root)
            if not row["approval"]:
                raise ValueError("Tested function lacks owner-approved field policy and finding decision")
        if status == "waived" and (not row["owner_approved_waiver"] or not row["approval"]):
            raise ValueError("Waiver lacks explicit owner approval and reason")
        if status in {"no_counterpart", "deferred"} and not row["approval"]:
            raise ValueError("Exclusion or deferral lacks owner decision")
        if status == "unreachable_oracle" and not row["deleted"]:
            raise ValueError("Unreachable declaration lacks deleted-declaration source evidence")
        if row["scientific_candidate"] != (row["mapping_status"] == "candidate_mapping"):
            raise ValueError("Scientific candidate classification changed")
        if status in {"pending_pilot", "not_in_pilot"} and (row["dataset"] or row["execution_evidence"]):
            raise ValueError("Pending function must not claim numerical coverage evidence")
        if not isinstance(row["route_ids"], list) or not row["route_ids"]:
            raise ValueError("Public function lacks route bundle")
        if len(row["route_ids"]) != len(set(row["route_ids"])):
            raise ValueError("Duplicate route assignment")
        functions[row["id"]] = row
        statuses[status] += 1
        mapping_counts[row["mapping_status"]] += 1
    identities = sorted((row["id"], row["signature"], row["lexically_qualified_signature"]) for row in rows)
    digest = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    if digest != PUBLIC_IDENTITY_SHA256 or digest != inventory["public_identity_sha256"]:
        raise ValueError("Public identity/signature snapshot changed")
    if mapping_counts != {"candidate_mapping": 266, "pending_mapping": 508, "no_current_counterpart": 620}:
        raise ValueError("Hybrid scientific/state/no-counterpart counts changed")
    if statuses["no_counterpart"] != 486 or statuses["deferred"] != 134:
        raise ValueError("Approved scope exclusions or containment deferrals changed")
    if all(row["function_status"] == "waived" for row in rows):
        raise ValueError("Universally waived inventory is not coverage")
    route_ids = set()
    executable = set()
    for route in inventory["routes"]:
        _require_fields(route, {"route_id", "method_ids", "assigned_method_ids", "status", "executable_now",
                                "dataset", "execution_evidence", "contract", "guard_ids"}, "route")
        route_id = route["route_id"]
        if route_id in route_ids:
            raise ValueError("Duplicate route bundle")
        route_ids.add(route_id)
        assigned = sorted(row["id"] for row in rows if route_id in row["route_ids"])
        if assigned != route["assigned_method_ids"]:
            raise ValueError(f"Missing or changed route membership: {route_id}")
        if not route["contract"] or not route["status"]:
            raise ValueError(f"Missing route state/lifecycle contract: {route_id}")
        if route["executable_now"]:
            if not route["dataset"] or not route["execution_evidence"]:
                raise ValueError("Executable route lacks dataset and execution evidence")
            _check_execution_evidence(route["execution_evidence"], route["dataset"], reference_root)
            if any(functions[identity]["scientific_candidate"] for identity in assigned):
                if not any(functions[identity]['scientific_candidate']
                           and functions[identity]['function_status'] == 'tested' for identity in assigned):
                    raise ValueError('Executable scientific route lacks per-function witnesses')
                executable.add(route_id)
    if any(not set(row["route_ids"]) <= route_ids for row in rows):
        raise ValueError("Missing route bundle for public function")
    if require_ready and not executable:
        raise ValueError("No executable scientific routes; pilot findings and owner approval remain pending")
    if require_ready:
        reference_root = Path(reference_root).resolve()
        root = reference_root.parent.parent.parent
        if root / 'tests/data/cpp_reference' != reference_root:
            raise ValueError('Complete qualification requires a canonical reference directory')
        policy_path = reference_root / 'tolerance-policy.json'
        policy = json.loads(policy_path.read_text(encoding='utf-8')) if policy_path.is_file() else {}
        _require_fields(policy, {'approved', 'fire_size'}, 'scientific qualification policy')
        fire_size = policy['fire_size']
        if policy['approved'] is not True or not isinstance(fire_size, dict) or fire_size.get('approved') is not True:
            raise ValueError('Scientific qualification policy is inactive')
        # Use the explicit sibling script, independent of caller imports and module caches.
        # Its normal structural inventory call does not request readiness, so this cannot recurse.
        result = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('cpp_reference.py')),
                                 '--root', str(root), '--check'], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise ValueError('Scientific qualification failed: ' + result.stderr[-4000:])
    if len(inventory["nonpublic_functions"]) != 194:
        raise ValueError("Nonpublic join scope changed")
    for row in inventory["nonpublic_functions"]:
        _require_fields(row, {"id", "qualified_name", "signature", "lexically_qualified_signature",
                              "file", "line", "python_targets", "rust_targets", "access"}, "nonpublic function")
        if row["id"] in functions:
            raise ValueError("Duplicate C++ declaration identity")
        functions[row["id"]] = row
    _check_reverse_joins(inventory, functions)
    return {"public_functions": len(rows), "scientific_candidates": mapping_counts["candidate_mapping"],
            "state_lifecycle_methods": mapping_counts["pending_mapping"], "bundles": len(route_ids),
            "tested_functions": statuses["tested"], "executable_scientific_bundles": len(executable),
            "no_counterpart": statuses["no_counterpart"], "deferred": statuses["deferred"]}


if __name__ == "__main__":
    main()
