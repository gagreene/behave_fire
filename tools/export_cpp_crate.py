"""Copy reviewed numeric C++ pilot inputs into the standalone crate's tests.

This command never runs C++, calculates goldens, or copies upstream source or
notice text. Generation of canonical inputs is a separate explicit action.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN = "f3b2943e69abca942e795c5b1b45276f6338abb5"
INPUTS = {
    "cases.jsonl": "surface/cases.jsonl",
    "results.jsonl": "surface/results.jsonl",
    "tolerance-policy.json": "tolerance-policy.json",
}


def export(source, destination, check=False, dataset="surface"):
    """Copy or verify an explicit, byte-preserving fixture allowlist.

    :param source: Canonical tests/data/cpp_reference directory.
    :param destination: Crate-local dataset directory. Custom Surface requires the policy already exported
        in the sibling surface directory.
    :param check: Check existing copies without changing files when True.
    :param dataset: Surface, custom Surface, FireSize, Ignite, or Mortality observations.
    :return: Sorted payload and manifest names; raises for missing inputs or drift.
    """
    if dataset not in ("surface", "surface_custom", "fire_size", "ignite", "mortality"):
        raise ValueError("Unknown C++ dataset")
    paths = {name: relative.replace("surface/", dataset + "/") for name, relative in INPUTS.items()}
    payloads = {name: (source / relative).read_bytes() for name, relative in paths.items()}
    shared_policy = payloads.pop("tolerance-policy.json") if dataset == "surface_custom" else None
    if dataset in ("fire_size", "ignite", "mortality"):
        policy = json.loads(payloads["tolerance-policy.json"])[dataset]
        payloads["tolerance-policy.json"] = (json.dumps(policy, indent=2, sort_keys=True) + "\n").encode("utf-8")
    canonical = json.loads((source / dataset / "manifest.json").read_text(encoding="utf-8"))
    if canonical["upstream_sha"] != PIN:
        raise ValueError("canonical C++ reference pin changed")
    manifest = {
        "schema_version": 1,
        "upstream_sha": PIN,
        "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
    }
    if shared_policy is not None:
        shared = destination.parent / "surface" / "tolerance-policy.json"
        if not shared.is_file() or shared.read_bytes() != shared_policy:
            raise ValueError("crate-local C++ shared policy drift: surface/tolerance-policy.json")
        duplicate = destination / "tolerance-policy.json"
        if duplicate.exists() or duplicate.is_symlink():
            raise ValueError("crate-local C++ duplicate policy: surface_custom/tolerance-policy.json")
        manifest["shared_policy_sha256"] = hashlib.sha256(shared_policy).hexdigest()
    payloads["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if check:
        for name, payload in payloads.items():
            target = destination / name
            if not target.is_file() or target.read_bytes() != payload:
                raise ValueError(f"crate-local C++ fixture drift: {name}")
    else:
        destination.mkdir(parents=True, exist_ok=True)
        for name, payload in payloads.items():
            (destination / name).write_bytes(payload)
    return sorted(payloads)


def main():
    """Copy canonical pilot files or check the crate-local copies for drift.

    :return: None; writes only the allowlisted crate fixtures unless --check is set.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    for dataset in ("surface", "surface_custom", "fire_size", "ignite", "mortality"):
        names = export(
            ROOT / "tests/data/cpp_reference",
            ROOT / "rust/crates/behave-fire/tests/data/cpp_reference" / dataset,
            arguments.check, dataset,
        )
        print(dataset + ": " + ", ".join(names))


if __name__ == "__main__":
    main()
