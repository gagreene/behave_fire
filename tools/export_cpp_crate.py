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


def export(source, destination, check=False):
    """Copy or verify an explicit, byte-preserving fixture allowlist.

    :param source: Canonical tests/data/cpp_reference directory.
    :param destination: Crate-local tests/data/cpp_reference/surface directory.
    :param check: Check existing copies without changing files when True.
    :return: Sorted payload and manifest names; raises for missing inputs or drift.
    """
    payloads = {name: (source / relative).read_bytes() for name, relative in INPUTS.items()}
    canonical = json.loads((source / "surface/manifest.json").read_text(encoding="utf-8"))
    if canonical["upstream_sha"] != PIN:
        raise ValueError("canonical C++ reference pin changed")
    manifest = {
        "schema_version": 1,
        "upstream_sha": PIN,
        "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
    }
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
    names = export(
        ROOT / "tests/data/cpp_reference",
        ROOT / "rust/crates/behave-fire/tests/data/cpp_reference/surface",
        arguments.check,
    )
    print("\n".join(names))


if __name__ == "__main__":
    main()
