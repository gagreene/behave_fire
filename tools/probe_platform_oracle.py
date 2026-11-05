"""Probe the original exporter inputs using explicit Python; optionally native.

Run this file from an existing checkout/CI environment. No fixtures are written,
and the default run never imports the native namespace. JSON retains exact
encoded inputs, outputs, warnings, cast fingerprints, and frozen diagnostics.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import warnings

import numpy as np


ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "pytest.ini").is_file())
sys.path.insert(0, str(ROOT / "src"))
CASES = (
    ("crown", "coerce_crown_fire_inputs", "nan"),
    ("crown", "coerce_crown_fire_inputs", "positive-infinity"),
    ("ignite", "calculate_lightning_ignition_probability", "positive-infinity"),
    ("fine_dead_fuel_moisture_tool", "calculate_fine_dead_fuel_moisture", "nan"),
)


def _exporter():
    """Load the checkout's original exporter.

    :return: Explicit Python fixture exporter module.
    """
    spec = importlib.util.spec_from_file_location("platform_probe_exporter", ROOT / "tools/export_rust_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    """Emit exact original exporter cases and host conversion fingerprints.

    :param argv: Optional command-line arguments.
    :return: Zero; observed mismatches remain diagnostic JSON fields.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true", help="Also execute native calls in an environment with an installed extension")
    options = parser.parse_args(argv)
    exporter = _exporter()
    report = {"host": {"machine": platform.machine(), "platform": platform.platform(),
                       "python": platform.python_version(), "numpy": np.__version__},
              "cast_fingerprints": [], "cases": []}
    for source_dtype in ("float32", "float64"):
        source = np.array([np.nan, np.inf, -np.inf, 1e30, -1e30], dtype=source_dtype)
        for target_dtype in ("int32", "int64"):
            with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
                warnings.simplefilter("always")
                output = np.asarray(source, dtype=target_dtype)
            report["cast_fingerprints"].append({"input": exporter.encode_value(source),
                "target_dtype": target_dtype, "output": exporter.encode_value(output),
                "warnings": sorted(set(type(item.message).__name__ for item in caught))})
    for domain, name, profile in CASES:
        module = getattr(exporter, domain)
        function = getattr(module, name)
        _, shape, dtype, replacement = next(item for item in exporter._profiles() if item[0] == profile)
        values = exporter._context(shape, dtype)
        arguments = exporter._arguments(function, values)
        targets = [key for key, value in arguments.items() if isinstance(value, np.ndarray) and key != "coeffs"]
        values[targets[0]] = replacement
        current = exporter._case(function, values, f"{name}/{profile}")
        frozen_document = json.loads((ROOT / "tests/data/rust_port" / f"{domain}.json").read_text(encoding="utf-8"))
        frozen = next(case for case in frozen_document["cases"] if case["id"] == current["id"])
        assert current["inputs"] == frozen["inputs"], (domain, current["id"], "original inputs drifted")
        record = {"domain": domain, "explicit_python": current, "frozen": frozen,
                  "python_matches_frozen": exporter.contracts_match(current.get("output"), frozen.get("output"), True)}
        source = exporter._arguments(function, values)[targets[0]]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            record["first_input_int32"] = exporter.encode_value(np.asarray(source, dtype=np.int32))
        if options.native:
            from behave_fire import rust
            native = exporter._case(getattr(getattr(rust, domain), name), values, current["id"])
            record["native"] = native
            record["native_matches_python"] = exporter.contracts_match(native.get("output"), current.get("output"), True)
            record["native_matches_frozen"] = exporter.contracts_match(native.get("output"), frozen.get("output"), True)
            record["native_exception_matches_python"] = native.get("exception") == current.get("exception")
            record["native_warnings_match_python"] = native["warnings"] == current["warnings"]
        report["cases"].append(record)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
