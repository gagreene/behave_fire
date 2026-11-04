"""Generate or check embedded Rust tables against the explicit Python source.

This development tool is never invoked by Cargo. Regeneration is explicit; use
``--check`` in CI to compare all records, fields, and fixture cells without writes.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


sys.dont_write_bytecode = True
SOURCE_REVISION = "7933789f00470c6bbfbe942dc22bdd7f74e192f9"
FUEL_NUMBERS = (
    "fuel_bed_depth", "moisture_of_extinction_dead", "heat_of_combustion_dead",
    "heat_of_combustion_live", "dead_1h", "dead_10h", "dead_100h", "live_herb",
    "live_woody", "savr_1h", "savr_live_herb", "savr_live_woody",
)
FUEL_FLAGS = ("is_dynamic", "is_reserved", "is_defined")
REGIONS = (
    "alaska", "california", "eastern_area", "great_basin", "northern_rockies",
    "northwest", "rocky_mountain", "southern_area", "southwest",
)


def _literal(value):
    """Format one portable Rust literal with full floating precision.

    :param value: Finite source scalar, string, or boolean.
    :return: Rust literal text.
    """
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False) + ".into()"
    return repr(float(value))


def _load_component(root, name):
    """Load one explicit source file without importing the root facade.

    :param root: Repository root path.
    :param name: Component module basename.
    :return: Loaded Python component module.
    """
    path = root / "src" / "behave_fire" / "components" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _outputs(root):
    """Build deterministic Rust literals and exhaustive source fixtures.

    :param root: Repository root path.
    :return: Mapping of relative output paths to complete UTF-8 file contents.
    """
    _load_component(root, "behave_units")
    fuel_module = _load_component(root, "fuel_models")
    species_module = _load_component(root, "species_master_table")
    moisture = _load_component(root, "fine_dead_fuel_moisture_tool")
    header = f"// Generated from explicit Python reference {SOURCE_REVISION}.\n// Regenerate with tools/check_rust_tables.py; do not edit by hand.\n"
    fuels = fuel_module.FuelModels().fuel_models_
    species = species_module.SpeciesMasterTable().records
    result = {}
    fuel_lines = [header, "#[rustfmt::skip]", "fn standard_fuel_models() -> std::collections::BTreeMap<i32, FuelModel> {", "    ["]
    fuel_fixture = []
    for number, record in sorted(fuels.items()):
        fields = [f"{key}: {_literal(record[key])}" for key in ("code", "name", *FUEL_NUMBERS, *FUEL_FLAGS)]
        fuel_lines.append(f"        ({number}, FuelModel {{ " + ", ".join(fields) + " }),")
        fuel_fixture.append("\t".join([str(number), record["code"], record["name"], *[repr(float(record[key])) for key in FUEL_NUMBERS], *[str(record[key]).lower() for key in FUEL_FLAGS]]))
    fuel_lines.extend(["    ].into_iter().collect()", "}", ""])
    result["src/generated/fuel_models.rs"] = "\n".join(fuel_lines)
    species_lines = [header, "#[rustfmt::skip]", "fn standard_species() -> Vec<SpeciesRecord> {", "    vec!["]
    species_fixture = []
    for record in species:
        fields = [f"{key}: {_literal(getattr(record, key))}" for key in ("species_code", "scientific_name", "common_name")]
        fields.extend(f"{key}: {getattr(record, key)}" for key in ("mortality_equation_number", "bark_equation_number", "crown_coefficient_code"))
        fields.extend(f"{key}: {_literal(getattr(record, key))}" for key in REGIONS)
        fields.extend([f"equation_type: EquationType::{record.equation_type.name}", f"crown_damage_equation_code: CrownDamageEquationCode::{record.crown_damage_equation_code.name}"])
        species_lines.append("        SpeciesRecord { " + ", ".join(fields) + " },")
        species_fixture.append("\t".join([record.species_code, record.scientific_name, record.common_name, str(record.mortality_equation_number), str(record.bark_equation_number), str(record.crown_coefficient_code), *[str(getattr(record, key)).lower() for key in REGIONS], str(int(record.equation_type)), str(int(record.crown_damage_equation_code))]))
    species_lines.extend(["    ]", "}", ""])
    result["src/generated/species.rs"] = "\n".join(species_lines)
    moisture_lines = [header]
    moisture_fixture = []
    for name, table in [("REF", moisture._REF), ("CORR", moisture._CORR)]:
        moisture_lines.extend([f"/// Complete {name} moisture lookup table from the explicit Python reference.", "#[rustfmt::skip]", f"pub const {name}: [[f64; {table.shape[1]}]; {table.shape[0]}] = ["])
        for i, row in enumerate(table):
            moisture_lines.append("    [" + ", ".join(_literal(float(value)) for value in row) + "],")
            moisture_fixture.extend(f"{name}\t{i}\t{j}\t{float(value)!r}" for j, value in enumerate(row))
        moisture_lines.extend(["];", ""])
    result["src/generated/fine_dead_fuel_moisture.rs"] = "\n".join(moisture_lines)
    result["tests/data/fuels.tsv"] = "\n".join(fuel_fixture) + "\n"
    result["tests/data/species.tsv"] = "\n".join(species_fixture) + "\n"
    result["tests/data/moisture.tsv"] = "\n".join(moisture_fixture) + "\n"
    names = ["behave_units", "fuel_models", "species_master_table", "fine_dead_fuel_moisture_tool"]
    hashes = {name: hashlib.sha256((root / "src" / "behave_fire" / "components" / f"{name}.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in names}
    result["tests/data/tables_metadata.json"] = json.dumps({"reference_revision": SOURCE_REVISION, "source_sha256_lf": hashes, "fuel_records": len(fuels), "species_records": len(species), "moisture_cells": sum(table.size for table in (moisture._REF, moisture._CORR))}, indent=2, sort_keys=True) + "\n"
    return result


def main():
    """Check or regenerate all deterministic table files.

    :return: Exit status; generation writes embedded tables and fixtures only.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare without writing files.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, help="Override the core crate output path (development drafts).")
    args = parser.parse_args()
    output = args.output or args.root / "rust" / "crates" / "behave-fire"
    mismatches = []
    for relative, expected in _outputs(args.root).items():
        path = output / relative
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                mismatches.append(relative)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(expected, encoding="utf-8", newline="\n")
    if mismatches:
        print("Rust table drift: " + ", ".join(mismatches))
        return 1
    print("Rust table check passed." if args.check else "Rust tables regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
