"""Export reviewed, portable contracts from the explicit Python implementation.

Run from any directory with a source checkout. Generation is an explicit review
action; ``--check`` only reads fixtures and reports drift. The exporter imports
``behave_fire.behave`` and component definitions, never an automatic facade.
"""

import argparse
from enum import Enum
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "behave_fire"
REFERENCE_SHA = "7933789f00470c6bbfbe942dc22bdd7f74e192f9"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from behave_fire.behave import BehaveRun
from behave_fire.components import behave_units, crown, surface
from behave_fire.components import contain, fine_dead_fuel_moisture_tool, ignite, mortality
from behave_fire.components.fuel_models import FuelModels
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays
from behave_fire.components.mortality import build_mortality_lookup
from behave_fire.components.fire_size import FireSize
from behave_fire.components.slope_tool import SlopeTool
from behave_fire.components.species_master_table import SpeciesMasterTable
from behave_fire.components.species_master_table import SpeciesMasterTableRecord


def _arguments(function, values):
    """Select explicit arguments without replacing defaults.

    :param function: Callable being characterized.
    :param values: Named reference input values.
    :return: Keyword arguments for the callable; missing required names raise.
    """
    result = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if name in values:
            result[name] = values[name]
        elif parameter.default is inspect.Parameter.empty:
            raise KeyError(f"No reference input for {function.__qualname__}.{name}")
    return result


def _boundary_cases():
    """Record heterogeneous fuel, equation, lightning, and terminal-state inputs.

    :return: Domain-grouped portable scientific boundary cases.
    """
    cases = []
    for charge in range(3):
        values = _context((8,))
        values.update({"fuel_bed_type_grid": np.arange(8), "charge_type": charge})
        cases.append(_case(ignite.calculate_lightning_ignition_probability, values, f"lightning/charge-{charge}"))
    values = _context((33,))
    values["equation_number_grid"] = np.array([-1, 0, *range(1, 21), *range(100, 110), 999])
    cases.append(_case(mortality.calculate_crown_scorch_mortality, values, "mortality/all-equations"))
    values = _context((261,))
    values["fuel_model_grid"] = np.arange(261)
    runner = BehaveRun(FuelModels())
    cases.append(_case(runner.do_surface_run, values, "surface/all-fuel-slots"))
    for orientation in ("RelativeToUpslope", "RelativeToNorth", "unsupported"):
        values = _context((4,))
        values.update({"wind_direction": np.array([0., 90., 180., 270.]),
                       "aspect": np.array([0., 90., 180., 270.]), "wind_orientation_mode": orientation})
        cases.append(_case(runner.do_surface_run, values, f"surface/wind-{orientation}"))
    values = _context((3,))
    values.update({"report_spread_rate": np.ones(3), "lw_ratio": 1.,
                   "report_size": np.array([1., 1e-8, 1.]), "production_rate": np.array([1000., 0., 0.]),
                   "attack_time": np.array([0., 1000., 1000.]), "max_steps": 10})
    cases.append(_case(contain.run_contain_sim_array, values, "containment/mixed-terminal"))
    values["max_steps"] = 0
    cases.append(_case(contain.run_contain_sim_array, values, "containment/no-steps"))
    return cases


def _case(function, values, label):
    """Capture a real calculation, exception, and emitted warnings.

    :param function: Explicit Python callable.
    :param values: Input value dictionary.
    :param label: Deterministic scenario name.
    :return: Portable case with inputs, outputs, and warning categories.
    """
    arguments = _arguments(function, values)
    inputs = {key: {"reference": "fuels:standard_lookup"} if key == "lut"
              else {"reference": "species:mortality_coefficients"} if key == "coeffs"
              else encode_value(value) for key, value in arguments.items()}
    record = {"id": label, "entry_point": f"{function.__module__}.{function.__qualname__}",
              "inputs": inputs}
    with warnings.catch_warnings(record=True) as caught, np.errstate(all="ignore"):
        warnings.simplefilter("always")
        try:
            record["output"] = encode_value(function(**arguments))
        except Exception as error:
            record["exception"] = type(error).__name__
        record["warnings"] = sorted(set(type(item.message).__name__ for item in caught))
    return record


def _context(shape=(1,), dtype="float64"):
    """Build a complete, physically representative pipeline input context.

    :param shape: Spatial shape; an empty tuple requests scalar arrays.
    :param dtype: Floating input dtype.
    :return: Inputs and independently computed intermediate stage dictionaries.
    """
    def array(value):
        """Create one spatial input.

        :param value: Constant cell value.
        :return: Array in the requested shape and dtype.
        """
        return np.full(shape, value, dtype=dtype)

    fuels = FuelModels()
    values = {"fuel_models": fuels, "lut": build_fuel_lookup_arrays(fuels),
              "fuel_model_grid": np.full(shape, 124, dtype=np.int64),
              "S": shape or (1,), "m1h": array(.06), "m10h": array(.07),
              "m100h": array(.08), "mlh": array(.6), "mlw": array(.9),
              "wind_speed": array(5), "wind_speed_units": 5,
              "wind_direction": array(0), "wind_orientation_mode": "RelativeToUpslope",
              "slope": array(30), "slope_deg": array(30), "slope_units": 0,
              "aspect": array(0), "canopy_cover": array(.5), "canopy_height": array(30),
              "canopy_base_height": array(6), "crown_ratio": array(.5),
              "canopy_bulk_density": array(.02), "moisture_foliar": array(100),
              "coeffs": build_mortality_lookup(), "species_master_table": SpeciesMasterTable()}
    constants = {
        "air_temp_f": 80, "air_temperature": 80, "air_temp": 80,
        "sun_shade_fraction": .5, "moisture_1h_fraction": .06,
        "moisture_100h_fraction": .08, "duff_depth_inches": 2,
        "fireline_intensity_btu_ft_s": 100, "midflame_wind_mph": 5,
        "scorch_height_ft": 20, "scorch_height": 20, "tree_height_ft": 60,
        "tree_height": 60, "height": 60, "dbh_inches": 12, "dbh": 12,
        "flame_height_ft": 5, "flame_length_ft": 5, "flame_length": 5,
        "flame_height": 5, "wind_mph": 5, "cover_height_ft": 10,
        "cover_height": 10, "dbh_in": 12, "height_ft": 60, "count": 3,
        "number_of_personnel": 4, "area_per_person_sqft": 50,
        "number_of_equipment": 1, "area_per_equipment_sqft": 300,
        "temperature": 80, "relative_humidity": .3, "value": 1,
        "forward_ros": 10, "backing_ros": 1, "lwr": 2,
        "spread_rate": 10, "backing_spread_rate": 1, "fireline_intensity": 100,
        "no_wind_no_slope_spread_rate": 1, "eccentricity": .5,
        "fire_length_to_width_ratio": 2, "effective_wind_speed": 5,
        "wind_b": 1, "wind_c": .1, "wind_e": .5, "relative_packing_ratio": 1,
        "rate_vector": 2, "x_component": 1, "y_component": 1,
        "sigma": 1500, "heat_per_unit_area": 1000, "reaction_intensity": 1000,
        "residence_time": .2, "propagating_flux": .03, "heat_sink": 100,
        "waf": .3, "twenty_foot_wind_speed": 440, "midflame_wind_speed": 100, "depth": 1,
        "packing_ratio": .01, "wind_phi": 1, "slope_phi": 1,
        "surface_ros": 10, "surface_hpua": 1000, "surface_fli": 100,
        "surface_critical_ros": 5, "crowning_surface_ros": 20,
        "crown_ros": 40, "crown_hpua": 2000, "crown_fli": 500,
        "crown_fl": 10, "critical_surface_fli": 150, "critical_crown_ros": 30,
        "passive_ros": 15, "passive_hpua": 1500, "passive_fli": 200,
        "passive_fl": 7, "canopy_hpua": 1000, "cfb": .4,
        "transition_ratio": 1, "active_ratio": 1, "report_spread_rate": 1,
        "lw_ratio": 2, "report_size": 1, "production_rate": 1000, "attack_time": 0,
    }
    values.update({name: array(value) for name, value in constants.items()})
    values.update({"units": 0, "temp_units": 0, "rh_units": 0, "elapsed_min": 10.,
                   "elapsed_time": 10., "charge_type": 0, "max_steps": 10,
                   "fireline_intensity_units": 0, "temperature_units": 0,
                   "flame_length_units": 0, "flame_height_units": 0,
                   "cover_height_units": 0, "dbh_units": 1, "height_units": 0,
                   "ros_units": 0, "elapsed_time_units": 0,
                   "is_surface_fire": np.full(shape, True),
                   "is_passive_crown": np.full(shape, False),
                   "equation_number_grid": np.full(shape, 1, dtype=np.int64),
                   "fuel_bed_type_grid": np.full(shape, 0, dtype=np.int64)})
    values.update({name: np.zeros(shape, dtype=np.int64) for name in
                   ("dry_bulb_i", "rh_i", "slope_i", "aspect_i", "shading_i",
                    "month_i", "elev_i", "time_i")})
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        values["p"] = surface.build_particle_arrays(**_arguments(surface.build_particle_arrays, values))
        values["ib"] = surface.calculate_fuelbed_intermediates(values["p"])
        values["ri"] = surface.calculate_reaction_intensity(values["ib"])
        values["surface_results"] = surface.run_surface_fire(**_arguments(surface.run_surface_fire, values))
        values["ib_crown"] = values["ib"]
        values["ri_crown"] = values["ri"]
    return values


def _domain(module):
    """Map Python module names onto the planned standalone Rust modules.

    :param module: Canonical Python module name.
    :return: Rust domain name.
    """
    return {"behave": "model", "behave_units": "units", "fuel_models_array": "fuel_models",
            "species_master_table": "species", "fine_dead_fuel_moisture_tool": "fine_dead_fuel_moisture",
            "vapor_pressure_deficit_calculator": "vapor_pressure_deficit",
            "slope_tool": "slope"}.get(module, module)


def _envelope(payload):
    """Add deterministic provenance to one fixture document.

    :param payload: Document fields.
    :return: Document with versioned provenance.
    """
    hashes = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
              for path in sorted(SOURCE.rglob("*.py"))}
    return {"schema_version": 1, "provenance": {"python_reference_sha": REFERENCE_SHA,
            "source_sha256": hashes, "oracle": "explicit Python"}, **payload}


def _modules():
    """Load source modules without selecting an automatic backend.

    :return: Explicit Python implementation modules.
    """
    paths = [SOURCE / "behave.py", *sorted((SOURCE / "components").glob("*.py"))]
    return [importlib.import_module(".".join(path.relative_to(ROOT / "src").with_suffix("").parts))
            for path in paths if path.name != "__init__.py"]


def _profiles():
    """Define compact dtype, layout, and scientific-boundary probes.

    :return: Tuples of scenario name, shape, dtype, and replacement array.
    """
    profiles = [("baseline", (1,), "float64", None)]
    for dtype in ("float16", "float32", "float64", "int64", "uint32", "bool", "object"):
        profiles.append((f"dtype-{dtype}", (1,), dtype, None))
    for name, shape in (("scalar", ()), ("2d", (2, 2)), ("3d", (1, 2, 2)), ("empty", (0,))):
        profiles.append((name, shape, "float64", None))
    for name, value in (("zero", 0.), ("negative", -1.), ("nan", np.nan),
                        ("positive-infinity", np.inf), ("negative-infinity", -np.inf),
                        ("below-threshold", np.nextafter(1e-7, 0)),
                        ("at-threshold", 1e-7), ("above-threshold", np.nextafter(1e-7, 1)),
                        ("float16-overflow", 300.), ("float16-underflow", 1e-8)):
        profiles.append((name, (1,), "float16" if "float16" in name else "float64", np.array([value])))
    profiles.extend([("masked", (2,), "float64", np.ma.array([1., 2.], mask=[False, True])),
                     ("strided", (2,), "float64", np.arange(4.)[::2]),
                     ("mixed-dtype", (1,), "float64", np.array([1.], dtype=np.float32)),
                     ("unsupported-object", (1,), "object", np.array(["invalid"], dtype=object)),
                     ("incompatible-shape", (2,), "float64", np.ones(3))])
    readonly = np.ones(2)
    readonly.flags.writeable = False
    profiles.append(("read-only", (2,), "float64", readonly))
    return profiles


def _registry_cases():
    """Characterize public registry methods with valid and absent keys.

    :return: Cases recording return values, errors, and mutation summaries.
    """
    cases = []
    values = {"fuel_model_number": 250, "code": "CUSTOM", "name": "Contract custom",
              "fuel_bed_depth": 2., "length_units": 0, "moisture_of_extinction_dead": .24,
              "moisture_units": 0, "heat_of_combustion_dead": 8000., "heat_of_combustion_live": 8000.,
              "heat_of_combustion_units": 0, "fuel_load_one_hour": .01, "fuel_load_ten_hour": .02,
              "fuel_load_hundred_hour": .03, "fuel_load_live_herbaceous": .04,
              "fuel_load_live_woody": .05, "loading_units": 0, "savr_one_hour": 1800.,
              "savr_live_herbaceous": 1500., "savr_live_woody": 1200., "savr_units": 0,
              "is_dynamic": True, "species_code": "pipo", "scientific_name": "Contract species",
              "common_name": "Contract species", "mortality_equation": 2, "bark_equation": 31,
              "crown_coefficient_code": 12, "equation_type": 0, "crown_damage_equation_code": -1,
              "alaska": False, "california": True, "eastern_area": False, "great_basin": True,
              "northern_rockies": True, "northwest": True, "rocky_mountain": False,
              "southern_area": False, "southwest": True, "gacc": 2, "index": 0,
              "record": SpeciesMasterTableRecord("CUSTOM", "Contract species", "Contract species")}
    for cls in (FuelModels, SpeciesMasterTable, SpeciesMasterTableRecord):
        for name, member in sorted(vars(cls).items()):
            if not inspect.isfunction(member) or name.startswith("_"):
                continue
            for profile in ("baseline", "absent"):
                instance = cls()
                if isinstance(instance, FuelModels):
                    instance.set_custom_fuel_model(**_arguments(instance.set_custom_fuel_model, values))
                inputs = dict(values)
                if profile == "absent":
                    inputs.update({"fuel_model_number": 261, "species_code": "MISSING", "index": -1})
                case = _case(getattr(instance, name), inputs, f"{cls.__name__}.{name}/{profile}")
                if isinstance(instance, FuelModels):
                    case["selected_record_after"] = encode_value(instance.fuel_models_.get(inputs["fuel_model_number"]))
                elif isinstance(instance, SpeciesMasterTable):
                    case["record_count_after"] = instance.get_number_of_records()
                cases.append(case)
    return cases


def _stateful_cases():
    """Exercise public stateful geometry helpers and every unit-class method.

    :return: Domain names mapped to portable calculation cases.
    """
    cases = {"fire_size": [], "slope_tool": [], "unit_classes": []}
    values = _context(())
    values.update({"is_crown": False, "effective_wind_speed": 5., "forward_spread_rate": 10.,
                   "wind_speed_units": 5, "spread_rate_units": 0, "length_units": 0,
                   "time_units": 0, "area_units": 0, "distance_units": 0,
                   "calculated_map_distance": 10., "max_slope_steepness": 30.,
                   "slope_units": 0, "map_representative_fraction": 24000,
                   "map_distance": 1., "contour_interval": 20., "number_of_contours": 3,
                   "contour_units": 0, "map_distance_units": 0, "elevation_units": 0})
    for domain, cls in (("fire_size", FireSize), ("slope_tool", SlopeTool)):
        for method in sorted(name for name in vars(cls) if not name.startswith("_")):
            for profile, index in (("baseline", 0), ("invalid-index", -1)):
                helper = cls()
                if domain == "fire_size":
                    helper.calculate_fire_basic_dimensions(False, 5., 5, 10., 0)
                else:
                    helper.calculate_horizontal_distance(10., 0, 30., 0)
                    helper.calculate_slope_from_map_measurements(24000, 1., 0, 20., 3, 0)
                inputs = {**values, "index": index}
                before = encode_value(vars(helper))
                record = _case(getattr(helper, method), inputs, f"{cls.__name__}.{method}/{profile}")
                record["state_before"] = before
                record["state_after"] = encode_value(vars(helper))
                cases[domain].append(record)
    for name, cls in sorted(vars(behave_units).items()):
        if not inspect.isclass(cls) or cls.__module__ != behave_units.__name__:
            continue
        enum = next(value for key, value in vars(cls).items() if key.endswith("Enum"))
        for method in ("toBaseUnits", "fromBaseUnits"):
            function = getattr(cls, method)
            for unit_name, unit in sorted(vars(enum).items()):
                if unit_name.startswith("_"):
                    continue
                for dtype in ("float16", "float32", "float64", "int64", "uint32", "bool", "object"):
                    inputs = {"value": np.array([0., 1., 60.], dtype=dtype), "units": int(unit)}
                    label = f"{name}.{method}/{unit_name}/{dtype}"
                    record = _case(function, inputs, label)
                    record["policy"] = "diagnostic" if dtype == "float16" else "acceptance"
                    cases["unit_classes"].append(record)
                    if dtype == "float16":
                        record = _case(function, normalize_float16(inputs), label + "/normalized")
                        record["policy"] = "acceptance"
                        cases["unit_classes"].append(record)
    return cases


def build_documents():
    """Calculate all versioned inventory, table, and scenario documents.

    :return: Mapping from repository-relative fixture paths to JSON objects.
    """
    documents = {"tests/data/rust_api_inventory.json": build_inventory()}
    cases = {}
    contexts = {(shape, dtype): _context(shape, dtype) for _, shape, dtype, _ in _profiles()}
    for module in _modules():
        functions = [(name, obj) for name, obj in vars(module).items()
                     if not name.startswith("_") and inspect.isfunction(obj) and obj.__module__ == module.__name__]
        if module.__name__.endswith(".behave"):
            runner = BehaveRun(FuelModels())
            functions += [(f"BehaveRun.{name}", getattr(runner, name)) for name in vars(BehaveRun)
                          if not name.startswith("_") and callable(getattr(runner, name))]
        domain_cases = []
        for name, function in sorted(functions):
            if name.startswith("build_") and name not in ("build_particle_arrays",):
                profiles = _profiles()[:1]
            else:
                profiles = _profiles()
            for profile, shape, dtype, replacement in profiles:
                values = dict(contexts[shape, dtype])
                arguments = _arguments(function, values)
                if replacement is not None:
                    targets = [key for key, value in arguments.items() if isinstance(value, np.ndarray)
                               and key not in ("coeffs",)]
                    if targets:
                        replacement = replacement.astype(dtype) if "float16" in profile else replacement
                        values[targets[0]] = replacement
                record = _case(function, values, f"{name}/{profile}")
                record["policy"] = "diagnostic" if dtype == "float16" else "acceptance"
                domain_cases.append(record)
                if dtype == "float16":
                    normalized = normalize_float16(values)
                    normal_record = _case(function, normalized, f"{name}/{profile}/normalized")
                    normal_record["policy"] = "acceptance"
                    domain_cases.append(normal_record)
        if domain_cases:
            cases[module.__name__.rsplit(".", 1)[1]] = domain_cases
    cases.update(_stateful_cases())
    cases["registries"] = _registry_cases()
    cases["boundaries"] = _boundary_cases()
    for domain, records in cases.items():
        documents[f"tests/data/rust_port/{domain}.json"] = _envelope({"cases": records})
    fuels = FuelModels()
    slots = [{"number": number, "record": encode_value(fuels.fuel_models_.get(number)),
              "defined": bool(fuels.is_fuel_model_defined(number)),
              "reserved": bool(fuels.is_fuel_model_reserved(number))} for number in range(261)]
    custom = []
    for number in range(261):
        registry = FuelModels()
        accepted = registry.set_custom_fuel_model(number, "CUSTOM", "Contract fuel", 1, 0, .24, 0,
                   8000, 8000, 0, .01, .02, .03, .04, .05, 0, 1800, 1500, 1200, 0, True)
        custom.append({"number": number, "accepted": bool(accepted),
                       "record": encode_value(registry.fuel_models_.get(number)),
                       "clear_result": encode_value(registry.clear_custom_fuel_model(number))})
    documents["tests/data/rust_port/fuels.json"] = _envelope({"slots": slots, "custom_slots": custom,
                                                             "lookup": encode_value(build_fuel_lookup_arrays(fuels))})
    species = SpeciesMasterTable()
    documents["tests/data/rust_port/species.json"] = _envelope({
        "records": [encode_value(vars(record)) for record in species.get_all_records()],
        "mortality_coefficients": encode_value(build_mortality_lookup(species))})
    groups = {"units": ["behave_units"], "surface": ["surface"], "crown": ["crown"],
              "effects": ["mortality", "ignite", "spot", "safety"],
              "tools": ["fine_dead_fuel_moisture_tool", "vapor_pressure_deficit_calculator", "fire_size", "slope_tool"],
              "containment": ["contain"]}
    for group, domains in groups.items():
        compact = [case for domain in domains for case in cases.get(domain, [])
                   if case["id"].endswith("/baseline")]
        documents[f"rust/crates/behave-fire/tests/data/{group}.json"] = _envelope({"cases": compact})
    documents["rust/crates/behave-fire/tests/data/fuels.json"] = _envelope({
        "slots": slots, "lookup": encode_value(build_fuel_lookup_arrays(fuels))})
    documents["rust/crates/behave-fire/tests/data/species.json"] = documents["tests/data/rust_port/species.json"]
    documents["tests/data/rust_port/tables.json"] = _envelope({
        "fine_dead_fuel_moisture_reference": encode_value(fine_dead_fuel_moisture_tool._REF),
        "fine_dead_fuel_moisture_correction": encode_value(fine_dead_fuel_moisture_tool._CORR),
        "lightning_coefficients": encode_value(ignite._LIG_COEFFS)})
    for group, prefix in (("effects", "lightning/"), ("effects", "mortality/"),
                          ("surface", "surface/wind-"), ("containment", "containment/")):
        documents[f"rust/crates/behave-fire/tests/data/{group}.json"]["cases"].extend(
            case for case in cases["boundaries"] if case["id"].startswith(prefix))
    return documents


def build_inventory():
    """Inventory definitions, class methods, enums, and public export aliases.

    :return: Versioned inventory with planned core and binding destinations.
    """
    aliases = {}
    # Read export declarations statically: importing the root automatic facade
    # would select a backend in future releases and invalidate this oracle.
    import ast
    for path, route in ((SOURCE / "__init__.py", "behave_fire"),
                        (SOURCE / "components/__init__.py", "behave_fire.components")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.level:
                base = "behave_fire" if route == "behave_fire" else route
                for name in node.names:
                    canonical = f"{base}.{node.module}.{name.name}"
                    aliases.setdefault(canonical, []).append(f"{route}.{name.asname or name.name}")
    entries = []
    def add(obj, name, domain, kind):
        """Append one canonical API member.

        :param obj: Python member being inventoried.
        :param name: Fully qualified Python name.
        :param domain: Planned Rust module.
        :param kind: Member category.
        :return: None; appends an inventory entry.
        """
        try:
            signature = str(inspect.signature(obj))
        except (TypeError, ValueError):
            signature = None
        parts = name.split(".")
        suffix = "::".join(parts[3:] if ".components." in name else parts[2:])
        entry = {"python_name": name, "kind": kind, "signature": signature,
                 "aliases": sorted(aliases.get(name, [])), "rust_core": f"{domain}::{suffix}",
                 "binding_route": name.replace("behave_fire.components.", "behave_fire.rust.")
                                      .replace("behave_fire.behave.", "behave_fire.rust.core."),
                 "status": "planned", "role": "computation or state"}
        if parts[-1] in ("coerce_crown_fire_inputs", "get_surface_fire_inputs", "build_fuel_lookup_arrays"):
            entry["role"] = "Python array adaptation with native data transformation counterpart"
        if inspect.isclass(obj) and issubclass(obj, Enum):
            entry["members"] = {key: value.value for key, value in obj.__members__.items()}
            entry["kind"] = "enum"
        elif inspect.isclass(obj) and name.endswith("Enum"):
            entry["members"] = {key: value for key, value in vars(obj).items() if not key.startswith("_")}
            entry["kind"] = "enum namespace"
        entries.append(entry)
    for module in _modules():
        domain = _domain(module.__name__.rsplit(".", 1)[1])
        for name, obj in sorted(vars(module).items()):
            if name.startswith("STATUS_"):
                entries.append({"python_name": f"{module.__name__}.{name}", "kind": "status constant",
                                "value": int(obj), "rust_core": f"{domain}::{name}",
                                "binding_route": f"behave_fire.rust.contain.{name}",
                                "aliases": [], "status": "planned", "role": "status value"})
                continue
            if name.startswith("_") or getattr(obj, "__module__", None) != module.__name__:
                continue
            canonical = f"{module.__name__}.{name}"
            if inspect.isfunction(obj):
                add(obj, canonical, domain, "function")
            elif inspect.isclass(obj):
                add(obj, canonical, domain, "class")
                for member, value in sorted(vars(obj).items()):
                    if member.startswith("_") and member not in ("__init__", "__contains__", "__getitem__", "__repr__"):
                        continue
                    if isinstance(value, staticmethod):
                        value = value.__func__
                    if inspect.isfunction(value) or inspect.isclass(value):
                        add(value, f"{canonical}.{member}", domain, "method")
    return _envelope({"entries": sorted(entries, key=lambda entry: entry["python_name"]),
                      "coverage_note": "Destinations are planned, not claims of native implementation. Numerical scenarios are separate from registry/state contracts."})


def contracts_match(actual, expected, calculated=False, dtype="float64"):
    """Compare exact contracts while allowing documented numeric output error.

    :param actual: Newly calculated portable document or value.
    :param expected: Reviewed portable document or value.
    :param calculated: Whether values belong to a computed output or state.
    :param dtype: Dtype controlling floating tolerance within an output array.
    :return: True when structure, inputs, classifications, and outputs agree.
    """
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            return False
        dtype = expected.get("dtype", dtype)
        computed_fields = ("output", "state_before", "state_after", "p", "ib", "ib_crown", "ri", "ri_crown", "surface_results")
        return all(contracts_match(actual[key], value, calculated or key in computed_fields, dtype)
                   for key, value in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            contracts_match(left, right, calculated, dtype) for left, right in zip(actual, expected))
    if calculated and isinstance(expected, float):
        if actual == expected == 0.:
            return math.copysign(1., actual) == math.copysign(1., expected)
        relative, absolute = {"float16": (5e-3, 5e-4), "float32": (1e-6, 1e-7)}.get(dtype, (1e-9, 1e-12))
        return math.isclose(actual, expected, rel_tol=relative, abs_tol=absolute)
    return actual == expected


def encode_value(value):
    """Encode scalars, arrays, mappings, and registry inputs as strict JSON.

    :param value: Python reference value.
    :return: Portable representation retaining dtype, shape, masks, and signs.
    """
    if isinstance(value, np.ndarray):
        result = {"dtype": str(value.dtype), "shape": list(value.shape),
                  "values": [encode_value(item) for item in np.ma.getdata(value).ravel().tolist()]}
        if np.ma.isMaskedArray(value):
            result["mask"] = np.ma.getmaskarray(value).ravel().tolist()
        if not value.flags.c_contiguous:
            result["strides"] = list(value.strides)
        if not value.flags.writeable:
            result["read_only"] = True
        return result
    if isinstance(value, np.generic):
        return {"dtype": str(value.dtype), "shape": [], "values": [encode_value(value.item())]}
    if isinstance(value, dict):
        return {str(key): encode_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [encode_value(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, FuelModels):
        return {"registry": "standard FuelModels"}
    if isinstance(value, SpeciesMasterTable):
        return {"registry": "standard SpeciesMasterTable"}
    if isinstance(value, SpeciesMasterTableRecord):
        return encode_value(vars(value))
    raise TypeError(f"Unsupported reference type: {type(value).__name__}")


def main(argv=None):
    """Write reviewed fixtures or check existing artifacts without modifying them.

    :param argv: Optional command-line argument list.
    :return: Zero for success; one for drift.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report missing or changed fixtures without writing")
    parser.add_argument("--output-dir", type=Path, default=ROOT, help="Fixture repository root")
    options = parser.parse_args(argv)
    changed = []
    for name, document in sorted(build_documents().items()):
        target = options.output_dir / name
        serialized = json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if options.check:
            try:
                matches = contracts_match(document, json.loads(target.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                matches = False
            if not matches:
                changed.append(name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(serialized, encoding="utf-8", newline="\n")
    if changed:
        print("Reference fixture drift:\n" + "\n".join(changed))
        return 1
    print("Reference fixtures match" if options.check else "Reference fixtures exported")
    return 0


def normalize_float16(value):
    """Widen float16 inputs recursively while preserving mask and shape metadata.

    :param value: Input object or nested mapping.
    :return: Float16 arrays/scalars widened to float32; other values unchanged.
    """
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype == np.float16:
        return value.astype(np.float32)
    if isinstance(value, dict):
        return {key: normalize_float16(item) for key, item in value.items()}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
