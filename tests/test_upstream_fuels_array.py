"""Fuel metadata and numerical regressions against pinned upstream records."""

import json
from pathlib import Path

import numpy as np
import pytest

from behave_fire import BehaveRun
from behave_fire.components.fuel_models import FuelModels
from behave_fire.components.fuel_models_array import build_fuel_lookup_arrays

STATIC_MODELS = [*range(142, 149), 162, 164, 165, *range(181, 190), *range(201, 205)]
SCOTT_BURGAN_MODELS = [
    *range(101, 110), *range(121, 125), *range(141, 150),
    *range(161, 166), *range(181, 190), *range(201, 205),
]
REFERENCE = json.loads(
    (Path(__file__).parent / "data/upstream_metric_fuels.json").read_text(encoding="utf-8")
)
METRIC_MODELS = REFERENCE["metric_models"]
LUT_FIELDS = {
    "depth": "fuel_bed_depth", "moe_dead": "moisture_of_extinction_dead",
    "hoc_dead": "heat_of_combustion_dead", "hoc_live": "heat_of_combustion_live",
    "dead_1h": "dead_1h", "dead_10h": "dead_10h", "dead_100h": "dead_100h",
    "live_herb": "live_herb", "live_woody": "live_woody", "savr_1h": "savr_1h",
    "savr_lh": "savr_live_herb", "savr_lw": "savr_live_woody",
    "is_dynamic": "is_dynamic", "is_defined": "is_defined",
}


def _reference_fuels():
    """Register independent imperial records in available custom slots."""
    fuels = FuelModels()
    for number in METRIC_MODELS:
        r = REFERENCE["records"][str(number)]["local_record"]
        assert fuels.set_custom_fuel_model(
            211 + METRIC_MODELS.index(number), "REF", "Upstream reference",
            r["fuel_bed_depth"], 0, r["moisture_of_extinction_dead"], 0,
            r["heat_of_combustion_dead"], r["heat_of_combustion_live"], 0,
            r["dead_1h"], r["dead_10h"], r["dead_100h"], r["live_herb"], r["live_woody"], 0,
            r["savr_1h"], r["savr_live_herb"], r["savr_live_woody"], 0, r["is_dynamic"],
        )
    return fuels


def _surface(fuels, grid):
    """Run consistent surface scenarios using a fresh facade and lookup."""
    shape = np.shape(grid)
    return BehaveRun(fuels).do_surface_run(
        fuel_model_grid=grid, m1h=np.full(shape, 0.06),
        m10h=np.full(shape, 0.07), m100h=np.full(shape, 0.08),
        mlh=np.full(shape, 0.60), mlw=np.full(shape, 0.90),
        wind_speed=5.0, wind_speed_units=5,
        wind_direction=0.0, wind_orientation_mode="RelativeToUpslope",
        slope=0.30, slope_units=1,
    )


@pytest.mark.parametrize("number", METRIC_MODELS + REFERENCE["control_models"])
def test_fuel_records_and_lookup_match_upstream_reference(number):
    fuels = FuelModels()
    record = fuels.fuel_models_[number]
    reference = REFERENCE["records"][str(number)]["local_record"]
    for key, expected in reference.items():
        if isinstance(expected, (str, bool)):
            assert record[key] == expected, key
        else:
            assert record[key] == pytest.approx(expected, rel=1e-12, abs=1e-12), key
    lut = build_fuel_lookup_arrays(fuels)
    for key, field in LUT_FIELDS.items():
        assert lut[key][number] == pytest.approx(reference[field], rel=1e-12, abs=1e-12), key


@pytest.mark.parametrize("grid", [*METRIC_MODELS, np.array(METRIC_MODELS).reshape(3, 6)])
def test_metric_surface_matches_independent_imperial_custom_records(grid):
    fuels = _reference_fuels()
    reference_grid = np.asarray(grid).copy()
    for index, number in enumerate(METRIC_MODELS):
        reference_grid = np.where(np.asarray(grid) == number, 211 + index, reference_grid)
    actual = _surface(FuelModels(), grid)
    expected = _surface(fuels, reference_grid)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], rtol=1e-12, atol=1e-12, err_msg=key)


@pytest.mark.parametrize("number", SCOTT_BURGAN_MODELS)
def test_scott_burgan_dynamic_flags_follow_herbaceous_load(number):
    fuels = FuelModels()
    expected = fuels.fuel_models_[number]["live_herb"] > 0
    assert fuels.get_is_dynamic(number) == expected
    assert build_fuel_lookup_arrays(fuels)["is_dynamic"][number] == expected


@pytest.mark.parametrize("grid", [*STATIC_MODELS, np.array(STATIC_MODELS + [124]).reshape(4, 6)])
def test_static_flag_change_preserves_surface_results(grid):
    corrected = FuelModels()
    legacy = FuelModels()
    for number in STATIC_MODELS:
        legacy.fuel_models_[number]["is_dynamic"] = True
    actual = _surface(corrected, grid)
    expected = _surface(legacy, grid)
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
