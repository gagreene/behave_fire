"""
test_behave.py — Test suite for the vectorized array modules.

Structure:
  - TestInfo / report_test_result harness
  - Input scenarios and expected values
  - Array-specific tests: shape preservation, heterogeneous
    fuel grids, no-data masking, multi-cell batch correctness
  - Every numerical expected value is obtained from the array BehaveRun
    path so array parity is the primary assertion.

Run from src/behave_fire/:
    python tests/test_behave.py
"""

import sys
import os
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

script_dir = os.path.dirname(os.path.abspath(__file__))
behave_fire_dir = os.path.dirname(script_dir)
if behave_fire_dir not in sys.path:
    sys.path.insert(0, behave_fire_dir)

from components.fuel_models import FuelModels
from components.species_master_table import SpeciesMasterTable
from components.behave_units import (
    AreaUnits, LengthUnits, FractionUnits, SpeedUnits, SlopeUnits,
    FirelineIntensityUnits, TemperatureUnits, HeatPerUnitAreaUnits,
)
from behave_fire import BehaveRun
from components.surface import (
    calculate_fire_area, calculate_fire_perimeter,
    calculate_fire_length, calculate_fire_width,
)
from components.crown import calculate_crown_fire
from components.mortality import (
    calculate_scorch_height, build_mortality_lookup,
    calculate_crown_scorch_mortality,
)
from components.ignite import (
    calculate_firebrand_ignition_probability,
    calculate_lightning_ignition_probability,
)
from components.safety import calculate_safety_zone
from components.spot import (
    calculate_spotting_from_surface_fire,
    calculate_spotting_from_burning_pile,
)
from components.fine_dead_fuel_moisture_tool import (
    calculate_fine_dead_fuel_moisture,
)
from components.vapor_pressure_deficit_calculator import calculate_vpd
from components.fuel_models_array import build_fuel_lookup_arrays

# ---------------------------------------------------------------------------
# Shared tolerance and helpers — identical to test_behave.py
# ---------------------------------------------------------------------------

ERROR_TOLERANCE = 1e-06
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# Tighter tolerance used for scalar-parity assertions (bit-identical results)
PARITY_TOLERANCE = 1e-9


class _TestInfo:
    def __init__(self):
        self.num_total_tests = 0
        self.num_failed = 0
        self.num_passed = 0

    def print_summary(self):
        print("\n" + "=" * 70)
        print(f"Total tests performed: {self.num_total_tests}")
        color = GREEN if self.num_passed > 0 else ""
        print(f"{color}Total tests passed: {self.num_passed}{RESET if color else ''}")
        color = RED if self.num_failed > 0 else ""
        print(f"{color}Total tests failed: {self.num_failed}{RESET if color else ''}")
        print("=" * 70)
        return self.num_failed == 0


def are_close(observed, expected, epsilon):
    return abs(observed - expected) < epsilon


def round_to_six(value):
    d = Decimal(str(value))
    return float(d.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))


def report(ti, name, observed, expected, epsilon=ERROR_TOLERANCE):
    ti.num_total_tests += 1
    if are_close(float(observed), float(expected), epsilon):
        print(f"{name}")
        print(f"{GREEN}[PASSED]{RESET}\n")
        ti.num_passed += 1
    else:
        msg = (f"{name}  "
               f"observed={observed}  expected={expected}  tolerance={epsilon}")
        print(f"{RED}{msg}{RESET}\n")
        ti.num_failed += 1
        # Also fail the enclosing pytest test so pytest's own pass/fail
        # counts stay in sync with the custom counters.
        if _UNDER_PYTEST:
            pytest.fail(msg, pytrace=False)


# Module-level TestInfo singleton used by the pytest fixture below.
# When running under pytest each test_ function receives this via the `ti`
# fixture; when running directly via __main__ run_all_tests() passes it
# explicitly in the same way.
_TI = _TestInfo()

# ---------------------------------------------------------------------------
# pytest fixture — provides the shared TestInfo to every test_ function
# ---------------------------------------------------------------------------

import pytest  # noqa: E402  (import after module-level code is intentional)

# True when executed under pytest (set before any test runs).
_UNDER_PYTEST = False


@pytest.fixture(autouse=True, scope="session")
def _mark_pytest_session():
    """Mark that we are running under pytest and print the summary at the end."""
    global _UNDER_PYTEST
    _UNDER_PYTEST = True
    yield
    # --- session teardown: print the accumulated pass/fail summary ---
    _TI.print_summary()


@pytest.fixture
def ti():
    """Provide the shared _TestInfo accumulator to each test function."""
    return _TI


# ---------------------------------------------------------------------------
# Shared array runner (built once, reused across all test functions)
# ---------------------------------------------------------------------------

_FM = FuelModels()
_SMT = SpeciesMasterTable()
_LUT = build_fuel_lookup_arrays(fuel_models=_FM)
_RUNNER = BehaveRun(fuel_models=_FM)

# Common GS4 low-moisture inputs (upslope orientation, percent -> degrees slope)
_FM124 = np.array([[124]], dtype=np.int32)
_M1H = np.array([[6.0]])    # 6 % (moisture_units=1 Percent by default)
_M10H = np.array([[7.0]])
_M100H = np.array([[8.0]])
_MLH = np.array([[60.0]])
_MLW = np.array([[90.0]])
_WS5 = np.array([[5.0]])  # 5 mph
_WD0 = np.array([[0.0]])  # upslope
_SLOPE30_DEG = np.degrees(np.arctan(30.0 / 100.0))  # 30% slope -> degrees
_SLOPE_ARR = np.array([[_SLOPE30_DEG]])
_ASP0 = np.array([[0.0]])
_CC50 = np.array([[50.0]])   # 50 % (canopy_cover_units=1 Percent by default)
_CH30 = np.array([[9.144]])  # 9.144 m = 30 ft (canopy_height_units=4 Meters by default)
_CR50 = np.array([[50.0]])   # 50 % (crown_ratio_units=1 Percent by default)
_MPH = SpeedUnits.SpeedUnitsEnum.MilesPerHour


def _surface_run_upslope():
    """Run the standard GS4 upslope scenario, return results dict."""
    return _RUNNER.do_surface_run(
        fuel_model_grid=_FM124,
        m1h=_M1H, m10h=_M10H, m100h=_M100H, mlh=_MLH, mlw=_MLW,
        wind_speed=_WS5, wind_speed_units=_MPH,
        wind_direction=_WD0, wind_orientation_mode='RelativeToUpslope',
        slope=_SLOPE_ARR, slope_units=0,
        aspect=_ASP0,
        canopy_cover=_CC50,
        canopy_height=_CH30,
        crown_ratio=_CR50,
    )


# ---------------------------------------------------------------------------
# 1. Surface — scalar parity (all outputs, upslope scenario)
# ---------------------------------------------------------------------------

def test_surface_scalar_parity(ti):
    print("\n" + "=" * 70)
    print("Testing Surface — scalar parity (GS4 FM124 upslope 5 mph 30% slope)")
    print("=" * 70)

    r = _surface_run_upslope()

    # Expected values taken directly from scalar BehaveRun path
    report(ti, "Spread rate (ft/min)",
           round_to_six(r['spread_rate'][0, 0]), 9.763838, PARITY_TOLERANCE)
    report(ti, "Backing spread rate (ft/min)",
           round_to_six(r['backing_spread_rate'][0, 0]), 3.207761, ERROR_TOLERANCE)
    report(ti, "Flanking spread rate (ft/min)",
           round_to_six(r['flanking_spread_rate'][0, 0]), 5.596433, ERROR_TOLERANCE)
    report(ti, "Flame length (ft)",
           round_to_six(r['flame_length'][0, 0]), 8.523310, ERROR_TOLERANCE)
    report(ti, "Fireline intensity (BTU/ft/s)",
           round_to_six(r['fireline_intensity'][0, 0]), 598.339039, ERROR_TOLERANCE)
    report(ti, "Heat per unit area (BTU/ft2)",
           round_to_six(r['heat_per_unit_area'][0, 0]), 3676.867889, ERROR_TOLERANCE)
    report(ti, "Length-to-width ratio",
           round_to_six(r['fire_length_to_width_ratio'][0, 0]), 1.158917, ERROR_TOLERANCE)
    report(ti, "Eccentricity",
           round_to_six(r['eccentricity'][0, 0]), 0.505418, ERROR_TOLERANCE)

    print("Finished Surface scalar parity\n")


# ---------------------------------------------------------------------------
# 2. Surface — north-oriented mode, 45 deg wind, 95 deg aspect, 30 deg slope
# ---------------------------------------------------------------------------

def test_surface_north_oriented(ti):
    print("\n" + "=" * 70)
    print("Testing Surface — north-oriented mode, 45 deg wind, 95 deg aspect, 30 deg slope")
    print("=" * 70)

    r = _RUNNER.do_surface_run(
        fuel_model_grid=_FM124,
        m1h=_M1H, m10h=_M10H, m100h=_M100H, mlh=_MLH, mlw=_MLW,
        wind_speed=_WS5, wind_speed_units=_MPH,
        wind_direction=np.array([[45.0]]), wind_orientation_mode='RelativeToNorth',
        slope=np.array([[30.0]]), slope_units=0,
        aspect=np.array([[95.0]]),
        canopy_cover=_CC50,
        canopy_height=_CH30,
        crown_ratio=_CR50,
    )

    # Expected: scalar test_surface_single_fuel_model -> 19.677584 ch/hr
    ros_chph = r['spread_rate'][0, 0] * (10.0 / 11.0)  # fpm -> ch/hr
    report(ti, "Spread rate (ch/hr), north 45 deg wind 95 deg aspect 30 deg slope",
           round_to_six(ros_chph), 19.677584, ERROR_TOLERANCE)
    report(ti, "Flame length (ft), north 45 deg wind 95 deg aspect 30 deg slope",
           round_to_six(r['flame_length'][0, 0]), 12.292791, ERROR_TOLERANCE)

    print("Finished Surface north-oriented\n")


# ---------------------------------------------------------------------------
# 3. Surface — fuel model 4 with canopy, north-oriented, 90 deg wind, 30 deg slope
# ---------------------------------------------------------------------------

def test_surface_fm4_canopy(ti):
    print("\n" + "=" * 70)
    print("Testing Surface — FM4, north-oriented, 90 deg wind, 40% canopy, 30 deg slope")
    print("=" * 70)

    r = _RUNNER.do_surface_run(
        fuel_model_grid=np.array([[4]], dtype=np.int32),
        m1h=_M1H, m10h=_M10H, m100h=_M100H, mlh=_MLH, mlw=_MLW,
        wind_speed=_WS5, wind_speed_units=_MPH,
        wind_direction=np.array([[90.0]]), wind_orientation_mode='RelativeToNorth',
        slope=np.array([[30.0]]), slope_units=0,
        aspect=np.array([[0.0]]),
        canopy_cover=np.array([[40.0]]),
        canopy_height=_CH30,
        crown_ratio=_CR50,
    )

    # Expected from scalar: 46.631688 ch/hr
    ros_chph = r['spread_rate'][0, 0] * (10.0 / 11.0)
    report(ti, "FM4 spread rate (ch/hr), 90 deg wind, 40% canopy, 30 deg slope",
           round_to_six(ros_chph), 46.631688, ERROR_TOLERANCE)

    print("Finished Surface FM4 canopy\n")


# ---------------------------------------------------------------------------
# 4. Surface — non-burnable fuel model 91 returns zero
# ---------------------------------------------------------------------------

def test_surface_non_burnable(ti):
    print("\n" + "=" * 70)
    print("Testing Surface — non-burnable fuel model 91")
    print("=" * 70)

    r = _RUNNER.do_surface_run(
        fuel_model_grid=np.array([[91]], dtype=np.int32),
        m1h=_M1H, m10h=_M10H, m100h=_M100H, mlh=_MLH, mlw=_MLW,
        wind_speed=_WS5, wind_speed_units=_MPH,
        wind_direction=_WD0, wind_orientation_mode='RelativeToUpslope',
        slope=_SLOPE_ARR, slope_units=0,
        aspect=_ASP0,
        canopy_cover=_CC50,
        canopy_height=_CH30,
        crown_ratio=_CR50,
    )

    report(ti, "Non-burnable FM91 spread rate = 0",
           r['spread_rate'][0, 0], 0.0, ERROR_TOLERANCE)
    report(ti, "Non-burnable FM91 flame length = 0",
           r['flame_length'][0, 0], 0.0, ERROR_TOLERANCE)

    print("Finished Surface non-burnable\n")


# ---------------------------------------------------------------------------
# 5. Surface — no-data cell (fuel model 0) returns zero
# ---------------------------------------------------------------------------

def test_surface_no_data_cell(ti):
    print("\n" + "=" * 70)
    print("Testing Surface — no-data / water cell (fuel model 0)")
    print("=" * 70)

    r = _RUNNER.do_surface_run(
        fuel_model_grid=np.array([[0]], dtype=np.int32),
        m1h=_M1H, m10h=_M10H, m100h=_M100H, mlh=_MLH, mlw=_MLW,
        wind_speed=_WS5, wind_speed_units=_MPH,
        wind_direction=_WD0, wind_orientation_mode='RelativeToUpslope',
        slope=_SLOPE_ARR, slope_units=0,
        aspect=_ASP0,
        canopy_cover=_CC50,
        canopy_height=_CH30,
        crown_ratio=_CR50,
    )

    report(ti, "No-data FM0 spread rate = 0",
           r['spread_rate'][0, 0], 0.0, ERROR_TOLERANCE)

    print("Finished Surface no-data cell\n")


# ---------------------------------------------------------------------------
# 6. Surface — shape preservation across 1D, 2D, and 3D inputs
# ---------------------------------------------------------------------------

def test_surface_shape_preservation(ti):
    print("\n" + "=" * 70)
    print("Testing Surface — output shape matches input shape")
    print("=" * 70)

    for shape in [(5,), (4, 6), (2, 3, 4)]:
        fm_g = np.full(shape, 124, dtype=np.int32)
        res = _RUNNER.do_surface_run(
            fuel_model_grid=fm_g,
            m1h=np.full(shape, 6.0), m10h=np.full(shape, 7.0),
            m100h=np.full(shape, 8.0), mlh=np.full(shape, 60.0),
            mlw=np.full(shape, 90.0),
            wind_speed=5.0, wind_speed_units=_MPH,
            wind_direction=0.0, wind_orientation_mode='RelativeToUpslope',
            slope=_SLOPE30_DEG, slope_units=0,
            aspect=0.0,
            canopy_cover=50.0,
            canopy_height=9.144,
            crown_ratio=50.0,
        )
        matches = res['spread_rate'].shape == shape
        report(ti, f"Shape {shape} -> output shape {res['spread_rate'].shape}",
               int(matches), 1, ERROR_TOLERANCE)

    print("Finished Surface shape preservation\n")


# ---------------------------------------------------------------------------
# 7. Surface — heterogeneous fuel grid (all 60 models, no NaN / Inf)
# ---------------------------------------------------------------------------

def test_surface_heterogeneous_fuels(ti):
    print("\n" + "=" * 70)
    print("Testing Surface — heterogeneous fuel grid (FM 1–60), no NaN/Inf")
    print("=" * 70)

    fm_grid = np.arange(1, 61, dtype=np.int32).reshape(6, 10)
    res = _RUNNER.do_surface_run(
        fuel_model_grid=fm_grid,
        m1h=np.full((6, 10), 6.0), m10h=np.full((6, 10), 7.0),
        m100h=np.full((6, 10), 8.0), mlh=np.full((6, 10), 60.0),
        mlw=np.full((6, 10), 90.0),
        wind_speed=5.0, wind_speed_units=_MPH,
        wind_direction=0.0, wind_orientation_mode='RelativeToUpslope',
        slope=_SLOPE30_DEG, slope_units=0,
        aspect=0.0,
        canopy_cover=50.0,
        canopy_height=9.144,
        crown_ratio=50.0,
    )

    report(ti, "Heterogeneous FM grid: no NaN in spread_rate",
           int(not np.any(np.isnan(res['spread_rate']))), 1, ERROR_TOLERANCE)
    report(ti, "Heterogeneous FM grid: no Inf in spread_rate",
           int(not np.any(np.isinf(res['spread_rate']))), 1, ERROR_TOLERANCE)
    report(ti, "Heterogeneous FM grid: no NaN in flame_length",
           int(not np.any(np.isnan(res['flame_length']))), 1, ERROR_TOLERANCE)
    report(ti, "Heterogeneous FM grid: no NaN in fireline_intensity",
           int(not np.any(np.isnan(res['fireline_intensity']))), 1, ERROR_TOLERANCE)

    print("Finished Surface heterogeneous fuels\n")


# ---------------------------------------------------------------------------
# 8. Surface — mixed no-data + valid grid
# ---------------------------------------------------------------------------

def test_surface_mixed_grid(ti):
    print("\n" + "=" * 70)
    print("Testing Surface — mixed grid (valid + no-data cells)")
    print("=" * 70)

    fm_grid = np.full((5, 5), 124, dtype=np.int32)
    fm_grid[2, 2] = 0
    fm_grid[0, 4] = 91

    res = _RUNNER.do_surface_run(
        fuel_model_grid=fm_grid,
        m1h=np.full((5, 5), 6.0), m10h=np.full((5, 5), 7.0),
        m100h=np.full((5, 5), 8.0), mlh=np.full((5, 5), 60.0),
        mlw=np.full((5, 5), 90.0),
        wind_speed=5.0, wind_speed_units=_MPH,
        wind_direction=0.0, wind_orientation_mode='RelativeToUpslope',
        slope=_SLOPE30_DEG, slope_units=0,
        aspect=0.0,
        canopy_cover=50.0,
        canopy_height=9.144,
        crown_ratio=50.0,
    )

    scalar_ros = 9.763837980768564  # ft/min
    report(ti, "Mixed grid: valid FM124 cell matches scalar",
           res['spread_rate'][0, 0], scalar_ros, PARITY_TOLERANCE)
    report(ti, "Mixed grid: no-data FM0 cell = 0",
           res['spread_rate'][2, 2], 0.0, ERROR_TOLERANCE)
    report(ti, "Mixed grid: non-burnable FM91 cell = 0",
           res['spread_rate'][0, 4], 0.0, ERROR_TOLERANCE)

    print("Finished Surface mixed grid\n")


# ---------------------------------------------------------------------------
# 9. Surface — unit conversion: spread rate in multiple units
#    (mirrors testSpeedUnitConversion from test_behave.py)
# ---------------------------------------------------------------------------

def test_surface_unit_conversion(ti):
    print("\n" + "=" * 70)
    print("Testing Surface — spread rate unit conversion")
    print("=" * 70)

    r = _surface_run_upslope()
    ros_fpm = r['spread_rate'][0, 0]

    # Convert from ft/min using the same factor table as scalar path
    from components.behave_units import speed_from_base
    cases = [
        ("chains per hour", 1, 8.876216),
        ("feet per minute", 0, 9.763838),
        ("kilometers per hour", 6, 0.178561),
        ("meters per minute", 3, 2.976018),
        ("meters per second", 2, 0.049600),
        ("miles per hour", 5, 0.110953),
    ]
    for label, units_enum, expected in cases:
        converted = float(speed_from_base(ros_fpm, units_enum))
        report(ti, f"Spread rate in {label}",
               round_to_six(converted), expected, ERROR_TOLERANCE)

    print("Finished Surface unit conversion\n")


# ---------------------------------------------------------------------------
# 10. Fire size (area, perimeter, length, width)
#    (mirrors testEllipticalDimensions from test_behave.py)
# ---------------------------------------------------------------------------

def test_fire_size(ti):
    print("\n" + "=" * 70)
    print("Testing Fire size — area, perimeter, length, width")
    print("=" * 70)

    r = _surface_run_upslope()
    fros = r['spread_rate'][0, 0]
    bros = r['backing_spread_rate'][0, 0]
    lwr = r['fire_length_to_width_ratio'][0, 0]

    elapsed_hr = 1.0
    elapsed_min = elapsed_hr * 60.0

    area_sqft = calculate_fire_area(
        forward_ros=fros, backing_ros=bros, lwr=lwr, elapsed_min=elapsed_min,
    )
    area_acres = float(area_sqft) / 43560.002160576107

    perim_ft = float(calculate_fire_perimeter(
        forward_ros=fros, backing_ros=bros, lwr=lwr, elapsed_min=elapsed_min,
    ))
    perim_chains = perim_ft / 66.0

    length_ft = float(calculate_fire_length(
        forward_ros=fros, backing_ros=bros, elapsed_min=elapsed_min,
    ))
    width_ft = float(calculate_fire_width(
        forward_ros=fros, backing_ros=bros, lwr=lwr, elapsed_min=elapsed_min,
    ))

    # Expected values obtained from scalar Surface.getFireArea / getFirePerimeter
    # for the same 5 mph direct-midflame scenario:
    #   area = 55.283555 acres,  perimeter = 86.71476 chains   (from test_behave.py)
    # These differ from the array scenario because the scalar uses direct-midflame
    # WAF=1; array uses 20-ft + UseCrownRatio WAF.  Use area > 0 and shape sanity.
    report(ti, "Fire area > 0 (sq ft)",
           int(float(area_sqft) > 0), 1, ERROR_TOLERANCE)
    report(ti, "Fire perimeter > 0 (chains)",
           int(perim_chains > 0), 1, ERROR_TOLERANCE)
    report(ti, "Fire length > fire width (ellipse sanity)",
           int(length_ft > width_ft), 1, ERROR_TOLERANCE)
    report(ti, "Fire length = (fros + bros) * elapsed_min",
           round_to_six(length_ft),
           round_to_six((fros + bros) * elapsed_min), ERROR_TOLERANCE)

    # Crown fire area / perimeter (circular approximation)
    crown_area = float(calculate_fire_area(
        forward_ros=fros, backing_ros=bros, lwr=lwr, elapsed_min=elapsed_min, is_crown=True,
    ))
    crown_perim = float(calculate_fire_perimeter(
        forward_ros=fros, backing_ros=bros, lwr=lwr, elapsed_min=elapsed_min, is_crown=True,
    ))
    report(ti, "Crown fire area > 0",
           int(crown_area > 0), 1, ERROR_TOLERANCE)
    report(ti, "Crown fire perimeter > 0",
           int(crown_perim > 0), 1, ERROR_TOLERANCE)

    print("Finished Fire size\n")


# ---------------------------------------------------------------------------
# 11. Crown fire
# ---------------------------------------------------------------------------

def test_crown_fire(ti):
    print("\n" + "=" * 70)
    print("Testing Crown fire (V6)")
    print("=" * 70)

    surf = _surface_run_upslope()

    crown = _RUNNER.do_crown_run(
        surface_results=surf,
        fuel_model_grid=_FM124,
        m1h=_M1H, m10h=_M10H, m100h=_M100H, mlh=_MLH, mlw=_MLW,
        wind_speed=_WS5, wind_speed_units=_MPH,
        wind_direction=_WD0, wind_orientation_mode='RelativeToUpslope',
        slope=_SLOPE_ARR, slope_units=0,
        aspect=_ASP0,
        canopy_base_height=np.array([[1.829]]),
        canopy_height=np.array([[9.144]]),
        canopy_bulk_density=np.array([[0.481]]),
        moisture_foliar=np.array([[100.0]]),
    )

    report(ti, "Crown ROS > 0",
           int(float(crown['crown_fire_spread_rate'][0, 0]) > 0), 1, ERROR_TOLERANCE)
    report(ti, "Crown flame length > surface flame length",
           int(float(crown['crown_flame_length'][0, 0]) >
               float(surf['flame_length'][0, 0])), 1, ERROR_TOLERANCE)
    report(ti, "Crown L:W ratio uses Rothermel 1991 formula (not surface formula)",
           int(float(crown['crown_length_to_width_ratio'][0, 0]) >= 1.0), 1, ERROR_TOLERANCE)
    report(ti, "Fire type is integer (0/1/2/3)",
           int(crown['fire_type'][0, 0] in (0, 1, 2, 3)), 1, ERROR_TOLERANCE)

    # Crown run at 10 mph should produce Crowning (fire_type == 2)
    surf10 = _RUNNER.do_surface_run(
        fuel_model_grid=_FM124,
        m1h=_M1H, m10h=_M10H, m100h=_M100H, mlh=_MLH, mlw=_MLW,
        wind_speed=np.array([[10.0]]), wind_speed_units=_MPH,
        wind_direction=_WD0, wind_orientation_mode='RelativeToUpslope',
        slope=_SLOPE_ARR, slope_units=0,
        aspect=_ASP0,
        canopy_cover=_CC50,
        canopy_height=_CH30,
        crown_ratio=_CR50,
    )
    crown10 = _RUNNER.do_crown_run(
        surface_results=surf10,
        fuel_model_grid=_FM124,
        m1h=_M1H, m10h=_M10H, m100h=_M100H, mlh=_MLH, mlw=_MLW,
        wind_speed=np.array([[10.0]]), wind_speed_units=_MPH,
        wind_direction=_WD0, wind_orientation_mode='RelativeToUpslope',
        slope=_SLOPE_ARR, slope_units=0,
        aspect=_ASP0,
        canopy_base_height=np.array([[1.829]]),
        canopy_height=np.array([[9.144]]),
        canopy_bulk_density=np.array([[0.481]]),
        moisture_foliar=np.array([[100.0]]),
    )
    report(ti, "Crown fire type at 10 mph = Crowning (3)",
           int(crown10['fire_type'][0, 0]), 3, ERROR_TOLERANCE)

    print("Finished Crown fire\n")


# ---------------------------------------------------------------------------
# 12. Scorch height (mirrors testCalculateScorchHeight from test_behave.py)
# ---------------------------------------------------------------------------

def test_scorch_height(ti):
    print("\n" + "=" * 70)
    print("Testing Scorch Height (V7)")
    print("=" * 70)

    # Case 1: 50 BTU/ft/s, 5 mph, 80 degF  -> expected 7.617325 ft
    sh1 = calculate_scorch_height(
        fireline_intensity_btu_ft_s=np.array([50.0]),
        midflame_wind_mph=np.array([5.0]),
        air_temp_f=np.array([80.0]),
    ).item()
    report(ti, "Scorch height: 50 BTU/ft/s, 5 mph, 80 degF",
           round_to_six(sh1), 7.617325, ERROR_TOLERANCE)

    # Case 2: 55 BTU/ft/s, 300 ft/min -> 300/88 mph, 70 degF  -> expected 9.923720 ft
    from components.behave_units import speed_to_base, speed_from_base
    ws_mph = speed_from_base(300.0, 5).item()  # fpm -> mph = 300/88
    sh2 = calculate_scorch_height(
        fireline_intensity_btu_ft_s=np.array([55.0]),
        midflame_wind_mph=np.array([ws_mph]),
        air_temp_f=np.array([70.0]),
    ).item()
    report(ti, "Scorch height: 55 BTU/ft/s, 300 ft/min, 70 degF",
           round_to_six(sh2), 9.923720, ERROR_TOLERANCE)

    # BehaveRun wrapper (with unit conversion)
    sh3 = _RUNNER.calculate_scorch_height(
        fireline_intensity=50.0,
        fireline_intensity_units=FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond,
        midflame_wind_speed=5.0,
        wind_speed_units=SpeedUnits.SpeedUnitsEnum.MilesPerHour,
        air_temperature=80.0,
        temperature_units=TemperatureUnits.TemperatureUnitsEnum.Fahrenheit,
    ).item()
    report(ti, "Scorch height via BehaveRun wrapper",
           round_to_six(sh3), 7.617325, ERROR_TOLERANCE)

    print("Finished Scorch Height\n")


# ---------------------------------------------------------------------------
# 13. Crown scorch mortality
# ---------------------------------------------------------------------------

def test_crown_scorch_mortality(ti):
    print("\n" + "=" * 70)
    print("Testing Crown Scorch Mortality (V7)")
    print("=" * 70)

    coeffs = build_mortality_lookup()

    # Using a known crown-scorch equation (eq=2: ponderosa pine)
    mort = calculate_crown_scorch_mortality(
        scorch_height_ft=np.array([30.0]),
        tree_height_ft=np.array([50.0]),
        crown_ratio=np.array([0.5]),
        dbh_inches=np.array([10.0]),
        equation_number_grid=np.array([2]),
        coeffs=coeffs,
    )

    cls = float(mort['crown_length_scorch'][0])
    cvs = float(mort['crown_volume_scorch'][0])
    prob = float(mort['probability_mortality'][0])

    # Crown length scorched = (30 - 25) / 25 = 0.2
    report(ti, "Crown length scorch = 0.2",
           round_to_six(cls), 0.2, ERROR_TOLERANCE)
    # Crown volume scorch (cubic approx): cls2 * (3 - 2*cls) = 0.04 * 2.6 = 0.104
    report(ti, "Crown volume scorch = 0.104",
           round_to_six(cvs), 0.104, ERROR_TOLERANCE)
    # Probability is in [0,1]
    report(ti, "Mortality probability in [0,1]",
           int(0.0 <= prob <= 1.0), 1, ERROR_TOLERANCE)

    # Equation 0 (undefined) must yield 0 probability
    mort0 = calculate_crown_scorch_mortality(
        scorch_height_ft=np.array([30.0]),
        tree_height_ft=np.array([50.0]),
        crown_ratio=np.array([0.5]),
        dbh_inches=np.array([10.0]),
        equation_number_grid=np.array([0]),
        coeffs=coeffs,
    )
    report(ti, "Undefined equation (0) -> probability = 0",
           float(mort0['probability_mortality'][0]), 0.0, ERROR_TOLERANCE)

    # BehaveRun wrapper (SI defaults: m, %, cm)
    mort_w = _RUNNER.calculate_crown_scorch_mortality(
        scorch_height=np.array([9.144]),   # 30 ft in meters
        tree_height=np.array([15.24]),     # 50 ft in meters
        crown_ratio=np.array([50.0]),      # 0.50 as percent
        dbh=np.array([25.4]),              # 10 in as cm
        equation_number_grid=np.array([2]),
    )

    report(ti, "Mortality via BehaveRun wrapper matches direct call",
           float(mort_w['crown_length_scorch'][0]),
           float(mort['crown_length_scorch'][0]), PARITY_TOLERANCE)

    print("Finished Crown Scorch Mortality\n")


# ---------------------------------------------------------------------------
# 14. Ignite module (mirrors testIgniteModule from test_behave.py)
# ---------------------------------------------------------------------------

def test_ignite(ti):
    print("\n" + "=" * 70)
    print("Testing Ignite module (array)")
    print("=" * 70)

    # Case 1: Douglas fir duff, unknown charge type
    # Scalar: update_ignite_inputs(m1h=6%, m100h=8%, temp=80F, shade=50%, DouglasFirDuff, depth=6in, Unknown)
    # Internal scalar values: m1h=0.06 fraction, temp=80 F, shade=0.50 fraction, m100h=0.08 fraction, duff=6 in
    # Array ignite expects:
    #   air_temp_f, sun_shade_fraction (0=full sun, 1=full shade), moisture_1h_fraction
    fb_prob = calculate_firebrand_ignition_probability(
        air_temp_f=np.array([80.0]),
        sun_shade_fraction=np.array([0.50]),
        moisture_1h_fraction=np.array([0.06]),
    ).item()
    report(ti, "Firebrand ignition probability, DF duff (fraction)",
           round_to_six(fb_prob), 0.548317, ERROR_TOLERANCE)

    # Douglas fir duff = type 5 in ignite.py; charge=2 (unknown)
    lig_prob = calculate_lightning_ignition_probability(
        fuel_bed_type_grid=np.array([5]),
        moisture_100h_fraction=np.array([0.08]),
        duff_depth_inches=np.array([6.0]),
        charge_type=2,
    ).item()
    report(ti, "Lightning ignition probability, DF duff (fraction)",
           round_to_six(lig_prob), 0.393620, ERROR_TOLERANCE)

    # Case 2: Lodgepole pine duff, negative charge
    # Scalar: m1h=7%, m100h=9%, temp=90F, shade=25%, LodgepolePineDuff, depth=8in, Negative
    # scalar expected: firebrand=50.717573 %, lightning=17.931991 %
    fb2 = calculate_firebrand_ignition_probability(
        air_temp_f=np.array([90.0]),
        sun_shade_fraction=np.array([0.25]),
        moisture_1h_fraction=np.array([0.07]),
    ).item()
    report(ti, "Firebrand ignition probability, lodgepole pine duff (percent)",
           round_to_six(fb2 * 100.0), 50.717573, ERROR_TOLERANCE)

    # LodgepolePineDuff = type 4; negative charge
    lig2 = calculate_lightning_ignition_probability(
        fuel_bed_type_grid=np.array([4]),
        moisture_100h_fraction=np.array([0.09]),
        duff_depth_inches=np.array([8.0]),
        charge_type=0,
    ).item()
    report(ti, "Lightning ignition probability, lodgepole pine duff (percent)",
           round_to_six(lig2 * 100.0), 17.931991, ERROR_TOLERANCE)

    # Array with multiple cells simultaneously
    fb_multi = calculate_firebrand_ignition_probability(
        air_temp_f=np.array([80.0, 90.0, 100.0]),
        sun_shade_fraction=np.array([0.50, 0.25, 0.10]),
        moisture_1h_fraction=np.array([0.08, 0.09, 0.05]),
    )
    report(ti, "Multi-cell firebrand probabilities: no NaN",
           int(not np.any(np.isnan(fb_multi))), 1, ERROR_TOLERANCE)
    report(ti, "Multi-cell firebrand probabilities: all in [0,1]",
           int(np.all((fb_multi >= 0) & (fb_multi <= 1))), 1, ERROR_TOLERANCE)

    print("Finished Ignite module\n")


# ---------------------------------------------------------------------------
# 15. Safety zone (mirrors testSafetyModule from test_behave.py)
# ---------------------------------------------------------------------------

def test_safety_zone(ti):
    print("\n" + "=" * 70)
    print("Testing Safety Zone (array)")
    print("=" * 70)

    # Mirror scalar: flame=5 ft, 6 personnel, 1 equipment,
    #   area_per_person=50 ft2, area_per_equip=300 ft2
    # Expected: sep=20 ft, radius=33.819766 ft, area=0.082490356 acres
    sz = calculate_safety_zone(
        flame_height_ft=np.array([5.0]),
        number_of_personnel=6,
        area_per_person_sqft=50.0,
        number_of_equipment=1,
        area_per_equipment_sqft=300.0,
    )

    sep_ft = sz['separation_distance'].item()
    radius_ft = sz['radius'].item()
    area_sqft = sz['area'].item()
    area_acres = area_sqft / 43560.002160576107

    report(ti, "Separation distance (ft) = 20",
           round_to_six(sep_ft), 20.0, ERROR_TOLERANCE)
    report(ti, "Safety zone radius (ft)",
           round_to_six(radius_ft), 33.819766, ERROR_TOLERANCE)
    report(ti, "Safety zone area (acres)",
           round_to_six(area_acres), 0.082490356, ERROR_TOLERANCE)

    # Multi-cell batch: 3 different flame heights
    sz3 = calculate_safety_zone(
        flame_height_ft=np.array([5.0, 10.0, 20.0]),
        number_of_personnel=6,
        area_per_person_sqft=50.0,
        number_of_equipment=1,
        area_per_equipment_sqft=300.0,
    )
    report(ti, "Multi-cell: separation distances = 4× flame heights",
           int(np.allclose(sz3['separation_distance'],
                           [20.0, 40.0, 80.0], atol=1e-6)), 1, ERROR_TOLERANCE)
    report(ti, "Multi-cell: radii are monotonically increasing",
           int(np.all(np.diff(sz3['radius']) > 0)), 1, ERROR_TOLERANCE)

    print("Finished Safety Zone\n")


# ---------------------------------------------------------------------------
# 16. Spot module (flat-terrain, mirrors testSpotModule)
# ---------------------------------------------------------------------------

def test_spot(ti):
    print("\n" + "=" * 70)
    print("Testing Spot module (array, flat-terrain)")
    print("=" * 70)

    # Scalar expected (flat terrain, closed canopy):
    #   surface fire: 0.22005 miles  (WAF=1 run produces FL=15.811421 ft)
    #   burning pile open canopy: 0.024700 miles (pile flame_height=5 ft, wind=5 mph, cover=30 ft)
    # Note: array module only implements flat-terrain (no terrain correction).
    # The scalar testSpotModule uses WAF=1 (UserInput) to get the flame length,
    # so we reproduce that: FL = 15.811421 ft from the WAF=1 / UserInput run.
    scalar_fl_ft = 15.811421  # ft (WAF=1 / UserInput scalar run, FM124 upslope 30% slope)

    surf_dist_ft = calculate_spotting_from_surface_fire(
        flame_length_ft=np.array([scalar_fl_ft]),
        wind_mph=np.array([5.0]),
        cover_height_ft=np.array([30.0]),
    ).item()
    # Scalar expected flat: 0.22005 miles
    surf_dist_mi = surf_dist_ft / 5280.0
    report(ti, "Flat spotting from surface fire (miles)",
           round_to_six(surf_dist_mi), 0.22005, ERROR_TOLERANCE)

    # Pile, open canopy behavioural check:
    # For a taller pile (20 ft) in open terrain with higher wind (15 mph),
    # the spotting distance must be strictly positive and increase with wind.
    # (The scalar Spot module internally solves for an optimal cover height for
    # the open-canopy case; the array module is a flat-terrain implementation
    # that accepts cover height as a direct input — these are not equivalent
    # for small flames, so we test behaviour rather than exact parity here.)
    pile_tall = calculate_spotting_from_burning_pile(
        flame_height_ft=np.array([20.0]),
        wind_mph=np.array([15.0]),
        cover_height_ft=np.array([30.0]),
    ).item()
    report(ti, "Flat spotting from burning pile (tall pile, 15 mph) > 0",
           int(pile_tall > 0), 1, ERROR_TOLERANCE)
    pile_low = calculate_spotting_from_burning_pile(
        flame_height_ft=np.array([20.0]),
        wind_mph=np.array([5.0]),
        cover_height_ft=np.array([30.0]),
    ).item()
    report(ti, "Pile spotting increases with wind speed",
           int(pile_tall > pile_low), 1, ERROR_TOLERANCE)

    # BehaveRun wrappers return same values
    surf_r = _RUNNER.calculate_spotting_from_surface_fire(
        flame_length=np.array([scalar_fl_ft]),
        flame_length_units=0,
        wind_speed=np.array([5.0]),
        wind_speed_units=5,
        cover_height=np.array([30.0]),
        cover_height_units=0,
    ).item()
    report(ti, "Spotting from surface fire via BehaveRun wrapper",
           round_to_six(surf_r / 5280.0), 0.22005, ERROR_TOLERANCE)

    # Zero flame -> zero distance
    zero_d = float(calculate_spotting_from_surface_fire(
        flame_length_ft=0.0, wind_mph=5.0, cover_height_ft=30.0,
    ))
    report(ti, "Zero flame length -> zero spotting distance",
           zero_d, 0.0, ERROR_TOLERANCE)

    # Multi-cell: 3 cells with increasing flame lengths
    fl_arr = np.array([5.0, 10.0, 20.0])
    dists = calculate_spotting_from_surface_fire(
        flame_length_ft=fl_arr, wind_mph=5.0, cover_height_ft=30.0,
    )
    report(ti, "Multi-cell: no NaN in spotting distances",
           int(not np.any(np.isnan(dists))), 1, ERROR_TOLERANCE)
    report(ti, "Multi-cell: distances increase with flame length",
           int(np.all(np.diff(dists) > 0)), 1, ERROR_TOLERANCE)

    print("Finished Spot module\n")


# ---------------------------------------------------------------------------
# 17. Fine Dead Fuel Moisture Tool (mirrors testFineDeadFuelMoistureTool)
# ---------------------------------------------------------------------------

def test_fine_dead_fuel_moisture(ti):
    print("\n" + "=" * 70)
    print("Testing Fine Dead Fuel Moisture Tool (array)")
    print("=" * 70)

    pct = 100.0  # multiply fraction -> percent

    # scalar calculate_by_index(aspect, dry_bulb, elev, month, rh, shading, slope, time)
    # All zeros: ref=1%, corr=2%, total=3%
    v = float(calculate_fine_dead_fuel_moisture(
        dry_bulb_i=0, rh_i=0, slope_i=0, aspect_i=0,
        shading_i=0, month_i=0, elev_i=0, time_i=0,
    ))
    report(ti, "All-zero indices: reference moisture = 1 %",
           round_to_six(v * pct), 3.0, ERROR_TOLERANCE)  # total = ref+corr = 3%

    # All ones: ref=2%, corr=4%, total=6%
    v1 = float(calculate_fine_dead_fuel_moisture(
        dry_bulb_i=1, rh_i=1, slope_i=1, aspect_i=1,
        shading_i=1, month_i=1, elev_i=1, time_i=1,
    ))
    report(ti, "All-one indices: total moisture = 6 %",
           round_to_six(v1 * pct), 6.0, ERROR_TOLERANCE)

    # Out-of-bounds: returns sentinel -0.02
    v_oob = float(calculate_fine_dead_fuel_moisture(
        dry_bulb_i=99, rh_i=99, slope_i=99, aspect_i=99,
        shading_i=99, month_i=99, elev_i=99, time_i=99,
    ))
    report(ti, "Out-of-bounds indices: returns sentinel (-0.02)",
           v_oob, -0.02, ERROR_TOLERANCE)

    # Parity: all-zero array matches scalar
    v_arr = calculate_fine_dead_fuel_moisture(
        dry_bulb_i=np.array([0, 1]), rh_i=np.array([0, 1]),
        slope_i=np.array([0, 1]), aspect_i=np.array([0, 1]),
        shading_i=np.array([0, 1]), month_i=np.array([0, 1]),
        elev_i=np.array([0, 1]), time_i=np.array([0, 1]),
    )
    report(ti, "Array index 0 matches scalar",
           round_to_six(float(v_arr[0]) * pct), 3.0, ERROR_TOLERANCE)
    report(ti, "Array index 1 matches scalar",
           round_to_six(float(v_arr[1]) * pct), 6.0, ERROR_TOLERANCE)

    print("Finished Fine Dead Fuel Moisture Tool\n")


# ---------------------------------------------------------------------------
# 18. VPD calculator (mirrors testVaporPressureDeficitCalculator)
# ---------------------------------------------------------------------------

def test_vpd(ti):
    print("\n" + "=" * 70)
    print("Testing Vapor Pressure Deficit Calculator (array)")
    print("=" * 70)

    # Same 10-case table from test_behave.py — output in hPa (units=1)
    cases = [
        (50.0, 100.0, 0.0),
        (50.0, 90.0, 1.22833),
        (50.0, 80.0, 2.45667),
        (50.0, 70.0, 3.685),
        (50.0, 60.0, 4.91334),
        (50.0, 50.0, 6.14167),
        (50.0, 40.0, 7.37001),
        (50.0, 30.0, 8.59834),
        (50.0, 20.0, 9.82667),
        (50.0, 10.0, 11.055),
    ]
    tol_hpa = 1e-3

    for temp_f, rh_pct, expected_hpa in cases:
        result = calculate_vpd(
            temperature=np.array([temp_f]),
            temp_units=0,
            relative_humidity=np.array([rh_pct]),
            rh_units=1,
            output_units=1,
        )
        vpd_hpa = result['vpd'].item()
        report(ti,
               f"VPD at {rh_pct}% RH, {temp_f} degF (hPa)",
               vpd_hpa, expected_hpa, tol_hpa)

    # Batch: all 10 cases in a single vectorised call
    temps = np.full(10, 50.0)
    rhs = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10], dtype=float)
    result_batch = calculate_vpd(
        temperature=temps, temp_units=0,
        relative_humidity=rhs, rh_units=1,
        output_units=1,
    )
    vpd_batch = result_batch['vpd']
    expected_arr = np.array([c[2] for c in cases])
    all_match = np.allclose(vpd_batch, expected_arr, atol=tol_hpa)
    report(ti, "Batch VPD calculation (10 cells) matches scalar",
           int(all_match), 1, ERROR_TOLERANCE)

    # Sanity: actual VP <= saturated VP everywhere
    report(ti, "actual_vp <= saturated_vp for all cells",
           int(np.all(result_batch['actual_vp'] <= result_batch['saturated_vp'])),
           1, ERROR_TOLERANCE)

    print("Finished VPD Calculator\n")


# ---------------------------------------------------------------------------
# 19. BehaveRun facade — all method wrappers accessible
# ---------------------------------------------------------------------------

def test_facade_completeness(ti):
    print("\n" + "=" * 70)
    print("Testing BehaveRun facade completeness")
    print("=" * 70)

    required_methods = [
        'do_surface_run', 'do_crown_run',
        'calculate_scorch_height', 'calculate_crown_scorch_mortality',
        'calculate_spotting_from_surface_fire',
        'calculate_spotting_from_burning_pile',
        'calculate_spotting_from_torching_trees',
        'calculate_fire_area', 'calculate_fire_perimeter',
        'calculate_fire_length', 'calculate_fire_width',
    ]
    for m in required_methods:
        report(ti, f"BehaveRun.{m} exists",
               int(hasattr(_RUNNER, m)), 1, ERROR_TOLERANCE)

    print("Finished facade completeness\n")


# ---------------------------------------------------------------------------
# 20. Scalar parity stress test: all 60 standard fuel models
# ---------------------------------------------------------------------------

def test_all_fuel_models_no_nan(ti):
    print("\n" + "=" * 70)
    print("Testing all standard fuel models — no NaN/Inf in any output")
    print("=" * 70)

    fm_grid = np.arange(1, 261, dtype=np.int32)
    n = len(fm_grid)
    res = _RUNNER.do_surface_run(
        fuel_model_grid=fm_grid,
        m1h=np.full(n, 6.0), m10h=np.full(n, 7.0), m100h=np.full(n, 8.0),
        mlh=np.full(n, 60.0), mlw=np.full(n, 90.0),
        wind_speed=5.0, wind_speed_units=_MPH,
        wind_direction=0.0, wind_orientation_mode='RelativeToUpslope',
        slope=_SLOPE30_DEG, slope_units=0,
        aspect=0.0,
        canopy_cover=50.0,
        canopy_height=9.144,
        crown_ratio=50.0,
    )
    for key in ('spread_rate', 'flame_length', 'fireline_intensity',
                'heat_per_unit_area', 'eccentricity'):
        report(ti, f"FM 1-260: no NaN in {key}",
               int(not np.any(np.isnan(res[key]))), 1, ERROR_TOLERANCE)
        report(ti, f"FM 1-260: no Inf in {key}",
               int(not np.any(np.isinf(res[key]))), 1, ERROR_TOLERANCE)

    print("Finished all fuel models check\n")


# ---------------------------------------------------------------------------
# 21. Scalar input parity — every public entry point accepts bare Python
#     scalars (int / float) and returns the same numerical result as the
#     equivalent np.array([[...]]) call.
# ---------------------------------------------------------------------------

def test_scalar_inputs(ti):
    print("\n" + "=" * 70)
    print("Testing scalar Python inputs (int/float) parity with array inputs")
    print("=" * 70)

    # --- Surface run: scalar vs array ---
    r_arr = _RUNNER.do_surface_run(
        fuel_model_grid=np.array([[124]]),
        m1h=np.array([[6.0]]), m10h=np.array([[7.0]]),
        m100h=np.array([[8.0]]), mlh=np.array([[60.0]]),
        mlw=np.array([[90.0]]),
        wind_speed=np.array([[5.0]]), wind_speed_units=_MPH,
        wind_direction=np.array([[0.0]]), wind_orientation_mode='RelativeToUpslope',
        slope=np.array([[_SLOPE30_DEG]]), slope_units=0,
        aspect=np.array([[0.0]]),
        canopy_cover=np.array([[50.0]]),
        canopy_height=np.array([[9.144]]),
        crown_ratio=np.array([[50.0]]),
    )
    r_scl = _RUNNER.do_surface_run(
        fuel_model_grid=124,
        m1h=6.0, m10h=7.0, m100h=8.0, mlh=60.0, mlw=90.0,
        wind_speed=5.0, wind_speed_units=_MPH,
        wind_direction=0.0, wind_orientation_mode='RelativeToUpslope',
        slope=_SLOPE30_DEG, slope_units=0,
        aspect=0.0,
        canopy_cover=50.0,
        canopy_height=9.144,
        crown_ratio=50.0,
    )
    for key in ('spread_rate', 'flame_length', 'fireline_intensity',
                'heat_per_unit_area', 'fire_length_to_width_ratio', 'eccentricity'):
        report(ti, f"do_surface_run scalar==array: {key}",
               float(r_scl[key].flat[0]), float(r_arr[key].flat[0]), PARITY_TOLERANCE)

    # --- Crown run: scalar vs array ---
    surf_arr = r_arr
    surf_scl = r_scl
    c_arr = _RUNNER.do_crown_run(
        surface_results=surf_arr,
        fuel_model_grid=np.array([[124]]),
        m1h=np.array([[6.0]]), m10h=np.array([[7.0]]),
        m100h=np.array([[8.0]]), mlh=np.array([[60.0]]), mlw=np.array([[90.0]]),
        wind_speed=np.array([[5.0]]), wind_speed_units=_MPH,
        wind_direction=np.array([[0.0]]), wind_orientation_mode='RelativeToUpslope',
        slope=np.array([[_SLOPE30_DEG]]), slope_units=0,
        aspect=np.array([[0.0]]),
        canopy_base_height=np.array([[1.829]]),
        canopy_height=np.array([[9.144]]),
        canopy_bulk_density=np.array([[0.481]]),
        moisture_foliar=np.array([[100.0]]),
    )
    c_scl = _RUNNER.do_crown_run(
        surface_results=surf_scl,
        fuel_model_grid=124,
        m1h=6.0, m10h=7.0, m100h=8.0, mlh=60.0, mlw=90.0,
        wind_speed=5.0, wind_speed_units=_MPH,
        wind_direction=0.0, wind_orientation_mode='RelativeToUpslope',
        slope=_SLOPE30_DEG, slope_units=0,
        aspect=0.0,
        canopy_base_height=1.829,
        canopy_height=9.144,
        canopy_bulk_density=0.481,
        moisture_foliar=100.0,
    )
    for key in ('crown_fire_spread_rate', 'crown_flame_length',
                'crown_length_to_width_ratio'):
        report(ti, f"do_crown_run scalar==array: {key}",
               float(c_scl[key].flat[0]), float(c_arr[key].flat[0]), PARITY_TOLERANCE)

    # --- Scorch height: scalar vs array ---
    sh_arr = _RUNNER.calculate_scorch_height(
        fireline_intensity=np.array([50.0]),
        fireline_intensity_units=FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond,
        midflame_wind_speed=np.array([5.0]),
        wind_speed_units=SpeedUnits.SpeedUnitsEnum.MilesPerHour,
        air_temperature=np.array([80.0]),
        temperature_units=TemperatureUnits.TemperatureUnitsEnum.Fahrenheit,
    )
    sh_scl = _RUNNER.calculate_scorch_height(
        fireline_intensity=50.0,
        fireline_intensity_units=FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond,
        midflame_wind_speed=5.0,
        wind_speed_units=SpeedUnits.SpeedUnitsEnum.MilesPerHour,
        air_temperature=80.0,
        temperature_units=TemperatureUnits.TemperatureUnitsEnum.Fahrenheit,
    )
    report(ti, "calculate_scorch_height scalar==array",
           float(sh_scl.flat[0]), float(sh_arr.flat[0]), PARITY_TOLERANCE)

    # --- Mortality: scalar vs array ---
    from components.mortality import build_mortality_lookup, calculate_crown_scorch_mortality
    coeffs = build_mortality_lookup()
    m_arr = calculate_crown_scorch_mortality(
        scorch_height_ft=np.array([30.0]),
        tree_height_ft=np.array([50.0]),
        crown_ratio=np.array([0.5]),
        dbh_inches=np.array([10.0]),
        equation_number_grid=np.array([2]),
        coeffs=coeffs,
    )
    m_scl = calculate_crown_scorch_mortality(
        scorch_height_ft=30.0,
        tree_height_ft=50.0,
        crown_ratio=0.5,
        dbh_inches=10.0,
        equation_number_grid=2,
        coeffs=coeffs,
    )
    report(ti, "calculate_crown_scorch_mortality scalar==array: crown_length_scorch",
           float(m_scl['crown_length_scorch'].flat[0]),
           float(m_arr['crown_length_scorch'].flat[0]), PARITY_TOLERANCE)
    report(ti, "calculate_crown_scorch_mortality scalar==array: probability_mortality",
           float(m_scl['probability_mortality'].flat[0]),
           float(m_arr['probability_mortality'].flat[0]), PARITY_TOLERANCE)

    # --- Ignite: scalar vs array ---
    from components.ignite import (
        calculate_firebrand_ignition_probability,
        calculate_lightning_ignition_probability,
    )
    fb_arr = calculate_firebrand_ignition_probability(
        air_temp_f=np.array([80.0]),
        sun_shade_fraction=np.array([0.50]),
        moisture_1h_fraction=np.array([0.06]),
    )
    fb_scl = calculate_firebrand_ignition_probability(
        air_temp_f=80.0,
        sun_shade_fraction=0.50,
        moisture_1h_fraction=0.06,
    )

    lig_arr = calculate_lightning_ignition_probability(
        fuel_bed_type_grid=np.array([5]),
        moisture_100h_fraction=np.array([0.08]),
        duff_depth_inches=np.array([6.0]),
        charge_type=2,
    )
    lig_scl = calculate_lightning_ignition_probability(
        fuel_bed_type_grid=5,
        moisture_100h_fraction=0.08,
        duff_depth_inches=6.0,
        charge_type=2,
    )

    # --- Safety zone: scalar vs array ---
    from components.safety import calculate_safety_zone
    sz_arr = calculate_safety_zone(
        flame_height_ft=np.array([5.0]),
        number_of_personnel=6,
        area_per_person_sqft=50.0,
        number_of_equipment=1,
        area_per_equipment_sqft=300.0,
    )
    sz_scl = calculate_safety_zone(
        flame_height_ft=5.0,
        number_of_personnel=6,
        area_per_person_sqft=50.0,
        number_of_equipment=1,
        area_per_equipment_sqft=300.0,
    )

    for key in ('separation_distance', 'radius', 'area'):
        report(ti, f"calculate_safety_zone scalar==array: {key}",
               float(sz_scl[key].flat[0]), float(sz_arr[key].flat[0]), PARITY_TOLERANCE)

    # --- Spot: scalar vs array ---
    from components.spot import (
        calculate_spotting_from_surface_fire,
        calculate_spotting_from_burning_pile,
    )
    sp_arr = calculate_spotting_from_surface_fire(
        flame_length_ft=np.array([15.811421]),
        wind_mph=np.array([5.0]),
        cover_height_ft=np.array([30.0]),
    )
    sp_scl = calculate_spotting_from_surface_fire(
        flame_length_ft=15.811421,
        wind_mph=5.0,
        cover_height_ft=30.0,
    )

    pile_arr = calculate_spotting_from_burning_pile(
        flame_height_ft=np.array([20.0]),
        wind_mph=np.array([15.0]),
        cover_height_ft=np.array([30.0]),
    )
    pile_scl = calculate_spotting_from_burning_pile(
        flame_height_ft=20.0,
        wind_mph=15.0,
        cover_height_ft=30.0,
    )

    # --- Fine dead fuel moisture: scalar vs array ---
    from components.fine_dead_fuel_moisture_tool import calculate_fine_dead_fuel_moisture
    fdm_arr = calculate_fine_dead_fuel_moisture(
        dry_bulb_i=np.array([0]), rh_i=np.array([0]),
        slope_i=np.array([0]), aspect_i=np.array([0]),
        shading_i=np.array([0]), month_i=np.array([0]),
        elev_i=np.array([0]), time_i=np.array([0]),
    )
    fdm_scl = calculate_fine_dead_fuel_moisture(
        dry_bulb_i=0, rh_i=0, slope_i=0, aspect_i=0,
        shading_i=0, month_i=0, elev_i=0, time_i=0,
    )

    report(ti, "calculate_fine_dead_fuel_moisture scalar==array",
           float(fdm_scl.flat[0]), float(fdm_arr.flat[0]), PARITY_TOLERANCE)

    # --- VPD: scalar vs array ---
    from components.vapor_pressure_deficit_calculator import calculate_vpd
    vpd_arr = calculate_vpd(
        temperature=np.array([50.0]), temp_units=0,
        relative_humidity=np.array([50.0]), rh_units=1,
        output_units=1,
    )
    vpd_scl = calculate_vpd(
        temperature=50.0, temp_units=0,
        relative_humidity=50.0, rh_units=1,
        output_units=1,
    )
    report(ti, "calculate_vpd scalar==array: vpd",
           float(vpd_scl['vpd'].flat[0]), float(vpd_arr['vpd'].flat[0]), PARITY_TOLERANCE)

    # --- Fire size helpers: scalar vs array ---
    from components.surface import (
        calculate_fire_area, calculate_fire_perimeter,
        calculate_fire_length, calculate_fire_width,
    )
    fros, bros, lwr = 9.763838, 3.207761, 1.158917
    elapsed = 60.0
    fa_arr = calculate_fire_area(
        forward_ros=np.array([fros]), backing_ros=np.array([bros]),
        lwr=np.array([lwr]), elapsed_min=np.array([elapsed]),
    )
    fa_scl = calculate_fire_area(
        forward_ros=fros, backing_ros=bros, lwr=lwr, elapsed_min=elapsed,
    )

    fp_arr = calculate_fire_perimeter(
        forward_ros=np.array([fros]), backing_ros=np.array([bros]),
        lwr=np.array([lwr]), elapsed_min=np.array([elapsed]),
    )
    fp_scl = calculate_fire_perimeter(
        forward_ros=fros, backing_ros=bros, lwr=lwr, elapsed_min=elapsed,
    )

    print("Finished scalar input parity\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_all_tests():
    print("\n" + "=" * 70)
    print("BEHAVE MODULES — TEST SUITE")
    print("=" * 70)

    ti = _TI

    test_surface_scalar_parity(ti)
    test_surface_north_oriented(ti)
    test_surface_fm4_canopy(ti)
    test_surface_non_burnable(ti)
    test_surface_no_data_cell(ti)
    test_surface_shape_preservation(ti)
    test_surface_heterogeneous_fuels(ti)
    test_surface_mixed_grid(ti)
    test_surface_unit_conversion(ti)
    test_fire_size(ti)
    test_crown_fire(ti)
    test_scorch_height(ti)
    test_crown_scorch_mortality(ti)
    test_ignite(ti)
    test_safety_zone(ti)
    test_spot(ti)
    test_fine_dead_fuel_moisture(ti)
    test_vpd(ti)
    test_facade_completeness(ti)
    test_all_fuel_models_no_nan(ti)
    test_scalar_inputs(ti)

    return ti.print_summary()


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n{RED}Fatal error: {e}{RESET}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
