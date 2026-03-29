"""
Test suite for Behave fire behavior model - Python implementation.
Mirrors testBehave.cpp exactly - same test functions, same expected values, same order.
"""

import sys
import os
import math
from decimal import Decimal, ROUND_HALF_UP

script_dir = os.path.dirname(os.path.abspath(__file__))
behave_python_dir = os.path.dirname(script_dir)
if behave_python_dir not in sys.path:
    sys.path.insert(0, behave_python_dir)

from behave import BehaveRun
from components.fuel_models import FuelModels
from components.species_master_table import SpeciesMasterTable
from components.behave_units import (
    AreaUnits, LengthUnits, FractionUnits, SpeedUnits, SlopeUnits,
    DensityUnits, SurfaceAreaToVolumeUnits, HeatSourceAndReactionIntensityUnits,
    FirelineIntensityUnits, TemperatureUnits, TimeUnits
)

ERROR_TOLERANCE = 1e-06
RED   = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


class TestInfo:
    def __init__(self):
        self.num_total_tests = 0
        self.num_failed = 0
        self.num_passed = 0

    def print_summary(self):
        print("\n" + "="*70)
        print(f"Total tests performed: {self.num_total_tests}")
        color = GREEN if self.num_passed > 0 else ""
        print(f"{color}Total tests passed: {self.num_passed}{RESET if color else ''}")
        color = RED if self.num_failed > 0 else ""
        print(f"{color}Total tests failed: {self.num_failed}{RESET if color else ''}")
        print("="*70)
        return self.num_failed == 0


def are_close(observed, expected, epsilon):
    return abs(observed - expected) < epsilon


def round_to_six_decimal_places(value):
    d = Decimal(str(value))
    return float(d.quantize(Decimal('0.000001'), rounding=ROUND_HALF_UP))


def report_test_result(test_info, test_name, observed, expected, epsilon):
    test_info.num_total_tests += 1
    if are_close(observed, expected, epsilon):
        print(f"{test_name}")
        print(f"{GREEN}[PASSED]{RESET}\n")
        test_info.num_passed += 1
    else:
        print(f"{RED}{test_name}")
        print(f"[FAILED]  observed={observed}  expected={expected}  tolerance={epsilon}{RESET}\n")
        test_info.num_failed += 1


# ---------------------------------------------------------------------------
# Input helpers (mirror C++ setSurfaceInputs* / setCrownInputs* functions)
# ---------------------------------------------------------------------------

def set_surface_inputs_for_gs4_low_moisture_scenario(behave_run):
    """Mirrors C++ setSurfaceInputsForGS4LowMoistureScenario."""
    behave_run.surface.updateSurfaceInputs(
        124,
        6.0, 7.0, 8.0, 60.0, 90.0,
        FractionUnits.FractionUnitsEnum.Percent,
        5.0, SpeedUnits.SpeedUnitsEnum.MilesPerHour,
        'TwentyFoot',
        0,
        'RelativeToNorth',
        30.0, SlopeUnits.SlopeUnitsEnum.Percent,
        0,
        50, FractionUnits.FractionUnitsEnum.Percent,
        30.0, LengthUnits.LengthUnitsEnum.Feet,
        0.50, FractionUnits.FractionUnitsEnum.Fraction
    )


def set_crown_inputs_low_moisture_scenario(behave_run):
    """Mirrors C++ setCrownInputsLowMoistureScenario."""
    try:
        behave_run.crown.updateCrownInputs(
            124,
            6.0, 7.0, 8.0, 60.0, 90.0, 120.0,
            FractionUnits.FractionUnitsEnum.Percent,
            5.0, SpeedUnits.SpeedUnitsEnum.MilesPerHour,
            'TwentyFoot', 0, 'RelativeToNorth',
            30.0, SlopeUnits.SlopeUnitsEnum.Percent, 0,
            50, FractionUnits.FractionUnitsEnum.Percent,
            30.0, 6.0, LengthUnits.LengthUnitsEnum.Feet,
            0.50, FractionUnits.FractionUnitsEnum.Fraction,
            0.03, DensityUnits.DensityUnitsEnum.PoundsPerCubicFoot
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# testSurfaceSingleFuelModel
# ---------------------------------------------------------------------------

def test_surface_single_fuel_model(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Surface, single fuel model")
    print("="*70)

    mph = SpeedUnits.SpeedUnitsEnum.MilesPerHour
    twenty_ft = 'TwentyFoot'

    set_surface_inputs_for_gs4_low_moisture_scenario(behave_run)
    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_slope(30, SlopeUnits.SlopeUnitsEnum.Degrees)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_wind_speed(5, mph)
    behave_run.surface.set_wind_direction(45)
    behave_run.surface.set_aspect(95)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()

    report_test_result(test_info,
        "Test north oriented mode, 45 degree wind, 95 degree aspect, 5 mph 20 foot wind, 30 degree slope",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        19.677584, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test live moisture of extinction, 5 mph 20 foot upslope wind",
        round_to_six_decimal_places(behave_run.surface.getLiveFuelMoistureOfExtinction(FractionUnits.FractionUnitsEnum.Percent)),
        137.968551, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test characteristic live moisture, 5 mph 20 foot upslope wind",
        round_to_six_decimal_places(behave_run.surface.getCharacteristicMoistureByLifeState('Live', FractionUnits.FractionUnitsEnum.Percent)),
        85.874007, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test characteristic dead moisture, 5 mph 20 foot upslope wind",
        round_to_six_decimal_places(behave_run.surface.getCharacteristicMoistureByLifeState('Dead', FractionUnits.FractionUnitsEnum.Percent)),
        6.005463, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test characteristic SAVR for north oriented mode, 45 degree wind, 95 degree aspect, 5 mph 20 foot wind, 30 degree slope",
        round_to_six_decimal_places(behave_run.surface.getCharacteristicSAVR(SurfaceAreaToVolumeUnits.SurfaceAreaToVolumeUnitsEnum.SquareFeetOverCubicFeet)),
        1631.128734, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test heat source for north oriented mode, 45 degree wind, 95 degree aspect, 5 mph 20 foot wind, 30 degree slope",
        round_to_six_decimal_places(behave_run.surface.getHeatSource(HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.BtusPerSquareFootPerMinute)),
        5177.248579, ERROR_TOLERANCE)

    behave_run.surface.set_wind_direction(0)
    behave_run.surface.set_aspect(0)
    behave_run.surface.set_slope(0, SlopeUnits.SlopeUnitsEnum.Degrees)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test heat source for north oriented mode, 0 degree wind, 0 degree aspect, 5 mph 20 foot wind, 0 degree slope",
        round_to_six_decimal_places(behave_run.surface.getHeatSource(HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.BtusPerSquareFootPerMinute)),
        1164.267376, ERROR_TOLERANCE)

    behave_run.surface.set_fuel_model(124)
    behave_run.surface.set_wind_speed(5, mph)
    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToUpslope')
    behave_run.surface.set_wind_direction(0)
    behave_run.surface.set_slope(30, SlopeUnits.SlopeUnitsEnum.Percent)
    behave_run.surface.set_aspect(0)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()

    report_test_result(test_info,
        "Test upslope oriented mode, 5 mph 20 foot upslope wind",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        8.876216, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test backing spreadrate upslope oriented mode, 5 mph 20 foot upslope wind",
        behave_run.surface.getBackingSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour),
        2.91614659, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test flanking spreadrate upslope oriented mode, 5 mph 20 foot upslope wind",
        behave_run.surface.getFlankingSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour),
        5.08766627, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test spread distance upslope oriented mode, 5 mph 20 foot upslope wind",
        behave_run.surface.getSpreadDistance(LengthUnits.LengthUnitsEnum.Chains, 2.0, TimeUnits.TimeUnitsEnum.Hours),
        17.7524327, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test backing distance upslope oriented mode, 5 mph 20 foot upslope wind",
        behave_run.surface.getBackingSpreadDistance(LengthUnits.LengthUnitsEnum.Chains, 2.0, TimeUnits.TimeUnitsEnum.Hours),
        5.8322932, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test flanking distance upslope oriented mode, 5 mph 20 foot upslope wind",
        behave_run.surface.getFlankingSpreadDistance(LengthUnits.LengthUnitsEnum.Chains, 2.0, TimeUnits.TimeUnitsEnum.Hours),
        10.17533253, ERROR_TOLERANCE)

    set_surface_inputs_for_gs4_low_moisture_scenario(behave_run)
    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToUpslope')
    behave_run.surface.set_wind_speed(5, mph)
    behave_run.surface.set_wind_direction(90)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test upslope oriented mode, 5 mph 20 foot wind cross-slope left to right (90 degrees)",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        7.091665, ERROR_TOLERANCE)

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_wind_direction(0)
    behave_run.surface.set_aspect(0)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test north oriented mode, 20 foot North wind, zero aspect",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        8.876216, ERROR_TOLERANCE)

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_aspect(215)
    behave_run.surface.set_wind_direction(45)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test north oriented mode, 20 foot north-east wind (45 degree), 215 degree aspect",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        4.113265, ERROR_TOLERANCE)

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_aspect(5)
    behave_run.surface.set_wind_direction(45)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test north oriented mode, 20 foot 45 degree wind, 95 degree aspect",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        8.503960, ERROR_TOLERANCE)

    behave_run.surface.set_fuel_model(4)
    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_aspect(0)
    behave_run.surface.set_wind_direction(90)
    behave_run.surface.set_wind_speed(5, mph)
    behave_run.surface.set_slope(30, SlopeUnits.SlopeUnitsEnum.Degrees)
    behave_run.surface.set_canopy_cover(40, FractionUnits.FractionUnitsEnum.Percent)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test Fuel Model 4, north oriented mode, 20 foot 90 degree wind, 0 degree aspect, 40 percent canopy cover",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        46.631688, ERROR_TOLERANCE)

    behave_run.surface.set_fuel_model(91)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test Non-Burnable Fuel",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        0.0, ERROR_TOLERANCE)

    print("Finished testing Surface, single fuel model\n")


# ---------------------------------------------------------------------------
# testCalculateScorchHeight
# ---------------------------------------------------------------------------

def test_calculate_scorch_height(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Scorch Height")
    print("="*70)

    report_test_result(test_info,
        "Test calculate scorch height from 80 F air temperature, 5 mph wind and 50 Btu/ft/s fireline intensity",
        behave_run.mortality.calculate_scorch_height(
            50.0, FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond,
            5.0, SpeedUnits.SpeedUnitsEnum.MilesPerHour,
            80.0, TemperatureUnits.TemperatureUnitsEnum.Fahrenheit,
            LengthUnits.LengthUnitsEnum.Feet),
        7.617325, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test calculate scorch height from 70 F air temperature, 300 ft/min wind and 55 Btu/ft/s fireline intensity",
        behave_run.mortality.calculate_scorch_height(
            55.0, FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond,
            300.0, SpeedUnits.SpeedUnitsEnum.FeetPerMinute,
            70.0, TemperatureUnits.TemperatureUnitsEnum.Fahrenheit,
            LengthUnits.LengthUnitsEnum.Feet),
        9.923720, ERROR_TOLERANCE)


# ---------------------------------------------------------------------------
# testChaparral
# ---------------------------------------------------------------------------

def test_chaparral(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Chaparral")
    print("="*70)

    behave_run.surface.set_wind_orientation_mode('RelativeToUpslope')
    behave_run.surface.set_wind_direction(0.0)

    behave_run.surface.set_is_using_chaparral(True)
    behave_run.surface.set_chaparral_fuel_bed_depth(1.0, LengthUnits.LengthUnitsEnum.Feet)
    behave_run.surface.set_chaparral_fuel_type('NotSet')
    behave_run.surface.set_chaparral_fuel_load_input_mode('DirectFuelLoad')
    behave_run.surface.set_chaparral_fuel_dead_load_fraction(0.25)
    behave_run.surface.set_chaparral_total_fuel_load(0.333, 'PoundsPerSquareFoot')
    behave_run.surface.set_wind_speed(3.0, SpeedUnits.SpeedUnitsEnum.MilesPerHour)
    behave_run.surface.set_wind_height_input_mode('DirectMidflame')
    behave_run.surface.set_slope(0.0, SlopeUnits.SlopeUnitsEnum.Percent)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test Chaparral spread rate, direct fuel load input mode, depth 1 foot, 25% dead fuel, 0.333 lbs/ft^2, 3 mph wind, 0% slope",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        1.792546, ERROR_TOLERANCE)

    behave_run.surface.set_chaparral_fuel_bed_depth(2.0, LengthUnits.LengthUnitsEnum.Feet)
    behave_run.surface.set_chaparral_fuel_type('Chamise')
    behave_run.surface.set_chaparral_fuel_load_input_mode('FuelLoadFromDepthAndChaparralType')
    behave_run.surface.set_chaparral_fuel_dead_load_fraction(0.33)
    behave_run.surface.set_wind_height_input_mode('DirectMidflame')
    behave_run.surface.set_wind_speed(4.0, SpeedUnits.SpeedUnitsEnum.MilesPerHour)
    behave_run.surface.set_slope(10.0, SlopeUnits.SlopeUnitsEnum.Percent)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test Chaparral spread rate, fuel load from type chamise, depth 2 foot, 33% dead fuel, 4 mph wind, 10% slope",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        6.108717, ERROR_TOLERANCE)

    behave_run.surface.set_chaparral_fuel_bed_depth(3.0, LengthUnits.LengthUnitsEnum.Feet)
    behave_run.surface.set_chaparral_fuel_type('MixedBrush')
    behave_run.surface.set_chaparral_fuel_load_input_mode('FuelLoadFromDepthAndChaparralType')
    behave_run.surface.set_chaparral_fuel_dead_load_fraction(0.50)
    behave_run.surface.set_wind_height_input_mode('DirectMidflame')
    behave_run.surface.set_wind_speed(5.0, SpeedUnits.SpeedUnitsEnum.MilesPerHour)
    behave_run.surface.set_slope(20.0, SlopeUnits.SlopeUnitsEnum.Percent)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test Chaparral spread rate, fuel load from type mixed brush, depth 3 foot, 50% dead fuel, 5 mph wind, 20% slope",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        13.945025, ERROR_TOLERANCE)

    behave_run.surface.set_is_using_chaparral(False)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')


# ---------------------------------------------------------------------------
# testPalmettoGallberry
# ---------------------------------------------------------------------------

def test_palmetto_gallberry(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Palmetto-Gallberry")
    print("="*70)

    behave_run.surface.set_is_using_palmetto_gallberry(True)
    behave_run.surface.update_surface_inputs_for_palmetto_gallberry(
        6.0, 7.0, 8.0, 60.0, 90.0, FractionUnits.FractionUnitsEnum.Percent,
        5.0, SpeedUnits.SpeedUnitsEnum.MilesPerHour, 'TwentyFoot',
        0, 'RelativeToUpslope',
        10, 4, 50, 50, 'SquareFeetPerAcre',
        30.0, SlopeUnits.SlopeUnitsEnum.Percent, 0,
        50, FractionUnits.FractionUnitsEnum.Percent,
        30.0, LengthUnits.LengthUnitsEnum.Feet,
        0.50, FractionUnits.FractionUnitsEnum.Fraction)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test Palmetto-Gallberry spread rate",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        12.521131, ERROR_TOLERANCE)
    behave_run.surface.set_is_using_palmetto_gallberry(False)


# ---------------------------------------------------------------------------
# testWesternAspen
# ---------------------------------------------------------------------------

def test_western_aspen(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Western Aspen")
    print("="*70)

    behave_run.surface.set_is_using_western_aspen(True)
    behave_run.surface.update_surface_inputs_for_western_aspen(
        3, 50.0, FractionUnits.FractionUnitsEnum.Percent,
        'Low', 10.0, LengthUnits.LengthUnitsEnum.Inches,
        6.0, 7.0, 8.0, 60.0, 90.0, FractionUnits.FractionUnitsEnum.Percent,
        5.0, SpeedUnits.SpeedUnitsEnum.MilesPerHour, 'TwentyFoot',
        0, 'RelativeToUpslope',
        30.0, SlopeUnits.SlopeUnitsEnum.Percent, 0,
        50, FractionUnits.FractionUnitsEnum.Percent,
        30.0, LengthUnits.LengthUnitsEnum.Feet,
        0.50, FractionUnits.FractionUnitsEnum.Fraction)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test Western Aspen spread rate",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        0.847629, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test Western Aspen mortality",
        behave_run.surface.get_aspen_mortality(FractionUnits.FractionUnitsEnum.Fraction),
        0.267093, ERROR_TOLERANCE)

    behave_run.surface.set_is_using_western_aspen(False)


# ---------------------------------------------------------------------------
# testLengthToWidthRatio
# ---------------------------------------------------------------------------

def test_length_to_width_ratio(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing length-to-width-ratio")
    print("="*70)

    mph = SpeedUnits.SpeedUnitsEnum.MilesPerHour
    twenty_ft = 'TwentyFoot'
    set_surface_inputs_for_gs4_low_moisture_scenario(behave_run)

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_slope(0, SlopeUnits.SlopeUnitsEnum.Degrees)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_wind_speed(0, mph)
    behave_run.surface.set_wind_direction(0)
    behave_run.surface.set_aspect(0)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test length-to-width-ratio, north oriented mode, north wind, 0 mph 20 foot wind, 0 degree aspect, 0 degree slope",
        round_to_six_decimal_places(behave_run.surface.getFireLengthToWidthRatio()),
        1.0, ERROR_TOLERANCE)

    behave_run.surface.set_wind_height_input_mode('DirectMidflame')
    behave_run.surface.set_slope(0, SlopeUnits.SlopeUnitsEnum.Degrees)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_wind_speed(5, mph)
    behave_run.surface.set_wind_direction(0)
    behave_run.surface.set_aspect(0)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test length-to-width-ratio, north oriented mode, 0 degree wind, 5 mph midflame wind, 0 degree aspect, 0 degree slope",
        round_to_six_decimal_places(behave_run.surface.getFireLengthToWidthRatio()),
        1.590064, ERROR_TOLERANCE)

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_slope(30, SlopeUnits.SlopeUnitsEnum.Degrees)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_wind_speed(5, mph)
    behave_run.surface.set_wind_direction(45)
    behave_run.surface.set_aspect(95)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test length-to-width-ratio, north oriented mode, 45 degree wind, 5 mph 20 foot wind, 95 degree aspect, 30 degree slope",
        round_to_six_decimal_places(behave_run.surface.getFireLengthToWidthRatio()),
        1.375624, ERROR_TOLERANCE)

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_slope(30, SlopeUnits.SlopeUnitsEnum.Degrees)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_wind_speed(15, mph)
    behave_run.surface.set_wind_direction(45)
    behave_run.surface.set_aspect(95)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    report_test_result(test_info,
        "Test length-to-width-ratio, north oriented mode, 45 degree wind, 15 mph 20 foot wind, 95 degree aspect, 30 degree slope",
        round_to_six_decimal_places(behave_run.surface.getFireLengthToWidthRatio()),
        1.519936, ERROR_TOLERANCE)

    print("Finished testing length-to-width-ratio\n")


# ---------------------------------------------------------------------------
# testEllipticalDimensions
# ---------------------------------------------------------------------------

def test_elliptical_dimensions(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing elliptical dimensions")
    print("="*70)

    set_surface_inputs_for_gs4_low_moisture_scenario(behave_run)
    behave_run.surface.set_wind_orientation_mode('RelativeToUpslope')
    behave_run.surface.set_slope(0, SlopeUnits.SlopeUnitsEnum.Degrees)
    behave_run.surface.set_wind_speed(5, SpeedUnits.SpeedUnitsEnum.MilesPerHour)
    behave_run.surface.set_wind_height_input_mode('DirectMidflame')
    behave_run.surface.do_surface_run_in_direction_of_max_spread()

    elapsed = 3.5869124
    chains   = LengthUnits.LengthUnitsEnum.Chains
    minutes  = TimeUnits.TimeUnitsEnum.Minutes
    hours    = TimeUnits.TimeUnitsEnum.Hours

    report_test_result(test_info,
        "Test fire elliptical dimension a with 5 mph direct mid-flame, upslope mode, 3.5869124 minutes elapsed time",
        round_to_six_decimal_places(behave_run.surface.getEllipticalA(chains, elapsed, minutes)),
        0.628905, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test fire elliptical dimension b with 5 mph direct mid-flame, upslope mode, 3.5869124 minutes elapsed time",
        round_to_six_decimal_places(behave_run.surface.getEllipticalB(chains, elapsed, minutes)),
        1.0, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test fire elliptical dimension c with 5 mph direct mid-flame, upslope mode, 3.5869124 minutes elapsed time",
        round_to_six_decimal_places(behave_run.surface.getEllipticalC(chains, elapsed, minutes)),
        0.777482, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test fire heading to backing ratio with 5 mph direct mid-flame, upslope mode",
        round_to_six_decimal_places(behave_run.surface.getHeadingToBackingRatio()),
        7.988029, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test fire elliptical area in acres with direct mid-flame, upslope mode, 1 hour elapsed time",
        round_to_six_decimal_places(behave_run.surface.getFireArea(AreaUnits.AreaUnitsEnum.Acres, 1.0, hours)),
        55.283555, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test fire elliptical area in km^2 with direct mid-flame, upslope mode, 1 hour elapsed time",
        round_to_six_decimal_places(behave_run.surface.getFireArea(AreaUnits.AreaUnitsEnum.SquareKilometers, 1.0, hours)),
        0.223725, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test fire elliptical perimeter in chains with direct mid-flame, upslope mode, 1 hour elapsed time",
        round_to_six_decimal_places(behave_run.surface.getFirePerimeter(chains, 1.0, hours)),
        86.71476, ERROR_TOLERANCE)

    print("Finished testing elliptical dimensions\n")


# ---------------------------------------------------------------------------
# testDirectionOfInterest
# ---------------------------------------------------------------------------

def test_direction_of_interest(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing spread rate in direction of interest")
    print("="*70)

    set_surface_inputs_for_gs4_low_moisture_scenario(behave_run)
    from_perimeter = 'FromPerimeter'
    from_ignition  = 'FromIgnitionPoint'
    fpm   = SpeedUnits.SpeedUnitsEnum.FeetPerMinute
    chph  = SpeedUnits.SpeedUnitsEnum.ChainsPerHour
    twenty_ft = 'TwentyFoot'

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToUpslope')
    behave_run.surface.set_wind_direction(0)
    behave_run.surface.do_surface_run_in_direction_of_interest(90, from_perimeter)
    report_test_result(test_info,
        "Test perimeter spread direction mode upslope oriented mode, 20 foot wind, direction of interest 90 degrees from upslope, 45 degree wind",
        round_to_six_decimal_places(behave_run.surface.getSpreadRateInDirectionOfInterest(fpm)),
        5.596433, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test perimeter spread direction mode upslope oriented mode, direction of interest 90 degrees flame length",
        round_to_six_decimal_places(behave_run.surface.getFlameLengthInDirectionOfInterest(LengthUnits.LengthUnitsEnum.Feet)),
        6.598148, ERROR_TOLERANCE)

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToUpslope')
    behave_run.surface.set_wind_direction(290)
    behave_run.surface.do_surface_run_in_direction_of_interest(160, from_perimeter)
    report_test_result(test_info,
        "Test upslope oriented mode, 20 foot wind, direction of interest 160 degrees from upslope, 290 degree wind",
        round_to_six_decimal_places(behave_run.surface.getSpreadRateInDirectionOfInterest(chph)),
        2.766387, ERROR_TOLERANCE)

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToUpslope')
    behave_run.surface.set_wind_direction(215)
    behave_run.surface.do_surface_run_in_direction_of_interest(215, from_perimeter)
    report_test_result(test_info,
        "Test upslope oriented mode, 20 foot wind, direction of interest 215 degrees from upslope, 215 degree wind",
        round_to_six_decimal_places(behave_run.surface.getSpreadRateInDirectionOfInterest(chph)),
        2.818063, ERROR_TOLERANCE)

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_wind_direction(280)
    behave_run.surface.set_aspect(135)
    behave_run.surface.do_surface_run_in_direction_of_interest(30, from_ignition)
    report_test_result(test_info,
        "Test north oriented mode, 20 foot 135 degree wind, direction of interest 30 degrees from north, 263 degree aspect",
        round_to_six_decimal_places(behave_run.surface.getSpreadRateInDirectionOfInterest(chph)),
        4.180938, ERROR_TOLERANCE)

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_wind_direction(0)
    behave_run.surface.set_aspect(45)
    behave_run.surface.do_surface_run_in_direction_of_interest(90, from_ignition)
    report_test_result(test_info,
        "Test north oriented mode, 20 foot north wind, direction of interest 90 degrees from north, 45 degree aspect",
        round_to_six_decimal_places(behave_run.surface.getSpreadRateInDirectionOfInterest(chph)),
        3.438243, ERROR_TOLERANCE)

    behave_run.surface.set_wind_height_input_mode(twenty_ft)
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.set_wind_direction(280)
    behave_run.surface.set_aspect(263)
    behave_run.surface.do_surface_run_in_direction_of_interest(285, from_ignition)
    report_test_result(test_info,
        "Test north oriented mode, 20 foot 135 degree wind, direction of interest 285 degrees from north, 263 degree aspect",
        round_to_six_decimal_places(behave_run.surface.getSpreadRateInDirectionOfInterest(chph)),
        2.944975, ERROR_TOLERANCE)

    print("Finished testing spread rate in direction of interest\n")


# ---------------------------------------------------------------------------
# testFirelineIntensity
# ---------------------------------------------------------------------------

def test_fireline_intensity(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing fireline intensity")
    print("="*70)

    set_surface_inputs_for_gs4_low_moisture_scenario(behave_run)
    behave_run.surface.set_wind_height_input_mode('TwentyFoot')
    behave_run.surface.set_wind_orientation_mode('RelativeToUpslope')
    behave_run.surface.do_surface_run_in_direction_of_max_spread()

    report_test_result(test_info,
        "Test upslope oriented mode, 20 foot upslope wind (Btu/ft/s)",
        behave_run.surface.getFirelineIntensity(FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond),
        598.339039, ERROR_TOLERANCE)

    report_test_result(test_info,
        "Test upslope oriented mode, 20 foot upslope wind (kW/m)",
        behave_run.surface.getFirelineIntensity(FirelineIntensityUnits.FirelineIntensityUnitsEnum.KilowattsPerMeter),
        2072.730450, ERROR_TOLERANCE)

    print("Finished testing fireline intensity\n")


# ---------------------------------------------------------------------------
# testTwoFuelModels
# ---------------------------------------------------------------------------

def test_two_fuel_models(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Two Fuel Models, first fuel model 1, second fuel model 124")
    print("="*70)

    behave_run.surface.set_wind_height_input_mode('TwentyFoot')
    behave_run.surface.update_surface_inputs_for_two_fuel_models(
        1, 124,
        6.0, 7.0, 8.0, 60.0, 90.0, FractionUnits.FractionUnitsEnum.Percent,
        5.0, SpeedUnits.SpeedUnitsEnum.MilesPerHour, 'TwentyFoot',
        0, 'RelativeToNorth',
        0, FractionUnits.FractionUnitsEnum.Percent, 'TwoDimensional',
        30.0, SlopeUnits.SlopeUnitsEnum.Percent, 0,
        50, FractionUnits.FractionUnitsEnum.Percent,
        30.0, LengthUnits.LengthUnitsEnum.Feet,
        0.50, FractionUnits.FractionUnitsEnum.Fraction)

    pct  = FractionUnits.FractionUnitsEnum.Percent
    chph = SpeedUnits.SpeedUnitsEnum.ChainsPerHour
    cases = [
        (0,   8.876216),
        (10,  10.470801),
        (20,  12.189713),
        (30,  13.958900),
        (40,  15.706408),
        (50,  17.362382),
        (60,  18.859066),
        (70,  20.130802),
        (80,  21.114030),
        (90,  21.747289),
        (100, 21.971217),
    ]
    for coverage, expected in cases:
        behave_run.surface.set_two_fuel_models_first_fuel_model_coverage(coverage, pct)
        behave_run.surface.do_surface_run_in_direction_of_max_spread()
        report_test_result(test_info,
            f"First fuel model coverage {coverage}",
            round_to_six_decimal_places(behave_run.surface.getSpreadRate(chph)),
            expected, ERROR_TOLERANCE)

    print("Finished testing Two Fuel Models\n")


# ---------------------------------------------------------------------------
# testCrownModuleRothermel
# ---------------------------------------------------------------------------

def test_crown_module_rothermel(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Crown module, Rothermel")
    print("="*70)

    mph = SpeedUnits.SpeedUnitsEnum.MilesPerHour
    behave_run.crown.set_wind_height_input_mode('TwentyFoot')

    set_crown_inputs_low_moisture_scenario(behave_run)
    behave_run.crown.do_crown_run_rothermel()

    report_test_result(test_info, "Test crown Rothermel fire spread rate",
        round_to_six_decimal_places(behave_run.crown.get_crown_fire_spread_rate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        10.259921, ERROR_TOLERANCE)

    report_test_result(test_info, "Test crown fire Rothermel length-to-width ratio",
        round_to_six_decimal_places(behave_run.crown.get_crown_fire_length_to_width_ratio()),
        1.625, ERROR_TOLERANCE)

    report_test_result(test_info, "Test crown fire Rothermel area",
        round_to_six_decimal_places(behave_run.crown.get_crown_fire_area(AreaUnits.AreaUnitsEnum.Acres, 1.0, TimeUnits.TimeUnitsEnum.Hours)),
        5.087736, ERROR_TOLERANCE)

    report_test_result(test_info, "Test crown fire Rothermel perimeter",
        round_to_six_decimal_places(behave_run.crown.get_crown_fire_perimeter(LengthUnits.LengthUnitsEnum.Chains, 1.0, TimeUnits.TimeUnitsEnum.Hours)),
        26.033937, ERROR_TOLERANCE)

    report_test_result(test_info, "Test crown fire Rothermel flame length",
        round_to_six_decimal_places(behave_run.crown.get_crown_flame_length(LengthUnits.LengthUnitsEnum.Feet)),
        29.320557, ERROR_TOLERANCE)

    report_test_result(test_info, "Test crown Rothermel fireline intensity",
        round_to_six_decimal_places(behave_run.crown.get_crown_fireline_intensity(FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond)),
        1775.061222, ERROR_TOLERANCE)

    set_crown_inputs_low_moisture_scenario(behave_run)
    behave_run.crown.set_moisture_one_hour(20, FractionUnits.FractionUnitsEnum.Percent)
    behave_run.crown.do_crown_run_rothermel()
    report_test_result(test_info, "Test fire type Rothermel, Surface fire expected",
        behave_run.crown.get_fire_type(), 0, ERROR_TOLERANCE)

    set_crown_inputs_low_moisture_scenario(behave_run)
    behave_run.crown.do_crown_run_rothermel()
    report_test_result(test_info, "Test fire type Rothermel, Torching fire expected",
        behave_run.crown.get_fire_type(), 1, ERROR_TOLERANCE)

    set_crown_inputs_low_moisture_scenario(behave_run)
    behave_run.crown.set_wind_speed(10, mph)
    behave_run.crown.do_crown_run_rothermel()
    report_test_result(test_info, "Test fire type Rothermel, Crowning fire expected",
        behave_run.crown.get_fire_type(), 2, ERROR_TOLERANCE)

    set_crown_inputs_low_moisture_scenario(behave_run)
    behave_run.crown.set_canopy_height(60, LengthUnits.LengthUnitsEnum.Feet)
    behave_run.crown.set_canopy_base_height(30, LengthUnits.LengthUnitsEnum.Feet)
    behave_run.crown.set_canopy_bulk_density(0.06, DensityUnits.DensityUnitsEnum.PoundsPerCubicFoot)
    behave_run.crown.set_wind_speed(5, mph)
    behave_run.crown.do_crown_run_rothermel()
    report_test_result(test_info, "Test fire type Rothermel, Conditional crown fire expected",
        behave_run.crown.get_fire_type(), 3, ERROR_TOLERANCE)

    print("Finished testing Crown module, Rothermel\n")


# ---------------------------------------------------------------------------
# testCrownModuleScottAndReinhardt
# ---------------------------------------------------------------------------

def test_crown_module_scott_and_reinhardt(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Crown module, Scott and Reinhardt")
    print("="*70)

    behave_run.crown.set_wind_adjustment_factor_calculation_method('UserInput')
    behave_run.crown.set_user_provided_wind_adjustment_factor(0.4)
    behave_run.crown.update_crown_inputs(
        10, 8.0, 9.0, 10.0, 0.0, 117.0, 100.0, FractionUnits.FractionUnitsEnum.Percent,
        2187.226624, SpeedUnits.SpeedUnitsEnum.FeetPerMinute, 'TwentyFoot',
        0, 'RelativeToUpslope',
        20, SlopeUnits.SlopeUnitsEnum.Percent, 0,
        50, FractionUnits.FractionUnitsEnum.Percent,
        38.104626, 2.952756, LengthUnits.LengthUnitsEnum.Feet,
        0.50, FractionUnits.FractionUnitsEnum.Fraction,
        0.01311, DensityUnits.DensityUnitsEnum.PoundsPerCubicFoot)

    behave_run.crown.do_crown_run_scott_and_reinhardt()

    report_test_result(test_info, "Test Scott and Reinhardt crown fire spread rate",
        round_to_six_decimal_places(behave_run.crown.get_final_spread_rate(SpeedUnits.SpeedUnitsEnum.FeetPerMinute)),
        65.221842, ERROR_TOLERANCE)

    report_test_result(test_info, "Test Scott and Reinhardt crown flame length",
        round_to_six_decimal_places(behave_run.crown.get_final_flame_length(LengthUnits.LengthUnitsEnum.Feet)),
        60.744542, ERROR_TOLERANCE)

    report_test_result(test_info, "Test Scott and Reinhardt crown fireline intensity",
        round_to_six_decimal_places(behave_run.crown.get_final_fireline_intensity(FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond)),
        5293.170672, ERROR_TOLERANCE)

    report_test_result(test_info, "Test Scott and Reinhardt crown fire critical open wind speed",
        behave_run.crown.get_critical_open_wind_speed(SpeedUnits.SpeedUnitsEnum.FeetPerMinute),
        1717.916785, ERROR_TOLERANCE)

    report_test_result(test_info, "Test Scott and Reinhardt crown fire type, crown fire expected",
        behave_run.crown.get_fire_type(), 2, ERROR_TOLERANCE)

    behave_run.crown.set_moisture_input_mode('AllAggregate')
    behave_run.crown.set_moisture_dead_aggregate(9.0, FractionUnits.FractionUnitsEnum.Percent)
    behave_run.crown.set_moisture_live_aggregate(100.0, FractionUnits.FractionUnitsEnum.Percent)
    behave_run.crown.set_wind_adjustment_factor_calculation_method('UseCrownRatio')
    behave_run.crown.set_wind_speed(2187.2266239, SpeedUnits.SpeedUnitsEnum.FeetPerMinute)
    behave_run.crown.do_crown_run_scott_and_reinhardt()
    report_test_result(test_info, "Test Scott and Reinhardt crown fire spread rate with aggregate moisture",
        round_to_six_decimal_places(behave_run.crown.get_final_spread_rate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        64.016394, ERROR_TOLERANCE)

    behave_run.crown.set_moisture_input_mode('MoistureScenario')
    behave_run.crown.set_current_moisture_scenario_by_name('D3L2')
    behave_run.crown.do_crown_run_scott_and_reinhardt()
    report_test_result(test_info, "Test Scott and Reinhardt crown fire spread rate with moisture scenario D3L2",
        round_to_six_decimal_places(behave_run.crown.get_final_spread_rate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        68.334996, ERROR_TOLERANCE)

    behave_run.crown.set_moisture_input_mode('BySizeClass')
    behave_run.crown.set_wind_adjustment_factor_calculation_method('UserInput')
    behave_run.crown.set_user_provided_wind_adjustment_factor(0.15)
    behave_run.crown.update_crown_inputs(
        5, 5.0, 6.0, 8.0, 0.0, 117.0, 100.0, FractionUnits.FractionUnitsEnum.Percent,
        24.854848, SpeedUnits.SpeedUnitsEnum.MilesPerHour, 'TwentyFoot',
        0, 'RelativeToUpslope',
        20, SlopeUnits.SlopeUnitsEnum.Percent, 0,
        50, FractionUnits.FractionUnitsEnum.Percent,
        71.631562, 4.92126, LengthUnits.LengthUnitsEnum.Feet,
        0.50, FractionUnits.FractionUnitsEnum.Fraction,
        0.003746, DensityUnits.DensityUnitsEnum.PoundsPerCubicFoot)

    behave_run.crown.do_crown_run_scott_and_reinhardt()

    report_test_result(test_info, "Test crown fire spread rate",
        round_to_six_decimal_places(behave_run.crown.get_final_spread_rate(SpeedUnits.SpeedUnitsEnum.FeetPerMinute)),
        29.475388, ERROR_TOLERANCE)

    report_test_result(test_info, "Test crown flame length",
        round_to_six_decimal_places(behave_run.crown.get_final_flame_length(LengthUnits.LengthUnitsEnum.Feet)),
        12.759447, ERROR_TOLERANCE)

    report_test_result(test_info, "Test crown fireline intensity",
        round_to_six_decimal_places(behave_run.crown.get_final_fireline_intensity(FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond)),
        509.568753, ERROR_TOLERANCE)

    report_test_result(test_info, "Test crown fire critical open wind speed",
        behave_run.crown.get_critical_open_wind_speed(SpeedUnits.SpeedUnitsEnum.FeetPerMinute),
        3874.421988, ERROR_TOLERANCE)

    report_test_result(test_info, "Test crown fire type, torching fire expected",
        behave_run.crown.get_fire_type(), 1, ERROR_TOLERANCE)

    behave_run.surface.set_wind_adjustment_factor_calculation_method('UseCrownRatio')
    print("Finished testing Crown module, Scott and Reinhardt\n")


# ---------------------------------------------------------------------------
# testSpotModule
# ---------------------------------------------------------------------------

def test_spot_module(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Spot module")
    print("="*70)

    set_surface_inputs_for_gs4_low_moisture_scenario(behave_run)
    behave_run.surface.set_wind_height_input_mode('TwentyFoot')
    behave_run.surface.set_user_provided_wind_adjustment_factor(1.0)
    behave_run.surface.set_wind_adjustment_factor_calculation_method('UserInput')
    behave_run.surface.do_surface_run_in_direction_of_max_spread()
    flame_length = behave_run.surface.getFlameLength(LengthUnits.LengthUnitsEnum.Feet)

    location = 'RidgeTop'
    rtv_dist = 1.0
    rtv_elev = 2000.0
    cover_ht = 30.0
    pile_flame = 5.0
    wind_20ft = 5.0
    mph   = SpeedUnits.SpeedUnitsEnum.MilesPerHour
    miles = LengthUnits.LengthUnitsEnum.Miles
    feet  = LengthUnits.LengthUnitsEnum.Feet

    behave_run.spot.update_spot_inputs_for_burning_pile(
        location, rtv_dist, miles, rtv_elev, feet, cover_ht, feet, 'Closed',
        pile_flame, feet, wind_20ft, mph)
    behave_run.spot.calculate_spotting_distance_from_burning_pile()
    report_test_result(test_info, "Test mountain spotting distance from burning pile, closed downwind canopy",
        round_to_six_decimal_places(behave_run.spot.get_max_mountainous_terrain_spotting_distance_from_burning_pile(miles)),
        0.021330, ERROR_TOLERANCE)
    report_test_result(test_info, "Test flat spotting distance from burning pile, closed downwind canopy",
        round_to_six_decimal_places(behave_run.spot.get_max_flat_terrain_spotting_distance_from_burning_pile(miles)),
        0.017067, ERROR_TOLERANCE)

    behave_run.spot.update_spot_inputs_for_burning_pile(
        location, rtv_dist, miles, rtv_elev, feet, cover_ht, feet, 'Open',
        pile_flame, feet, wind_20ft, mph)
    behave_run.spot.calculate_spotting_distance_from_burning_pile()
    report_test_result(test_info, "Test mountain spotting distance from burning pile, open downwind canopy",
        round_to_six_decimal_places(behave_run.spot.get_max_mountainous_terrain_spotting_distance_from_burning_pile(miles)),
        0.030863, ERROR_TOLERANCE)
    report_test_result(test_info, "Test flat spotting distance from burning pile, open downwind canopy",
        round_to_six_decimal_places(behave_run.spot.get_max_flat_terrain_spotting_distance_from_burning_pile(miles)),
        0.024700, ERROR_TOLERANCE)

    behave_run.spot.update_spot_inputs_for_surface_fire(
        location, rtv_dist, miles, rtv_elev, feet, cover_ht, feet, 'Closed',
        wind_20ft, mph, flame_length, feet)
    behave_run.spot.calculate_spotting_distance_from_surface_fire()
    report_test_result(test_info, "Test mountain spotting distance from surface fire, closed downwind canopy",
        round_to_six_decimal_places(behave_run.spot.get_max_mountainous_terrain_spotting_distance_from_surface_fire(miles)),
        0.267467, ERROR_TOLERANCE)
    report_test_result(test_info, "Test flat spotting distance from surface fire, closed downwind canopy",
        round_to_six_decimal_places(behave_run.spot.get_max_flat_terrain_spotting_distance_from_surface_fire(miles)),
        0.22005, ERROR_TOLERANCE)

    behave_run.spot.update_spot_inputs_for_torching_trees(
        location, rtv_dist, miles, rtv_elev, feet, cover_ht, feet, 'Closed',
        15, 20.0, LengthUnits.LengthUnitsEnum.Inches,
        30.0, feet, 'ENGELMANN_SPRUCE', wind_20ft, mph)
    behave_run.spot.calculate_spotting_distance_from_torching_trees()
    report_test_result(test_info, "Test mountain spotting distance from torching trees, closed downwind canopy",
        round_to_six_decimal_places(behave_run.spot.get_max_mountainous_terrain_spotting_distance_from_torching_trees(miles)),
        0.222396, ERROR_TOLERANCE)
    report_test_result(test_info, "Test flat spotting distance from torching trees, closed downwind canopy",
        round_to_six_decimal_places(behave_run.spot.get_max_flat_terrain_spotting_distance_from_torching_trees(miles)),
        0.181449, ERROR_TOLERANCE)

    behave_run.spot.update_spot_inputs_for_active_crown_fire(
        location, rtv_dist, miles, rtv_elev, feet, 30.0, feet, 'Closed',
        wind_20ft, mph, 20.0, feet)
    behave_run.spot.calculate_spotting_distance_from_active_crown()
    report_test_result(test_info, "Test mountain spotting distance from active crown fire, closed downwind canopy",
        round_to_six_decimal_places(behave_run.spot.get_max_mountainous_terrain_spotting_distance_from_active_crown(miles)),
        0.400473, ERROR_TOLERANCE)

    print("Finished testing Spot module\n")


# ---------------------------------------------------------------------------
# testSpeedUnitConversion
# ---------------------------------------------------------------------------

def test_speed_unit_conversion(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing speed unit conversion")
    print("="*70)

    behave_run.surface.set_wind_adjustment_factor_calculation_method('UseCrownRatio')
    set_surface_inputs_for_gs4_low_moisture_scenario(behave_run)
    behave_run.surface.set_fuel_model(124)
    behave_run.surface.set_wind_speed(5, SpeedUnits.SpeedUnitsEnum.MilesPerHour)
    behave_run.surface.set_wind_height_input_mode('TwentyFoot')
    behave_run.surface.set_wind_orientation_mode('RelativeToUpslope')
    behave_run.surface.set_wind_direction(0)
    behave_run.surface.set_slope(30, SlopeUnits.SlopeUnitsEnum.Percent)
    behave_run.surface.set_aspect(0)
    behave_run.surface.do_surface_run_in_direction_of_max_spread()

    cases = [
        ("chains per hour",     SpeedUnits.SpeedUnitsEnum.ChainsPerHour,     8.876216),
        ("feet per minute",     SpeedUnits.SpeedUnitsEnum.FeetPerMinute,      9.763838),
        ("kilometers per hour", SpeedUnits.SpeedUnitsEnum.KilometersPerHour,  0.178561),
        ("meters per minute",   SpeedUnits.SpeedUnitsEnum.MetersPerMinute,    2.976018),
        ("meters per second",   SpeedUnits.SpeedUnitsEnum.MetersPerSecond,    0.049600),
        ("miles per hour",      SpeedUnits.SpeedUnitsEnum.MilesPerHour,       0.110953),
    ]
    for label, units, expected in cases:
        report_test_result(test_info, f"Test surface spread rate in {label}",
            round_to_six_decimal_places(behave_run.surface.getSpreadRate(units)),
            expected, ERROR_TOLERANCE)

    print("Finished testing speed unit conversion\n")


# ---------------------------------------------------------------------------
# testIgniteModule
# ---------------------------------------------------------------------------

def test_ignite_module(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Ignite module")
    print("="*70)

    pct  = FractionUnits.FractionUnitsEnum.Percent
    frac = FractionUnits.FractionUnitsEnum.Fraction

    behave_run.ignite.update_ignite_inputs(
        6.0, 8.0, pct,
        80.0, TemperatureUnits.TemperatureUnitsEnum.Fahrenheit,
        50.0, pct,
        'DouglasFirDuff', 6.0, LengthUnits.LengthUnitsEnum.Inches, 'Unknown')

    report_test_result(test_info, "Test firebrand ignition probability for Douglas fir duff",
        behave_run.ignite.calculate_firebrand_ignition_probability(frac),
        0.54831705, ERROR_TOLERANCE)

    report_test_result(test_info, "Test lightning ignition probability for Douglas fir duff",
        behave_run.ignite.calculate_lightning_ignition_probability(frac),
        0.39362018, ERROR_TOLERANCE)

    behave_run.ignite.update_ignite_inputs(
        7.0, 9.0, pct,
        90.0, TemperatureUnits.TemperatureUnitsEnum.Fahrenheit,
        25.0, pct,
        'LodgepolePineDuff', 8.0, LengthUnits.LengthUnitsEnum.Inches, 'Negative')

    report_test_result(test_info, "Test firebrand ignition probability for Lodgepole pine duff",
        behave_run.ignite.calculate_firebrand_ignition_probability(pct),
        50.717573, ERROR_TOLERANCE)

    report_test_result(test_info, "Test lightning ignition probability for Lodgepole pine duff",
        behave_run.ignite.calculate_lightning_ignition_probability(pct),
        17.931991, ERROR_TOLERANCE)

    print("Finished testing Ignite module\n")


# ---------------------------------------------------------------------------
# testSafetyModule
# ---------------------------------------------------------------------------

def test_safety_module(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Safety module")
    print("="*70)

    feet = LengthUnits.LengthUnitsEnum.Feet
    sqft = AreaUnits.AreaUnitsEnum.SquareFeet
    acres = AreaUnits.AreaUnitsEnum.Acres

    behave_run.safety.update_safety_inputs(
        5.0, feet, 6, 1, 50.0, 300.0, sqft)
    behave_run.safety.calculate_safety_zone()

    report_test_result(test_info, "Test separation distance",
        behave_run.safety.get_separation_distance(feet), 20.0, ERROR_TOLERANCE)

    report_test_result(test_info, "Test safety zone area",
        behave_run.safety.get_safety_zone_area(acres), 0.082490356, ERROR_TOLERANCE)

    report_test_result(test_info, "Test safety zone radius",
        behave_run.safety.get_safety_zone_radius(feet), 33.819766, ERROR_TOLERANCE)

    print("Finished testing Safety module\n")


# ---------------------------------------------------------------------------
# testContainModule
# ---------------------------------------------------------------------------

def test_contain_module(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Contain module")
    print("="*70)

    chph   = SpeedUnits.SpeedUnitsEnum.ChainsPerHour
    chains = LengthUnits.LengthUnitsEnum.Chains
    acres  = AreaUnits.AreaUnitsEnum.Acres
    minutes = TimeUnits.TimeUnitsEnum.Minutes
    hours   = TimeUnits.TimeUnitsEnum.Hours

    behave_run.contain.set_attack_distance(0, chains)
    behave_run.contain.set_lw_ratio(3)
    behave_run.contain.set_report_rate(5, chph)
    behave_run.contain.set_report_size(1, acres)
    behave_run.contain.set_tactic('HeadAttack')
    behave_run.contain.add_resource(2, 8, hours, 20, chph, "test")
    behave_run.contain.do_contain_run()

    report_test_result(test_info, "Test final fire line length",
        behave_run.contain.get_final_fire_line_length(chains), 39.539849615, ERROR_TOLERANCE)
    report_test_result(test_info, "Test observed perimeter at initial attack",
        behave_run.contain.get_perimeter_at_initial_attack(chains), 37.51917991, ERROR_TOLERANCE)
    report_test_result(test_info, "Test observed perimeter at containment",
        behave_run.contain.get_perimeter_at_containment(chains), 39.539849615, ERROR_TOLERANCE)
    report_test_result(test_info, "Test observed fire size at initial attack",
        behave_run.contain.get_fire_size_at_initial_attack(acres), 8.954501709, ERROR_TOLERANCE)
    report_test_result(test_info, "Test observed final fire size",
        behave_run.contain.get_final_fire_size(acres), 9.42749714, ERROR_TOLERANCE)
    report_test_result(test_info, "Test observed final containment area",
        behave_run.contain.get_final_containment_area(acres), 9.42749714, ERROR_TOLERANCE)
    report_test_result(test_info, "Test observed final time since report",
        behave_run.contain.get_final_time_since_report(minutes), 238.75, ERROR_TOLERANCE)
    report_test_result(test_info, "Test observed containment status",
        behave_run.contain.get_containment_status(), 1, ERROR_TOLERANCE)

    print("Finished testing Contain module\n")


# ---------------------------------------------------------------------------
# testFineDeadFuelMoistureTool
# ---------------------------------------------------------------------------

def test_fine_dead_fuel_moisture_tool(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Fine Dead Fuel Moisture Tool")
    print("="*70)

    pct  = FractionUnits.FractionUnitsEnum.Percent
    tool = behave_run.fine_dead_fuel_moisture_tool

    tool.calculate_by_index(0, 0, 0, 0, 0, 0, 0, 0)
    report_test_result(test_info, "Test reference moisture for all zero index values",
        tool.get_reference_moisture(pct), 1, ERROR_TOLERANCE)
    report_test_result(test_info, "Test correction moisture for all zero index values",
        tool.get_correction_moisture(pct), 2, ERROR_TOLERANCE)
    report_test_result(test_info, "Test fine dead fuel moisture for all zero index values",
        tool.get_fine_dead_fuel_moisture(pct), 3, ERROR_TOLERANCE)

    tool.calculate_by_index(1, 1, 1, 1, 1, 1, 1, 1)
    report_test_result(test_info, "Test reference moisture for all one index values",
        tool.get_reference_moisture(pct), 2, ERROR_TOLERANCE)
    report_test_result(test_info, "Test correction moisture for all one index values",
        tool.get_correction_moisture(pct), 4, ERROR_TOLERANCE)
    report_test_result(test_info, "Test fine dead fuel moisture for all one index values",
        tool.get_fine_dead_fuel_moisture(pct), 6, ERROR_TOLERANCE)

    max_a  = tool.get_aspect_index_size() - 1
    max_d  = tool.get_dry_bulb_temperature_index_size() - 1
    max_e  = tool.get_elevation_index_size() - 1
    max_m  = tool.get_month_index_size() - 1
    max_r  = tool.get_relative_humidity_index_size() - 1
    max_s  = tool.get_shading_index_size() - 1
    max_sl = tool.get_slope_index_size() - 1
    max_t  = tool.get_time_of_day_index_size() - 1
    tool.calculate_by_index(max_a, max_d, max_e, max_m, max_r, max_s, max_sl, max_t)
    report_test_result(test_info, "Test reference moisture for all max index values",
        tool.get_reference_moisture(pct), 12, ERROR_TOLERANCE)
    report_test_result(test_info, "Test correction moisture for all max index values",
        tool.get_correction_moisture(pct), 6, ERROR_TOLERANCE)
    report_test_result(test_info, "Test fine dead fuel moisture for max one index values",
        tool.get_fine_dead_fuel_moisture(pct), 18, ERROR_TOLERANCE)

    tool.calculate_by_index(max_a+1, max_d+1, max_e+1, max_m+1, max_r+1, max_s+1, max_sl+1, max_t+1)
    report_test_result(test_info, "Test reference moisture for all indices out of bounds",
        tool.get_reference_moisture(pct), -1, ERROR_TOLERANCE)
    report_test_result(test_info, "Test correction moisture for all indices out of bounds",
        tool.get_correction_moisture(pct), -1, ERROR_TOLERANCE)
    report_test_result(test_info, "Test fine dead fuel moisture all indices out of bounds",
        tool.get_fine_dead_fuel_moisture(pct), -1, ERROR_TOLERANCE)

    print("Finished testing Fine Dead Fuel Moisture Tool\n")


# ---------------------------------------------------------------------------
# testSlopeTool
# ---------------------------------------------------------------------------

def test_slope_tool(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Slope Tool")
    print("="*70)

    inches = LengthUnits.LengthUnitsEnum.Inches
    feet   = LengthUnits.LengthUnitsEnum.Feet
    cm     = LengthUnits.LengthUnitsEnum.Centimeters
    meters = LengthUnits.LengthUnitsEnum.Meters
    tool   = behave_run.slope_tool

    tool.calculate_slope_from_map_measurements(1980, 3.6, inches, 50.0, 4.1, feet)
    report_test_result(test_info, "Test slope in degrees from map measurements, imperial units",
        round(tool.get_slope_from_map_measurements(SlopeUnits.SlopeUnitsEnum.Degrees)), 19.0, ERROR_TOLERANCE)
    report_test_result(test_info, "Test slope in percent from map measurements, imperial units",
        round(tool.get_slope_from_map_measurements(SlopeUnits.SlopeUnitsEnum.Percent)), 35.0, ERROR_TOLERANCE)
    report_test_result(test_info, "Test slope elevation change from map measurements, imperial units",
        round(tool.get_slope_elevation_change_from_map_measurements(feet)), 205.0, ERROR_TOLERANCE)
    report_test_result(test_info, "Test slope horizontal distance from map measurements, imperial units",
        round(tool.get_slope_horizontal_distance_from_map_measurements(feet)), 594.0, ERROR_TOLERANCE)

    tool.calculate_slope_from_map_measurements(3960, 3.0, cm, 15.0, 5.5, meters)
    report_test_result(test_info, "Test slope in degrees from map measurements, metric units",
        round(tool.get_slope_from_map_measurements(SlopeUnits.SlopeUnitsEnum.Degrees)), 35.0, ERROR_TOLERANCE)
    report_test_result(test_info, "Test slope in percent from map measurements, metric units",
        round(tool.get_slope_from_map_measurements(SlopeUnits.SlopeUnitsEnum.Percent)), 69.0, ERROR_TOLERANCE)
    report_test_result(test_info, "Test slope elevation change from map measurements, metric units",
        round(tool.get_slope_elevation_change_from_map_measurements(meters)), 82.0, ERROR_TOLERANCE)
    report_test_result(test_info, "Test slope horizontal distance from map measurements, metric units",
        round(tool.get_slope_horizontal_distance_from_map_measurements(meters)), 119.0, ERROR_TOLERANCE)

    num = tool.get_number_of_horizontal_distances()
    expected_hd = [2.9, 2.9, 2.9, 2.9, 3.0, 3.0, 3.0]
    labels = ["UPSLOPE_ZERO_DEGREES","FIFTEEN_DEGREES_FROM_UPSLOPE","THIRTY_DEGREES_FROM_UPSLOPE",
              "FORTY_FIVE_DEGREES_FROM_UPSLOPE","SIXTY_DEGREES_FROM_UPSLOPE",
              "SEVENTY_FIVE_DEGREES_FROM_UPSLOPE","CROSS_SLOPE_NINETY_DEGREES"]
    tool.calculate_horizontal_distance(3.0, inches, 30.0, SlopeUnits.SlopeUnitsEnum.Percent)
    for i in range(min(num, len(expected_hd))):
        report_test_result(test_info,
            f"Test calculateHorizontalDistance() {labels[i]} distance in feet",
            round(tool.get_horizontal_distance_at_index(i, feet) * 10.0) / 10.0,
            expected_hd[i], ERROR_TOLERANCE)

    print("Finished testing Slope Tool\n")


# ---------------------------------------------------------------------------
# testVaporPressureDeficitCalculator
# ---------------------------------------------------------------------------

def test_vapor_pressure_deficit_calculator(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Vapor Pressure Deficit Calculator")
    print("="*70)

    vpd  = behave_run.vpd_calculator
    pct  = FractionUnits.FractionUnitsEnum.Percent
    fahr = TemperatureUnits.TemperatureUnitsEnum.Fahrenheit
    hpa  = 'HectoPascal'
    tol  = 1e-3

    cases = [
        (50.0, 100.0,  0.0),
        (50.0,  90.0,  1.22833),
        (50.0,  80.0,  2.45667),
        (50.0,  70.0,  3.685),
        (50.0,  60.0,  4.91334),
        (50.0,  50.0,  6.14167),
        (50.0,  40.0,  7.37001),
        (50.0,  30.0,  8.59834),
        (50.0,  20.0,  9.82667),
        (50.0,  10.0, 11.055),
    ]
    for temp, rh, expected in cases:
        vpd.set_temperature(temp, fahr)
        vpd.set_relative_humidity(rh, pct)
        vpd.run_calculation()
        report_test_result(test_info,
            f"Test {rh}% Relative Humidity, {temp} degF Air Temperature",
            vpd.get_vapor_pressure_deficit(hpa), expected, tol)

    print("Finished testing Vapor Pressure Deficit Calculator\n")


# ---------------------------------------------------------------------------
# testSimpleSurface
# ---------------------------------------------------------------------------

def test_simple_surface(test_info, behave_run):
    print("\n" + "="*70)
    print("Testing Simple Surface")
    print("="*70)

    behave_run.surface.updateSurfaceInputs(
        124, 6.0, 7.0, 8.0, 60.0, 90.0, FractionUnits.FractionUnitsEnum.Percent,
        5.0, SpeedUnits.SpeedUnitsEnum.MilesPerHour, 'TwentyFoot',
        0, 'RelativeToNorth',
        30.0, SlopeUnits.SlopeUnitsEnum.Percent, 0,
        0.0, FractionUnits.FractionUnitsEnum.Percent,
        0.0, LengthUnits.LengthUnitsEnum.Feet,
        0.0, FractionUnits.FractionUnitsEnum.Fraction)

    behave_run.surface.set_user_provided_wind_adjustment_factor(1.0)
    behave_run.surface.set_wind_adjustment_factor_calculation_method('UserInput')
    behave_run.surface.set_wind_height_input_mode('TwentyFoot')
    behave_run.surface.set_wind_orientation_mode('RelativeToNorth')
    behave_run.surface.do_surface_run_in_direction_of_max_spread()

    report_test_result(test_info, "Test Simple Surface Rate of Spread",
        round_to_six_decimal_places(behave_run.surface.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)),
        34.011429, ERROR_TOLERANCE)

    report_test_result(test_info, "Test Simple Surface Flame Length",
        round_to_six_decimal_places(behave_run.surface.getFlameLength(LengthUnits.LengthUnitsEnum.Feet)),
        15.811421, ERROR_TOLERANCE)

    report_test_result(test_info, "Test Simple Surface FirelineIntensity",
        round_to_six_decimal_places(behave_run.surface.getFirelineIntensity(FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond)),
        2292.684759, ERROR_TOLERANCE)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_all_tests():
    print("\n" + "="*70)
    print("BEHAVE FIRE BEHAVIOR MODEL - PYTHON TEST SUITE")
    print("="*70)

    test_info = TestInfo()
    fuel_models = FuelModels()
    species_master_table = SpeciesMasterTable()
    behave_run = BehaveRun(fuel_models, species_master_table)

    set_surface_inputs_for_gs4_low_moisture_scenario(behave_run)

    test_surface_single_fuel_model(test_info, behave_run)
    test_chaparral(test_info, behave_run)
    test_calculate_scorch_height(test_info, behave_run)
    test_palmetto_gallberry(test_info, behave_run)
    test_western_aspen(test_info, behave_run)
    test_length_to_width_ratio(test_info, behave_run)
    test_elliptical_dimensions(test_info, behave_run)
    test_direction_of_interest(test_info, behave_run)
    test_fireline_intensity(test_info, behave_run)
    test_two_fuel_models(test_info, behave_run)
    test_crown_module_rothermel(test_info, behave_run)
    test_crown_module_scott_and_reinhardt(test_info, behave_run)
    test_spot_module(test_info, behave_run)
    test_speed_unit_conversion(test_info, behave_run)
    test_ignite_module(test_info, behave_run)
    test_safety_module(test_info, behave_run)
    test_contain_module(test_info, behave_run)
    # testMortalityModule is an empty stub in C++ — skipped
    test_fine_dead_fuel_moisture_tool(test_info, behave_run)
    test_slope_tool(test_info, behave_run)
    test_vapor_pressure_deficit_calculator(test_info, behave_run)
    test_simple_surface(test_info, behave_run)

    return test_info.print_summary()


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n{RED}Fatal error: {e}{RESET}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

