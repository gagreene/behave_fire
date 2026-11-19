"""Boundary geometry and unit conversions for the Python FireSize component."""

from math import inf, nextafter, pi

from behave_fire.components.behave_units import AreaUnits, LengthUnits, SpeedUnits, TimeUnits
from behave_fire.components.fire_size import FireSize

AREA = AreaUnits.AreaUnitsEnum
LENGTH = LengthUnits.LengthUnitsEnum
SPEED = SpeedUnits.SpeedUnitsEnum
TIME = TimeUnits.TimeUnitsEnum


def _assert_zero_geometry(size, elapsed_minutes):
    length_args = (LENGTH.Feet, elapsed_minutes, TIME.Minutes)
    area_args = (AREA.SquareFeet, elapsed_minutes, TIME.Minutes)
    assert size.get_elliptical_a(*length_args) == 0.0
    assert size.get_elliptical_b(*length_args) == 0.0
    assert size.get_elliptical_c(*length_args) == 0.0
    assert size.get_fire_length(*length_args) == 0.0
    assert size.get_max_fire_width(*length_args) == 0.0
    assert size.get_fire_area(False, *area_args) == 0.0
    assert size.get_fire_area(True, *area_args) == 0.0
    assert size.get_fire_perimeter(False, *length_args) == 0.0
    assert size.get_fire_perimeter(True, *length_args) == 0.0


# ---- circular geometry tests ----

def test_calm_fire_geometry_distinguishes_surface_and_crown_extents():
    size = FireSize()
    size.calculate_fire_basic_dimensions(True, 0.0, SPEED.MilesPerHour, 4.0, SPEED.FeetPerMinute)
    length_args = (LENGTH.Feet, 2.0, TIME.Minutes)
    area_args = (AREA.SquareFeet, 2.0, TIME.Minutes)

    assert size.get_fire_length_to_width_ratio() == 1.0
    assert size.get_heading_to_backing_ratio() == 1.0
    assert size.get_eccentricity() == 0.0
    assert size.get_backing_spread_rate(SPEED.FeetPerMinute) == 4.0
    assert size.get_flanking_spread_rate(SPEED.FeetPerMinute) == 4.0
    assert size.get_elliptical_a(*length_args) == size.get_elliptical_b(*length_args) == 8.0
    assert size.get_elliptical_c(*length_args) == 0.0
    assert size.get_fire_length(*length_args) == size.get_max_fire_width(*length_args) == 16.0

    # Surface radius is 8 ft. Rothermel crown area and perimeter use 8 ft as the diameter.
    assert size.get_fire_area(False, *area_args) == 64.0 * pi
    assert size.get_fire_area(True, *area_args) == 16.0 * pi
    assert size.get_fire_perimeter(False, *length_args) == 16.0 * pi
    assert size.get_fire_perimeter(True, *length_args) == 8.0 * pi


def test_surface_geometry_becomes_elliptical_above_the_calm_wind_cutoff():
    size = FireSize()
    calm_cutoff_mph = 1e-7
    length_args = (LENGTH.Feet, 2.0, TIME.Minutes)
    size.calculate_fire_basic_dimensions(False, calm_cutoff_mph, SPEED.MilesPerHour, 4.0, SPEED.FeetPerMinute)
    assert size.get_fire_length_to_width_ratio() == 1.0
    assert size.get_eccentricity() == 0.0

    wind_mph = nextafter(calm_cutoff_mph, inf)
    size.calculate_fire_basic_dimensions(False, wind_mph, SPEED.MilesPerHour, 4.0, SPEED.FeetPerMinute)

    assert size.get_fire_length_to_width_ratio() > 1.0
    assert 0.0 < size.get_eccentricity() < 1.0
    assert 0.0 < size.get_elliptical_c(*length_args) < size.get_elliptical_a(*length_args)
    assert 0.0 < size.get_max_fire_width(*length_args) < size.get_fire_length(*length_args) < 16.0


# ---- degenerate geometry tests ----

def test_zero_elapsed_time_gives_zero_geometry():
    size = FireSize()
    size.calculate_fire_basic_dimensions(False, 5.0, SPEED.MilesPerHour, 8.0, SPEED.FeetPerMinute)
    assert size.get_elliptical_c(LENGTH.Feet, 2.0, TIME.Minutes) > 0.0

    _assert_zero_geometry(size, elapsed_minutes=0.0)


def test_zero_spread_clears_geometry_after_nonzero_spread():
    size = FireSize()
    size.calculate_fire_basic_dimensions(False, 5.0, SPEED.MilesPerHour, 8.0, SPEED.FeetPerMinute)
    assert size.get_elliptical_c(LENGTH.Feet, 2.0, TIME.Minutes) > 0.0

    size.calculate_fire_basic_dimensions(False, 5.0, SPEED.MilesPerHour, 0.0, SPEED.FeetPerMinute)

    _assert_zero_geometry(size, elapsed_minutes=2.0)
    assert size.get_backing_spread_rate(SPEED.FeetPerMinute) == 0.0
    assert size.get_flanking_spread_rate(SPEED.FeetPerMinute) == 0.0


# ---- unit conversion tests ----

def test_nonbase_spread_time_and_geometry_units():
    size = FireSize()
    size.calculate_fire_basic_dimensions(False, 0.0, SPEED.MilesPerHour, 2.0, SPEED.MilesPerHour)

    # At 2 mi/h the calm circle grows 88 ft in 30 seconds. One foot is exactly 0.0003048 km.
    assert size.get_fire_length(LENGTH.Inches, 30.0, TIME.Seconds) == 2112.0

    # The circle is larger than its inscribed square and smaller than its circumscribed square.
    radius_km = 88.0 * 0.0003048
    radius_squared = radius_km * radius_km
    area_square_km = size.get_fire_area(False, AREA.SquareKilometers, 30.0, TIME.Seconds)
    assert 2.0 * radius_squared < area_square_km < 4.0 * radius_squared
