"""Zero-distance and zero-rise boundaries for the Python SlopeTool component."""

from behave_fire.components.behave_units import LengthUnits, SlopeUnits
from behave_fire.components.slope_tool import SlopeTool

LENGTH = LengthUnits.LengthUnitsEnum
SLOPE = SlopeUnits.SlopeUnitsEnum

# ---- horizontal distance tests ----

def test_horizontal_distances_reset_to_zero_at_all_seven_directions():
    tool = SlopeTool()
    tool.calculate_horizontal_distance(8.0, LENGTH.Feet, 30.0, SLOPE.Degrees)
    assert all(tool.get_horizontal_distance_at_index(index, LENGTH.Feet) > 0.0 for index in range(7))

    tool.calculate_horizontal_distance(0.0, LENGTH.Feet, 30.0, SLOPE.Degrees)

    assert [tool.get_horizontal_distance_at_index(index, LENGTH.Feet) for index in range(7)] == [0.0] * 7
    assert tool.get_horizontal_distance_at_index(7, LENGTH.Inches) == -1.0


# ---- map slope tests ----

def test_map_slope_resets_to_zero_at_short_runs_and_zero_rise():
    tool = SlopeTool()
    runs = []
    for map_distance, contours, expected_rise in ((0.0, 3, 60.0), (2**-20, 3, 60.0), (0.125, 0, 0.0)):
        # Start each edge call from a nonzero slope to detect stale results.
        tool.calculate_slope_from_map_measurements(1980, 0.125, LENGTH.Feet, 20.0, 3, LENGTH.Feet)
        assert tool.get_slope_from_map_measurements_in_degrees() > 0.0
        assert tool.get_slope_from_map_measurements_in_percent() > 0.0

        tool.calculate_slope_from_map_measurements(1980, map_distance, LENGTH.Feet, 20.0, contours, LENGTH.Feet)

        runs.append(tool.get_slope_horizontal_distance_from_map_measurements(LENGTH.Feet))
        assert tool.get_slope_elevation_change_from_map_measurements(LENGTH.Feet) == expected_rise
        assert tool.get_slope_from_map_measurements_in_degrees() == 0.0
        assert tool.get_slope_from_map_measurements_in_percent() == 0.0

    zero_run, sub_cutoff_run, flat_run = runs
    assert zero_run == 0.0
    assert 0.0 < sub_cutoff_run < 0.01
    assert flat_run > 0.01
