"""Native stateful tools preserving public snapshot attributes."""
from ._bridge import _state
from ._serialization import NativeState as _NativeState

class SlopeTool(_NativeState):
    """SlopeTool calculates slope from map measurements and horizontal distances.
    Mirrors C++ SlopeTool.
    """

    def __init__(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', '__init__', locals())

    def calculate_horizontal_distance(
            self,
            calculated_map_distance,
            distance_units,
            max_slope_steepness,
            slope_units,
    ):
        """Execute the explicit native operation.

        :param calculated_map_distance: Input matching the public API.
        :param distance_units: Input matching the public API.
        :param max_slope_steepness: Input matching the public API.
        :param slope_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'calculate_horizontal_distance', locals())

    def calculate_slope_from_map_measurements(
            self,
            map_representative_fraction,
            map_distance,
            distance_units,
            contour_interval,
            number_of_contours,
            contour_units,
    ):
        """Execute the explicit native operation.

        :param map_representative_fraction: Input matching the public API.
        :param map_distance: Input matching the public API.
        :param distance_units: Input matching the public API.
        :param contour_interval: Input matching the public API.
        :param number_of_contours: Input matching the public API.
        :param contour_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'calculate_slope_from_map_measurements', locals())

    def get_number_of_horizontal_distances(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_number_of_horizontal_distances', locals())

    def get_horizontal_distance_at_index(self, index, map_distance_units):
        """Execute the explicit native operation.

        :param index: Input matching the public API.
        :param map_distance_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_horizontal_distance_at_index', locals())

    def get_slope_from_map_measurements(self, slope_units):
        """Execute the explicit native operation.

        :param slope_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_slope_from_map_measurements', locals())

    def get_slope_from_map_measurements_in_percent(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_slope_from_map_measurements_in_percent', locals())

    def get_slope_from_map_measurements_in_degrees(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_slope_from_map_measurements_in_degrees', locals())

    def get_slope_horizontal_distance_from_map_measurements(self, distance_units):
        """Execute the explicit native operation.

        :param distance_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_slope_horizontal_distance_from_map_measurements', locals())

    def get_slope_elevation_change_from_map_measurements(self, elevation_units):
        """Execute the explicit native operation.

        :param elevation_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_slope_elevation_change_from_map_measurements', locals())

    def get_number_of_representative_fractions(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_number_of_representative_fractions', locals())

    def get_representative_fraction_at_index(self, index):
        """Execute the explicit native operation.

        :param index: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_representative_fraction_at_index', locals())

    def get_inches_per_mile_at_index(self, index):
        """Execute the explicit native operation.

        :param index: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_inches_per_mile_at_index', locals())

    def get_miles_per_inch_at_index(self, index):
        """Execute the explicit native operation.

        :param index: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_miles_per_inch_at_index', locals())

    def get_centimeters_per_kilometer_at_index(self, index):
        """Execute the explicit native operation.

        :param index: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_centimeters_per_kilometer_at_index', locals())

    def get_kilometers_per_centimeter_at_index(self, index):
        """Execute the explicit native operation.

        :param index: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'slope_tool', 'get_kilometers_per_centimeter_at_index', locals())
