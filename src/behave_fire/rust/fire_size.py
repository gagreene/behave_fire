"""Native stateful tools preserving public snapshot attributes."""
from ._bridge import _state
from ._serialization import NativeState as _NativeState

class FireSize(_NativeState):
    """FireSize component for calculating various properties related to fire size.

    Uses elliptical fire model to calculate fire dimensions, perimeter, and area
    based on forward and backing spread rates, wind speed, and elapsed time.
    """

    def __init__(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', '__init__', locals())

    def calculate_fire_basic_dimensions(
            self,
            is_crown,
            effective_wind_speed,
            wind_speed_units,
            forward_spread_rate,
            spread_rate_units,
    ):
        """Execute the explicit native operation.

        :param is_crown: Input matching the public API.
        :param effective_wind_speed: Input matching the public API.
        :param wind_speed_units: Input matching the public API.
        :param forward_spread_rate: Input matching the public API.
        :param spread_rate_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'calculate_fire_basic_dimensions', locals())

    def calculate_surface_fire_length_to_width_ratio(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'calculate_surface_fire_length_to_width_ratio', locals())

    def calculate_crown_fire_length_to_width_ratio(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'calculate_crown_fire_length_to_width_ratio', locals())

    def calculate_fire_eccentricity(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'calculate_fire_eccentricity', locals())

    def calculate_elliptical_dimensions(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'calculate_elliptical_dimensions', locals())

    def calculate_backing_spread_rate(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'calculate_backing_spread_rate', locals())

    def calculate_flanking_spread_rate(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'calculate_flanking_spread_rate', locals())

    def get_fire_length_to_width_ratio(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_fire_length_to_width_ratio', locals())

    def get_eccentricity(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_eccentricity', locals())

    def get_heading_to_backing_ratio(self):
        """Execute the explicit native operation.


        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_heading_to_backing_ratio', locals())

    def get_backing_spread_rate(self, spread_rate_units):
        """Execute the explicit native operation.

        :param spread_rate_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_backing_spread_rate', locals())

    def get_flanking_spread_rate(self, spread_rate_units):
        """Execute the explicit native operation.

        :param spread_rate_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_flanking_spread_rate', locals())

    def get_elliptical_a(self, length_units, elapsed_time, time_units):
        """Execute the explicit native operation.

        :param length_units: Input matching the public API.
        :param elapsed_time: Input matching the public API.
        :param time_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_elliptical_a', locals())

    def get_elliptical_b(self, length_units, elapsed_time, time_units):
        """Execute the explicit native operation.

        :param length_units: Input matching the public API.
        :param elapsed_time: Input matching the public API.
        :param time_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_elliptical_b', locals())

    def get_elliptical_c(self, length_units, elapsed_time, time_units):
        """Execute the explicit native operation.

        :param length_units: Input matching the public API.
        :param elapsed_time: Input matching the public API.
        :param time_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_elliptical_c', locals())

    def get_fire_length(self, length_units, elapsed_time, time_units):
        """Execute the explicit native operation.

        :param length_units: Input matching the public API.
        :param elapsed_time: Input matching the public API.
        :param time_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_fire_length', locals())

    def get_max_fire_width(self, length_units, elapsed_time, time_units):
        """Execute the explicit native operation.

        :param length_units: Input matching the public API.
        :param elapsed_time: Input matching the public API.
        :param time_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_max_fire_width', locals())

    def get_fire_perimeter(self, is_crown, length_units, elapsed_time, time_units):
        """Execute the explicit native operation.

        :param is_crown: Input matching the public API.
        :param length_units: Input matching the public API.
        :param elapsed_time: Input matching the public API.
        :param time_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_fire_perimeter', locals())

    def get_fire_area(self, is_crown, area_units, elapsed_time, time_units):
        """Execute the explicit native operation.

        :param is_crown: Input matching the public API.
        :param area_units: Input matching the public API.
        :param elapsed_time: Input matching the public API.
        :param time_units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _state(self, 'fire_size', 'get_fire_area', locals())
