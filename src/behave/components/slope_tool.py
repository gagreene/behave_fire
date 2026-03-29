"""
SlopeTool component
Python port of C++ slopeTool.cpp/.h

Calculates slope from map measurements and horizontal distances.
"""
import math

try:
    from .behave_units import LengthUnits, SlopeUnits
except ImportError:
    from behave_units import LengthUnits, SlopeUnits


class SlopeTool:
    """
    SlopeTool calculates slope from map measurements and horizontal distances.
    Mirrors C++ SlopeTool.
    """

    def __init__(self):
        NUM_HORIZONTAL_DISTANCE_VALUES = 7
        self.horizontal_distances_ = [0.0] * NUM_HORIZONTAL_DISTANCE_VALUES

        self.max_slope_in_degrees_ = -1.0
        self.max_slope_in_percent_ = -1.0
        self.slope_from_map_measurements_ = -1.0
        self.slope_horizontal_distance_ = -1.0
        self.slope_elevation_change_ = -1.0

        self.representative_fractions_ = [
            1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500,
            63360, 100000, 126720, 250000, 253440, 506880, 1000000, 1013760
        ]
        self.inches_per_mile_ = [
            32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138,
            1.0, 0.6336, 0.5, 0.2534, 0.25, 0.125, 0.0634, 0.0625
        ]
        self.miles_per_inch_ = [
            0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.33330, 0.37879, 0.5,
            0.78914, 0.98643, 1.0, 1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0
        ]
        self.centimeters_per_kilometer_ = [
            50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566,
            2.0, 1.6, 1.5783, 1.0, 0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986
        ]
        self.kilometers_per_centimeter_ = [
            0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168,
            0.5, 0.625, 0.6336, 1.0, 1.2672, 2.50, 2.5344, 5.0688, 10.0, 10.1376
        ]

    def calculate_horizontal_distance(self, calculated_map_distance, distance_units,
                                       max_slope_steepness, slope_units):
        """
        Calculate horizontal distances for 7 compass directions given a map measurement
        and maximum slope.

        Args:
            calculated_map_distance: Map distance measured
            distance_units: LengthUnitsEnum
            max_slope_steepness: Maximum slope value
            slope_units: SlopeUnitsEnum
        """
        self.max_slope_in_degrees_ = SlopeUnits.toBaseUnits(max_slope_steepness, slope_units)
        self.max_slope_in_percent_ = SlopeUnits.fromBaseUnits(
            self.max_slope_in_degrees_, SlopeUnits.SlopeUnitsEnum.Percent)

        # Replicate C++ behavior: groundDistanceInFeet = toBase(input, units)
        # but groundDistanceInInches = fromBase(raw_input, Inches)  [C++ treats raw as feet]
        ground_distance_in_feet = LengthUnits.toBaseUnits(calculated_map_distance, distance_units)
        # C++ line: fromBaseUnits(calulatedMapDistance, Inches) — uses RAW input treated as feet
        ground_distance_in_inches = LengthUnits.fromBaseUnits(
            calculated_map_distance, LengthUnits.LengthUnitsEnum.Inches)

        for i in range(len(self.horizontal_distances_)):
            direction = 15.0 * float(i)
            a = ground_distance_in_inches * math.cos(direction * math.pi / 180.0)
            b = ground_distance_in_inches * math.sin(direction * math.pi / 180.0)
            c = a * math.cos(self.max_slope_in_degrees_ * math.pi / 180.0)
            d = math.sqrt(c * c + b * b)
            # Store as base units (feet)
            self.horizontal_distances_[i] = LengthUnits.toBaseUnits(d, LengthUnits.LengthUnitsEnum.Inches)

    def calculate_slope_from_map_measurements(self, map_representative_fraction, map_distance,
                                               distance_units, contour_interval, number_of_contours,
                                               contour_units):
        """
        Calculate slope from map measurements.

        Args:
            map_representative_fraction: Map scale (e.g. 1980 for 1:1980)
            map_distance: Distance measured on map
            distance_units: LengthUnitsEnum for map distance
            contour_interval: Vertical interval between contours
            number_of_contours: Number of contours crossed
            contour_units: LengthUnitsEnum for contour interval
        """
        distance_feet = LengthUnits.toBaseUnits(map_distance, distance_units)
        distance_inches = LengthUnits.fromBaseUnits(distance_feet, LengthUnits.LengthUnitsEnum.Inches)

        # Elevation change = contour_interval * number_of_contours in base units (feet)
        self.slope_elevation_change_ = LengthUnits.toBaseUnits(
            contour_interval * number_of_contours, contour_units)

        # Horizontal distance = map scale * map distance (in inches) -> feet
        self.slope_horizontal_distance_ = LengthUnits.toBaseUnits(
            map_representative_fraction * distance_inches, LengthUnits.LengthUnitsEnum.Inches)

        if self.slope_horizontal_distance_ < 0.01:
            slope_in_percent = 0.0
            self.slope_from_map_measurements_ = 0.0
        else:
            slope_in_percent = self.slope_elevation_change_ / self.slope_horizontal_distance_
            # Store as degrees (base unit for SlopeUnits)
            self.slope_from_map_measurements_ = math.atan(slope_in_percent) * 180.0 / math.pi

    def get_number_of_horizontal_distances(self):
        """Return number of horizontal distance values."""
        return len(self.horizontal_distances_)

    def get_horizontal_distance_at_index(self, index, map_distance_units):
        """
        Get horizontal distance at specified index.

        Args:
            index: Index into horizontal distances array
            map_distance_units: LengthUnitsEnum

        Returns:
            Horizontal distance in requested units, or -1.0 if out of bounds.
        """
        if 0 <= index < len(self.horizontal_distances_):
            return LengthUnits.fromBaseUnits(self.horizontal_distances_[index], map_distance_units)
        return -1.0

    def get_slope_from_map_measurements(self, slope_units):
        """
        Get slope from map measurements in requested units.

        Args:
            slope_units: SlopeUnitsEnum

        Returns:
            Slope in requested units.
        """
        return SlopeUnits.fromBaseUnits(self.slope_from_map_measurements_, slope_units)

    def get_slope_from_map_measurements_in_percent(self):
        """Get slope from map measurements in percent."""
        return SlopeUnits.fromBaseUnits(self.slope_from_map_measurements_, SlopeUnits.SlopeUnitsEnum.Percent)

    def get_slope_from_map_measurements_in_degrees(self):
        """Get slope from map measurements in degrees."""
        return SlopeUnits.fromBaseUnits(self.slope_from_map_measurements_, SlopeUnits.SlopeUnitsEnum.Degrees)

    def get_slope_horizontal_distance_from_map_measurements(self, distance_units):
        """
        Get slope horizontal distance from map measurements.

        Args:
            distance_units: LengthUnitsEnum

        Returns:
            Horizontal distance in requested units.
        """
        return LengthUnits.fromBaseUnits(self.slope_horizontal_distance_, distance_units)

    def get_slope_elevation_change_from_map_measurements(self, elevation_units):
        """
        Get slope elevation change from map measurements.

        Args:
            elevation_units: LengthUnitsEnum

        Returns:
            Elevation change in requested units.
        """
        return LengthUnits.fromBaseUnits(self.slope_elevation_change_, elevation_units)

    def get_number_of_representative_fractions(self):
        """Return number of representative fractions in table."""
        return len(self.representative_fractions_)

    def get_representative_fraction_at_index(self, index):
        """Get representative fraction at index."""
        if 0 <= index < len(self.representative_fractions_):
            return self.representative_fractions_[index]
        return -1

    def get_inches_per_mile_at_index(self, index):
        """Get inches per mile at index."""
        if 0 <= index < len(self.inches_per_mile_):
            return self.inches_per_mile_[index]
        return -1.0

    def get_miles_per_inch_at_index(self, index):
        """Get miles per inch at index."""
        if 0 <= index < len(self.miles_per_inch_):
            return self.miles_per_inch_[index]
        return -1.0

    def get_centimeters_per_kilometer_at_index(self, index):
        """Get centimeters per kilometer at index."""
        if 0 <= index < len(self.centimeters_per_kilometer_):
            return self.centimeters_per_kilometer_[index]
        return -1.0

    def get_kilometers_per_centimeter_at_index(self, index):
        """Get kilometers per centimeter at index."""
        if 0 <= index < len(self.kilometers_per_centimeter_):
            return self.kilometers_per_centimeter_[index]
        return -1.0


