"""
Vapor Pressure Deficit Calculator
Python port of C++ vaporPressureDeficitCalculator.cpp
"""
import math

try:
    from .behave_units import TemperatureUnits, FractionUnits, PressureUnits
except ImportError:
    from behave_units import TemperatureUnits, FractionUnits, PressureUnits


def _parse_pressure_units(units):
    """Convert string or int pressure units to PressureUnitsEnum."""
    if isinstance(units, int):
        return units
    s = str(units).upper().replace(' ', '').replace('_', '')
    _MAP = {
        'PASCAL': PressureUnits.PressureUnitsEnum.Pascal,
        'PA': PressureUnits.PressureUnitsEnum.Pascal,
        'HECTOPASCAL': PressureUnits.PressureUnitsEnum.HectoPascal,
        'HPA': PressureUnits.PressureUnitsEnum.HectoPascal,
        'KILOPASCAL': PressureUnits.PressureUnitsEnum.KiloPascal,
        'KPA': PressureUnits.PressureUnitsEnum.KiloPascal,
        'MEGAPASCAL': PressureUnits.PressureUnitsEnum.MegaPascal,
        'GIGAPASCAL': PressureUnits.PressureUnitsEnum.GigaPascal,
        'BAR': PressureUnits.PressureUnitsEnum.Bar,
        'ATMOSPHERE': PressureUnits.PressureUnitsEnum.Atmosphere,
        'ATM': PressureUnits.PressureUnitsEnum.Atmosphere,
        'PSI': PressureUnits.PressureUnitsEnum.PoundPerSquareInch,
    }
    return _MAP.get(s, PressureUnits.PressureUnitsEnum.Pascal)


class VaporPressureDeficitCalculator:
    """
    Calculates vapor pressure deficit from temperature and relative humidity.
    Mirrors C++ VaporPressureDeficitCalculator.
    """

    def __init__(self):
        self.temperature_ = 0.0           # Fahrenheit (base)
        self.relative_humidity_ = 0.0     # fraction (base)
        self.actual_vapor_pressure_ = 0.0  # Pa (base)
        self.saturated_vapor_pressure_ = 0.0  # Pa (base)
        self.vapor_pressure_deficit_ = 0.0    # Pa (base)

    def set_temperature(self, temp, units):
        """Set temperature with unit conversion (base = Fahrenheit)."""
        self.temperature_ = TemperatureUnits.toBaseUnits(temp, units)

    def set_relative_humidity(self, rh, units):
        """Set relative humidity with unit conversion (base = Fraction)."""
        self.relative_humidity_ = FractionUnits.toBaseUnits(rh, units)

    def run_calculation(self):
        """Calculate vapor pressure deficit."""
        # Convert temperature to Celsius
        temp_c = TemperatureUnits.fromBaseUnits(self.temperature_, TemperatureUnits.TemperatureUnitsEnum.Celsius)

        # Tetens formula: saturated vapor pressure (hPa)
        temp_k = temp_c + 237.3
        sat_vp_hpa = 6.11 * math.pow(10.0, (7.5 * temp_c) / temp_k)

        # Store saturated vapor pressure in Pa (base)
        self.saturated_vapor_pressure_ = PressureUnits.toBaseUnits(sat_vp_hpa, PressureUnits.PressureUnitsEnum.HectoPascal)

        # Actual vapor pressure (hPa): rh (fraction) * sat_vp (hPa)
        actual_vp_hpa = self.relative_humidity_ * sat_vp_hpa
        self.actual_vapor_pressure_ = PressureUnits.toBaseUnits(actual_vp_hpa, PressureUnits.PressureUnitsEnum.HectoPascal)

        # Deficit (hPa) = sat - actual
        vpd_hpa = sat_vp_hpa - actual_vp_hpa
        self.vapor_pressure_deficit_ = PressureUnits.toBaseUnits(vpd_hpa, PressureUnits.PressureUnitsEnum.HectoPascal)

    def get_vapor_pressure_deficit(self, units):
        """Get vapor pressure deficit in requested units."""
        return PressureUnits.fromBaseUnits(self.vapor_pressure_deficit_, _parse_pressure_units(units))

    def get_actual_vapor_pressure(self, units):
        """Get actual vapor pressure in requested units."""
        return PressureUnits.fromBaseUnits(self.actual_vapor_pressure_, _parse_pressure_units(units))

    def get_saturated_vapor_pressure(self, units):
        """Get saturated vapor pressure in requested units."""
        return PressureUnits.fromBaseUnits(self.saturated_vapor_pressure_, _parse_pressure_units(units))

