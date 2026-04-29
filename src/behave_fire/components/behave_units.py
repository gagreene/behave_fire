"""
behave_units_array.py — NumPy-native unit conversion (V1)

Vectorized equivalents of every conversion in behave_units.py.
All functions:
  - Accept scalars or any-shape NumPy arrays.
  - Always return an ndarray (never a bare Python scalar) so downstream
    np.where / arithmetic always receives a consistent type (G1 fix).
  - 'units' arguments are scalar integer enums (same values as behave_units.py).

Base units (same as scalar module):
  Speed          → FeetPerMinute      (0)
  Length         → Feet               (0)
  Area           → SquareFeet         (0)
  Fraction       → Fraction           (0)
  Temperature    → Fahrenheit         (0)
  Slope          → Degrees            (0)
  Pressure       → Pascal             (0)
  FirelineIntensity → BtusPerFootPerSecond (0)
  HeatPerUnitArea   → BtusPerSquareFoot   (0)
  ReactionIntensity → BtusPerSqFtPerMin   (0)
  Loading           → PoundsPerSquareFoot (0)
  Density           → PoundsPerCubicFoot  (0)
  Time              → Minutes             (0)
"""

import numpy as np
from typing import Union

# ---------------------------------------------------------------------------
# Speed  (base = FeetPerMinute = 0)
# ---------------------------------------------------------------------------
_SPEED_TO_FPM = np.array([
    1.0,           # 0 FeetPerMinute
    1.1,           # 1 ChainsPerHour
    196.8503937,   # 2 MetersPerSecond
    3.28084,       # 3 MetersPerMinute
    0.0547,        # 4 MetersPerHour   (ft/min per m/hr)
    88.0,          # 5 MilesPerHour
    54.680665,     # 6 KilometersPerHour
])

_SPEED_FROM_FPM = np.array([
    1.0,                # 0 FeetPerMinute
    10.0 / 11.0,        # 1 ChainsPerHour
    0.00508,            # 2 MetersPerSecond
    0.3048,             # 3 MetersPerMinute
    18.288,             # 4 MetersPerHour
    0.01136363636,      # 5 MilesPerHour
    0.018288,           # 6 KilometersPerHour
])


def speed_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert speed to ft/min (base unit).

    :param value: Speed value (*S) or scalar, in ``units``.
    :param units: ``SpeedUnitsEnum`` integer (0=FeetPerMinute, 5=MilesPerHour, etc.).
    :return: ndarray — speed in ft/min.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _SPEED_TO_FPM[units]


def speed_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert speed from ft/min to the requested units.

    :param value: Speed value (*S) or scalar in ft/min.
    :param units: ``SpeedUnitsEnum`` integer for desired output.
    :return: ndarray — speed in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _SPEED_FROM_FPM[units]


# ---------------------------------------------------------------------------
# Length  (base = Feet = 0)
# ---------------------------------------------------------------------------
_LENGTH_TO_FT = np.array([
    1.0,              # 0 Feet
    0.08333333333333, # 1 Inches
    0.003280839895,   # 2 Millimeters
    0.03280839895,    # 3 Centimeters
    3.2808398950131,  # 4 Meters
    66.0,             # 5 Chains
    5280.0,           # 6 Miles
    3280.8398950131,  # 7 Kilometers
])

_LENGTH_FROM_FT = np.array([
    1.0,                      # 0 Feet
    12.0,                     # 1 Inches
    1.0 / 0.003280839895,     # 2 Millimeters  (≈304.8)
    30.480,                   # 3 Centimeters
    0.3048,                   # 4 Meters
    0.0151515151515,          # 5 Chains
    0.0001893939393939394,    # 6 Miles
    0.0003048,                # 7 Kilometers
])


def length_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert length to feet (base unit).

    :param value: Length value (*S) or scalar, in ``units``.
    :param units: ``LengthUnitsEnum`` integer (0=Feet, 4=Meters, etc.).
    :return: ndarray — length in feet.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _LENGTH_TO_FT[units]


def length_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert length from feet to the requested units.

    :param value: Length value (*S) or scalar in feet.
    :param units: ``LengthUnitsEnum`` integer for desired output.
    :return: ndarray — length in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _LENGTH_FROM_FT[units]


# ---------------------------------------------------------------------------
# Area  (base = SquareFeet = 0)
# ---------------------------------------------------------------------------
_AREA_TO_SQFT = np.array([
    1.0,                     # 0 SquareFeet
    43560.002160576107,      # 1 Acres
    107639.10416709723,      # 2 Hectares
    10.76391041671,          # 3 SquareMeters
    27878400.0,              # 4 SquareMiles
    10763910.416709721,      # 5 SquareKilometers
])

_AREA_FROM_SQFT = np.array([
    1.0,                  # 0 SquareFeet
    2.295684e-05,         # 1 Acres
    0.0000092903036,      # 2 Hectares
    0.0929030353835,      # 3 SquareMeters
    3.5870064279e-08,     # 4 SquareMiles
    9.290304e-08,         # 5 SquareKilometers
])


def area_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert area to square feet (base unit).

    :param value: Area value (*S) or scalar, in ``units``.
    :param units: ``AreaUnitsEnum`` integer (0=SquareFeet, 1=Acres, etc.).
    :return: ndarray — area in square feet.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _AREA_TO_SQFT[units]


def area_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert area from square feet to the requested units.

    :param value: Area value (*S) or scalar in square feet.
    :param units: ``AreaUnitsEnum`` integer for desired output.
    :return: ndarray — area in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _AREA_FROM_SQFT[units]


# ---------------------------------------------------------------------------
# Fraction / Percentage  (base = Fraction = 0)
# ---------------------------------------------------------------------------

def fraction_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert fraction or percent to fraction (base unit, 0–1).

    :param value: Value (*S) or scalar, in ``units``.
    :param units: ``FractionUnitsEnum`` integer (0=Fraction, 1=Percent).
    :return: ndarray — value as a fraction (0–1).
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    # units == 1 → Percent
    return arr / 100.0


def fraction_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert fraction (0–1) to the requested units.

    :param value: Fraction value (*S) or scalar (0–1).
    :param units: ``FractionUnitsEnum`` integer for desired output.
    :return: ndarray — value in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * 100.0


# ---------------------------------------------------------------------------
# Temperature  (base = Fahrenheit = 0)
# ---------------------------------------------------------------------------

def temp_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert temperature to °F (base unit).

    :param value: Temperature value (*S) or scalar, in ``units``.
    :param units: ``TemperatureUnitsEnum`` integer (0=Fahrenheit, 1=Celsius, 2=Kelvin).
    :return: ndarray — temperature in °F.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    if units == 1:   # Celsius
        return arr * (9.0 / 5.0) + 32.0
    if units == 2:   # Kelvin
        return (arr - 273.15) * (9.0 / 5.0) + 32.0
    return arr


def temp_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert temperature from °F to the requested units.

    :param value: Temperature value (*S) or scalar in °F.
    :param units: ``TemperatureUnitsEnum`` integer for desired output.
    :return: ndarray — temperature in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    if units == 1:   # Celsius
        return (arr - 32.0) * (5.0 / 9.0)
    if units == 2:   # Kelvin
        return (arr - 32.0) * (5.0 / 9.0) + 273.15
    return arr


# ---------------------------------------------------------------------------
# Slope  (base = Degrees = 0)
# ---------------------------------------------------------------------------

def slope_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert slope to degrees (base unit).

    :param value: Slope value (*S) or scalar, in ``units``.
    :param units: ``SlopeUnitsEnum`` integer (0=Degrees, 1=Percent).
    :return: ndarray — slope in degrees.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    # units == 1 → Percent: convert via arctan
    return np.degrees(np.arctan(arr / 100.0))


def slope_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert slope from degrees to the requested units.

    :param value: Slope value (*S) or scalar in degrees.
    :param units: ``SlopeUnitsEnum`` integer for desired output.
    :return: ndarray — slope in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return np.tan(np.radians(arr)) * 100.0


# ---------------------------------------------------------------------------
# Pressure  (base = Pascal = 0)
# ---------------------------------------------------------------------------
_PRESSURE_TO_PA = np.array([
    1.0,          # 0 Pascal
    100.0,        # 1 HectoPascal
    1000.0,       # 2 KiloPascal
    1e6,          # 3 MegaPascal
    1e9,          # 4 GigaPascal
    100000.0,     # 5 Bar
    101325.0,     # 6 Atmosphere
    98066.5,      # 7 TechnicalAtmosphere
    6894.757,     # 8 PoundPerSquareInch
])

_PRESSURE_FROM_PA = np.array([
    1.0,                     # 0 Pascal
    0.01,                    # 1 HectoPascal
    0.001,                   # 2 KiloPascal
    1e-6,                    # 3 MegaPascal
    1e-9,                    # 4 GigaPascal
    1e-5,                    # 5 Bar
    1.0 / 101325.0,          # 6 Atmosphere
    1.0 / 98066.5,           # 7 TechnicalAtmosphere
    1.0 / 6894.757,          # 8 PoundPerSquareInch
])


def pressure_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert pressure to Pascals (base unit).

    :param value: Pressure value (*S) or scalar, in ``units``.
    :param units: ``PressureUnitsEnum`` integer (0=Pascal, 1=HectoPascal, etc.).
    :return: ndarray — pressure in Pascals.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _PRESSURE_TO_PA[units]


def pressure_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert pressure from Pascals to the requested units.

    :param value: Pressure value (*S) or scalar in Pascals.
    :param units: ``PressureUnitsEnum`` integer for desired output.
    :return: ndarray — pressure in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _PRESSURE_FROM_PA[units]


# ---------------------------------------------------------------------------
# Fireline Intensity  (base = BtusPerFootPerSecond = 0)
# ---------------------------------------------------------------------------
_FLI_TO_BASE = np.array([
    1.0,                    # 0 BtusPerFootPerSecond
    0.01666666666666667,    # 1 BtusPerFootPerMinute
    0.2886719,              # 2 KilojoulesPerMeterPerSecond
    0.00481120819,          # 3 KilojoulesPerMeterPerMinute
    0.2886719,              # 4 KilowattsPerMeter
])

_FLI_FROM_BASE = np.array([
    1.0,           # 0 BtusPerFootPerSecond
    60.0,          # 1 BtusPerFootPerMinute
    3.464140419,   # 2 KilojoulesPerMeterPerSecond
    207.848,       # 3 KilojoulesPerMeterPerMinute
    3.464140419,   # 4 KilowattsPerMeter
])


def fireline_intensity_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert fireline intensity to BTU/ft/s (base unit).

    :param value: Fireline intensity (*S) or scalar, in ``units``.
    :param units: ``FirelineIntensityUnitsEnum`` integer
        (0=BtusPerFootPerSecond, 4=KilowattsPerMeter, etc.).
    :return: ndarray — fireline intensity in BTU/ft/s.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _FLI_TO_BASE[units]


def fireline_intensity_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert fireline intensity from BTU/ft/s to the requested units.

    :param value: Fireline intensity (*S) or scalar in BTU/ft/s.
    :param units: ``FirelineIntensityUnitsEnum`` integer for desired output.
    :return: ndarray — fireline intensity in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _FLI_FROM_BASE[units]


# ---------------------------------------------------------------------------
# Heat Per Unit Area  (base = BtusPerSquareFoot = 0)
# ---------------------------------------------------------------------------
_HPUA_TO_BASE = np.array([
    1.0,    # 0 BtusPerSquareFoot
    0.088,  # 1 KilojoulesPerSquareMeter
    0.088,  # 2 KilowattSecondsPerSquareMeter
])

_HPUA_FROM_BASE = np.array([
    1.0,               # 0 BtusPerSquareFoot
    1.0 / 0.088,       # 1 KilojoulesPerSquareMeter
    1.0 / 0.088,       # 2 KilowattSecondsPerSquareMeter
])


def hpua_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert heat per unit area to BTU/ft² (base unit).

    :param value: Heat per unit area (*S) or scalar, in ``units``.
    :param units: ``HeatPerUnitAreaUnitsEnum`` integer
        (0=BtusPerSquareFoot, 1=KilojoulesPerSquareMeter, 2=KilowattSecondsPerSquareMeter).
    :return: ndarray — heat per unit area in BTU/ft².
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _HPUA_TO_BASE[units]


def hpua_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert heat per unit area from BTU/ft² to the requested units.

    :param value: Heat per unit area (*S) or scalar in BTU/ft².
    :param units: ``HeatPerUnitAreaUnitsEnum`` integer for desired output.
    :return: ndarray — heat per unit area in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _HPUA_FROM_BASE[units]


# ---------------------------------------------------------------------------
# Reaction Intensity / Heat Source  (base = BtusPerSqFtPerMinute = 0)
# ---------------------------------------------------------------------------
_RI_TO_BASE = np.array([
    1.0,    # 0 BtusPerSquareFootPerMinute
    60.0,   # 1 BtusPerSquareFootPerSecond
    5.28,   # 2 KilojoulesPerSquareMeterPerSecond
    0.088,  # 3 KilojoulesPerSquareMeterPerMinute
    0.528,  # 4 KilowattsPerSquareMeter
])

_RI_FROM_BASE = np.array([
    1.0,              # 0 BtusPerSquareFootPerMinute
    1.0 / 60.0,       # 1 BtusPerSquareFootPerSecond
    1.0 / 5.28,       # 2 KilojoulesPerSquareMeterPerSecond
    1.0 / 0.088,      # 3 KilojoulesPerSquareMeterPerMinute
    1.0 / 0.528,      # 4 KilowattsPerSquareMeter
])


def reaction_intensity_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert reaction intensity to BTU/ft²/min (base unit).

    :param value: Reaction intensity (*S) or scalar, in ``units``.
    :param units: ``HeatSourceAndReactionIntensityUnitsEnum`` integer
        (0=BtusPerSquareFootPerMinute, 4=KilowattsPerSquareMeter, etc.).
    :return: ndarray — reaction intensity in BTU/ft²/min.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _RI_TO_BASE[units]


def reaction_intensity_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert reaction intensity from BTU/ft²/min to the requested units.

    :param value: Reaction intensity (*S) or scalar in BTU/ft²/min.
    :param units: ``HeatSourceAndReactionIntensityUnitsEnum`` integer for
        desired output.
    :return: ndarray — reaction intensity in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _RI_FROM_BASE[units]


# ---------------------------------------------------------------------------
# Loading  (base = PoundsPerSquareFoot = 0)
# ---------------------------------------------------------------------------
_LOAD_TO_BASE = np.array([
    1.0,              # 0 PoundsPerSquareFoot
    0.02296841643,    # 1 TonsPerAcre
    0.10197162129,    # 2 TonnesPerHectare
    0.20481754075,    # 3 KilogramsPerSquareMeter
])

_LOAD_FROM_BASE = np.array([
    1.0,                           # 0 PoundsPerSquareFoot
    1.0 / 0.02296841643,           # 1 TonsPerAcre
    1.0 / 0.10197162129,           # 2 TonnesPerHectare
    1.0 / 0.20481754075,           # 3 KilogramsPerSquareMeter
])


def loading_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert fuel loading to lb/ft² (base unit).

    :param value: Loading value (*S) or scalar, in ``units``.
    :param units: ``LoadingUnitsEnum`` integer
        (0=PoundsPerSquareFoot, 1=TonsPerAcre, etc.).
    :return: ndarray — loading in lb/ft².
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _LOAD_TO_BASE[units]


def loading_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert fuel loading from lb/ft² to the requested units.

    :param value: Loading value (*S) or scalar in lb/ft².
    :param units: ``LoadingUnitsEnum`` integer for desired output.
    :return: ndarray — loading in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _LOAD_FROM_BASE[units]


# ---------------------------------------------------------------------------
# Density  (base = PoundsPerCubicFoot = 0)
# ---------------------------------------------------------------------------
_DENSITY_TO_BASE = np.array([
    1.0,           # 0 PoundsPerCubicFoot
    0.062427961,   # 1 KilogramsPerCubicMeter
])

_DENSITY_FROM_BASE = np.array([
    1.0,                     # 0 PoundsPerCubicFoot
    1.0 / 0.062427961,       # 1 KilogramsPerCubicMeter
])


def density_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert density to lb/ft³ (base unit).

    :param value: Density value (*S) or scalar, in ``units``.
    :param units: ``DensityUnitsEnum`` integer
        (0=PoundsPerCubicFoot, 1=KilogramsPerCubicMeter).
    :return: ndarray — density in lb/ft³.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _DENSITY_TO_BASE[units]


def density_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert density from lb/ft³ to the requested units.

    :param value: Density value (*S) or scalar in lb/ft³.
    :param units: ``DensityUnitsEnum`` integer for desired output.
    :return: ndarray — density in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _DENSITY_FROM_BASE[units]


# ---------------------------------------------------------------------------
# Time  (base = Minutes = 0)
# ---------------------------------------------------------------------------
_TIME_TO_MIN = np.array([
    1.0,                       # 0 Minutes
    1.0 / 60.0,                # 1 Seconds
    60.0,                      # 2 Hours
    24.0 * 60.0,               # 3 Days
    365.25 * 24.0 * 60.0,      # 4 Years
])

_TIME_FROM_MIN = np.array([
    1.0,                            # 0 Minutes
    60.0,                           # 1 Seconds
    1.0 / 60.0,                     # 2 Hours
    1.0 / (24.0 * 60.0),            # 3 Days
    1.0 / (365.25 * 24.0 * 60.0),  # 4 Years
])


def time_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert time to minutes (base unit).

    :param value: Time value (*S) or scalar, in ``units``.
    :param units: ``TimeUnitsEnum`` integer (0=Minutes, 1=Seconds, 2=Hours, etc.).
    :return: ndarray — time in minutes.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _TIME_TO_MIN[units]


def time_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """
    Convert time from minutes to the requested units.

    :param value: Time value (*S) or scalar in minutes.
    :param units: ``TimeUnitsEnum`` integer for desired output.
    :return: ndarray — time in the requested units.
    """
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _TIME_FROM_MIN[units]


# ---------------------------------------------------------------------------
# Unit enum classes — defined here so that code which relies on e.g.
# SpeedUnits.SpeedUnitsEnum continues to work unchanged.
# ---------------------------------------------------------------------------

class AreaUnits:
    class AreaUnitsEnum:
        SquareFeet = 0
        Acres = 1
        Hectares = 2
        SquareMeters = 3
        SquareMiles = 4
        SquareKilometers = 5

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert area to square feet (base unit). Delegates to ``area_to_base()``."""
        return area_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert area from square feet to requested units. Delegates to ``area_from_base()``."""
        return area_from_base(value, units)

class BasalAreaUnits:
    class BasalAreaUnitsEnum:
        SquareFeetPerAcre = 0
        SquareMetersPerHectare = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert basal area to ft²/acre (base unit, units=0 pass-through)."""
        arr = np.asarray(value, dtype=float)
        if units == 0:
            return arr
        # 1 m²/ha ≈ 4.35889 ft²/acre
        return arr * 4.35889

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert basal area from ft²/acre to requested units."""
        arr = np.asarray(value, dtype=float)
        if units == 0:
            return arr
        return arr / 4.35889

class LengthUnits:
    class LengthUnitsEnum:
        Feet = 0
        Inches = 1
        Millimeters = 2
        Centimeters = 3
        Meters = 4
        Chains = 5
        Miles = 6
        Kilometers = 7

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert length to feet (base unit). Delegates to ``length_to_base()``."""
        return length_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert length from feet to requested units. Delegates to ``length_from_base()``."""
        return length_from_base(value, units)

class LoadingUnits:
    class LoadingUnitsEnum:
        PoundsPerSquareFoot = 0
        TonsPerAcre = 1
        TonnesPerHectare = 2
        KilogramsPerSquareMeter = 3

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert fuel loading to lb/ft² (base unit). Delegates to ``loading_to_base()``."""
        return loading_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert fuel loading from lb/ft² to requested units. Delegates to ``loading_from_base()``."""
        return loading_from_base(value, units)

class PressureUnits:
    class PressureUnitsEnum:
        Pascal = 0
        HectoPascal = 1
        KiloPascal = 2
        MegaPascal = 3
        GigaPascal = 4
        Bar = 5
        Atmosphere = 6
        TechnicalAtmosphere = 7
        PoundPerSquareInch = 8

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert pressure to Pascals (base unit). Delegates to ``pressure_to_base()``."""
        return pressure_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert pressure from Pascals to requested units. Delegates to ``pressure_from_base()``."""
        return pressure_from_base(value, units)

class SurfaceAreaToVolumeUnits:
    class SurfaceAreaToVolumeUnitsEnum:
        SquareFeetOverCubicFeet = 0
        SquareMetersOverCubicMeters = 1
        SquareInchesOverCubicInches = 2
        SquareCentimetersOverCubicCentimeters = 3

    # Conversion factors to ft²/ft³ (base)
    _TO_BASE = [1.0, 0.3048, 12.0, 0.032808]
    _FROM_BASE = [1.0, 1.0 / 0.3048, 1.0 / 12.0, 1.0 / 0.032808]

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert SAVR to ft²/ft³ (base unit)."""
        arr = np.asarray(value, dtype=float)
        if units == 0:
            return arr
        factors = [1.0, 0.3048, 12.0, 0.032808]
        return arr * factors[units]

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert SAVR from ft²/ft³ to requested units."""
        arr = np.asarray(value, dtype=float)
        if units == 0:
            return arr
        factors = [1.0, 1.0 / 0.3048, 1.0 / 12.0, 1.0 / 0.032808]
        return arr * factors[units]

class SpeedUnits:
    class SpeedUnitsEnum:
        FeetPerMinute = 0
        ChainsPerHour = 1
        MetersPerSecond = 2
        MetersPerMinute = 3
        MetersPerHour = 4
        MilesPerHour = 5
        KilometersPerHour = 6

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert speed to ft/min (base unit). Delegates to ``speed_to_base()``."""
        return speed_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert speed from ft/min to requested units. Delegates to ``speed_from_base()``."""
        return speed_from_base(value, units)

class FractionUnits:
    class FractionUnitsEnum:
        Fraction = 0
        Percent = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert fraction/percent to fraction (base unit). Delegates to ``fraction_to_base()``."""
        return fraction_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert fraction to requested units. Delegates to ``fraction_from_base()``."""
        return fraction_from_base(value, units)

class SlopeUnits:
    class SlopeUnitsEnum:
        Degrees = 0
        Percent = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert slope to degrees (base unit). Delegates to ``slope_to_base()``."""
        return slope_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert slope from degrees to requested units. Delegates to ``slope_from_base()``."""
        return slope_from_base(value, units)

class DensityUnits:
    class DensityUnitsEnum:
        PoundsPerCubicFoot = 0
        KilogramsPerCubicMeter = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert density to lb/ft³ (base unit). Delegates to ``density_to_base()``."""
        return density_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert density from lb/ft³ to requested units. Delegates to ``density_from_base()``."""
        return density_from_base(value, units)

class HeatOfCombustionUnits:
    class HeatOfCombustionUnitsEnum:
        BtusPerPound = 0
        KilojoulesPerKilogram = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert heat of combustion to BTU/lb (base unit)."""
        arr = np.asarray(value, dtype=float)
        if units == 0:
            return arr
        # 1 kJ/kg ≈ 0.429923 BTU/lb
        return arr * 0.429923

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert heat of combustion from BTU/lb to requested units."""
        arr = np.asarray(value, dtype=float)
        if units == 0:
            return arr
        return arr / 0.429923


class HeatSinkUnits:
    class HeatSinkUnitsEnum:
        BtusPerCubicFoot = 0
        KilojoulesPerCubicMeter = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert heat sink to BTU/ft³ (base unit)."""
        arr = np.asarray(value, dtype=float)
        if units == 0:
            return arr
        # 1 kJ/m³ ≈ 0.026839 BTU/ft³
        return arr * 0.026839

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert heat sink from BTU/ft³ to requested units."""
        arr = np.asarray(value, dtype=float)
        if units == 0:
            return arr
        return arr / 0.026839

class HeatPerUnitAreaUnits:
    class HeatPerUnitAreaUnitsEnum:
        BtusPerSquareFoot = 0
        KilojoulesPerSquareMeter = 1
        KilowattSecondsPerSquareMeter = 2

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert heat per unit area to BTU/ft² (base unit). Delegates to ``hpua_to_base()``."""
        return hpua_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert heat per unit area from BTU/ft² to requested units. Delegates to ``hpua_from_base()``."""
        return hpua_from_base(value, units)

class HeatSourceAndReactionIntensityUnits:
    class HeatSourceAndReactionIntensityUnitsEnum:
        BtusPerSquareFootPerMinute = 0
        BtusPerSquareFootPerSecond = 1
        KilojoulesPerSquareMeterPerSecond = 2
        KilojoulesPerSquareMeterPerMinute = 3
        KilowattsPerSquareMeter = 4

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert reaction intensity to BTU/ft²/min (base unit). Delegates to ``reaction_intensity_to_base()``."""
        return reaction_intensity_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert reaction intensity from BTU/ft²/min to requested units. Delegates to ``reaction_intensity_from_base()``."""
        return reaction_intensity_from_base(value, units)

class FirelineIntensityUnits:
    class FirelineIntensityUnitsEnum:
        BtusPerFootPerSecond = 0
        BtusPerFootPerMinute = 1
        KilojoulesPerMeterPerSecond = 2
        KilojoulesPerMeterPerMinute = 3
        KilowattsPerMeter = 4

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert fireline intensity to BTU/ft/s (base unit). Delegates to ``fireline_intensity_to_base()``."""
        return fireline_intensity_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert fireline intensity from BTU/ft/s to requested units. Delegates to ``fireline_intensity_from_base()``."""
        return fireline_intensity_from_base(value, units)

class TemperatureUnits:
    class TemperatureUnitsEnum:
        Fahrenheit = 0
        Celsius = 1
        Kelvin = 2

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert temperature to °F (base unit). Delegates to ``temp_to_base()``."""
        return temp_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert temperature from °F to requested units. Delegates to ``temp_from_base()``."""
        return temp_from_base(value, units)

class TimeUnits:
    class TimeUnitsEnum:
        Minutes = 0
        Seconds = 1
        Hours = 2
        Days = 3
        Years = 4

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert time to minutes (base unit). Delegates to ``time_to_base()``."""
        return time_to_base(value, units)

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Convert time from minutes to requested units. Delegates to ``time_from_base()``."""
        return time_from_base(value, units)

