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


def speed_to_base(value, units):
    """Convert speed to ft/min.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _SPEED_TO_FPM[units]


def speed_from_base(value, units):
    """Convert speed from ft/min to requested units.  Always returns ndarray."""
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


def length_to_base(value, units):
    """Convert length to feet.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _LENGTH_TO_FT[units]


def length_from_base(value, units):
    """Convert length from feet to requested units.  Always returns ndarray."""
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


def area_to_base(value, units):
    """Convert area to sq ft.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _AREA_TO_SQFT[units]


def area_from_base(value, units):
    """Convert area from sq ft.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _AREA_FROM_SQFT[units]


# ---------------------------------------------------------------------------
# Fraction / Percentage  (base = Fraction = 0)
# ---------------------------------------------------------------------------

def fraction_to_base(value, units):
    """Convert fraction/percent to fraction (base).  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    # units == 1 → Percent
    return arr / 100.0


def fraction_from_base(value, units):
    """Convert fraction to requested units.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * 100.0


# ---------------------------------------------------------------------------
# Temperature  (base = Fahrenheit = 0)
# ---------------------------------------------------------------------------

def temp_to_base(value, units):
    """Convert temperature to °F.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    if units == 1:   # Celsius
        return arr * (9.0 / 5.0) + 32.0
    if units == 2:   # Kelvin
        return (arr - 273.15) * (9.0 / 5.0) + 32.0
    return arr


def temp_from_base(value, units):
    """Convert temperature from °F to requested units.  Always returns ndarray."""
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

def slope_to_base(value, units):
    """Convert slope to degrees.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    # units == 1 → Percent
    return np.degrees(np.arctan(arr / 100.0))


def slope_from_base(value, units):
    """Convert slope from degrees to requested units.  Always returns ndarray."""
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


def pressure_to_base(value, units):
    """Convert pressure to Pascals.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _PRESSURE_TO_PA[units]


def pressure_from_base(value, units):
    """Convert pressure from Pascals to requested units.  Always returns ndarray."""
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


def fireline_intensity_to_base(value, units):
    """Convert fireline intensity to BTU/ft/s.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _FLI_TO_BASE[units]


def fireline_intensity_from_base(value, units):
    """Convert fireline intensity from BTU/ft/s.  Always returns ndarray."""
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


def hpua_to_base(value, units):
    """Convert heat per unit area to BTU/ft².  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _HPUA_TO_BASE[units]


def hpua_from_base(value, units):
    """Convert heat per unit area from BTU/ft².  Always returns ndarray."""
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


def reaction_intensity_to_base(value, units):
    """Convert reaction intensity to BTU/ft²/min.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _RI_TO_BASE[units]


def reaction_intensity_from_base(value, units):
    """Convert reaction intensity from BTU/ft²/min.  Always returns ndarray."""
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


def loading_to_base(value, units):
    """Convert loading to lb/ft².  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _LOAD_TO_BASE[units]


def loading_from_base(value, units):
    """Convert loading from lb/ft².  Always returns ndarray."""
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


def density_to_base(value, units):
    """Convert density to lb/ft³.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _DENSITY_TO_BASE[units]


def density_from_base(value, units):
    """Convert density from lb/ft³.  Always returns ndarray."""
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


def time_to_base(value, units):
    """Convert time to minutes.  Always returns ndarray."""
    arr = np.asarray(value, dtype=float)
    if units == 0:
        return arr
    return arr * _TIME_TO_MIN[units]


def time_from_base(value, units):
    """Convert time from minutes.  Always returns ndarray."""
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

class BasalAreaUnits:
    class BasalAreaUnitsEnum:
        SquareFeetPerAcre = 0
        SquareMetersPerHectare = 1

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

class LoadingUnits:
    class LoadingUnitsEnum:
        PoundsPerSquareFoot = 0
        TonsPerAcre = 1
        TonnesPerHectare = 2
        KilogramsPerSquareMeter = 3

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

class SurfaceAreaToVolumeUnits:
    class SurfaceAreaToVolumeUnitsEnum:
        SquareFeetOverCubicFeet = 0
        SquareMetersOverCubicMeters = 1
        SquareInchesOverCubicInches = 2
        SquareCentimetersOverCubicCentimeters = 3

class SpeedUnits:
    class SpeedUnitsEnum:
        FeetPerMinute = 0
        ChainsPerHour = 1
        MetersPerSecond = 2
        MetersPerMinute = 3
        MetersPerHour = 4
        MilesPerHour = 5
        KilometersPerHour = 6

class FractionUnits:
    class FractionUnitsEnum:
        Fraction = 0
        Percent = 1

class SlopeUnits:
    class SlopeUnitsEnum:
        Degrees = 0
        Percent = 1

class DensityUnits:
    class DensityUnitsEnum:
        PoundsPerCubicFoot = 0
        KilogramsPerCubicMeter = 1

class HeatOfCombustionUnits:
    class HeatOfCombustionUnitsEnum:
        BtusPerPound = 0
        KilojoulesPerKilogram = 1

class HeatSinkUnits:
    class HeatSinkUnitsEnum:
        BtusPerCubicFoot = 0
        KilojoulesPerCubicMeter = 1

class HeatPerUnitAreaUnits:
    class HeatPerUnitAreaUnitsEnum:
        BtusPerSquareFoot = 0
        KilojoulesPerSquareMeter = 1
        KilowattSecondsPerSquareMeter = 2

class HeatSourceAndReactionIntensityUnits:
    class HeatSourceAndReactionIntensityUnitsEnum:
        BtusPerSquareFootPerMinute = 0
        BtusPerSquareFootPerSecond = 1
        KilojoulesPerSquareMeterPerSecond = 2
        KilojoulesPerSquareMeterPerMinute = 3
        KilowattsPerSquareMeter = 4

class FirelineIntensityUnits:
    class FirelineIntensityUnitsEnum:
        BtusPerFootPerSecond = 0
        BtusPerFootPerMinute = 1
        KilojoulesPerMeterPerSecond = 2
        KilojoulesPerMeterPerMinute = 3
        KilowattsPerMeter = 4

class TemperatureUnits:
    class TemperatureUnitsEnum:
        Fahrenheit = 0
        Celsius = 1
        Kelvin = 2

class TimeUnits:
    class TimeUnitsEnum:
        Minutes = 0
        Seconds = 1
        Hours = 2
        Days = 3
        Years = 4

