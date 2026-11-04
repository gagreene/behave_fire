"""Native unit converters with the historical enum namespaces."""
from __future__ import annotations
from typing import Union, Optional
import numpy as np
from ._bridge import _call

def area_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Area.from_base', locals())

def area_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Area.to_base', locals())

def density_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Density.from_base', locals())

def density_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Density.to_base', locals())

def fireline_intensity_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.FirelineIntensity.from_base', locals())

def fireline_intensity_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.FirelineIntensity.to_base', locals())

def fraction_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Fraction.from_base', locals())

def fraction_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Fraction.to_base', locals())

def hpua_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.HeatPerUnitArea.from_base', locals())

def hpua_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.HeatPerUnitArea.to_base', locals())

def length_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Length.from_base', locals())

def length_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Length.to_base', locals())

def loading_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Loading.from_base', locals())

def loading_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Loading.to_base', locals())

def pressure_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Pressure.from_base', locals())

def pressure_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Pressure.to_base', locals())

def reaction_intensity_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.HeatSourceAndReactionIntensity.from_base', locals())

def reaction_intensity_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.HeatSourceAndReactionIntensity.to_base', locals())

def slope_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Slope.from_base', locals())

def slope_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Slope.to_base', locals())

def speed_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Speed.from_base', locals())

def speed_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Speed.to_base', locals())

def temp_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Temperature.from_base', locals())

def temp_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Temperature.to_base', locals())

def time_from_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Time.from_base', locals())

def time_to_base(value: Union[float, np.ndarray], units: int) -> np.ndarray:
    """Execute the explicit native operation.

    :param value: Input matching the public API.
    :param units: Input matching the public API.
    :return: Calculation result or updated receiver state.
    """
    return _call('units.Time.to_base', locals())

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
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Area.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Area.from_base', locals())

class BasalAreaUnits:

    class BasalAreaUnitsEnum:
        SquareFeetPerAcre = 0
        SquareMetersPerHectare = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.BasalArea.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.BasalArea.from_base', locals())

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
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Length.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Length.from_base', locals())

class LoadingUnits:

    class LoadingUnitsEnum:
        PoundsPerSquareFoot = 0
        TonsPerAcre = 1
        TonnesPerHectare = 2
        KilogramsPerSquareMeter = 3

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Loading.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Loading.from_base', locals())

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
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Pressure.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Pressure.from_base', locals())

class SurfaceAreaToVolumeUnits:

    class SurfaceAreaToVolumeUnitsEnum:
        SquareFeetOverCubicFeet = 0
        SquareMetersOverCubicMeters = 1
        SquareInchesOverCubicInches = 2
        SquareCentimetersOverCubicCentimeters = 3

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.SurfaceAreaToVolume.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.SurfaceAreaToVolume.from_base', locals())

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
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Speed.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Speed.from_base', locals())

class FractionUnits:

    class FractionUnitsEnum:
        Fraction = 0
        Percent = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Fraction.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Fraction.from_base', locals())

class SlopeUnits:

    class SlopeUnitsEnum:
        Degrees = 0
        Percent = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Slope.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Slope.from_base', locals())

class DensityUnits:

    class DensityUnitsEnum:
        PoundsPerCubicFoot = 0
        KilogramsPerCubicMeter = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Density.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Density.from_base', locals())

class HeatOfCombustionUnits:

    class HeatOfCombustionUnitsEnum:
        BtusPerPound = 0
        KilojoulesPerKilogram = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.HeatOfCombustion.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.HeatOfCombustion.from_base', locals())

class HeatSinkUnits:

    class HeatSinkUnitsEnum:
        BtusPerCubicFoot = 0
        KilojoulesPerCubicMeter = 1

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.HeatSink.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.HeatSink.from_base', locals())

class HeatPerUnitAreaUnits:

    class HeatPerUnitAreaUnitsEnum:
        BtusPerSquareFoot = 0
        KilojoulesPerSquareMeter = 1
        KilowattSecondsPerSquareMeter = 2

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.HeatPerUnitArea.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.HeatPerUnitArea.from_base', locals())

class HeatSourceAndReactionIntensityUnits:

    class HeatSourceAndReactionIntensityUnitsEnum:
        BtusPerSquareFootPerMinute = 0
        BtusPerSquareFootPerSecond = 1
        KilojoulesPerSquareMeterPerSecond = 2
        KilojoulesPerSquareMeterPerMinute = 3
        KilowattsPerSquareMeter = 4

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.HeatSourceAndReactionIntensity.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.HeatSourceAndReactionIntensity.from_base', locals())

class FirelineIntensityUnits:

    class FirelineIntensityUnitsEnum:
        BtusPerFootPerSecond = 0
        BtusPerFootPerMinute = 1
        KilojoulesPerMeterPerSecond = 2
        KilojoulesPerMeterPerMinute = 3
        KilowattsPerMeter = 4

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.FirelineIntensity.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.FirelineIntensity.from_base', locals())

class TemperatureUnits:

    class TemperatureUnitsEnum:
        Fahrenheit = 0
        Celsius = 1
        Kelvin = 2

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Temperature.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Temperature.from_base', locals())

class TimeUnits:

    class TimeUnitsEnum:
        Minutes = 0
        Seconds = 1
        Hours = 2
        Days = 3
        Years = 4

    @staticmethod
    def toBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Time.to_base', locals())

    @staticmethod
    def fromBaseUnits(value: Union[float, np.ndarray], units: int) -> np.ndarray:
        """Execute the explicit native operation.

        :param value: Input matching the public API.
        :param units: Input matching the public API.
        :return: Calculation result or updated receiver state.
        """
        return _call('units.Time.from_base', locals())
