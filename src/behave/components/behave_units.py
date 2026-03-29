"""
Behave Fire Behavior Model - Unit Conversion Module
A collection of units conversion functions and enums for the Behave model.

This module provides comprehensive unit conversion capabilities for all measurements
used in fire behavior modeling, including lengths, areas, speeds, temperatures, etc.
Each unit category has a base unit defined, with conversion functions to/from base units.
"""

import math


class AreaUnits:
    """Area unit conversions. Base unit: Square Feet"""
    
    class AreaUnitsEnum:
        SquareFeet = 0          # base area unit
        Acres = 1
        Hectares = 2
        SquareMeters = 3
        SquareMiles = 4
        SquareKilometers = 5
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert area to base units (Square Feet)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants to Square Feet (match C++ behaveUnits.cpp exactly)
        ACRES_TO_SQ_FT = 43560.002160576107
        HECTARES_TO_SQ_FT = 107639.10416709723
        SQ_METERS_TO_SQ_FT = 10.76391041671
        SQ_MILES_TO_SQ_FT = 27878400.0
        SQ_KM_TO_SQ_FT = 10763910.416709721
        
        if units == AreaUnits.AreaUnitsEnum.SquareFeet:
            pass  # Already in base
        elif units == AreaUnits.AreaUnitsEnum.Acres:
            value *= ACRES_TO_SQ_FT
        elif units == AreaUnits.AreaUnitsEnum.Hectares:
            value *= HECTARES_TO_SQ_FT
        elif units == AreaUnits.AreaUnitsEnum.SquareMeters:
            value *= SQ_METERS_TO_SQ_FT
        elif units == AreaUnits.AreaUnitsEnum.SquareMiles:
            value *= SQ_MILES_TO_SQ_FT
        elif units == AreaUnits.AreaUnitsEnum.SquareKilometers:
            value *= SQ_KM_TO_SQ_FT
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert area from base units (Square Feet)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants from Square Feet (match C++ behaveUnits.cpp exactly)
        SQ_FT_TO_ACRES = 2.295684e-05
        SQ_FT_TO_HECTARES = 0.0000092903036
        SQ_FT_TO_SQ_METERS = 0.0929030353835
        SQ_FT_TO_SQ_MILES = 3.5870064279e-08
        SQ_FT_TO_SQ_KM = 9.290304e-08
        
        if units == AreaUnits.AreaUnitsEnum.SquareFeet:
            pass  # Already in base
        elif units == AreaUnits.AreaUnitsEnum.Acres:
            value *= SQ_FT_TO_ACRES
        elif units == AreaUnits.AreaUnitsEnum.Hectares:
            value *= SQ_FT_TO_HECTARES
        elif units == AreaUnits.AreaUnitsEnum.SquareMeters:
            value *= SQ_FT_TO_SQ_METERS
        elif units == AreaUnits.AreaUnitsEnum.SquareMiles:
            value *= SQ_FT_TO_SQ_MILES
        elif units == AreaUnits.AreaUnitsEnum.SquareKilometers:
            value *= SQ_FT_TO_SQ_KM
        
        return value


class BasalAreaUnits:
    """Basal Area unit conversions. Base unit: Square Feet Per Acre"""
    
    class BasalAreaUnitsEnum:
        SquareFeetPerAcre = 0       # base basal area unit
        SquareMetersPerHectare = 1
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert basal area to base units (Square Feet Per Acre)"""
        if value == 0.0:
            return 0.0
        
        SQ_M_PER_HA_TO_SQ_FT_PER_ACRE = 0.229568411
        
        if units == BasalAreaUnits.BasalAreaUnitsEnum.SquareFeetPerAcre:
            pass  # Already in base
        elif units == BasalAreaUnits.BasalAreaUnitsEnum.SquareMetersPerHectare:
            value *= SQ_M_PER_HA_TO_SQ_FT_PER_ACRE
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert basal area from base units (Square Feet Per Acre)"""
        if value == 0.0:
            return 0.0
        
        SQ_FT_PER_ACRE_TO_SQ_M_PER_HA = 1.0 / 0.229568411
        
        if units == BasalAreaUnits.BasalAreaUnitsEnum.SquareFeetPerAcre:
            pass  # Already in base
        elif units == BasalAreaUnits.BasalAreaUnitsEnum.SquareMetersPerHectare:
            value *= SQ_FT_PER_ACRE_TO_SQ_M_PER_HA
        
        return value


class LengthUnits:
    """Length unit conversions. Base unit: Feet"""
    
    class LengthUnitsEnum:
        Feet = 0                   # base length unit
        Inches = 1
        Millimeters = 2
        Centimeters = 3
        Meters = 4
        Chains = 5
        Miles = 6
        Kilometers = 7
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert length to base units (Feet)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants to Feet
        INCHES_TO_FEET = 0.08333333333333
        METERS_TO_FEET = 3.2808398950131
        MILLIMETERS_TO_FEET = 0.003280839895
        CENTIMETERS_TO_FEET = 0.03280839895
        CHAINS_TO_FEET = 66.0
        MILES_TO_FEET = 5280.0
        KILOMETERS_TO_FEET = 3280.8398950131
        
        if units == LengthUnits.LengthUnitsEnum.Feet:
            pass  # Already in base
        elif units == LengthUnits.LengthUnitsEnum.Inches:
            value *= INCHES_TO_FEET
        elif units == LengthUnits.LengthUnitsEnum.Millimeters:
            value *= MILLIMETERS_TO_FEET
        elif units == LengthUnits.LengthUnitsEnum.Centimeters:
            value *= CENTIMETERS_TO_FEET
        elif units == LengthUnits.LengthUnitsEnum.Meters:
            value *= METERS_TO_FEET
        elif units == LengthUnits.LengthUnitsEnum.Chains:
            value *= CHAINS_TO_FEET
        elif units == LengthUnits.LengthUnitsEnum.Miles:
            value *= MILES_TO_FEET
        elif units == LengthUnits.LengthUnitsEnum.Kilometers:
            value *= KILOMETERS_TO_FEET
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert length from base units (Feet)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants from Feet
        FEET_TO_INCHES = 12.0
        FEET_TO_CENTIMETERS = 30.480
        FEET_TO_METERS = 0.3048
        FEET_TO_CHAINS = 0.0151515151515
        FEET_TO_MILES = 0.0001893939393939394
        FEET_TO_KILOMETERS = 0.0003048
        
        if units == LengthUnits.LengthUnitsEnum.Feet:
            pass  # Already in base
        elif units == LengthUnits.LengthUnitsEnum.Inches:
            value *= FEET_TO_INCHES
        elif units == LengthUnits.LengthUnitsEnum.Centimeters:
            value *= FEET_TO_CENTIMETERS
        elif units == LengthUnits.LengthUnitsEnum.Meters:
            value *= FEET_TO_METERS
        elif units == LengthUnits.LengthUnitsEnum.Chains:
            value *= FEET_TO_CHAINS
        elif units == LengthUnits.LengthUnitsEnum.Miles:
            value *= FEET_TO_MILES
        elif units == LengthUnits.LengthUnitsEnum.Kilometers:
            value *= FEET_TO_KILOMETERS
        
        return value


class LoadingUnits:
    """Fuel loading unit conversions. Base unit: Pounds Per Square Foot"""
    
    class LoadingUnitsEnum:
        PoundsPerSquareFoot = 0     # base loading unit
        TonsPerAcre = 1
        TonnesPerHectare = 2
        KilogramsPerSquareMeter = 3
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert loading to base units (Pounds Per Square Foot)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants to Pounds Per Square Foot
        TONS_PER_ACRE_TO_LBS_PER_SQ_FT = 0.02296841643
        TONNES_PER_HA_TO_LBS_PER_SQ_FT = 0.10197162129
        KG_PER_SQ_M_TO_LBS_PER_SQ_FT = 0.20481754075
        
        if units == LoadingUnits.LoadingUnitsEnum.PoundsPerSquareFoot:
            pass  # Already in base
        elif units == LoadingUnits.LoadingUnitsEnum.TonsPerAcre:
            value *= TONS_PER_ACRE_TO_LBS_PER_SQ_FT
        elif units == LoadingUnits.LoadingUnitsEnum.TonnesPerHectare:
            value *= TONNES_PER_HA_TO_LBS_PER_SQ_FT
        elif units == LoadingUnits.LoadingUnitsEnum.KilogramsPerSquareMeter:
            value *= KG_PER_SQ_M_TO_LBS_PER_SQ_FT
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert loading from base units (Pounds Per Square Foot)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants from Pounds Per Square Foot
        LBS_PER_SQ_FT_TO_TONS_PER_ACRE = 1.0 / 0.02296841643
        LBS_PER_SQ_FT_TO_TONNES_PER_HA = 1.0 / 0.10197162129
        LBS_PER_SQ_FT_TO_KG_PER_SQ_M = 1.0 / 0.20481754075
        
        if units == LoadingUnits.LoadingUnitsEnum.PoundsPerSquareFoot:
            pass  # Already in base
        elif units == LoadingUnits.LoadingUnitsEnum.TonsPerAcre:
            value *= LBS_PER_SQ_FT_TO_TONS_PER_ACRE
        elif units == LoadingUnits.LoadingUnitsEnum.TonnesPerHectare:
            value *= LBS_PER_SQ_FT_TO_TONNES_PER_HA
        elif units == LoadingUnits.LoadingUnitsEnum.KilogramsPerSquareMeter:
            value *= LBS_PER_SQ_FT_TO_KG_PER_SQ_M
        
        return value


class PressureUnits:
    """Pressure unit conversions. Base unit: Pascal"""
    
    class PressureUnitsEnum:
        Pascal = 0                 # base pressure unit
        HectoPascal = 1            # hPa
        KiloPascal = 2             # kPa
        MegaPascal = 3             # MPa
        GigaPascal = 4             # GPa
        Bar = 5                    # bar
        Atmosphere = 6             # atm
        TechnicalAtmosphere = 7    # at
        PoundPerSquareInch = 8     # psi
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert pressure to base units (Pascal)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants to Pascal
        HPA_TO_PA = 100.0
        KPA_TO_PA = 1000.0
        MPA_TO_PA = 1000000.0
        GPA_TO_PA = 1000000000.0
        BAR_TO_PA = 100000.0
        ATM_TO_PA = 101325.0
        TECHNICAL_ATM_TO_PA = 98066.5
        PSI_TO_PA = 6894.757
        
        if units == PressureUnits.PressureUnitsEnum.Pascal:
            pass  # Already in base
        elif units == PressureUnits.PressureUnitsEnum.HectoPascal:
            value *= HPA_TO_PA
        elif units == PressureUnits.PressureUnitsEnum.KiloPascal:
            value *= KPA_TO_PA
        elif units == PressureUnits.PressureUnitsEnum.MegaPascal:
            value *= MPA_TO_PA
        elif units == PressureUnits.PressureUnitsEnum.GigaPascal:
            value *= GPA_TO_PA
        elif units == PressureUnits.PressureUnitsEnum.Bar:
            value *= BAR_TO_PA
        elif units == PressureUnits.PressureUnitsEnum.Atmosphere:
            value *= ATM_TO_PA
        elif units == PressureUnits.PressureUnitsEnum.TechnicalAtmosphere:
            value *= TECHNICAL_ATM_TO_PA
        elif units == PressureUnits.PressureUnitsEnum.PoundPerSquareInch:
            value *= PSI_TO_PA
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert pressure from base units (Pascal)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants from Pascal
        PA_TO_HPA = 0.01
        PA_TO_KPA = 0.001
        PA_TO_MPA = 0.000001
        PA_TO_GPA = 0.000000001
        PA_TO_BAR = 0.00001
        PA_TO_ATM = 1.0 / 101325.0
        PA_TO_TECHNICAL_ATM = 1.0 / 98066.5
        PA_TO_PSI = 1.0 / 6894.757
        
        if units == PressureUnits.PressureUnitsEnum.Pascal:
            pass  # Already in base
        elif units == PressureUnits.PressureUnitsEnum.HectoPascal:
            value *= PA_TO_HPA
        elif units == PressureUnits.PressureUnitsEnum.KiloPascal:
            value *= PA_TO_KPA
        elif units == PressureUnits.PressureUnitsEnum.MegaPascal:
            value *= PA_TO_MPA
        elif units == PressureUnits.PressureUnitsEnum.GigaPascal:
            value *= PA_TO_GPA
        elif units == PressureUnits.PressureUnitsEnum.Bar:
            value *= PA_TO_BAR
        elif units == PressureUnits.PressureUnitsEnum.Atmosphere:
            value *= PA_TO_ATM
        elif units == PressureUnits.PressureUnitsEnum.TechnicalAtmosphere:
            value *= PA_TO_TECHNICAL_ATM
        elif units == PressureUnits.PressureUnitsEnum.PoundPerSquareInch:
            value *= PA_TO_PSI
        
        return value


class SurfaceAreaToVolumeUnits:
    """Surface Area to Volume (SAVR) unit conversions. Base unit: Square Feet Over Cubic Feet"""
    
    class SurfaceAreaToVolumeUnitsEnum:
        SquareFeetOverCubicFeet = 0         # base SAVR unit
        SquareMetersOverCubicMeters = 1
        SquareInchesOverCubicInches = 2
        SquareCentimetersOverCubicCentimeters = 3
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert SAVR to base units (Square Feet Over Cubic Feet)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants to Square Feet Over Cubic Feet
        SQ_M_OVER_CUB_M_TO_SQ_FT_OVER_CUB_FT = 3.280839895
        SQ_IN_OVER_CUB_IN_TO_SQ_FT_OVER_CUB_FT = 1.0 / 12.0
        SQ_CM_OVER_CUB_CM_TO_SQ_FT_OVER_CUB_FT = 0.03280839895
        
        if units == SurfaceAreaToVolumeUnits.SurfaceAreaToVolumeUnitsEnum.SquareFeetOverCubicFeet:
            pass  # Already in base
        elif units == SurfaceAreaToVolumeUnits.SurfaceAreaToVolumeUnitsEnum.SquareMetersOverCubicMeters:
            value *= SQ_M_OVER_CUB_M_TO_SQ_FT_OVER_CUB_FT
        elif units == SurfaceAreaToVolumeUnits.SurfaceAreaToVolumeUnitsEnum.SquareInchesOverCubicInches:
            value *= SQ_IN_OVER_CUB_IN_TO_SQ_FT_OVER_CUB_FT
        elif units == SurfaceAreaToVolumeUnits.SurfaceAreaToVolumeUnitsEnum.SquareCentimetersOverCubicCentimeters:
            value *= SQ_CM_OVER_CUB_CM_TO_SQ_FT_OVER_CUB_FT
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert SAVR from base units (Square Feet Over Cubic Feet)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants from Square Feet Over Cubic Feet
        SQ_FT_OVER_CUB_FT_TO_SQ_M_OVER_CUB_M = 1.0 / 3.280839895
        SQ_FT_OVER_CUB_FT_TO_SQ_IN_OVER_CUB_IN = 12.0
        SQ_FT_OVER_CUB_FT_TO_SQ_CM_OVER_CUB_CM = 1.0 / 0.03280839895
        
        if units == SurfaceAreaToVolumeUnits.SurfaceAreaToVolumeUnitsEnum.SquareFeetOverCubicFeet:
            pass  # Already in base
        elif units == SurfaceAreaToVolumeUnits.SurfaceAreaToVolumeUnitsEnum.SquareMetersOverCubicMeters:
            value *= SQ_FT_OVER_CUB_FT_TO_SQ_M_OVER_CUB_M
        elif units == SurfaceAreaToVolumeUnits.SurfaceAreaToVolumeUnitsEnum.SquareInchesOverCubicInches:
            value *= SQ_FT_OVER_CUB_FT_TO_SQ_IN_OVER_CUB_IN
        elif units == SurfaceAreaToVolumeUnits.SurfaceAreaToVolumeUnitsEnum.SquareCentimetersOverCubicCentimeters:
            value *= SQ_FT_OVER_CUB_FT_TO_SQ_CM_OVER_CUB_CM
        
        return value


class SpeedUnits:
    """Speed/Velocity unit conversions. Base unit: Feet Per Minute"""
    
    class SpeedUnitsEnum:
        FeetPerMinute = 0          # base velocity unit
        ChainsPerHour = 1
        MetersPerSecond = 2
        MetersPerMinute = 3
        MetersPerHour = 4
        MilesPerHour = 5
        KilometersPerHour = 6
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert speed to base units (Feet Per Minute)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants to Feet Per Minute
        METERS_PER_SECOND_TO_FEET_PER_MINUTE = 196.8503937
        METERS_PER_MINUTE_TO_FEET_PER_MINUTE = 3.28084
        METERS_PER_HOUR_TO_FEET_PER_MINUTE = 0.0547
        CHAINS_PER_HOUR_TO_FEET_PER_MINUTE = 1.1
        MILES_PER_HOUR_TO_FEET_PER_MINUTE = 88.0
        KILOMETERS_PER_HOUR_TO_FEET_PER_MINUTE = 54.680665
        
        if units == SpeedUnits.SpeedUnitsEnum.FeetPerMinute:
            pass  # Already in base
        elif units == SpeedUnits.SpeedUnitsEnum.MetersPerSecond:
            value *= METERS_PER_SECOND_TO_FEET_PER_MINUTE
        elif units == SpeedUnits.SpeedUnitsEnum.MetersPerMinute:
            value *= METERS_PER_MINUTE_TO_FEET_PER_MINUTE
        elif units == SpeedUnits.SpeedUnitsEnum.MetersPerHour:
            value *= METERS_PER_HOUR_TO_FEET_PER_MINUTE
        elif units == SpeedUnits.SpeedUnitsEnum.ChainsPerHour:
            value *= CHAINS_PER_HOUR_TO_FEET_PER_MINUTE
        elif units == SpeedUnits.SpeedUnitsEnum.MilesPerHour:
            value *= MILES_PER_HOUR_TO_FEET_PER_MINUTE
        elif units == SpeedUnits.SpeedUnitsEnum.KilometersPerHour:
            value *= KILOMETERS_PER_HOUR_TO_FEET_PER_MINUTE
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert speed from base units (Feet Per Minute)"""
        if value == 0.0:
            return 0.0
        
        # Conversion constants from Feet Per Minute
        FEET_PER_MINUTE_TO_METERS_PER_SECOND = 0.00508
        FEET_PER_MINUTE_TO_METERS_PER_MINUTE = 0.3048
        FEET_PER_MINUTE_TO_METERS_PER_HOUR = 18.288
        FEET_PER_MINUTE_TO_CHAINS_PER_HOUR = 10.0 / 11.0
        FEET_PER_MINUTE_TO_MILES_PER_HOUR = 0.01136363636
        FEET_PER_MINUTE_TO_KILOMETERS_PER_HOUR = 0.018288
        
        if units == SpeedUnits.SpeedUnitsEnum.FeetPerMinute:
            pass  # Already in base
        elif units == SpeedUnits.SpeedUnitsEnum.MetersPerSecond:
            value *= FEET_PER_MINUTE_TO_METERS_PER_SECOND
        elif units == SpeedUnits.SpeedUnitsEnum.MetersPerMinute:
            value *= FEET_PER_MINUTE_TO_METERS_PER_MINUTE
        elif units == SpeedUnits.SpeedUnitsEnum.MetersPerHour:
            value *= FEET_PER_MINUTE_TO_METERS_PER_HOUR
        elif units == SpeedUnits.SpeedUnitsEnum.ChainsPerHour:
            value *= FEET_PER_MINUTE_TO_CHAINS_PER_HOUR
        elif units == SpeedUnits.SpeedUnitsEnum.MilesPerHour:
            value *= FEET_PER_MINUTE_TO_MILES_PER_HOUR
        elif units == SpeedUnits.SpeedUnitsEnum.KilometersPerHour:
            value *= FEET_PER_MINUTE_TO_KILOMETERS_PER_HOUR
        
        return value


class FractionUnits:
    """Fraction/Percentage unit conversions. Base unit: Fraction"""
    
    class FractionUnitsEnum:
        Fraction = 0               # base fraction unit
        Percent = 1
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert fraction to base units (Fraction)"""
        if value == 0.0:
            return 0.0
        
        if units == FractionUnits.FractionUnitsEnum.Percent:
            value /= 100.0
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert fraction from base units (Fraction)"""
        if value == 0.0:
            return 0.0
        
        if units == FractionUnits.FractionUnitsEnum.Percent:
            value *= 100.0
        
        return value


class SlopeUnits:
    """Slope unit conversions. Base unit: Degrees"""
    
    class SlopeUnitsEnum:
        Degrees = 0                # base slope unit
        Percent = 1
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert slope to base units (Degrees)"""
        if value == 0.0:
            return 0.0
        
        PI = math.pi
        
        if units == SlopeUnits.SlopeUnitsEnum.Percent:
            # Convert percent slope to degrees
            value = (180.0 / PI) * math.atan(value / 100.0)
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert slope from base units (Degrees)"""
        if value == 0.0:
            return 0.0
        
        PI = math.pi
        
        if units == SlopeUnits.SlopeUnitsEnum.Percent:
            # Convert degrees to percent slope
            value = math.tan(value * (PI / 180.0)) * 100.0
        
        return value


class DensityUnits:
    """Density unit conversions. Base unit: Pounds Per Cubic Foot"""
    
    class DensityUnitsEnum:
        PoundsPerCubicFoot = 0     # base density unit
        KilogramsPerCubicMeter = 1
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert density to base units (Pounds Per Cubic Foot)"""
        if value == 0.0:
            return 0.0
        
        KG_PER_CUB_M_TO_LBS_PER_CUB_FT = 0.062427961
        
        if units == DensityUnits.DensityUnitsEnum.PoundsPerCubicFoot:
            pass  # Already in base
        elif units == DensityUnits.DensityUnitsEnum.KilogramsPerCubicMeter:
            value *= KG_PER_CUB_M_TO_LBS_PER_CUB_FT
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert density from base units (Pounds Per Cubic Foot)"""
        if value == 0.0:
            return 0.0
        
        LBS_PER_CUB_FT_TO_KG_PER_CUB_M = 1.0 / 0.062427961
        
        if units == DensityUnits.DensityUnitsEnum.PoundsPerCubicFoot:
            pass  # Already in base
        elif units == DensityUnits.DensityUnitsEnum.KilogramsPerCubicMeter:
            value *= LBS_PER_CUB_FT_TO_KG_PER_CUB_M
        
        return value


class HeatOfCombustionUnits:
    """Heat of Combustion unit conversions. Base unit: BTU Per Pound"""
    
    class HeatOfCombustionUnitsEnum:
        BtusPerPound = 0           # base heat of combustion unit
        KilojoulesPerKilogram = 1
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert heat of combustion to base units (BTU Per Pound)"""
        if value == 0.0:
            return 0.0
        
        KJ_PER_KG_TO_BTU_PER_LB = 0.42992250433
        
        if units == HeatOfCombustionUnits.HeatOfCombustionUnitsEnum.BtusPerPound:
            pass  # Already in base
        elif units == HeatOfCombustionUnits.HeatOfCombustionUnitsEnum.KilojoulesPerKilogram:
            value *= KJ_PER_KG_TO_BTU_PER_LB
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert heat of combustion from base units (BTU Per Pound)"""
        if value == 0.0:
            return 0.0
        
        BTU_PER_LB_TO_KJ_PER_KG = 1.0 / 0.42992250433
        
        if units == HeatOfCombustionUnits.HeatOfCombustionUnitsEnum.BtusPerPound:
            pass  # Already in base
        elif units == HeatOfCombustionUnits.HeatOfCombustionUnitsEnum.KilojoulesPerKilogram:
            value *= BTU_PER_LB_TO_KJ_PER_KG
        
        return value


class HeatSinkUnits:
    """Heat Sink unit conversions. Base unit: BTU Per Cubic Foot"""
    
    class HeatSinkUnitsEnum:
        BtusPerCubicFoot = 0       # base heat sink unit
        KilojoulesPerCubicMeter = 1
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert heat sink to base units (BTU Per Cubic Foot)"""
        if value == 0.0:
            return 0.0
        
        KJ_PER_CUB_M_TO_BTU_PER_CUB_FT = 0.0269281218
        
        if units == HeatSinkUnits.HeatSinkUnitsEnum.BtusPerCubicFoot:
            pass  # Already in base
        elif units == HeatSinkUnits.HeatSinkUnitsEnum.KilojoulesPerCubicMeter:
            value *= KJ_PER_CUB_M_TO_BTU_PER_CUB_FT
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert heat sink from base units (BTU Per Cubic Foot)"""
        if value == 0.0:
            return 0.0
        
        BTU_PER_CUB_FT_TO_KJ_PER_CUB_M = 1.0 / 0.0269281218
        
        if units == HeatSinkUnits.HeatSinkUnitsEnum.BtusPerCubicFoot:
            pass  # Already in base
        elif units == HeatSinkUnits.HeatSinkUnitsEnum.KilojoulesPerCubicMeter:
            value *= BTU_PER_CUB_FT_TO_KJ_PER_CUB_M
        
        return value


class HeatPerUnitAreaUnits:
    """Heat Per Unit Area unit conversions. Base unit: BTU Per Square Foot"""
    
    class HeatPerUnitAreaUnitsEnum:
        BtusPerSquareFoot = 0      # base HPUA unit
        KilojoulesPerSquareMeter = 1
        KilowattSecondsPerSquareMeter = 2
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert heat per unit area to base units (BTU Per Square Foot)"""
        if value == 0.0:
            return 0.0
        
        KJ_PER_SQ_M_TO_BTU_PER_SQ_FT = 0.088
        KW_S_PER_SQ_M_TO_BTU_PER_SQ_FT = 0.088
        
        if units == HeatPerUnitAreaUnits.HeatPerUnitAreaUnitsEnum.BtusPerSquareFoot:
            pass  # Already in base
        elif units == HeatPerUnitAreaUnits.HeatPerUnitAreaUnitsEnum.KilojoulesPerSquareMeter:
            value *= KJ_PER_SQ_M_TO_BTU_PER_SQ_FT
        elif units == HeatPerUnitAreaUnits.HeatPerUnitAreaUnitsEnum.KilowattSecondsPerSquareMeter:
            value *= KW_S_PER_SQ_M_TO_BTU_PER_SQ_FT
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert heat per unit area from base units (BTU Per Square Foot)"""
        if value == 0.0:
            return 0.0
        
        BTU_PER_SQ_FT_TO_KJ_PER_SQ_M = 1.0 / 0.088
        BTU_PER_SQ_FT_TO_KW_S_PER_SQ_M = 1.0 / 0.088
        
        if units == HeatPerUnitAreaUnits.HeatPerUnitAreaUnitsEnum.BtusPerSquareFoot:
            pass  # Already in base
        elif units == HeatPerUnitAreaUnits.HeatPerUnitAreaUnitsEnum.KilojoulesPerSquareMeter:
            value *= BTU_PER_SQ_FT_TO_KJ_PER_SQ_M
        elif units == HeatPerUnitAreaUnits.HeatPerUnitAreaUnitsEnum.KilowattSecondsPerSquareMeter:
            value *= BTU_PER_SQ_FT_TO_KW_S_PER_SQ_M
        
        return value


class HeatSourceAndReactionIntensityUnits:
    """Heat Source and Reaction Intensity unit conversions. Base unit: BTU Per Square Foot Per Minute"""
    
    class HeatSourceAndReactionIntensityUnitsEnum:
        BtusPerSquareFootPerMinute = 0      # base reaction intensity unit
        BtusPerSquareFootPerSecond = 1
        KilojoulesPerSquareMeterPerSecond = 2
        KilojoulesPerSquareMeterPerMinute = 3
        KilowattsPerSquareMeter = 4
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert heat source/reaction intensity to base units (BTU Per Square Foot Per Minute)"""
        if value == 0.0:
            return 0.0
        
        BTU_PER_SQ_FT_PER_SEC_TO_PER_MIN = 60.0
        KJ_PER_SQ_M_PER_SEC_TO_BTU_PER_SQ_FT_PER_MIN = 5.28
        KJ_PER_SQ_M_PER_MIN_TO_BTU_PER_SQ_FT_PER_MIN = 0.088
        KW_PER_SQ_M_TO_BTU_PER_SQ_FT_PER_MIN = 0.528
        
        if units == HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.BtusPerSquareFootPerMinute:
            pass  # Already in base
        elif units == HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.BtusPerSquareFootPerSecond:
            value *= BTU_PER_SQ_FT_PER_SEC_TO_PER_MIN
        elif units == HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.KilojoulesPerSquareMeterPerSecond:
            value *= KJ_PER_SQ_M_PER_SEC_TO_BTU_PER_SQ_FT_PER_MIN
        elif units == HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.KilojoulesPerSquareMeterPerMinute:
            value *= KJ_PER_SQ_M_PER_MIN_TO_BTU_PER_SQ_FT_PER_MIN
        elif units == HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.KilowattsPerSquareMeter:
            value *= KW_PER_SQ_M_TO_BTU_PER_SQ_FT_PER_MIN
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert heat source/reaction intensity from base units (BTU Per Square Foot Per Minute)"""
        if value == 0.0:
            return 0.0
        
        BTU_PER_SQ_FT_PER_MIN_TO_PER_SEC = 1.0 / 60.0
        BTU_PER_SQ_FT_PER_MIN_TO_KJ_PER_SQ_M_PER_SEC = 1.0 / 5.28
        BTU_PER_SQ_FT_PER_MIN_TO_KJ_PER_SQ_M_PER_MIN = 1.0 / 0.088
        BTU_PER_SQ_FT_PER_MIN_TO_KW_PER_SQ_M = 1.0 / 0.528
        
        if units == HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.BtusPerSquareFootPerMinute:
            pass  # Already in base
        elif units == HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.BtusPerSquareFootPerSecond:
            value *= BTU_PER_SQ_FT_PER_MIN_TO_PER_SEC
        elif units == HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.KilojoulesPerSquareMeterPerSecond:
            value *= BTU_PER_SQ_FT_PER_MIN_TO_KJ_PER_SQ_M_PER_SEC
        elif units == HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.KilojoulesPerSquareMeterPerMinute:
            value *= BTU_PER_SQ_FT_PER_MIN_TO_KJ_PER_SQ_M_PER_MIN
        elif units == HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum.KilowattsPerSquareMeter:
            value *= BTU_PER_SQ_FT_PER_MIN_TO_KW_PER_SQ_M
        
        return value


class FirelineIntensityUnits:
    """Fireline Intensity unit conversions. Base unit: BTU Per Foot Per Second"""
    
    class FirelineIntensityUnitsEnum:
        BtusPerFootPerSecond = 0   # base fireline intensity unit
        BtusPerFootPerMinute = 1
        KilojoulesPerMeterPerSecond = 2
        KilojoulesPerMeterPerMinute = 3
        KilowattsPerMeter = 4
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert fireline intensity to base units (BTU Per Foot Per Second)"""
        if value == 0.0:
            return 0.0
        
        BTU_PER_FT_PER_MIN_TO_PER_SEC = 0.01666666666666667
        KJ_PER_M_PER_SEC_TO_BTU_PER_FT_PER_SEC = 0.2886719
        KJ_PER_M_PER_MIN_TO_BTU_PER_FT_PER_SEC = 0.00481120819
        KW_PER_M_TO_BTU_PER_FT_PER_SEC = 0.2886719
        
        if units == FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond:
            pass  # Already in base
        elif units == FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerMinute:
            value *= BTU_PER_FT_PER_MIN_TO_PER_SEC
        elif units == FirelineIntensityUnits.FirelineIntensityUnitsEnum.KilojoulesPerMeterPerSecond:
            value *= KJ_PER_M_PER_SEC_TO_BTU_PER_FT_PER_SEC
        elif units == FirelineIntensityUnits.FirelineIntensityUnitsEnum.KilojoulesPerMeterPerMinute:
            value *= KJ_PER_M_PER_MIN_TO_BTU_PER_FT_PER_SEC
        elif units == FirelineIntensityUnits.FirelineIntensityUnitsEnum.KilowattsPerMeter:
            value *= KW_PER_M_TO_BTU_PER_FT_PER_SEC
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert fireline intensity from base units (BTU Per Foot Per Second)"""
        if value == 0.0:
            return 0.0
        
        BTU_PER_FT_PER_SEC_TO_PER_MIN = 60.0
        BTU_PER_FT_PER_SEC_TO_KJ_PER_M_PER_SEC = 3.464140419  # matches C++ BTUS_PER_FOOT_PER_SECOND_TO_KILOWATTS_PER_METER
        BTU_PER_FT_PER_SEC_TO_KJ_PER_M_PER_MIN = 207.848      # matches C++ BTUS_PER_FOOT_PER_SECOND_TO_KILOJOULES_PER_METER_PER_MINUTE
        BTU_PER_FT_PER_SEC_TO_KW_PER_M = 3.464140419          # same as KJ/m/s
        
        if units == FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerSecond:
            pass  # Already in base
        elif units == FirelineIntensityUnits.FirelineIntensityUnitsEnum.BtusPerFootPerMinute:
            value *= BTU_PER_FT_PER_SEC_TO_PER_MIN
        elif units == FirelineIntensityUnits.FirelineIntensityUnitsEnum.KilojoulesPerMeterPerSecond:
            value *= BTU_PER_FT_PER_SEC_TO_KJ_PER_M_PER_SEC
        elif units == FirelineIntensityUnits.FirelineIntensityUnitsEnum.KilojoulesPerMeterPerMinute:
            value *= BTU_PER_FT_PER_SEC_TO_KJ_PER_M_PER_MIN
        elif units == FirelineIntensityUnits.FirelineIntensityUnitsEnum.KilowattsPerMeter:
            value *= BTU_PER_FT_PER_SEC_TO_KW_PER_M
        
        return value


class TemperatureUnits:
    """Temperature unit conversions. Base unit: Fahrenheit"""
    
    class TemperatureUnitsEnum:
        Fahrenheit = 0             # base temperature unit
        Celsius = 1
        Kelvin = 2
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert temperature to base units (Fahrenheit)"""
        if units == TemperatureUnits.TemperatureUnitsEnum.Fahrenheit:
            pass  # Already in base
        elif units == TemperatureUnits.TemperatureUnitsEnum.Celsius:
            value = (value * 9.0 / 5.0) + 32.0
        elif units == TemperatureUnits.TemperatureUnitsEnum.Kelvin:
            value = (value - 273.15) * 9.0 / 5.0 + 32.0
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert temperature from base units (Fahrenheit)"""
        if units == TemperatureUnits.TemperatureUnitsEnum.Fahrenheit:
            pass  # Already in base
        elif units == TemperatureUnits.TemperatureUnitsEnum.Celsius:
            value = (value - 32.0) * 5.0 / 9.0
        elif units == TemperatureUnits.TemperatureUnitsEnum.Kelvin:
            value = ((value - 32.0) * 5.0 / 9.0) + 273.15
        
        return value


class TimeUnits:
    """Time unit conversions. Base unit: Minutes"""
    
    class TimeUnitsEnum:
        Minutes = 0                # base time unit
        Seconds = 1
        Hours = 2
        Days = 3
        Years = 4
    
    @staticmethod
    def toBaseUnits(value, units):
        """Convert time to base units (Minutes)"""
        if value == 0.0:
            return 0.0
        
        SECONDS_TO_MINUTES = 1.0 / 60.0
        HOURS_TO_MINUTES = 60.0
        DAYS_TO_MINUTES = 24.0 * 60.0
        YEARS_TO_MINUTES = 365.25 * 24.0 * 60.0
        
        if units == TimeUnits.TimeUnitsEnum.Minutes:
            pass  # Already in base
        elif units == TimeUnits.TimeUnitsEnum.Seconds:
            value *= SECONDS_TO_MINUTES
        elif units == TimeUnits.TimeUnitsEnum.Hours:
            value *= HOURS_TO_MINUTES
        elif units == TimeUnits.TimeUnitsEnum.Days:
            value *= DAYS_TO_MINUTES
        elif units == TimeUnits.TimeUnitsEnum.Years:
            value *= YEARS_TO_MINUTES
        
        return value
    
    @staticmethod
    def fromBaseUnits(value, units):
        """Convert time from base units (Minutes)"""
        if value == 0.0:
            return 0.0
        
        MINUTES_TO_SECONDS = 60.0
        MINUTES_TO_HOURS = 1.0 / 60.0
        MINUTES_TO_DAYS = 1.0 / (24.0 * 60.0)
        MINUTES_TO_YEARS = 1.0 / (365.25 * 24.0 * 60.0)
        
        if units == TimeUnits.TimeUnitsEnum.Minutes:
            pass  # Already in base
        elif units == TimeUnits.TimeUnitsEnum.Seconds:
            value *= MINUTES_TO_SECONDS
        elif units == TimeUnits.TimeUnitsEnum.Hours:
            value *= MINUTES_TO_HOURS
        elif units == TimeUnits.TimeUnitsEnum.Days:
            value *= MINUTES_TO_DAYS
        elif units == TimeUnits.TimeUnitsEnum.Years:
            value *= MINUTES_TO_YEARS
        
        return value

