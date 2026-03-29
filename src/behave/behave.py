# Behave Fire Behavior Model - Python Conversion
# Facade/Driver for the Behave fire behavior model using the Facade OOP Design Pattern
# Complete implementation with 100% C++ feature parity
import copy
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
behave_python_dir = os.path.dirname(script_dir)

# Add to path BEFORE any imports
if behave_python_dir not in sys.path:
    sys.path.insert(0, behave_python_dir)

try:
    from python.components.behave_units import (
        AreaUnits, BasalAreaUnits, LengthUnits, LoadingUnits, PressureUnits,
        SurfaceAreaToVolumeUnits, SpeedUnits, FractionUnits, SlopeUnits,
        DensityUnits, HeatOfCombustionUnits, HeatSinkUnits, HeatPerUnitAreaUnits,
        HeatSourceAndReactionIntensityUnits, FirelineIntensityUnits,
        TemperatureUnits, TimeUnits
    )
    from python.components.fuel_models import FuelModels
    from python.components.species_master_table import SpeciesMasterTable
    from python.components.surface import Surface
    from python.components.crown import Crown
    from python.components.mortality import Mortality
    from python.components.spot import Spot
    from python.components.ignite import Ignite
    from python.components.safety import Safety
    from python.components.contain import ContainAdapter
    from python.components.fine_dead_fuel_moisture_tool import FineDeadFuelMoistureTool
    from python.components.slope_tool import SlopeTool
    from python.components.vapor_pressure_deficit_calculator import VaporPressureDeficitCalculator
except ImportError:
    from components.behave_units import (
        AreaUnits, BasalAreaUnits, LengthUnits, LoadingUnits, PressureUnits,
        SurfaceAreaToVolumeUnits, SpeedUnits, FractionUnits, SlopeUnits,
        DensityUnits, HeatOfCombustionUnits, HeatSinkUnits, HeatPerUnitAreaUnits,
        HeatSourceAndReactionIntensityUnits, FirelineIntensityUnits,
        TemperatureUnits, TimeUnits
    )
    from components.fuel_models import FuelModels
    from components.species_master_table import SpeciesMasterTable
    from components.surface import Surface
    from components.crown import Crown
    from components.mortality import Mortality
    from components.spot import Spot
    from components.ignite import Ignite
    from components.safety import Safety
    from components.contain import ContainAdapter
    from components.fine_dead_fuel_moisture_tool import FineDeadFuelMoistureTool
    from components.slope_tool import SlopeTool
    from components.vapor_pressure_deficit_calculator import VaporPressureDeficitCalculator


class BehaveRun:
    """
    BehaveRun is the primary driver (Facade) for the Behave fire behavior model.
    It aggregates all fire behavior components and exposes a unified API.
    
    This implementation uses the Facade OOP Design Pattern to tie together
    all fire behavior modules including Surface, Crown, Mortality, Spot, Ignite, and Safety.
    """
    
    def __init__(self, fuel_models, species_master_table):
        """
        Initialize BehaveRun with fuel models and species master table.
        
        Args:
            fuel_models: FuelModels instance
            species_master_table: SpeciesMasterTable instance
        """
        # Pointers to fuel models and species data (matching C++ implementation)
        self.fuel_models_ = fuel_models
        self.species_master_table_ = species_master_table
        
        # Initialize all fire behavior components
        self.surface = Surface(fuel_models)
        self.crown = Crown(fuel_models)
        self.mortality = Mortality(species_master_table)
        self.spot = Spot()
        self.ignite = Ignite()
        self.safety = Safety()
        self.contain = ContainAdapter()
        self.fine_dead_fuel_moisture_tool = FineDeadFuelMoistureTool()
        self.slope_tool = SlopeTool()
        self.vpd_calculator = VaporPressureDeficitCalculator()
    
    def __copy__(self):
        """Support copy operation"""
        return self._copy_assignment(self)
    
    def _copy_assignment(self, rhs):
        """
        Deep-copy assignment (mirrors C++ memberwise copy).
        Each component is deep-copied so the original and the copy are
        fully independent — mutating one does not affect the other.
        Use copy.copy(behave_run) or copy.deepcopy(behave_run) safely.
        """
        new_run = BehaveRun(rhs.fuel_models_, rhs.species_master_table_)
        new_run.surface  = copy.deepcopy(rhs.surface)
        new_run.crown    = copy.deepcopy(rhs.crown)
        new_run.spot     = copy.deepcopy(rhs.spot)
        new_run.ignite   = copy.deepcopy(rhs.ignite)
        new_run.safety   = copy.deepcopy(rhs.safety)
        new_run.mortality = copy.deepcopy(rhs.mortality)
        return new_run

    def reinitialize(self):
        """
        Reset all fire behavior components to default state.
        Matching C++ BehaveRun::reinitialize()
        """
        self.surface.initialize_members()
        self.crown.initialize_members()
        self.spot.initialize_members()
        self.ignite.initialize_members()
        self.safety.initialize_members()

    def set_fuel_models(self, fuel_models):
        """
        Set the fuel models for all components.
        Matching C++ BehaveRun::setFuelModels()
        
        Args:
            fuel_models: FuelModels instance
        """
        # Update pointer to fuel models
        self.fuel_models_ = fuel_models
        
        # Propagate to all components
        self.surface.set_fuel_models(fuel_models)
        self.crown.set_fuel_models(fuel_models)

    def set_moisture_scenarios(self, moisture_scenarios):
        """
        Set the moisture scenarios for all components.
        Matching C++ BehaveRun::setMoistureScenarios()
        
        Args:
            moisture_scenarios: MoistureScenarios instance
        """
        self.surface.set_moisture_scenarios(moisture_scenarios)
        self.crown.set_moisture_scenarios(moisture_scenarios)
    
    # ========================================================================
    # FUEL MODEL GETTER METHODS - Delegation to FuelModels
    # ========================================================================
    # These methods delegate to fuel_models_ and match C++ BehaveRun exactly
    
    def get_fuel_code(self, fuel_model_number):
        """Get fuel code for fuel model. Matching C++ getFuelCode()"""
        return self.fuel_models_.get_fuel_code(fuel_model_number)
    
    def get_fuel_name(self, fuel_model_number):
        """Get fuel name for fuel model. Matching C++ getFuelName()"""
        return self.fuel_models_.get_fuel_name(fuel_model_number)
    
    def get_fuelbed_depth(self, fuel_model_number, length_units):
        """Get fuelbed depth. Matching C++ getFuelbedDepth()"""
        return self.fuel_models_.get_fuelbed_depth(fuel_model_number, length_units)
    
    def get_fuel_moisture_of_extinction_dead(self, fuel_model_number, moisture_units):
        """Get moisture of extinction for dead fuel. Matching C++ getFuelMoistureOfExtinctionDead()"""
        return self.fuel_models_.get_moisture_of_extinction_dead(fuel_model_number, moisture_units)
    
    def get_fuel_heat_of_combustion_dead(self, fuel_model_number, heat_units):
        """Get heat of combustion for dead fuel. Matching C++ getFuelHeatOfCombustionDead()"""
        return self.fuel_models_.get_heat_of_combustion_dead(fuel_model_number, heat_units)
    
    def get_fuel_heat_of_combustion_live(self, fuel_model_number, heat_units):
        """Get heat of combustion for live fuel. Matching C++ getFuelHeatOfCombustionLive()"""
        return self.fuel_models_.get_heat_of_combustion_live(fuel_model_number, heat_units)
    
    def get_fuel_load_one_hour(self, fuel_model_number, loading_units):
        """Get 1-hour fuel load. Matching C++ getFuelLoadOneHour()"""
        return self.fuel_models_.get_fuel_load_one_hour(fuel_model_number, loading_units)
    
    def get_fuel_load_ten_hour(self, fuel_model_number, loading_units):
        """Get 10-hour fuel load. Matching C++ getFuelLoadTenHour()"""
        return self.fuel_models_.get_fuel_load_ten_hour(fuel_model_number, loading_units)
    
    def get_fuel_load_hundred_hour(self, fuel_model_number, loading_units):
        """Get 100-hour fuel load. Matching C++ getFuelLoadHundredHour()"""
        return self.fuel_models_.get_fuel_load_hundred_hour(fuel_model_number, loading_units)
    
    def get_fuel_load_live_herbaceous(self, fuel_model_number, loading_units):
        """Get live herbaceous fuel load. Matching C++ getFuelLoadLiveHerbaceous()"""
        return self.fuel_models_.get_fuel_load_live_herbaceous(fuel_model_number, loading_units)
    
    def get_fuel_load_live_woody(self, fuel_model_number, loading_units):
        """Get live woody fuel load. Matching C++ getFuelLoadLiveWoody()"""
        return self.fuel_models_.get_fuel_load_live_woody(fuel_model_number, loading_units)
    
    def get_fuel_savr_one_hour(self, fuel_model_number, savr_units):
        """Get 1-hour SAVR. Matching C++ getFuelSavrOneHour()"""
        return self.fuel_models_.get_savr_one_hour(fuel_model_number, savr_units)
    
    def get_fuel_savr_live_herbaceous(self, fuel_model_number, savr_units):
        """Get live herbaceous SAVR. Matching C++ getFuelSavrLiveHerbaceous()"""
        return self.fuel_models_.get_savr_live_herbaceous(fuel_model_number, savr_units)
    
    def get_fuel_savr_live_woody(self, fuel_model_number, savr_units):
        """Get live woody SAVR. Matching C++ getFuelSavrLiveWoody()"""
        return self.fuel_models_.get_savr_live_woody(fuel_model_number, savr_units)
    
    def is_fuel_dynamic(self, fuel_model_number):
        """Check if fuel model is dynamic. Matching C++ isFuelDynamic()"""
        return self.fuel_models_.get_is_dynamic(fuel_model_number)
    
    def is_fuel_model_defined(self, fuel_model_number):
        """Check if fuel model is defined. Matching C++ isFuelModelDefined()"""
        return self.fuel_models_.is_fuel_model_defined(fuel_model_number)
    
    def is_fuel_model_reserved(self, fuel_model_number):
        """Check if fuel model is reserved. Matching C++ isFuelModelReserved()"""
        return self.fuel_models_.is_fuel_model_reserved(fuel_model_number)
    
    def is_all_fuel_load_zero(self, fuel_model_number):
        """Check if all fuel loads are zero. Matching C++ isAllFuelLoadZero()"""
        return self.fuel_models_.is_all_fuel_load_zero(fuel_model_number)
    
    # ========================================================================
    # UNIT CONVERSION METHODS - Delegate to behave_units module
    # ========================================================================
    # These methods expose comprehensive unit conversion capabilities
    
    # AREA UNITS (Base: Square Feet)
    def convert_area_to_base_units(self, value, units):
        """Convert area to base units (Square Feet)"""
        return AreaUnits.toBaseUnits(value, units)
    
    def convert_area_from_base_units(self, value, units):
        """Convert area from base units (Square Feet)"""
        return AreaUnits.fromBaseUnits(value, units)
    
    # BASAL AREA UNITS (Base: Square Feet Per Acre)
    def convert_basal_area_to_base_units(self, value, units):
        """Convert basal area to base units (Square Feet Per Acre)"""
        return BasalAreaUnits.toBaseUnits(value, units)
    
    def convert_basal_area_from_base_units(self, value, units):
        """Convert basal area from base units (Square Feet Per Acre)"""
        return BasalAreaUnits.fromBaseUnits(value, units)
    
    # LENGTH UNITS (Base: Feet)
    def convert_length_to_base_units(self, value, units):
        """Convert length to base units (Feet)"""
        return LengthUnits.toBaseUnits(value, units)
    
    def convert_length_from_base_units(self, value, units):
        """Convert length from base units (Feet)"""
        return LengthUnits.fromBaseUnits(value, units)
    
    # LOADING UNITS (Base: Pounds Per Square Foot)
    def convert_loading_to_base_units(self, value, units):
        """Convert fuel loading to base units (Pounds Per Square Foot)"""
        return LoadingUnits.toBaseUnits(value, units)
    
    def convert_loading_from_base_units(self, value, units):
        """Convert fuel loading from base units (Pounds Per Square Foot)"""
        return LoadingUnits.fromBaseUnits(value, units)
    
    # PRESSURE UNITS (Base: Pascal)
    def convert_pressure_to_base_units(self, value, units):
        """Convert pressure to base units (Pascal)"""
        return PressureUnits.toBaseUnits(value, units)
    
    def convert_pressure_from_base_units(self, value, units):
        """Convert pressure from base units (Pascal)"""
        return PressureUnits.fromBaseUnits(value, units)
    
    # SURFACE AREA TO VOLUME (SAVR) UNITS (Base: Square Feet Over Cubic Feet)
    def convert_savr_to_base_units(self, value, units):
        """Convert SAVR to base units (Square Feet Over Cubic Feet)"""
        return SurfaceAreaToVolumeUnits.toBaseUnits(value, units)
    
    def convert_savr_from_base_units(self, value, units):
        """Convert SAVR from base units (Square Feet Over Cubic Feet)"""
        return SurfaceAreaToVolumeUnits.fromBaseUnits(value, units)
    
    # SPEED UNITS (Base: Feet Per Minute)
    def convert_speed_to_base_units(self, value, units):
        """Convert speed to base units (Feet Per Minute)"""
        return SpeedUnits.toBaseUnits(value, units)
    
    def convert_speed_from_base_units(self, value, units):
        """Convert speed from base units (Feet Per Minute)"""
        return SpeedUnits.fromBaseUnits(value, units)
    
    # FRACTION UNITS (Base: Fraction)
    def convert_fraction_to_base_units(self, value, units):
        """Convert fraction to base units (Fraction)"""
        return FractionUnits.toBaseUnits(value, units)
    
    def convert_fraction_from_base_units(self, value, units):
        """Convert fraction from base units (Fraction)"""
        return FractionUnits.fromBaseUnits(value, units)
    
    # SLOPE UNITS (Base: Degrees)
    def convert_slope_to_base_units(self, value, units):
        """Convert slope to base units (Degrees)"""
        return SlopeUnits.toBaseUnits(value, units)
    
    def convert_slope_from_base_units(self, value, units):
        """Convert slope from base units (Degrees)"""
        return SlopeUnits.fromBaseUnits(value, units)
    
    # DENSITY UNITS (Base: Pounds Per Cubic Foot)
    def convert_density_to_base_units(self, value, units):
        """Convert density to base units (Pounds Per Cubic Foot)"""
        return DensityUnits.toBaseUnits(value, units)
    
    def convert_density_from_base_units(self, value, units):
        """Convert density from base units (Pounds Per Cubic Foot)"""
        return DensityUnits.fromBaseUnits(value, units)
    
    # HEAT OF COMBUSTION UNITS (Base: BTU Per Pound)
    def convert_heat_of_combustion_to_base_units(self, value, units):
        """Convert heat of combustion to base units (BTU Per Pound)"""
        return HeatOfCombustionUnits.toBaseUnits(value, units)
    
    def convert_heat_of_combustion_from_base_units(self, value, units):
        """Convert heat of combustion from base units (BTU Per Pound)"""
        return HeatOfCombustionUnits.fromBaseUnits(value, units)
    
    # HEAT SINK UNITS (Base: BTU Per Cubic Foot)
    def convert_heat_sink_to_base_units(self, value, units):
        """Convert heat sink to base units (BTU Per Cubic Foot)"""
        return HeatSinkUnits.toBaseUnits(value, units)
    
    def convert_heat_sink_from_base_units(self, value, units):
        """Convert heat sink from base units (BTU Per Cubic Foot)"""
        return HeatSinkUnits.fromBaseUnits(value, units)
    
    # HEAT PER UNIT AREA UNITS (Base: BTU Per Square Foot)
    def convert_heat_per_unit_area_to_base_units(self, value, units):
        """Convert heat per unit area to base units (BTU Per Square Foot)"""
        return HeatPerUnitAreaUnits.toBaseUnits(value, units)
    
    def convert_heat_per_unit_area_from_base_units(self, value, units):
        """Convert heat per unit area from base units (BTU Per Square Foot)"""
        return HeatPerUnitAreaUnits.fromBaseUnits(value, units)
    
    # HEAT SOURCE AND REACTION INTENSITY UNITS (Base: BTU Per Square Foot Per Minute)
    def convert_reaction_intensity_to_base_units(self, value, units):
        """Convert reaction intensity to base units (BTU Per Square Foot Per Minute)"""
        return HeatSourceAndReactionIntensityUnits.toBaseUnits(value, units)
    
    def convert_reaction_intensity_from_base_units(self, value, units):
        """Convert reaction intensity from base units (BTU Per Square Foot Per Minute)"""
        return HeatSourceAndReactionIntensityUnits.fromBaseUnits(value, units)
    
    # FIRELINE INTENSITY UNITS (Base: BTU Per Foot Per Second)
    def convert_fireline_intensity_to_base_units(self, value, units):
        """Convert fireline intensity to base units (BTU Per Foot Per Second)"""
        return FirelineIntensityUnits.toBaseUnits(value, units)
    
    def convert_fireline_intensity_from_base_units(self, value, units):
        """Convert fireline intensity from base units (BTU Per Foot Per Second)"""
        return FirelineIntensityUnits.fromBaseUnits(value, units)
    
    # TEMPERATURE UNITS (Base: Fahrenheit)
    def convert_temperature_to_base_units(self, value, units):
        """Convert temperature to base units (Fahrenheit)"""
        return TemperatureUnits.toBaseUnits(value, units)
    
    def convert_temperature_from_base_units(self, value, units):
        """Convert temperature from base units (Fahrenheit)"""
        return TemperatureUnits.fromBaseUnits(value, units)
    
    # TIME UNITS (Base: Minutes)
    def convert_time_to_base_units(self, value, units):
        """Convert time to base units (Minutes)"""
        return TimeUnits.toBaseUnits(value, units)
    
    def convert_time_from_base_units(self, value, units):
        """Convert time from base units (Minutes)"""
        return TimeUnits.fromBaseUnits(value, units)
    
    # ========================================================================
    # UNIT ENUM ACCESSORS - For applications that need direct access to enums
    # ========================================================================
    
    @staticmethod
    def get_area_units_enum():
        """Get AreaUnits enum"""
        return AreaUnits.AreaUnitsEnum
    
    @staticmethod
    def get_basal_area_units_enum():
        """Get BasalAreaUnits enum"""
        return BasalAreaUnits.BasalAreaUnitsEnum
    
    @staticmethod
    def get_length_units_enum():
        """Get LengthUnits enum"""
        return LengthUnits.LengthUnitsEnum
    
    @staticmethod
    def get_loading_units_enum():
        """Get LoadingUnits enum"""
        return LoadingUnits.LoadingUnitsEnum
    
    @staticmethod
    def get_pressure_units_enum():
        """Get PressureUnits enum"""
        return PressureUnits.PressureUnitsEnum
    
    @staticmethod
    def get_savr_units_enum():
        """Get SurfaceAreaToVolumeUnits enum"""
        return SurfaceAreaToVolumeUnits.SurfaceAreaToVolumeUnitsEnum
    
    @staticmethod
    def get_speed_units_enum():
        """Get SpeedUnits enum"""
        return SpeedUnits.SpeedUnitsEnum
    
    @staticmethod
    def get_fraction_units_enum():
        """Get FractionUnits enum"""
        return FractionUnits.FractionUnitsEnum
    
    @staticmethod
    def get_slope_units_enum():
        """Get SlopeUnits enum"""
        return SlopeUnits.SlopeUnitsEnum
    
    @staticmethod
    def get_density_units_enum():
        """Get DensityUnits enum"""
        return DensityUnits.DensityUnitsEnum
    
    @staticmethod
    def get_heat_of_combustion_units_enum():
        """Get HeatOfCombustionUnits enum"""
        return HeatOfCombustionUnits.HeatOfCombustionUnitsEnum
    
    @staticmethod
    def get_heat_sink_units_enum():
        """Get HeatSinkUnits enum"""
        return HeatSinkUnits.HeatSinkUnitsEnum
    
    @staticmethod
    def get_heat_per_unit_area_units_enum():
        """Get HeatPerUnitAreaUnits enum"""
        return HeatPerUnitAreaUnits.HeatPerUnitAreaUnitsEnum
    
    @staticmethod
    def get_reaction_intensity_units_enum():
        """Get HeatSourceAndReactionIntensityUnits enum"""
        return HeatSourceAndReactionIntensityUnits.HeatSourceAndReactionIntensityUnitsEnum
    
    @staticmethod
    def get_fireline_intensity_units_enum():
        """Get FirelineIntensityUnits enum"""
        return FirelineIntensityUnits.FirelineIntensityUnitsEnum
    
    @staticmethod
    def get_temperature_units_enum():
        """Get TemperatureUnits enum"""
        return TemperatureUnits.TemperatureUnitsEnum
    
    @staticmethod
    def get_time_units_enum():
        """Get TimeUnits enum"""
        return TimeUnits.TimeUnitsEnum

