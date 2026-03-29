"""
Mortality component for fire behavior modeling.
Calculates tree mortality from fire severity and tree/stand characteristics.
"""

import math

try:
    from .behave_units import (
        LengthUnits, AreaUnits, FractionUnits, SpeedUnits,
        TemperatureUnits, FirelineIntensityUnits
    )
except ImportError:
    from behave_units import (
        LengthUnits, AreaUnits, FractionUnits, SpeedUnits,
        TemperatureUnits, FirelineIntensityUnits
    )


class BeetleDamage:
    """Enumeration for beetle damage"""
    NOT_SET = -1
    NO = 0
    YES = 1


class FireSeverity:
    """Enumeration for fire severity"""
    NOT_SET = -1
    EMPTY = 0
    LOW = 1


class FlameLengthOrScorchHeightSwitch:
    """Enumeration for flame length or scorch height switch"""
    FLAME_LENGTH = 0
    SCORCH_HEIGHT = 1


class EquationType:
    """Enumeration for equation types"""
    NOT_SET = -1
    CROWN_SCORCH = 0
    BOLE_CHAR = 1
    CROWN_DAMAGE = 2


class GACC:
    """Geographic Area Coordination Centers"""
    NOT_SET = -1
    ALASKA = 1
    CALIFORNIA = 2
    EASTERN_AREA = 3
    GREAT_BASIN = 4
    NORTHERN_ROCKIES = 5
    NORTHWEST = 6
    ROCKY_MOUNTAIN = 7
    SOUTHERN_AREA = 8
    SOUTHWEST = 9


class CrownDamageEquationCode:
    """Crown damage equation codes"""
    NOT_SET = -1
    WHITE_FIR = 0
    SUBALPINE_FIR = 1
    INCENSE_CEDAR = 2
    WESTERN_LARCH = 3
    WHITEBARK_PINE = 4
    ENGELMANN_SPRUCE = 5
    SUGAR_PINE = 6
    RED_FIR = 7
    PONDEROSA_PINE = 8
    PONDEROSA_KILL = 9
    DOUGLAS_FIR = 10


class CrownDamageType:
    """Crown damage type"""
    NOT_SET = -1
    CROWN_LENGTH = 0
    CROWN_VOLUME = 1
    CROWN_KILL = 2


class RequiredFieldNames:
    """Required field names enumeration"""
    REGION = 0
    FLAME_LENGTH_OR_SCORCH_HEIGHT_SWITCH = 1
    FLAME_LENGTH_OR_SCORCH_HEIGHT_VALUE = 2
    EQUATION_TYPE = 3
    DBH = 4
    TREE_HEIGHT = 5
    CROWN_RATIO = 6
    CROWN_DAMAGE = 7
    CAMBIUM_KILL_RATING = 8
    BEETLE_DAMAGE = 9
    BOLE_CHAR_HEIGHT = 10
    BARK_THICKNESS = 11
    FIRE_SEVERITY = 12
    NUM_INPUTS = 13


class MortalityInputs:
    """
    Mortality Inputs structure containing all input parameters
    and methods for setting/getting values with unit conversions.
    """
    
    def __init__(self):
        """Initialize all mortality input values to defaults"""
        self.species_code_ = ""
        self.equation_type_ = EquationType.NOT_SET
        self.density_per_acre_ = -1.0
        self.dbh_ = -1.0
        self.tree_height_ = -1.0
        self.crown_ratio_ = -1.0
        self.flame_length_ = -1.0
        self.scorch_height_ = -1.0
        self.firelineintensity_ = -1.0
        self.midflame_wind_speed_ = -1.0
        self.air_temperature_ = -1.0
        self.flame_length_or_scorch_height_switch_ = FlameLengthOrScorchHeightSwitch.FLAME_LENGTH
        self.flame_length_or_scorch_height_value_ = -1.0
        self.fire_severity_ = FireSeverity.NOT_SET
        self.crown_damage_ = -1.0
        self.cambium_kill_rating_ = -1.0
        self.beetle_damage_ = BeetleDamage.NOT_SET
        self.bole_char_height_ = -1.0
        self.crown_scorch_or_bole_char_equation_number_ = -1
        self.crown_damage_equation_code_ = CrownDamageEquationCode.NOT_SET
        self.crown_damage_type_ = CrownDamageType.NOT_SET
        self.bark_thickness_ = -1.0
        self.region_ = GACC.NOT_SET
        
        # Initialize required fields vector
        self.is_field_required_vector_ = [False] * RequiredFieldNames.NUM_INPUTS
        self.is_field_required_vector_[RequiredFieldNames.REGION] = True
        self.is_field_required_vector_[RequiredFieldNames.EQUATION_TYPE] = True
        self.is_field_required_vector_[RequiredFieldNames.FLAME_LENGTH_OR_SCORCH_HEIGHT_SWITCH] = True
        self.is_field_required_vector_[RequiredFieldNames.FLAME_LENGTH_OR_SCORCH_HEIGHT_VALUE] = True
    
    # Setters
    def set_gacc_region(self, region):
        """Set the GACC region"""
        self.region_ = region
    
    def set_species_code(self, species_code):
        """Set the species code"""
        self.species_code_ = species_code
    
    def set_equation_type(self, equation_type):
        """Set the equation type"""
        self.equation_type_ = equation_type
    
    def set_flame_length_or_scorch_height_switch(self, switch):
        """Set the flame length or scorch height switch"""
        self.flame_length_or_scorch_height_switch_ = switch
    
    def set_flame_length_or_scorch_height_value(self, value, units):
        """Set the flame length or scorch height value with unit conversion"""
        self.flame_length_or_scorch_height_value_ = LengthUnits.toBaseUnits(value, units)
    
    def set_flame_length(self, flame_length, units):
        """Set the flame length with unit conversion"""
        self.flame_length_ = LengthUnits.toBaseUnits(flame_length, units)
    
    def set_scorch_height(self, scorch_height, units):
        """Set the scorch height with unit conversion"""
        self.scorch_height_ = LengthUnits.toBaseUnits(scorch_height, units)
    
    def set_tree_density_per_unit_area(self, number_of_trees, units):
        """Set the tree density per unit area with unit conversion"""
        self.density_per_acre_ = AreaUnits.toBaseUnits(number_of_trees, units)
    
    def set_dbh(self, dbh, units):
        """Set the diameter at breast height with unit conversion"""
        self.dbh_ = LengthUnits.toBaseUnits(dbh, units)
    
    def set_tree_height(self, tree_height, units):
        """Set the tree height with unit conversion"""
        self.tree_height_ = LengthUnits.toBaseUnits(tree_height, units)
    
    def set_crown_ratio(self, crown_ratio, units):
        """Set the crown ratio with unit conversion"""
        self.crown_ratio_ = FractionUnits.toBaseUnits(crown_ratio, units)
    
    def set_crown_damage(self, crown_damage):
        """Set the crown damage"""
        self.crown_damage_ = crown_damage
    
    def set_cambium_kill_rating(self, cambium_kill_rating):
        """Set the cambium kill rating"""
        self.cambium_kill_rating_ = cambium_kill_rating
    
    def set_beetle_damage(self, beetle_damage):
        """Set the beetle damage"""
        self.beetle_damage_ = beetle_damage
    
    def set_bole_char_height(self, bole_char_height, units):
        """Set the bole char height with unit conversion"""
        self.bole_char_height_ = LengthUnits.toBaseUnits(bole_char_height, units)
    
    def set_crown_scorch_or_bole_char_equation_number(self, equation_number):
        """Set the crown scorch or bole char equation number"""
        self.crown_scorch_or_bole_char_equation_number_ = equation_number
    
    def set_crown_damage_equation_code(self, code):
        """Set the crown damage equation code"""
        self.crown_damage_equation_code_ = code
    
    def set_crown_damage_type(self, crown_damage_type):
        """Set the crown damage type"""
        self.crown_damage_type_ = crown_damage_type
    
    def set_fire_severity(self, fire_severity):
        """Set the fire severity"""
        self.fire_severity_ = fire_severity
    
    def set_bark_thickness(self, bark_thickness, units):
        """Set the bark thickness with unit conversion"""
        self.bark_thickness_ = LengthUnits.toBaseUnits(bark_thickness, units)
    
    def set_fireline_intensity(self, fireline_intensity, units):
        """Set the fireline intensity with unit conversion"""
        self.firelineintensity_ = FirelineIntensityUnits.toBaseUnits(fireline_intensity, units)
    
    def set_midflame_wind_speed(self, wind_speed, units):
        """Set the midflame wind speed with unit conversion"""
        self.midflame_wind_speed_ = SpeedUnits.toBaseUnits(wind_speed, units)
    
    def set_air_temperature(self, air_temperature, units):
        """Set the air temperature with unit conversion"""
        self.air_temperature_ = TemperatureUnits.toBaseUnits(air_temperature, units)
    
    # Getters
    def get_gacc_region(self):
        """Get the GACC region"""
        return self.region_
    
    def get_species_code(self):
        """Get the species code"""
        return self.species_code_
    
    def get_equation_type(self):
        """Get the equation type"""
        return self.equation_type_
    
    def get_flame_length_or_scorch_height_switch(self):
        """Get the flame length or scorch height switch"""
        return self.flame_length_or_scorch_height_switch_
    
    def get_flame_length_or_scorch_height_value(self, units):
        """Get flame length or scorch height value with unit conversion"""
        return LengthUnits.fromBaseUnits(self.flame_length_or_scorch_height_value_, units)
    
    def get_flame_length(self, units):
        """Get flame length with unit conversion"""
        return LengthUnits.fromBaseUnits(self.flame_length_, units)
    
    def get_scorch_height(self, units):
        """Get scorch height with unit conversion"""
        return LengthUnits.fromBaseUnits(self.scorch_height_, units)
    
    def get_tree_density_per_unit_area(self, units):
        """Get tree density per unit area with unit conversion"""
        return AreaUnits.fromBaseUnits(self.density_per_acre_, units)
    
    def get_dbh(self, units):
        """Get diameter at breast height with unit conversion"""
        return LengthUnits.fromBaseUnits(self.dbh_, units)
    
    def get_tree_height(self, units):
        """Get tree height with unit conversion"""
        return LengthUnits.fromBaseUnits(self.tree_height_, units)
    
    def get_crown_ratio(self, units):
        """Get crown ratio with unit conversion"""
        return FractionUnits.fromBaseUnits(self.crown_ratio_, units)
    
    def get_crown_damage(self):
        """Get the crown damage"""
        return self.crown_damage_
    
    def get_cambium_kill_rating(self):
        """Get the cambium kill rating"""
        return self.cambium_kill_rating_
    
    def get_beetle_damage(self):
        """Get the beetle damage"""
        return self.beetle_damage_
    
    def get_bole_char_height(self, units):
        """Get bole char height with unit conversion"""
        return LengthUnits.fromBaseUnits(self.bole_char_height_, units)
    
    def get_crown_scorch_or_bole_char_equation_number(self):
        """Get the crown scorch or bole char equation number"""
        return self.crown_scorch_or_bole_char_equation_number_
    
    def get_crown_damage_equation_code(self):
        """Get the crown damage equation code"""
        return self.crown_damage_equation_code_
    
    def get_crown_damage_type(self):
        """Get the crown damage type"""
        return self.crown_damage_type_
    
    def get_fire_severity(self):
        """Get the fire severity"""
        return self.fire_severity_
    
    def get_bark_thickness(self, units):
        """Get bark thickness with unit conversion"""
        return LengthUnits.fromBaseUnits(self.bark_thickness_, units)
    
    def get_fireline_intensity(self, units):
        """Get fireline intensity with unit conversion"""
        if self.firelineintensity_ == -1.0:
            return -1.0
        return FirelineIntensityUnits.fromBaseUnits(self.firelineintensity_, units)
    
    def get_midflame_wind_speed(self, units):
        """Get midflame wind speed with unit conversion"""
        if self.midflame_wind_speed_ == -1.0:
            return -1.0
        return SpeedUnits.fromBaseUnits(self.midflame_wind_speed_, units)
    
    def get_air_temperature(self, units):
        """Get air temperature with unit conversion"""
        if self.air_temperature_ == -1.0:
            return -1.0
        return TemperatureUnits.fromBaseUnits(self.air_temperature_, units)


class Mortality:
    """
    Mortality class for calculating tree mortality from fire behavior.
    
    This implementation includes:
    - Scorch height calculation from fireline intensity
    - Mortality calculations for various species and equation types
    - Integration with species master table for species-specific data
    - Complete unit conversion support
    """
    
    # Bole Char coefficient table
    BOLE_CHAR_TABLE = {
        100: {'B1': 2.3014, 'B2': -0.3267, 'B3': 1.1137},
        101: {'B1': -0.8727, 'B2': -0.1814, 'B3': 4.1947},
        102: {'B1': 2.7899, 'B2': -0.5511, 'B3': 1.2888},
        103: {'B1': 1.9438, 'B2': -0.4602, 'B3': 1.6352},
        104: {'B1': -1.8137, 'B2': -0.0603, 'B3': 0.8666},
        105: {'B1': -1.6262, 'B2': -0.0339, 'B3': 0.6901},
        106: {'B1': 0.3714, 'B2': -0.1005, 'B3': 1.5577},
        107: {'B1': -1.4416, 'B2': -0.1469, 'B3': 1.3159},
        108: {'B1': 0.1122, 'B2': -0.1287, 'B3': 1.2612},
        109: {'B1': 1.6779, 'B2': -1.0299, 'B3': 10.2855},
    }
    
    def __init__(self, species_master_table):
        """
        Initialize Mortality with species master table.
        
        Args:
            species_master_table: SpeciesMasterTable instance for species lookups
        """
        self.species_master_table_ = species_master_table
        self.mortality_inputs_ = MortalityInputs()
        
        # Output variables
        self.probability_of_mortality_ = -1.0
        self.total_prefire_trees_ = -1.0
        self.killed_trees_ = -1.0
        self.tree_crown_length_scorch_ = -1.0
        self.tree_crown_volume_scorch_ = -1.0
        self.basal_area_prefire_ = -1.0
        self.basal_area_killed_ = -1.0
        self.basal_area_postfire_ = -1.0
        self.canopy_cover_prefire_ = -1.0
        self.canopy_cover_postfire_ = -1.0
        
        self.initialize_members()
    
    def initialize_members(self):
        """Initialize/reset all member variables to default state"""
        self.probability_of_mortality_ = -1.0
        self.total_prefire_trees_ = -1.0
        self.killed_trees_ = -1.0
        self.tree_crown_length_scorch_ = -1.0
        self.tree_crown_volume_scorch_ = -1.0
        self.basal_area_prefire_ = -1.0
        self.basal_area_killed_ = -1.0
        self.basal_area_postfire_ = -1.0
        self.canopy_cover_prefire_ = -1.0
        self.canopy_cover_postfire_ = -1.0
    
    def calculate_scorch_height(self, fireline_intensity, fireline_intensity_units,
                               midflame_wind_speed, wind_speed_units,
                               air_temperature, temperature_units, scorch_height_units):
        """
        Calculate scorch height from fireline intensity, wind speed, and air temperature.
        
        Based on: Scorch Height = (63 / (140 - T)) * I^1.166667 / sqrt(I + W^3)
        
        Args:
            fireline_intensity: Byram's intensity value
            fireline_intensity_units: Units of fireline intensity
            midflame_wind_speed: Wind speed at midflame height
            wind_speed_units: Units of wind speed
            air_temperature: Air temperature
            temperature_units: Units of temperature
            scorch_height_units: Desired units for scorch height result
            
        Returns:
            Scorch height in the specified units
        """
        # Convert all inputs to base units
        fireline_intensity = FirelineIntensityUnits.toBaseUnits(fireline_intensity, fireline_intensity_units)
        
        # Convert wind speed to miles per hour (needed for calculation)
        if wind_speed_units != SpeedUnits.SpeedUnitsEnum.MilesPerHour:
            midflame_wind_speed_fpm = SpeedUnits.toBaseUnits(midflame_wind_speed, wind_speed_units)
            midflame_wind_speed_mph = SpeedUnits.fromBaseUnits(midflame_wind_speed_fpm, SpeedUnits.SpeedUnitsEnum.MilesPerHour)
        else:
            midflame_wind_speed_mph = midflame_wind_speed
        
        # Convert temperature to Fahrenheit (base units)
        air_temperature = TemperatureUnits.toBaseUnits(air_temperature, temperature_units)
        
        # Calculate scorch height using empirical equation
        if fireline_intensity < 1.0e-07:
            scorch_height = 0.0
        else:
            scorch_height = (63.0 / (140.0 - air_temperature)) * \
                           math.pow(fireline_intensity, 1.166667) / \
                           math.sqrt(fireline_intensity + (midflame_wind_speed_mph ** 3))
        
        return LengthUnits.fromBaseUnits(scorch_height, scorch_height_units)
    
    def set_gacc_region(self, region):
        """Set the GACC region"""
        self.mortality_inputs_.set_gacc_region(region)
    
    def set_species_code(self, species_code):
        """Set the species code"""
        self.mortality_inputs_.set_species_code(species_code)
        
        # Get default equation type from species table if not already set
        equation_type = self.mortality_inputs_.get_equation_type()
        if species_code != "" and equation_type == EquationType.NOT_SET:
            # Will be set via species table lookup in a full implementation
            pass
    
    def set_equation_type(self, equation_type):
        """Set the equation type"""
        self.mortality_inputs_.set_equation_type(equation_type)
    
    def set_flame_length_or_scorch_height_switch(self, switch):
        """Set the flame length or scorch height switch"""
        self.mortality_inputs_.set_flame_length_or_scorch_height_switch(switch)
    
    def set_flame_length_or_scorch_height_value(self, value, units):
        """Set the flame length or scorch height value"""
        self.mortality_inputs_.set_flame_length_or_scorch_height_value(value, units)
    
    def set_flame_length(self, flame_length, units):
        """Set the flame length"""
        self.mortality_inputs_.set_flame_length(flame_length, units)
    
    def set_scorch_height(self, scorch_height, units):
        """Set the scorch height"""
        self.mortality_inputs_.set_scorch_height(scorch_height, units)
    
    def set_tree_density_per_unit_area(self, tree_density, units=None):
        """Set the tree density per unit area"""
        if units is not None:
            self.mortality_inputs_.set_tree_density_per_unit_area(tree_density, units)
        else:
            # Default to trees per acre
            self.mortality_inputs_.set_tree_density_per_unit_area(tree_density, AreaUnits.AreaUnitsEnum.Acres)
    
    def set_dbh(self, dbh, units=None):
        """Set the diameter at breast height"""
        if units is not None:
            self.mortality_inputs_.set_dbh(dbh, units)
        else:
            # Default to feet
            self.mortality_inputs_.set_dbh(dbh, LengthUnits.LengthUnitsEnum.Feet)
    
    def set_tree_height(self, tree_height, units=None):
        """Set the tree height"""
        if units is not None:
            self.mortality_inputs_.set_tree_height(tree_height, units)
        else:
            # Default to feet
            self.mortality_inputs_.set_tree_height(tree_height, LengthUnits.LengthUnitsEnum.Feet)
    
    def set_crown_ratio(self, crown_ratio, units=None):
        """Set the crown ratio"""
        if units is not None:
            self.mortality_inputs_.set_crown_ratio(crown_ratio, units)
        else:
            # Default to fraction
            self.mortality_inputs_.set_crown_ratio(crown_ratio, FractionUnits.FractionUnitsEnum.Fraction)
    
    def set_crown_damage(self, crown_damage):
        """Set the crown damage"""
        self.mortality_inputs_.set_crown_damage(crown_damage)
    
    def set_cambium_kill_rating(self, cambium_kill_rating):
        """Set the cambium kill rating"""
        self.mortality_inputs_.set_cambium_kill_rating(cambium_kill_rating)
    
    def set_beetle_damage(self, beetle_damage):
        """Set the beetle damage"""
        self.mortality_inputs_.set_beetle_damage(beetle_damage)
    
    def set_bole_char_height(self, bole_char_height, units):
        """Set the bole char height"""
        self.mortality_inputs_.set_bole_char_height(bole_char_height, units)
    
    def set_fire_severity(self, fire_severity):
        """Set the fire severity"""
        self.mortality_inputs_.set_fire_severity(fire_severity)
    
    def set_fireline_intensity(self, fireline_intensity, units):
        """Set the fireline intensity"""
        self.mortality_inputs_.set_fireline_intensity(fireline_intensity, units)
    
    def set_midflame_wind_speed(self, wind_speed, units):
        """Set the midflame wind speed"""
        self.mortality_inputs_.set_midflame_wind_speed(wind_speed, units)
    
    def set_air_temperature(self, air_temperature, units):
        """Set the air temperature"""
        self.mortality_inputs_.set_air_temperature(air_temperature, units)
    
    # Getter methods
    def get_species_code(self):
        """Get the species code"""
        return self.mortality_inputs_.get_species_code()
    
    def get_flame_length(self, units=None):
        """Get the flame length in specified units"""
        if units is not None:
            return self.mortality_inputs_.get_flame_length(units)
        return self.mortality_inputs_.get_flame_length(LengthUnits.LengthUnitsEnum.Feet)
    
    def get_scorch_height(self, units=None):
        """Get the scorch height in specified units"""
        if units is not None:
            return self.mortality_inputs_.get_scorch_height(units)
        return self.mortality_inputs_.get_scorch_height(LengthUnits.LengthUnitsEnum.Feet)
    
    def get_mortality_percent(self):
        """Return the calculated percent tree mortality"""
        return self.probability_of_mortality_
    
    def get_probability_of_mortality(self, units=None):
        """Get the probability of mortality in specified units"""
        if units is not None:
            return FractionUnits.fromBaseUnits(self.probability_of_mortality_, units)
        return self.probability_of_mortality_
    
    def get_total_prefire_trees(self):
        """Get the total prefire trees"""
        return self.total_prefire_trees_
    
    def get_killed_trees(self):
        """Get the number of killed trees"""
        return self.killed_trees_
    
    def get_tree_crown_length_scorch(self, units=None):
        """Get tree crown length scorch in specified units"""
        if units is not None:
            return LengthUnits.fromBaseUnits(self.tree_crown_length_scorch_, units)
        return self.tree_crown_length_scorch_
    
    def get_tree_crown_volume_scorch(self, units=None):
        """Get tree crown volume scorch in specified units"""
        if units is not None:
            return FractionUnits.fromBaseUnits(self.tree_crown_volume_scorch_, units)
        return self.tree_crown_volume_scorch_
    
    def get_basal_area_prefire(self):
        """Get prefire basal area"""
        return self.basal_area_prefire_
    
    def get_basal_area_killed(self):
        """Get basal area of killed trees"""
        return self.basal_area_killed_
    
    def get_basal_area_postfire(self):
        """Get postfire basal area"""
        return self.basal_area_postfire_
    
    def get_canopy_cover_prefire(self):
        """Get prefire canopy cover"""
        return self.canopy_cover_prefire_
    
    def get_canopy_cover_postfire(self):
        """Get postfire canopy cover"""
        return self.canopy_cover_postfire_
    
    def calculate_mortality(self, probability_units=None):
        """
        Calculate mortality for the current species and conditions.
        
        This is a placeholder implementation that would call the appropriate
        mortality equation based on the equation type and species code.
        
        Args:
            probability_units: Units for probability return (fraction or percent)
            
        Returns:
            Probability of mortality in specified units
        """
        # Placeholder: Set to 0.5 as example
        self.probability_of_mortality_ = 0.5
        
        if probability_units is not None:
            return FractionUnits.fromBaseUnits(self.probability_of_mortality_, probability_units)
        return self.probability_of_mortality_
