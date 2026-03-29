"""
Ignite component for fire ignition probability calculations.
Calculates ignition probability from firebrands and lightning strikes.
"""

import math

try:
    from .behave_units import LengthUnits, FractionUnits, TemperatureUnits
except ImportError:
    from behave_units import LengthUnits, FractionUnits, TemperatureUnits


class IgnitionFuelBedType:
    """Enumeration for ignition fuel bed types"""
    PONDEROSA_PINE_LITTER = 0
    PUNKY_WOOD_ROTTEN_CHUNKY = 1
    PUNKY_WOOD_POWDER_DEEP = 2
    PUNK_WOOD_POWDER_SHALLOW = 3
    LODGEPOLE_PINE_DUFF = 4
    DOUGLAS_FIR_DUFF = 5
    HIGH_ALTITUDE_MIXED = 6
    PEAT_MOSS = 7


class LightningCharge:
    """Enumeration for lightning charge type"""
    NEGATIVE = 0
    POSITIVE = 1
    UNKNOWN = 2


def _parse_fuel_bed_type(v):
    """Convert string or int to IgnitionFuelBedType enum int."""
    if isinstance(v, int):
        return v
    s = str(v).upper().replace(' ', '').replace('_', '').replace('-', '')
    _MAP = {
        'PONDEROSAFINELITTER': IgnitionFuelBedType.PONDEROSA_PINE_LITTER,
        'PONDEROSAPINELITTER': IgnitionFuelBedType.PONDEROSA_PINE_LITTER,
        'PUNKYWOODROTTENCHUNKY': IgnitionFuelBedType.PUNKY_WOOD_ROTTEN_CHUNKY,
        'PUNKYWOODPOWDERDEEP': IgnitionFuelBedType.PUNKY_WOOD_POWDER_DEEP,
        'PUNKWOODPOWDERSHALLOW': IgnitionFuelBedType.PUNK_WOOD_POWDER_SHALLOW,
        'LODGEPOLEPINEDUFF': IgnitionFuelBedType.LODGEPOLE_PINE_DUFF,
        'DOUGLASFIRDUFF': IgnitionFuelBedType.DOUGLAS_FIR_DUFF,
        'HIGHALTITUDEMIXED': IgnitionFuelBedType.HIGH_ALTITUDE_MIXED,
        'PEATMOSS': IgnitionFuelBedType.PEAT_MOSS,
    }
    return _MAP.get(s, IgnitionFuelBedType.PONDEROSA_PINE_LITTER)


def _parse_lightning_charge(v):
    """Convert string or int to LightningCharge enum int."""
    if isinstance(v, int):
        return v
    s = str(v).upper()
    if 'NEG' in s:  return LightningCharge.NEGATIVE
    if 'POS' in s:  return LightningCharge.POSITIVE
    return LightningCharge.UNKNOWN


class IgniteInputs:
    """
    Input management for Ignite calculations.
    Stores all parameters with automatic unit conversion.
    """
    
    def __init__(self):
        """Initialize all input parameters"""
        self.moisture_one_hour_ = 0.0
        self.moisture_hundred_hour_ = 0.0
        self.air_temperature_ = 0.0
        self.sun_shade_ = 0.0
        self.fuel_bed_type_ = IgnitionFuelBedType.PONDEROSA_PINE_LITTER
        self.duff_depth_ = 0.0
        self.lightning_charge_type_ = LightningCharge.UNKNOWN
    
    def initialize_members(self):
        """Reset all input parameters to defaults"""
        self.moisture_one_hour_ = 0.0
        self.moisture_hundred_hour_ = 0.0
        self.air_temperature_ = 0.0
        self.sun_shade_ = 0.0
        self.fuel_bed_type_ = IgnitionFuelBedType.PONDEROSA_PINE_LITTER
        self.duff_depth_ = 0.0
        self.lightning_charge_type_ = LightningCharge.UNKNOWN
    
    def update_ignite_inputs(self, moisture_one_hour, moisture_hundred_hour, moisture_units,
                            air_temperature, temperature_units, sun_shade, sun_shade_units,
                            fuel_bed_type, duff_depth, duff_depth_units, lightning_charge_type):
        """Update all inputs at once"""
        self.set_moisture_one_hour(moisture_one_hour, moisture_units)
        self.set_moisture_hundred_hour(moisture_hundred_hour, moisture_units)
        self.set_air_temperature(air_temperature, temperature_units)
        self.set_sun_shade(sun_shade, sun_shade_units)
        self.fuel_bed_type_ = _parse_fuel_bed_type(fuel_bed_type)
        self.set_duff_depth(duff_depth, duff_depth_units)
        self.lightning_charge_type_ = _parse_lightning_charge(lightning_charge_type)
    
    # Setters with unit conversion
    def set_moisture_one_hour(self, moisture, units):
        """Set 1-hour fuel moisture with unit conversion"""
        self.moisture_one_hour_ = FractionUnits.toBaseUnits(moisture, units)
    
    def set_moisture_hundred_hour(self, moisture, units):
        """Set 100-hour fuel moisture with unit conversion"""
        self.moisture_hundred_hour_ = FractionUnits.toBaseUnits(moisture, units)
    
    def set_air_temperature(self, temperature, units):
        """Set air temperature with unit conversion"""
        self.air_temperature_ = TemperatureUnits.toBaseUnits(temperature, units)
    
    def set_sun_shade(self, sun_shade, units):
        """Set sun/shade factor with unit conversion"""
        self.sun_shade_ = FractionUnits.toBaseUnits(sun_shade, units)
    
    def set_ignition_fuel_bed_type(self, fuel_bed_type):
        """Set fuel bed type"""
        self.fuel_bed_type_ = _parse_fuel_bed_type(fuel_bed_type)
    
    def set_duff_depth(self, depth, units):
        """Set duff depth with unit conversion"""
        self.duff_depth_ = LengthUnits.toBaseUnits(depth, units)
    
    def set_lightning_charge_type(self, charge_type):
        """Set lightning charge type"""
        self.lightning_charge_type_ = _parse_lightning_charge(charge_type)
    
    # Getters with unit conversion
    def get_moisture_one_hour(self, units):
        """Get 1-hour fuel moisture with unit conversion"""
        return FractionUnits.fromBaseUnits(self.moisture_one_hour_, units)
    
    def get_moisture_hundred_hour(self, units):
        """Get 100-hour fuel moisture with unit conversion"""
        return FractionUnits.fromBaseUnits(self.moisture_hundred_hour_, units)
    
    def get_air_temperature(self, units):
        """Get air temperature with unit conversion"""
        return TemperatureUnits.fromBaseUnits(self.air_temperature_, units)
    
    def get_sun_shade(self, units):
        """Get sun/shade factor with unit conversion"""
        return FractionUnits.fromBaseUnits(self.sun_shade_, units)
    
    def get_ignition_fuel_bed_type(self):
        """Get fuel bed type"""
        return self.fuel_bed_type_
    
    def get_duff_depth(self, units):
        """Get duff depth with unit conversion"""
        return LengthUnits.fromBaseUnits(self.duff_depth_, units)
    
    def get_lightning_charge_type(self):
        """Get lightning charge type"""
        return self.lightning_charge_type_


class Ignite:
    """
    Ignite component for calculating ignition probability from firebrands and lightning.
    
    Calculates the probability of fire ignition based on fuel characteristics,
    moisture content, temperature, and lightning properties.
    """
    
    def __init__(self):
        """Initialize Ignite component"""
        self.ignite_inputs_ = IgniteInputs()
        self.fuel_temperature_ = 0.0
    
    def initialize_members(self):
        """Reset all members to default state"""
        self.ignite_inputs_.initialize_members()
        self.fuel_temperature_ = 0.0
    
    # Input setters
    def set_moisture_one_hour(self, moisture, units):
        """Set 1-hour fuel moisture"""
        self.ignite_inputs_.set_moisture_one_hour(moisture, units)
    
    def set_moisture_hundred_hour(self, moisture, units):
        """Set 100-hour fuel moisture"""
        self.ignite_inputs_.set_moisture_hundred_hour(moisture, units)
    
    def set_air_temperature(self, temperature, units):
        """Set air temperature"""
        self.ignite_inputs_.set_air_temperature(temperature, units)
    
    def set_sun_shade(self, sun_shade, units):
        """Set sun/shade factor"""
        self.ignite_inputs_.set_sun_shade(sun_shade, units)
    
    def set_ignition_fuel_bed_type(self, fuel_bed_type):
        """Set fuel bed type"""
        self.ignite_inputs_.set_ignition_fuel_bed_type(fuel_bed_type)
    
    def set_duff_depth(self, depth, units):
        """Set duff depth"""
        self.ignite_inputs_.set_duff_depth(depth, units)
    
    def set_lightning_charge_type(self, charge_type):
        """Set lightning charge type"""
        self.ignite_inputs_.set_lightning_charge_type(charge_type)
    
    def update_ignite_inputs(self, moisture_one_hour, moisture_hundred_hour, moisture_units,
                            air_temperature, temperature_units, sun_shade, sun_shade_units,
                            fuel_bed_type, duff_depth, duff_depth_units, lightning_charge_type):
        """Update all ignite inputs at once"""
        self.ignite_inputs_.update_ignite_inputs(moisture_one_hour, moisture_hundred_hour, moisture_units,
                                                 air_temperature, temperature_units, sun_shade, sun_shade_units,
                                                 fuel_bed_type, duff_depth, duff_depth_units, lightning_charge_type)
    
    # Calculation methods
    def calculate_firebrand_ignition_probability(self, desired_units):
        """
        Calculate probability of ignition from a firebrand.
        
        Uses empirical equation based on fuel temperature and 1-hour fuel moisture.
        
        Args:
            desired_units: FractionUnits enum for return value (Fraction or Percent)
            
        Returns:
            Probability of ignition from firebrand (0.0 to 1.0 or 0 to 100%)
        """
        # Calculate fuel temperature
        self.calculate_fuel_temperature()
        
        # Get fuel temperature in Celsius
        fuel_temperature = self.get_fuel_temperature(TemperatureUnits.TemperatureUnitsEnum.Celsius)
        
        # Get 1-hour moisture as fraction
        fuel_moisture = self.ignite_inputs_.get_moisture_one_hour(FractionUnits.FractionUnitsEnum.Fraction)
        
        # Calculate heat of ignition using empirical equation
        heat_of_ignition = (144.51
                           - 0.26600 * fuel_temperature
                           - 0.00058 * fuel_temperature * fuel_temperature
                           - fuel_temperature * fuel_moisture
                           + 18.5400 * (1.0 - math.exp(-15.1 * fuel_moisture))
                           + 640.000 * fuel_moisture)
        
        # Cap heat of ignition at 400
        if heat_of_ignition > 400.0:
            heat_of_ignition = 400.0
        
        # Calculate ignition probability
        x = 0.1 * (400.0 - heat_of_ignition)
        probability_of_ignition = (0.000048 * math.pow(x, 4.3)) / 50.0
        
        # Bound probability to [0, 1]
        if probability_of_ignition > 1.0:
            probability_of_ignition = 1.0
        elif probability_of_ignition < 0.0:
            probability_of_ignition = 0.0
        
        return FractionUnits.fromBaseUnits(probability_of_ignition, desired_units)
    
    def calculate_lightning_ignition_probability(self, desired_units):
        """
        Calculate probability of ignition from lightning strike.
        
        Uses Latham's equations with fuel bed type and charge type specific parameters.
        
        Args:
            desired_units: FractionUnits enum for return value (Fraction or Percent)
            
        Returns:
            Probability of ignition from lightning (0.0 to 1.0 or 0 to 100%)
        """
        # Probability of continuing current by charge type (Latham)
        cc_neg = 0.2
        cc_pos = 0.9
        
        # Relative frequency by charge type (Latham and Schlieter)
        freq_neg = 0.723
        freq_pos = 0.277
        
        # Convert duff depth to centimeters and restrict to maximum of 10 cm
        duff_depth = self.ignite_inputs_.get_duff_depth(LengthUnits.LengthUnitsEnum.Centimeters)
        duff_depth *= 2.54  # Convert inches to cm
        if duff_depth > 10.0:
            duff_depth = 10.0
        
        # Use 100-hour moisture as duff moisture, convert to percent, restrict to max 40%
        fuel_moisture = self.ignite_inputs_.get_moisture_hundred_hour(FractionUnits.FractionUnitsEnum.Percent)
        if fuel_moisture > 40.0:
            fuel_moisture = 40.0
        
        # Initialize probabilities
        p_pos = 0.0
        p_neg = 0.0
        
        # Calculate fuel bed type specific probabilities
        fuel_type = self.ignite_inputs_.get_ignition_fuel_bed_type()
        
        if fuel_type == IgnitionFuelBedType.PONDEROSA_PINE_LITTER:
            p_pos = 0.92 * math.exp(-0.087 * fuel_moisture)
            p_neg = 1.04 * math.exp(-0.054 * fuel_moisture)
        elif fuel_type == IgnitionFuelBedType.PUNKY_WOOD_ROTTEN_CHUNKY:
            p_pos = 0.44 * math.exp(-0.110 * fuel_moisture)
            p_neg = 0.59 * math.exp(-0.094 * fuel_moisture)
        elif fuel_type == IgnitionFuelBedType.PUNKY_WOOD_POWDER_DEEP:
            p_pos = 0.86 * math.exp(-0.060 * fuel_moisture)
            p_neg = 0.90 * math.exp(-0.056 * fuel_moisture)
        elif fuel_type == IgnitionFuelBedType.PUNK_WOOD_POWDER_SHALLOW:
            p_pos = 0.60 - (0.011 * fuel_moisture)
            p_neg = 0.73 - (0.011 * fuel_moisture)
        elif fuel_type == IgnitionFuelBedType.LODGEPOLE_PINE_DUFF:
            p_pos = 1.0 / (1.0 + math.exp(5.13 - 0.68 * duff_depth))
            p_neg = 1.0 / (1.0 + math.exp(3.84 - 0.60 * duff_depth))
        elif fuel_type == IgnitionFuelBedType.DOUGLAS_FIR_DUFF:
            p_pos = 1.0 / (1.0 + math.exp(6.69 - 1.39 * duff_depth))
            p_neg = 1.0 / (1.0 + math.exp(5.48 - 1.28 * duff_depth))
        elif fuel_type == IgnitionFuelBedType.HIGH_ALTITUDE_MIXED:
            p_pos = 0.62 * math.exp(-0.050 * fuel_moisture)
            p_neg = 0.80 - (0.014 * fuel_moisture)
        elif fuel_type == IgnitionFuelBedType.PEAT_MOSS:
            p_pos = 0.71 * math.exp(-0.070 * fuel_moisture)
            p_neg = 0.84 * math.exp(-0.060 * fuel_moisture)
        
        # Calculate probability based on charge type
        charge = self.ignite_inputs_.get_lightning_charge_type()
        
        if charge == LightningCharge.NEGATIVE:
            probability_of_lightning_ignition = cc_neg * p_neg
        elif charge == LightningCharge.POSITIVE:
            probability_of_lightning_ignition = cc_pos * p_pos
        elif charge == LightningCharge.UNKNOWN:
            # For unknown, use weighted average
            probability_of_lightning_ignition = (freq_neg * cc_neg * p_neg) + (freq_pos * cc_pos * p_pos)
        else:
            probability_of_lightning_ignition = 0.0
        
        # Bound probability to [0, 1]
        if probability_of_lightning_ignition > 1.0:
            probability_of_lightning_ignition = 1.0
        elif probability_of_lightning_ignition < 0.0:
            probability_of_lightning_ignition = 0.0
        
        return FractionUnits.fromBaseUnits(probability_of_lightning_ignition, desired_units)
    
    def calculate_fuel_temperature(self):
        """
        Calculate fuel temperature from air temperature and sun/shade factor.
        
        Returns:
            Fuel temperature in Fahrenheit (base units)
        """
        # Get inputs
        sun_shade = self.ignite_inputs_.get_sun_shade(FractionUnits.FractionUnitsEnum.Fraction)
        air_temperature = self.ignite_inputs_.get_air_temperature(TemperatureUnits.TemperatureUnitsEnum.Fahrenheit)
        
        # Calculate temperature differential based on sun/shade
        temperature_differential = 25.0 - (20.0 * sun_shade)
        
        # Calculate fuel temperature
        self.fuel_temperature_ = air_temperature + temperature_differential
        return self.fuel_temperature_
    
    # Getter methods
    def get_air_temperature(self, units):
        """Get air temperature with unit conversion"""
        return self.ignite_inputs_.get_air_temperature(units)
    
    def get_fuel_temperature(self, units):
        """Get fuel temperature with unit conversion"""
        return TemperatureUnits.fromBaseUnits(self.fuel_temperature_, units)
    
    def get_moisture_one_hour(self, units):
        """Get 1-hour fuel moisture with unit conversion"""
        return self.ignite_inputs_.get_moisture_one_hour(units)
    
    def get_moisture_hundred_hour(self, units):
        """Get 100-hour fuel moisture with unit conversion"""
        return self.ignite_inputs_.get_moisture_hundred_hour(units)
    
    def get_sun_shade(self, units):
        """Get sun/shade factor with unit conversion"""
        return self.ignite_inputs_.get_sun_shade(units)
    
    def get_duff_depth(self, units):
        """Get duff depth with unit conversion"""
        return self.ignite_inputs_.get_duff_depth(units)
    
    def get_fuel_bed_type(self):
        """Get fuel bed type"""
        return self.ignite_inputs_.get_ignition_fuel_bed_type()
    
    def get_lightning_charge_type(self):
        """Get lightning charge type"""
        return self.ignite_inputs_.get_lightning_charge_type()
    
    def is_fuel_depth_needed(self):
        """
        Determine if fuel depth input is needed based on fuel bed type.
        
        Returns:
            True if fuel bed type requires duff depth input, False otherwise
        """
        fuel_type = self.ignite_inputs_.get_ignition_fuel_bed_type()
        
        # Fuel types that require duff depth
        requires_depth = [
            IgnitionFuelBedType.LODGEPOLE_PINE_DUFF,
            IgnitionFuelBedType.DOUGLAS_FIR_DUFF
        ]
        
        return fuel_type in requires_depth
