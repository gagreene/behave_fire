"""
Safety component for firefighter safety zone calculations.
Calculates safety zone size, separation distance, and radius for protection from radiant heat.
"""

import math

try:
    from .behave_units import LengthUnits, AreaUnits
except ImportError:
    from behave_units import LengthUnits, AreaUnits


class Safety:
    """
    Safety component for calculating safety zone dimensions for firefighter protection.
    
    This class calculates the minimum size of a circular safety zone that protects
    a specified number of personnel and heavy equipment from radiant burn injury
    based on flame height.
    """
    
    def __init__(self):
        """Initialize Safety component"""
        # Input variables
        self.flame_height_ = 0.0
        self.number_of_personnel_ = 0
        self.area_per_person_ = 0.0
        self.number_of_equipment_ = 0
        self.area_per_equipment_ = 0.0
        
        # Output variables
        self.separation_distance_ = 0.0
        self.safety_zone_radius_ = 0.0
        self.safety_zone_area_ = 0.0
    
    def initialize_members(self):
        """Reset all input and output variables to defaults"""
        self.flame_height_ = 0.0
        self.number_of_personnel_ = 0
        self.area_per_person_ = 0.0
        self.number_of_equipment_ = 0
        self.area_per_equipment_ = 0.0
        self.separation_distance_ = 0.0
        self.safety_zone_radius_ = 0.0
        self.safety_zone_area_ = 0.0
    
    # Input setters
    def set_flame_height(self, flame_height, length_units):
        """
        Set the flame height with unit conversion.
        
        Args:
            flame_height: Flame height value
            length_units: LengthUnits enum for the input units
        """
        self.flame_height_ = LengthUnits.toBaseUnits(flame_height, length_units)
    
    def set_number_of_personnel(self, number_of_personnel):
        """
        Set the number of personnel in the safety zone.
        
        Args:
            number_of_personnel: Number of personnel (integer or float)
        """
        self.number_of_personnel_ = number_of_personnel
    
    def set_area_per_person(self, area_per_person, area_units):
        """
        Set the mean area required per person within safety zone.
        
        Args:
            area_per_person: Area per person value
            area_units: AreaUnits enum for the input units
        """
        self.area_per_person_ = AreaUnits.toBaseUnits(area_per_person, area_units)
    
    def set_number_of_equipment(self, number_of_equipment):
        """
        Set the number of equipment pieces in the safety zone.
        
        Args:
            number_of_equipment: Number of equipment pieces (integer or float)
        """
        self.number_of_equipment_ = number_of_equipment
    
    def set_area_per_equipment(self, area_per_equipment, area_units):
        """
        Set the mean area required per piece of equipment within safety zone.
        
        Args:
            area_per_equipment: Area per equipment value
            area_units: AreaUnits enum for the input units
        """
        self.area_per_equipment_ = AreaUnits.toBaseUnits(area_per_equipment, area_units)
    
    def update_safety_inputs(self, flame_height, flame_height_units, number_of_personnel, number_of_equipment, area_per_person, area_per_equipment, area_units):
        """
        Update all safety inputs at once.
        
        Args:
            flame_height: Flame height value
            flame_height_units: LengthUnits enum for flame height
            number_of_personnel: Number of personnel
            number_of_equipment: Number of equipment pieces
            area_per_person: Area per person value
            area_per_equipment: Area per equipment value
            area_units: AreaUnits enum for both area values
        """
        self.set_flame_height(flame_height, flame_height_units)
        self.set_number_of_personnel(number_of_personnel)
        self.set_number_of_equipment(number_of_equipment)
        self.set_area_per_person(area_per_person, area_units)
        self.set_area_per_equipment(area_per_equipment, area_units)
    
    # Calculation methods
    def calculate_safety_zone(self):
        """
        Calculate the safety zone dimensions.
        
        Computes:
        1. Separation distance (4 × flame height)
        2. Core radius needed for personnel and equipment
        3. Total safety zone radius
        4. Total safety zone area
        """
        # Calculate separation distance from flame height
        self.calculate_safety_zone_separation_distance()
        
        # Calculate space needed by firefighters and equipment in core of safety zone
        # Core area = area per person × number of personnel + area per equipment × number of equipment
        core_area = (self.area_per_person_ * self.number_of_personnel_ +
                     self.number_of_equipment_ * self.area_per_equipment_)
        
        # Convert core area to core radius: Area = π × r²  →  r = √(Area / π)
        core_radius = math.sqrt(core_area / math.pi) if core_area > 1.0e-07 else 0.0
        
        # Total safety zone radius = separation distance + core radius
        self.safety_zone_radius_ = self.separation_distance_ + core_radius
        
        # Calculate safety zone area: Area = π × r²
        self.safety_zone_area_ = math.pi * self.safety_zone_radius_ * self.safety_zone_radius_
    
    def calculate_safety_zone_separation_distance(self):
        """
        Calculate the separation distance from flame height.
        
        Separation distance = 4 × flame height
        
        This is the minimum distance a firefighter in protective clothing
        must be to prevent radiant heat injury.
        
        Returns:
            Separation distance in base units (feet)
        """
        self.separation_distance_ = 4.0 * self.flame_height_
        return self.separation_distance_
    
    # Output getters with unit conversion
    def get_separation_distance(self, length_units):
        """
        Get the separation distance with unit conversion.
        
        Args:
            length_units: LengthUnits enum for desired output units
            
        Returns:
            Separation distance in requested units
        """
        return LengthUnits.fromBaseUnits(self.separation_distance_, length_units)
    
    def get_safety_zone_radius(self, length_units):
        """
        Get the safety zone radius with unit conversion.
        
        Args:
            length_units: LengthUnits enum for desired output units
            
        Returns:
            Safety zone radius in requested units
        """
        return LengthUnits.fromBaseUnits(self.safety_zone_radius_, length_units)
    
    def get_safety_zone_area(self, area_units):
        """
        Get the safety zone area with unit conversion.
        
        Args:
            area_units: AreaUnits enum for desired output units
            
        Returns:
            Safety zone area in requested units
        """
        return AreaUnits.fromBaseUnits(self.safety_zone_area_, area_units)
