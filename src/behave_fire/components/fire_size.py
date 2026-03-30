"""
FireSize component for fire ellipse and dimension calculations.
Calculates fire dimensions based on spread rates and wind speed using elliptical fire models.
"""

import math

try:
    from .behave_units import LengthUnits, SpeedUnits, TimeUnits, AreaUnits
except ImportError:
    from behave_units import LengthUnits, SpeedUnits, TimeUnits, AreaUnits


class FireSize:
    """
    FireSize component for calculating various properties related to fire size.
    
    Uses elliptical fire model to calculate fire dimensions, perimeter, and area
    based on forward and backing spread rates, wind speed, and elapsed time.
    """
    
    def __init__(self):
        """Initialize FireSize component"""
        # Inputs (stored in base units)
        self.effective_wind_speed_ = 0.0  # mph
        self.forward_spread_rate_ = 0.0   # ft/min
        self.elapsed_time_ = 0.0          # minutes
        
        # Outputs
        self.elliptical_a_ = 0.0          # semi-minor axis (ft/min) - width direction
        self.elliptical_b_ = 0.0          # semi-major axis (ft/min) - length direction
        self.elliptical_c_ = 0.0          # distance from center to focus (ft/min)
        self.eccentricity_ = 0.0          # measure of deviance from circle [0, 1)
        self.backing_spread_rate_ = 0.0   # spread rate at 180° from max (ft/min)
        self.flanking_spread_rate_ = 0.0  # spread rate at widest width (ft/min)
        self.fire_length_to_width_ratio_ = 1.0  # L:W ratio
        self.heading_to_backing_ratio_ = 0.0    # Alexander 1985 ratio
    
    def calculate_fire_basic_dimensions(self, is_crown, effective_wind_speed, wind_speed_units,
                                       forward_spread_rate, spread_rate_units):
        """
        Calculate all basic fire dimensions.
        
        Args:
            is_crown: Boolean - True for crown fire, False for surface fire
            effective_wind_speed: Wind speed value
            wind_speed_units: SpeedUnits enum for wind speed
            forward_spread_rate: Forward spread rate value
            spread_rate_units: SpeedUnits enum for spread rate
        """
        # Convert spread rate to base units (ft/min)
        self.forward_spread_rate_ = SpeedUnits.toBaseUnits(forward_spread_rate, spread_rate_units)
        
        # Convert wind speed to mph
        if wind_speed_units != SpeedUnits.SpeedUnitsEnum.MilesPerHour:
            # First convert to base units (ft/min), then to mph
            wind_fpm = SpeedUnits.toBaseUnits(effective_wind_speed, wind_speed_units)
            self.effective_wind_speed_ = SpeedUnits.fromBaseUnits(wind_fpm, SpeedUnits.SpeedUnitsEnum.MilesPerHour)
        else:
            self.effective_wind_speed_ = effective_wind_speed
        
        # Calculate length-to-width ratio based on fire type
        if is_crown:
            self.calculate_crown_fire_length_to_width_ratio()
        else:
            self.calculate_surface_fire_length_to_width_ratio()
        
        # Calculate remaining fire properties
        self.calculate_fire_eccentricity()
        self.calculate_backing_spread_rate()
        self.calculate_flanking_spread_rate()
        self.calculate_elliptical_dimensions()
    
    def calculate_surface_fire_length_to_width_ratio(self):
        """
        Calculate length-to-width ratio for surface fire.
        Uses empirical equation based on wind speed.
        """
        if self.effective_wind_speed_ > 1.0e-07:
            # Empirical equation for surface fire L:W ratio
            self.fire_length_to_width_ratio_ = (0.936 * math.exp(0.1147 * self.effective_wind_speed_) +
                                               0.461 * math.exp(-0.0692 * self.effective_wind_speed_) -
                                               0.397)
            # Cap at maximum eccentricity
            if self.fire_length_to_width_ratio_ > 8.0:
                self.fire_length_to_width_ratio_ = 8.0
        else:
            self.fire_length_to_width_ratio_ = 1.0
    
    def calculate_crown_fire_length_to_width_ratio(self):
        """
        Calculate length-to-width ratio for crown fire.
        Uses Rothermel 1991 equation 10.
        """
        if self.effective_wind_speed_ > 1.0e-07:
            # Rothermel 1991, Equation 10
            self.fire_length_to_width_ratio_ = 1.0 + 0.125 * self.effective_wind_speed_
        else:
            self.fire_length_to_width_ratio_ = 1.0
    
    def calculate_fire_eccentricity(self):
        """
        Calculate fire ellipse eccentricity.
        Measure of deviance from perfect circle, ranges from [0, 1).
        """
        self.eccentricity_ = 0.0
        x = (self.fire_length_to_width_ratio_ * self.fire_length_to_width_ratio_) - 1.0
        if x > 0.0:
            self.eccentricity_ = math.sqrt(x) / self.fire_length_to_width_ratio_
    
    def calculate_elliptical_dimensions(self):
        """
        Calculate elliptical dimensions (a, b, c).
        Also calculates Alexander 1985 heading to backing ratio.
        """
        self.elliptical_a_ = 0.0
        self.elliptical_b_ = 0.0
        self.elliptical_c_ = 0.0
        self.heading_to_backing_ratio_ = 0.0
        
        # Internally A, B, and C are in terms of ft travelled in one minute
        self.elliptical_b_ = (self.forward_spread_rate_ + self.backing_spread_rate_) / 2.0
        
        if self.fire_length_to_width_ratio_ > 1e-07:
            # Alexander 1985 heading/backing ratio
            part = math.sqrt(self.fire_length_to_width_ratio_ ** 2 - 1)
            self.heading_to_backing_ratio_ = ((self.fire_length_to_width_ratio_ + part) /
                                             (self.fire_length_to_width_ratio_ - part))
            
            # Semi-minor axis
            self.elliptical_a_ = self.elliptical_b_ / self.fire_length_to_width_ratio_
        
        # Distance from center to focus
        self.elliptical_c_ = self.elliptical_b_ - self.backing_spread_rate_
    
    def calculate_backing_spread_rate(self):
        """
        Calculate backing spread rate (opposite direction from forward spread).
        Uses eccentricity to modify forward spread rate.
        """
        self.backing_spread_rate_ = (self.forward_spread_rate_ *
                                    (1.0 - self.eccentricity_) /
                                    (1.0 + self.eccentricity_))
    
    def calculate_flanking_spread_rate(self):
        """
        Calculate flanking spread rate (perpendicular to spread direction).
        """
        fire_length = self.backing_spread_rate_ + self.forward_spread_rate_
        width = fire_length / self.fire_length_to_width_ratio_
        self.flanking_spread_rate_ = width * 0.5
    
    # Getter methods
    def get_fire_length_to_width_ratio(self):
        """Get the length-to-width ratio"""
        return self.fire_length_to_width_ratio_
    
    def get_eccentricity(self):
        """Get the fire ellipse eccentricity"""
        return self.eccentricity_
    
    def get_heading_to_backing_ratio(self):
        """Get the Alexander 1985 heading to backing ratio"""
        return self.heading_to_backing_ratio_
    
    def get_backing_spread_rate(self, spread_rate_units):
        """
        Get backing spread rate with unit conversion.
        
        Args:
            spread_rate_units: SpeedUnits enum for desired output
            
        Returns:
            Backing spread rate in requested units
        """
        return SpeedUnits.fromBaseUnits(self.backing_spread_rate_, spread_rate_units)
    
    def get_flanking_spread_rate(self, spread_rate_units):
        """
        Get flanking spread rate with unit conversion.
        
        Args:
            spread_rate_units: SpeedUnits enum for desired output
            
        Returns:
            Flanking spread rate in requested units
        """
        return SpeedUnits.fromBaseUnits(self.flanking_spread_rate_, spread_rate_units)
    
    def get_elliptical_a(self, length_units, elapsed_time, time_units):
        """
        Get elliptical semi-minor axis (fire width).
        
        Args:
            length_units: LengthUnits enum for desired output
            elapsed_time: Elapsed time value
            time_units: TimeUnits enum for elapsed time
            
        Returns:
            Elliptical a in requested units
        """
        elapsed_time_minutes = TimeUnits.toBaseUnits(elapsed_time, time_units)
        return LengthUnits.fromBaseUnits((self.elliptical_a_ * elapsed_time_minutes), length_units)
    
    def get_elliptical_b(self, length_units, elapsed_time, time_units):
        """
        Get elliptical semi-major axis (fire length).
        
        Args:
            length_units: LengthUnits enum for desired output
            elapsed_time: Elapsed time value
            time_units: TimeUnits enum for elapsed time
            
        Returns:
            Elliptical b in requested units
        """
        elapsed_time_minutes = TimeUnits.toBaseUnits(elapsed_time, time_units)
        return LengthUnits.fromBaseUnits((self.elliptical_b_ * elapsed_time_minutes), length_units)
    
    def get_elliptical_c(self, length_units, elapsed_time, time_units):
        """
        Get distance from ellipse center to focus.
        
        Args:
            length_units: LengthUnits enum for desired output
            elapsed_time: Elapsed time value
            time_units: TimeUnits enum for elapsed time
            
        Returns:
            Elliptical c in requested units
        """
        elapsed_time_minutes = TimeUnits.toBaseUnits(elapsed_time, time_units)
        return LengthUnits.fromBaseUnits((self.elliptical_c_ * elapsed_time_minutes), length_units)
    
    def get_fire_length(self, length_units, elapsed_time, time_units):
        """
        Get total fire length (head to tail).
        Fire length = 2 × elliptical_b
        
        Args:
            length_units: LengthUnits enum for desired output
            elapsed_time: Elapsed time value
            time_units: TimeUnits enum for elapsed time
            
        Returns:
            Fire length in requested units
        """
        elapsed_time_minutes = TimeUnits.toBaseUnits(elapsed_time, time_units)
        return LengthUnits.fromBaseUnits((self.elliptical_b_ * elapsed_time_minutes * 2.0), length_units)
    
    def get_max_fire_width(self, length_units, elapsed_time, time_units):
        """
        Get maximum fire width (perpendicular to spread direction).
        Max width = 2 × elliptical_a
        
        Args:
            length_units: LengthUnits enum for desired output
            elapsed_time: Elapsed time value
            time_units: TimeUnits enum for elapsed time
            
        Returns:
            Maximum fire width in requested units
        """
        elapsed_time_minutes = TimeUnits.toBaseUnits(elapsed_time, time_units)
        return LengthUnits.fromBaseUnits((self.elliptical_a_ * elapsed_time_minutes * 2.0), length_units)
    
    def get_fire_perimeter(self, is_crown, length_units, elapsed_time, time_units):
        """
        Get fire perimeter.
        Uses different equations for crown vs surface fires.
        
        Args:
            is_crown: Boolean - True for crown fire
            length_units: LengthUnits enum for desired output
            elapsed_time: Elapsed time value
            time_units: TimeUnits enum for elapsed time
            
        Returns:
            Fire perimeter in requested units
        """
        elapsed_time_minutes = TimeUnits.toBaseUnits(elapsed_time, time_units)
        perimeter = 0.0
        
        if is_crown:
            # Crown fire perimeter (Rothermel 1991, equation 13)
            spread_distance = self.forward_spread_rate_ * elapsed_time_minutes
            perimeter = (0.5 * math.pi * spread_distance *
                        (1.0 + 1.0 / self.fire_length_to_width_ratio_))
        else:
            # Surface fire perimeter (ellipse approximation)
            my_elliptical_a = self.elliptical_a_ * elapsed_time_minutes
            my_elliptical_b = self.elliptical_b_ * elapsed_time_minutes
            
            if (my_elliptical_a + my_elliptical_b) > 1.0e-07:
                # Ramanujan's approximation for ellipse perimeter
                a_minus_b = my_elliptical_a - my_elliptical_b
                a_minus_b_squared = a_minus_b * a_minus_b
                a_plus_b = my_elliptical_a + my_elliptical_b
                a_plus_b_squared = a_plus_b * a_plus_b
                h = a_minus_b_squared / a_plus_b_squared
                
                perimeter = math.pi * a_plus_b * (1 + (h / 4.0) + ((h * h) / 64.0))
        
        return LengthUnits.fromBaseUnits(perimeter, length_units)
    
    def get_fire_area(self, is_crown, area_units, elapsed_time, time_units):
        """
        Get fire area.
        Uses different equations for crown vs surface fires.
        
        Args:
            is_crown: Boolean - True for crown fire
            area_units: AreaUnits enum for desired output
            elapsed_time: Elapsed time value
            time_units: TimeUnits enum for elapsed time
            
        Returns:
            Fire area in requested units
        """
        elapsed_time_minutes = TimeUnits.toBaseUnits(elapsed_time, time_units)
        area = 0.0
        
        if is_crown:
            # Crown fire area (Rothermel 1991, equation 11)
            # Uses only forward spread, ignores backing
            spread_distance = self.forward_spread_rate_ * elapsed_time_minutes
            area = AreaUnits.fromBaseUnits(
                (math.pi * spread_distance * spread_distance /
                 (4.0 * self.fire_length_to_width_ratio_)),
                area_units
            )
        else:
            # Surface fire area (ellipse area formula)
            area = AreaUnits.fromBaseUnits(
                (math.pi * self.elliptical_a_ * self.elliptical_b_ *
                 elapsed_time_minutes * elapsed_time_minutes),
                area_units
            )
        
        return area

