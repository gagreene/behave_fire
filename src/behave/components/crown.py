# Crown Fire Behavior Model - Python Conversion
# Complete implementation with Rothermel and Scott-Reinhardt crown fire models

from enum import IntEnum
try:
    from .surface import Surface
    from .fire_size import FireSize
    from .behave_units import (
        SpeedUnits, LengthUnits, AreaUnits, FirelineIntensityUnits,
        FractionUnits, SlopeUnits, DensityUnits, TimeUnits
    )
except ImportError:
    from components.surface import Surface
    from components.fire_size import FireSize
    from components.behave_units import (
        SpeedUnits, LengthUnits, AreaUnits, FirelineIntensityUnits,
        FractionUnits, SlopeUnits, DensityUnits, TimeUnits
    )

# ============================================================================
# ENUMS
# ============================================================================

class FireType(IntEnum):
    Surface = 0
    Torching = 1
    Crowning = 2
    ConditionalCrownFire = 3


class CrownModelType(IntEnum):
    Rothermel = 0
    ScottAndReinhardt = 1


# ============================================================================
# CROWN CLASS
# ============================================================================

class Crown:
    def __init__(self, fuel_models):
        self.fuel_models = fuel_models
        self.surface_fuel = Surface(fuel_models)
        self.crown_fuel = Surface(fuel_models)
        self.crown_fire_size = FireSize()
        self.crown_inputs = {
            'canopy_base_height': 0.0,
            'canopy_bulk_density': 0.0,
            'moisture_foliar': 0.0,
        }
        # Moisture input mode support
        self._moisture_input_mode = 'BySizeClass'
        self._moisture_scenarios = {}
        self._current_scenario_name = ''
        self._initialize_results()

    def _initialize_results(self):
        self.fire_type = FireType.Surface
        self.surface_fire_heat_per_unit_area = 0.0
        self.surface_fire_line_intensity = 0.0
        self.surface_fire_spread_rate = 0.0
        self.surface_fire_flame_length = 0.0
        self.surface_fire_critical_spread_rate = 0.0
        self.crown_fuel_load = 0.0
        self.canopy_heat_per_unit_area = 0.0
        self.crown_fire_heat_per_unit_area = 0.0
        self.crown_fire_line_intensity = 0.0
        self.crown_flame_length = 0.0
        self.crown_fire_spread_rate = 0.0
        self.crown_critical_surface_fire_line_intensity = 0.0
        self.crown_critical_fire_spread_rate = 0.0
        self.crown_critical_surface_flame_length = 0.0
        self.crown_fire_active_ratio = 0.0
        self.crown_fire_transition_ratio = 0.0
        self.crown_fire_length_to_width_ratio = 1.0
        self.crown_fire_active_wind_speed = 0.0
        self.crown_fraction_burned = 0.0
        self.crowning_surface_fire_ros = 0.0
        self.wind_speed_at_twenty_feet = 0.0
        self.passive_crown_fire_spread_rate = 0.0
        self.passive_crown_fire_heat_per_unit_area = 0.0
        self.passive_crown_fire_line_intensity = 0.0
        self.passive_crown_fire_flame_length = 0.0
        self.final_spread_rate = 0.0
        self.final_heat_per_unit_area = 0.0
        self.final_fire_line_intensity = 0.0
        self.final_flame_length = 0.0
        self.is_surface_fire = False
        self.is_passive_crown_fire = False
        self.is_active_crown_fire = False
        self.is_crown_fire = False

    def initialize_members(self):
        self._initialize_results()
        self.crown_inputs = {'canopy_base_height': 0.0, 'canopy_bulk_density': 0.0, 'moisture_foliar': 0.0}
        self.surface_fuel.initialize_members()
        self.crown_fuel.initialize_members()

    def set_fuel_models(self, fuel_models):
        self.fuel_models = fuel_models
        self.surface_fuel.set_fuel_models(fuel_models)
        self.crown_fuel.set_fuel_models(fuel_models)

    def set_moisture_scenarios(self, moisture_scenarios):
        self._moisture_scenarios = moisture_scenarios
        self.surface_fuel.set_moisture_scenarios(moisture_scenarios)
        self.crown_fuel.set_moisture_scenarios(moisture_scenarios)

    # ========================================================================
    # updateCrownInputs
    # ========================================================================

    def updateCrownInputs(self, fuel_model_number,
                          moisture_one_hour, moisture_ten_hour, moisture_hundred_hour,
                          moisture_live_herbaceous, moisture_live_woody, moisture_foliar,
                          moisture_units,
                          wind_speed, wind_speed_units, wind_height_input_mode,
                          wind_direction, wind_orientation_mode,
                          slope, slope_units, aspect,
                          canopy_cover, canopy_cover_units,
                          canopy_height, canopy_base_height, canopy_height_units,
                          crown_ratio, crown_ratio_units,
                          canopy_bulk_density, density_units):
        """Update all crown inputs at once."""
        self.surface_fuel.updateSurfaceInputs(
            fuel_model_number,
            moisture_one_hour, moisture_ten_hour, moisture_hundred_hour,
            moisture_live_herbaceous, moisture_live_woody, moisture_units,
            wind_speed, wind_speed_units, wind_height_input_mode,
            wind_direction, wind_orientation_mode,
            slope, slope_units, aspect,
            canopy_cover, canopy_cover_units,
            canopy_height, canopy_height_units,
            crown_ratio, crown_ratio_units
        )
        self.set_moisture_foliar(moisture_foliar, moisture_units)
        cbh_ft = LengthUnits.toBaseUnits(canopy_base_height, canopy_height_units)
        self.crown_inputs['canopy_base_height'] = cbh_ft
        cbd_lb_ft3 = canopy_bulk_density
        if density_units == DensityUnits.DensityUnitsEnum.KilogramsPerCubicMeter:
            cbd_lb_ft3 = canopy_bulk_density / 16.0185
        self.crown_inputs['canopy_bulk_density'] = cbd_lb_ft3

    def update_crown_inputs(self, *args, **kwargs):
        """snake_case alias for updateCrownInputs."""
        return self.updateCrownInputs(*args, **kwargs)

    # ========================================================================
    # MAIN CALCULATION METHODS
    # ========================================================================

    def _get_wind_speed_ft_per_min(self):
        """Get 20-ft wind speed in ft/min from surface_fuel state."""
        ws = self.surface_fuel.state.get('wind_speed', 0.0)
        wu = self.surface_fuel.state.get('wind_speed_units', SpeedUnits.SpeedUnitsEnum.FeetPerMinute)
        ws_base = SpeedUnits.toBaseUnits(ws, wu)
        return SpeedUnits.fromBaseUnits(ws_base, SpeedUnits.SpeedUnitsEnum.FeetPerMinute)

    def do_crown_run_rothermel(self):
        """Crown fire using Rothermel 1991."""
        canopy_height = self.surface_fuel.get_canopy_height('ft')
        canopy_base_height = self.crown_inputs['canopy_base_height']
        crown_ratio = 0.0
        if canopy_height > 0:
            crown_ratio = (canopy_height - canopy_base_height) / canopy_height

        waf_method = self.surface_fuel.state.get('waf_calculation_method', 'UseCrownRatio')
        if isinstance(waf_method, str) and 'crownratio' in waf_method.lower().replace('_', '').replace(' ', ''):
            self.surface_fuel.set_crown_ratio(crown_ratio, FractionUnits.FractionUnitsEnum.Fraction)

        self.surface_fuel.do_surface_run_in_direction_of_max_spread()
        self.surface_fire_heat_per_unit_area = self.surface_fuel.results.get('heat_per_unit_area', 0.0)
        self.surface_fire_line_intensity = self.surface_fuel.results.get('fireline_intensity', 0.0)
        self.surface_fire_spread_rate = self.surface_fuel.results.get('spread_rate', 0.0)
        self.surface_fire_flame_length = self.surface_fuel.results.get('flame_length', 0.0)

        # Crown fuel = fuel model 10 with WAF=0.4, slope=0, upslope mode
        self.crown_fuel = Surface(self.fuel_models)
        self.crown_fuel.state = self.surface_fuel.state.copy()
        self.crown_fuel.set_fuel_model_number(10)
        self.crown_fuel.set_wind_adjustment_factor_calculation_method('UserInput')
        self.crown_fuel.set_user_provided_wind_adjustment_factor(0.4)
        self.crown_fuel.set_slope(0.0, SlopeUnits.SlopeUnitsEnum.Degrees)
        self.crown_fuel.set_wind_direction(0.0)
        self.crown_fuel.set_wind_and_spread_orientation_mode('RelativeToUpslope')

        self.crown_fuel.do_surface_run_in_direction_of_max_spread()
        self.crown_fire_spread_rate = 3.34 * self.crown_fuel.results.get('spread_rate', 0.0)

        self._calculate_crown_fuel_load()
        self._calculate_canopy_heat_per_unit_area()
        self._calculate_crown_fire_heat_per_unit_area()
        self._calculate_crown_fire_line_intensity()
        self._calculate_crown_flame_length()
        self._calculate_crown_critical_fire_spread_rate()
        self._calculate_crown_fire_active_ratio()
        self._calculate_crown_critical_surface_fire_intensity()
        self._calculate_crown_critical_surface_flame_length()
        self._calculate_crown_fire_transition_ratio()
        self._calculate_wind_speed_at_twenty_feet()

        self.crown_fire_size.calculate_fire_basic_dimensions(
            True, self.wind_speed_at_twenty_feet, 'mph',
            self.crown_fire_spread_rate, 'ft/min'
        )
        self.crown_fire_length_to_width_ratio = self.crown_fire_size.get_fire_length_to_width_ratio()

        self._calculate_fire_type_rothermel()
        self._assign_final_fire_behavior(CrownModelType.Rothermel)

    def do_crown_run_scott_and_reinhardt(self):
        """Crown fire using Scott & Reinhardt 2001."""
        canopy_height = self.surface_fuel.get_canopy_height('ft')
        canopy_base_height = self.crown_inputs['canopy_base_height']
        crown_ratio = 0.0
        if canopy_height > 0:
            crown_ratio = (canopy_height - canopy_base_height) / canopy_height

        waf_method = self.surface_fuel.state.get('waf_calculation_method', 'UseCrownRatio')
        if isinstance(waf_method, str) and 'crownratio' in waf_method.lower().replace('_', '').replace(' ', ''):
            self.surface_fuel.set_crown_ratio(crown_ratio, FractionUnits.FractionUnitsEnum.Fraction)

        # Apply moisture input mode
        self._apply_moisture_input_mode()

        self.surface_fuel.do_surface_run_in_direction_of_max_spread()
        self.surface_fire_spread_rate = self.surface_fuel.results.get('spread_rate', 0.0)
        self.surface_fire_heat_per_unit_area = self.surface_fuel.results.get('heat_per_unit_area', 0.0)
        self.surface_fire_line_intensity = self.surface_fuel.results.get('fireline_intensity', 0.0)
        self.surface_fire_flame_length = self.surface_fuel.results.get('flame_length', 0.0)

        wind_speed_fpm = self._get_wind_speed_ft_per_min()

        self.crown_fuel = Surface(self.fuel_models)
        self.crown_fuel.state = self.surface_fuel.state.copy()
        self.crown_fuel.set_fuel_model_number(10)
        self.crown_fuel.set_wind_adjustment_factor_calculation_method('UserInput')
        self.crown_fuel.set_user_provided_wind_adjustment_factor(0.4)
        self.crown_fuel.set_slope(0.0, SlopeUnits.SlopeUnitsEnum.Degrees)
        self.crown_fuel.set_wind_direction(0.0)
        self.crown_fuel.set_wind_and_spread_orientation_mode('RelativeToUpslope')

        self.crown_fuel.do_surface_run_in_direction_of_max_spread()
        self.crown_fire_spread_rate = 3.34 * self.crown_fuel.results.get('spread_rate', 0.0)

        self._calculate_crown_fire_active_wind_speed()
        self._calculate_crown_fuel_load()
        self._calculate_canopy_heat_per_unit_area()
        self._calculate_crown_fire_heat_per_unit_area()
        self._calculate_crown_fire_line_intensity()
        self._calculate_crown_flame_length()
        self._calculate_crown_critical_fire_spread_rate()
        self._calculate_crown_critical_surface_fire_intensity()
        self._calculate_crown_critical_surface_flame_length()
        self._calculate_crown_fire_active_ratio()
        self._calculate_crown_fire_transition_ratio()
        self._calculate_wind_speed_at_twenty_feet()

        self.crown_fire_size.calculate_fire_basic_dimensions(
            True, self.wind_speed_at_twenty_feet, 'mph',
            self.crown_fire_spread_rate, 'ft/min'
        )
        self.crown_fire_length_to_width_ratio = self.crown_fire_size.get_fire_length_to_width_ratio()

        self._calculate_fire_type_rothermel()
        self._calculate_fire_type_scott_and_reinhardt()
        self._calculate_surface_fire_critical_spread_rate_scott_and_reinhardt()
        self._calculate_crowning_surface_fire_rate_of_spread()
        self._calculate_crown_fraction_burned()

        self.passive_crown_fire_spread_rate = (
            self.surface_fire_spread_rate +
            self.crown_fraction_burned * (self.crown_fire_spread_rate - self.surface_fire_spread_rate)
        )
        self.passive_crown_fire_heat_per_unit_area = (
            self.surface_fire_heat_per_unit_area +
            self.canopy_heat_per_unit_area * self.crown_fraction_burned
        )
        self.passive_crown_fire_line_intensity = (
            self.passive_crown_fire_heat_per_unit_area * self.passive_crown_fire_spread_rate / 60.0
        )
        self._calculate_passive_crown_flame_length()
        self._assign_final_fire_behavior(CrownModelType.ScottAndReinhardt)

    # ========================================================================
    # MOISTURE INPUT MODE
    # ========================================================================

    def _apply_moisture_input_mode(self):
        """Apply the current moisture input mode to surface_fuel state."""
        mode = self._moisture_input_mode
        if mode == 'AllAggregate':
            dead = self.surface_fuel.state.get('_moisture_dead_aggregate', 0.06)
            live = self.surface_fuel.state.get('_moisture_live_aggregate', 0.90)
            self.surface_fuel.state['moisture_1h'] = dead
            self.surface_fuel.state['moisture_10h'] = dead
            self.surface_fuel.state['moisture_100h'] = dead
            self.surface_fuel.state['moisture_live_herb'] = live
            self.surface_fuel.state['moisture_live_woody'] = live
        elif mode == 'MoistureScenario':
            name = self._current_scenario_name
            scenario = self._moisture_scenarios.get(name, {}) if isinstance(self._moisture_scenarios, dict) else {}
            if hasattr(self._moisture_scenarios, 'get_scenario'):
                try:
                    scenario = self._moisture_scenarios.get_scenario(name)
                except Exception:
                    scenario = {}
            if scenario:
                self.surface_fuel.state['moisture_1h'] = scenario.get('moisture_1h', self.surface_fuel.state.get('moisture_1h', 0.06))
                self.surface_fuel.state['moisture_10h'] = scenario.get('moisture_10h', self.surface_fuel.state.get('moisture_10h', 0.07))
                self.surface_fuel.state['moisture_100h'] = scenario.get('moisture_100h', self.surface_fuel.state.get('moisture_100h', 0.08))
                self.surface_fuel.state['moisture_live_herb'] = scenario.get('moisture_live_herb', self.surface_fuel.state.get('moisture_live_herb', 0.60))
                self.surface_fuel.state['moisture_live_woody'] = scenario.get('moisture_live_woody', self.surface_fuel.state.get('moisture_live_woody', 0.90))
        # BySizeClass: use existing per-class values (already set via setters)

    # ========================================================================
    # PRIVATE CALCULATION METHODS
    # ========================================================================

    def _calculate_crown_fuel_load(self):
        canopy_bulk_density = self.crown_inputs['canopy_bulk_density']
        canopy_base_height = self.crown_inputs['canopy_base_height']
        canopy_height = self.surface_fuel.get_canopy_height('ft')
        self.crown_fuel_load = canopy_bulk_density * (canopy_height - canopy_base_height)

    def _calculate_canopy_heat_per_unit_area(self):
        LOW_HEAT_OF_COMBUSTION = 8000.0
        self.canopy_heat_per_unit_area = self.crown_fuel_load * LOW_HEAT_OF_COMBUSTION

    def _calculate_crown_fire_heat_per_unit_area(self):
        self.crown_fire_heat_per_unit_area = (
            self.surface_fire_heat_per_unit_area + self.canopy_heat_per_unit_area
        )

    def _calculate_crown_fire_line_intensity(self):
        self.crown_fire_line_intensity = (self.crown_fire_spread_rate / 60.0) * self.crown_fire_heat_per_unit_area

    def _calculate_crown_flame_length(self):
        if self.crown_fire_line_intensity <= 0.0:
            self.crown_flame_length = 0.0
        else:
            self.crown_flame_length = 0.2 * (self.crown_fire_line_intensity ** (2.0 / 3.0))

    def _calculate_passive_crown_flame_length(self):
        if self.passive_crown_fire_line_intensity <= 0.0:
            self.passive_crown_fire_flame_length = 0.0
        else:
            self.passive_crown_fire_flame_length = 0.2 * (self.passive_crown_fire_line_intensity ** (2.0 / 3.0))

    def _calculate_crown_critical_surface_fire_intensity(self):
        moisture_foliar = self.crown_inputs['moisture_foliar']
        moisture_foliar = max(moisture_foliar, 30.0)
        crown_base_height_m = self.crown_inputs['canopy_base_height'] * 0.3048
        crown_base_height_m = max(crown_base_height_m, 0.1)
        critical_intensity_kw_m = (0.010 * crown_base_height_m * (460.0 + 25.9 * moisture_foliar)) ** 1.5
        # Convert kW/m to Btu/ft/s using C++ constant: KILOWATTS_PER_METER_TO_BTUS_PER_FOOT_PER_SECOND = 0.2886719
        self.crown_critical_surface_fire_line_intensity = FirelineIntensityUnits.toBaseUnits(
            critical_intensity_kw_m, FirelineIntensityUnits.FirelineIntensityUnitsEnum.KilowattsPerMeter
        )

    def _calculate_crown_critical_surface_flame_length(self):
        if self.crown_critical_surface_fire_line_intensity <= 0.0:
            self.crown_critical_surface_flame_length = 0.0
        else:
            self.crown_critical_surface_flame_length = 0.2 * (self.crown_critical_surface_fire_line_intensity ** (2.0 / 3.0))

    def _calculate_crown_critical_fire_spread_rate(self):
        cbd_kg_m3 = self.crown_inputs['canopy_bulk_density'] * 16.0185
        if cbd_kg_m3 < 1e-07:
            self.crown_critical_fire_spread_rate = 0.0
        else:
            self.crown_critical_fire_spread_rate = (3.0 / cbd_kg_m3) * 3.28084

    def _calculate_crown_fire_active_ratio(self):
        if self.crown_critical_fire_spread_rate < 1e-07:
            self.crown_fire_active_ratio = 0.0
        else:
            self.crown_fire_active_ratio = self.crown_fire_spread_rate / self.crown_critical_fire_spread_rate

    def _calculate_crown_fire_transition_ratio(self):
        if self.crown_critical_surface_fire_line_intensity < 1e-07:
            self.crown_fire_transition_ratio = 0.0
        else:
            self.crown_fire_transition_ratio = (
                self.surface_fire_line_intensity / self.crown_critical_surface_fire_line_intensity
            )

    def _calculate_crown_fire_active_wind_speed(self):
        cbd = self.crown_inputs['canopy_bulk_density'] * 16.0185
        if cbd < 1e-7:
            self.crown_fire_active_wind_speed = 0.0
            return
        ractive = 3.28084 * (3.0 / cbd)
        r10 = ractive / 3.34

        prop_flux = 0.048317062998571636
        wind_b = 1.4308256324729873
        wind_b_inv = 1.0 / wind_b
        wind_k = 0.0016102128596515481

        reaction_intensity = self.crown_fuel.results.get('reaction_intensity', 0.0)
        heat_sink = self.crown_fuel.intermediates.heat_sink

        if reaction_intensity <= 0 or heat_sink <= 0:
            self.crown_fire_active_wind_speed = 0.0
            return

        ros0 = reaction_intensity * prop_flux / heat_sink
        slope_factor = 0.0
        if ros0 < 1e-10:
            self.crown_fire_active_wind_speed = 0.0
            return
        a = ((r10 / ros0) - 1.0 - slope_factor) / wind_k
        if a <= 0:
            self.crown_fire_active_wind_speed = 0.0
            return
        u_mid = a ** wind_b_inv
        self.crown_fire_active_wind_speed = u_mid / 0.4

    def _calculate_wind_speed_at_twenty_feet(self):
        """Calculate wind speed at 20 ft (ft/min) from surface_fuel state."""
        wind_height_mode = self.surface_fuel.get_wind_height_input_mode()
        ws_fpm = self._get_wind_speed_ft_per_min()

        mode_lower = wind_height_mode.lower().replace('_', '').replace(' ', '') if isinstance(wind_height_mode, str) else ''
        if 'tenmeter' in mode_lower:
            self.wind_speed_at_twenty_feet = ws_fpm / 1.15
        elif 'directmidflame' in mode_lower:
            # reverse WAF to get 20ft speed
            waf = self.surface_fuel._calculate_wind_adjustment_factor()
            if waf > 1e-7:
                self.wind_speed_at_twenty_feet = ws_fpm / waf
            else:
                self.wind_speed_at_twenty_feet = ws_fpm
        else:
            # TwentyFoot - already at 20ft
            self.wind_speed_at_twenty_feet = ws_fpm

    def _calculate_crowning_surface_fire_rate_of_spread(self):
        saved_state = self.surface_fuel.state.copy()
        self.surface_fuel.set_wind_speed(self.crown_fire_active_wind_speed, SpeedUnits.SpeedUnitsEnum.FeetPerMinute)
        self.surface_fuel.set_wind_height_input_mode('TwentyFoot')
        self.surface_fuel.do_surface_run_in_direction_of_max_spread()
        self.crowning_surface_fire_ros = self.surface_fuel.results.get('spread_rate', 0.0)
        self.surface_fuel.state = saved_state

    def _calculate_crown_fraction_burned(self):
        numerator = self.surface_fire_spread_rate - self.surface_fire_critical_spread_rate
        denominator = self.crowning_surface_fire_ros - self.surface_fire_critical_spread_rate
        if denominator > 1e-07:
            self.crown_fraction_burned = numerator / denominator
        else:
            self.crown_fraction_burned = 0.0
        self.crown_fraction_burned = max(0.0, min(1.0, self.crown_fraction_burned))

    def _calculate_fire_type_rothermel(self):
        self.fire_type = FireType.Surface
        self.is_active_crown_fire = False
        self.is_passive_crown_fire = False
        self.is_surface_fire = True
        if self.crown_fire_transition_ratio < 1.0:
            if self.crown_fire_active_ratio < 1.0:
                self.fire_type = FireType.Surface
            else:
                # Active crown fire possible but surface not intense enough: ConditionalCrownFire
                self.fire_type = FireType.ConditionalCrownFire
        else:
            if self.crown_fire_active_ratio < 1.0:
                # Surface intensity sufficient but spread rate not: Torching (passive)
                self.fire_type = FireType.Torching
                self.is_passive_crown_fire = True
                self.is_surface_fire = False
            else:
                # Both thresholds exceeded: active crown fire
                self.fire_type = FireType.Crowning
                self.is_active_crown_fire = True
                self.is_surface_fire = False

    def _calculate_fire_type_scott_and_reinhardt(self):
        self.is_surface_fire = (self.fire_type == FireType.Surface or self.fire_type == FireType.ConditionalCrownFire)
        self.is_passive_crown_fire = (self.fire_type == FireType.Torching)
        self.is_active_crown_fire = (self.fire_type == FireType.Crowning)
        self.is_crown_fire = self.is_active_crown_fire or self.is_passive_crown_fire

    def _calculate_surface_fire_critical_spread_rate_scott_and_reinhardt(self):
        if self.surface_fire_heat_per_unit_area > 1e-07:
            self.surface_fire_critical_spread_rate = (
                (60.0 * self.crown_critical_surface_fire_line_intensity) / self.surface_fire_heat_per_unit_area
            )
        else:
            self.surface_fire_critical_spread_rate = 0.0

    def _assign_final_fire_behavior(self, crown_model_type):
        if crown_model_type == CrownModelType.ScottAndReinhardt:
            if self.is_surface_fire:
                self.final_spread_rate = self.surface_fire_spread_rate
                self.final_heat_per_unit_area = self.surface_fire_heat_per_unit_area
                self.final_fire_line_intensity = self.surface_fire_line_intensity
                self.final_flame_length = self.surface_fire_flame_length
            elif self.is_passive_crown_fire:
                self.final_spread_rate = self.passive_crown_fire_spread_rate
                self.final_heat_per_unit_area = self.passive_crown_fire_heat_per_unit_area
                self.final_fire_line_intensity = self.passive_crown_fire_line_intensity
                self.final_flame_length = self.passive_crown_fire_flame_length
            elif self.is_active_crown_fire:
                self.final_spread_rate = self.crown_fire_spread_rate
                self.final_heat_per_unit_area = self.crown_fire_heat_per_unit_area
                self.final_fire_line_intensity = self.crown_fire_line_intensity
                self.final_flame_length = self.crown_flame_length
        elif crown_model_type == CrownModelType.Rothermel:
            if self.is_surface_fire:
                self.final_spread_rate = self.surface_fire_spread_rate
                self.final_heat_per_unit_area = self.surface_fire_heat_per_unit_area
                self.final_fire_line_intensity = self.surface_fire_line_intensity
                self.final_flame_length = self.surface_fire_flame_length
            elif self.fire_type == FireType.Torching:
                self.final_spread_rate = self.surface_fire_spread_rate
                self.final_heat_per_unit_area = self.crown_fire_heat_per_unit_area
                self.final_fire_line_intensity = self.crown_fire_line_intensity
                self.final_flame_length = self.crown_flame_length
            elif self.fire_type == FireType.Crowning:
                self.final_spread_rate = self.crown_fire_spread_rate
                self.final_heat_per_unit_area = self.crown_fire_heat_per_unit_area
                self.final_fire_line_intensity = self.crown_fire_line_intensity
                self.final_flame_length = self.crown_flame_length

    # ========================================================================
    # GETTER METHODS — use SpeedUnits/LengthUnits/FirelineIntensityUnits properly
    # ========================================================================

    def get_crown_fire_spread_rate(self, speed_units):
        return SpeedUnits.fromBaseUnits(self.crown_fire_spread_rate, speed_units)

    def get_crown_fire_length_to_width_ratio(self):
        return self.crown_fire_length_to_width_ratio

    def get_crown_fire_area(self, area_units, elapsed_time, time_units):
        return self.crown_fire_size.get_fire_area(True, area_units, elapsed_time, time_units)

    def get_crown_fire_perimeter(self, length_units, elapsed_time, time_units):
        return self.crown_fire_size.get_fire_perimeter(True, length_units, elapsed_time, time_units)

    def get_crown_flame_length(self, length_units):
        return LengthUnits.fromBaseUnits(self.crown_flame_length, length_units)

    def get_crown_fireline_intensity(self, intensity_units):
        return FirelineIntensityUnits.fromBaseUnits(self.crown_fire_line_intensity, intensity_units)

    def get_fire_type(self):
        return int(self.fire_type)

    def get_final_spread_rate(self, speed_units):
        return SpeedUnits.fromBaseUnits(self.final_spread_rate, speed_units)

    def get_final_heat_per_unit_area(self, heat_units):
        return self.final_heat_per_unit_area

    def get_final_fireline_intensity(self, intensity_units):
        return FirelineIntensityUnits.fromBaseUnits(self.final_fire_line_intensity, intensity_units)

    def get_final_flame_length(self, length_units):
        return LengthUnits.fromBaseUnits(self.final_flame_length, length_units)

    def get_critical_open_wind_speed(self, speed_units):
        return SpeedUnits.fromBaseUnits(self.crown_fire_active_wind_speed, speed_units)

    def get_crown_fraction_burned(self):
        return self.crown_fraction_burned

    def get_surface_fire_spread_rate(self, speed_units):
        return SpeedUnits.fromBaseUnits(self.surface_fire_spread_rate, speed_units)

    # ========================================================================
    # SETTER METHODS
    # ========================================================================

    def set_canopy_base_height(self, canopy_base_height, height_units):
        self.crown_inputs['canopy_base_height'] = LengthUnits.toBaseUnits(canopy_base_height, height_units)

    def set_canopy_bulk_density(self, canopy_bulk_density, density_units):
        if density_units == DensityUnits.DensityUnitsEnum.KilogramsPerCubicMeter:
            self.crown_inputs['canopy_bulk_density'] = canopy_bulk_density / 16.0185
        else:
            self.crown_inputs['canopy_bulk_density'] = canopy_bulk_density

    def set_moisture_foliar(self, moisture_foliar, moisture_units):
        if moisture_units == FractionUnits.FractionUnitsEnum.Fraction:
            self.crown_inputs['moisture_foliar'] = moisture_foliar * 100.0
        else:
            self.crown_inputs['moisture_foliar'] = moisture_foliar

    def set_canopy_cover(self, canopy_cover, cover_units):
        self.surface_fuel.set_canopy_cover(canopy_cover, cover_units)

    def set_canopy_height(self, canopy_height, height_units):
        self.surface_fuel.set_canopy_height(canopy_height, height_units)

    def set_crown_ratio(self, crown_ratio, ratio_units):
        self.surface_fuel.set_crown_ratio(crown_ratio, ratio_units)

    def set_fuel_model_number(self, fuel_model_number):
        self.surface_fuel.set_fuel_model_number(fuel_model_number)

    def set_moisture_one_hour(self, moisture, moisture_units):
        self.surface_fuel.set_moisture_one_hour(moisture, moisture_units)

    def set_moisture_ten_hour(self, moisture, moisture_units):
        self.surface_fuel.set_moisture_ten_hour(moisture, moisture_units)

    def set_moisture_hundred_hour(self, moisture, moisture_units):
        self.surface_fuel.set_moisture_hundred_hour(moisture, moisture_units)

    def set_moisture_dead_aggregate(self, moisture, moisture_units):
        val = self.surface_fuel._convert_moisture(moisture, moisture_units)
        self.surface_fuel.state['_moisture_dead_aggregate'] = val

    def set_moisture_live_aggregate(self, moisture, moisture_units):
        val = self.surface_fuel._convert_moisture(moisture, moisture_units)
        self.surface_fuel.state['_moisture_live_aggregate'] = val

    def set_moisture_live_herbaceous(self, moisture, moisture_units):
        self.surface_fuel.set_moisture_live_herbaceous(moisture, moisture_units)

    def set_moisture_live_woody(self, moisture, moisture_units):
        self.surface_fuel.set_moisture_live_woody(moisture, moisture_units)

    def set_moisture_live_aggregate(self, moisture, moisture_units):
        val = self.surface_fuel._convert_moisture(moisture, moisture_units)
        self.surface_fuel.state['_moisture_live_aggregate'] = val

    def set_moisture_input_mode(self, mode):
        """Set moisture input mode: 'BySizeClass', 'AllAggregate', 'MoistureScenario'."""
        self._moisture_input_mode = mode

    def set_current_moisture_scenario_by_name(self, name):
        """Set the current moisture scenario by name. Looks up from moisture scenarios."""
        self._current_scenario_name = name
        # Try to apply from known scenarios
        known = {
            'D1L1': {'moisture_1h': 0.03, 'moisture_10h': 0.04, 'moisture_100h': 0.05, 'moisture_live_herb': 0.30, 'moisture_live_woody': 0.60},
            'D1L2': {'moisture_1h': 0.03, 'moisture_10h': 0.04, 'moisture_100h': 0.05, 'moisture_live_herb': 0.60, 'moisture_live_woody': 0.90},
            'D1L3': {'moisture_1h': 0.03, 'moisture_10h': 0.04, 'moisture_100h': 0.05, 'moisture_live_herb': 0.90, 'moisture_live_woody': 1.20},
            'D1L4': {'moisture_1h': 0.03, 'moisture_10h': 0.04, 'moisture_100h': 0.05, 'moisture_live_herb': 1.20, 'moisture_live_woody': 1.50},
            'D2L1': {'moisture_1h': 0.06, 'moisture_10h': 0.07, 'moisture_100h': 0.08, 'moisture_live_herb': 0.30, 'moisture_live_woody': 0.60},
            'D2L2': {'moisture_1h': 0.06, 'moisture_10h': 0.07, 'moisture_100h': 0.08, 'moisture_live_herb': 0.60, 'moisture_live_woody': 0.90},
            'D2L3': {'moisture_1h': 0.06, 'moisture_10h': 0.07, 'moisture_100h': 0.08, 'moisture_live_herb': 0.90, 'moisture_live_woody': 1.20},
            'D2L4': {'moisture_1h': 0.06, 'moisture_10h': 0.07, 'moisture_100h': 0.08, 'moisture_live_herb': 1.20, 'moisture_live_woody': 1.50},
            'D3L1': {'moisture_1h': 0.09, 'moisture_10h': 0.10, 'moisture_100h': 0.11, 'moisture_live_herb': 0.30, 'moisture_live_woody': 0.60},
            'D3L2': {'moisture_1h': 0.09, 'moisture_10h': 0.10, 'moisture_100h': 0.11, 'moisture_live_herb': 0.60, 'moisture_live_woody': 0.90},
            'D3L3': {'moisture_1h': 0.09, 'moisture_10h': 0.10, 'moisture_100h': 0.11, 'moisture_live_herb': 0.90, 'moisture_live_woody': 1.20},
            'D3L4': {'moisture_1h': 0.09, 'moisture_10h': 0.10, 'moisture_100h': 0.11, 'moisture_live_herb': 1.20, 'moisture_live_woody': 1.50},
            'D4L1': {'moisture_1h': 0.12, 'moisture_10h': 0.13, 'moisture_100h': 0.14, 'moisture_live_herb': 0.30, 'moisture_live_woody': 0.60},
            'D4L2': {'moisture_1h': 0.12, 'moisture_10h': 0.13, 'moisture_100h': 0.14, 'moisture_live_herb': 0.60, 'moisture_live_woody': 0.90},
            'D4L3': {'moisture_1h': 0.12, 'moisture_10h': 0.13, 'moisture_100h': 0.14, 'moisture_live_herb': 0.90, 'moisture_live_woody': 1.20},
            'D4L4': {'moisture_1h': 0.12, 'moisture_10h': 0.13, 'moisture_100h': 0.14, 'moisture_live_herb': 1.20, 'moisture_live_woody': 1.50},
        }
        if name in known:
            s = known[name]
            for k, v in s.items():
                self.surface_fuel.state[k] = v

    def set_slope(self, slope, slope_units):
        self.surface_fuel.set_slope(slope, slope_units)

    def set_aspect(self, aspect):
        self.surface_fuel.set_aspect(aspect)

    def set_wind_speed(self, wind_speed, wind_speed_units, wind_height_mode=None):
        self.surface_fuel.set_wind_speed(wind_speed, wind_speed_units, wind_height_mode)

    def set_wind_direction(self, wind_direction):
        self.surface_fuel.set_wind_direction(wind_direction)

    def set_wind_height_input_mode(self, wind_height_mode):
        self.surface_fuel.set_wind_height_input_mode(wind_height_mode)

    def set_wind_and_spread_orientation_mode(self, orientation_mode):
        self.surface_fuel.set_wind_and_spread_orientation_mode(orientation_mode)

    def set_user_provided_wind_adjustment_factor(self, waf):
        self.surface_fuel.set_user_provided_wind_adjustment_factor(waf)

    def set_wind_adjustment_factor_calculation_method(self, method):
        self.surface_fuel.set_wind_adjustment_factor_calculation_method(method)

    # ========================================================================
    # GETTER DELEGATION
    # ========================================================================

    def get_fuel_model_number(self):
        return self.surface_fuel.get_fuel_model_number() if hasattr(self.surface_fuel, 'get_fuel_model_number') else self.surface_fuel.state.get('fuel_model_number')

    def get_canopy_cover(self, cover_units=None):
        return self.surface_fuel.get_canopy_cover(cover_units)

    def get_canopy_height(self, height_units='ft'):
        return self.surface_fuel.get_canopy_height(height_units)

    def get_crown_ratio(self, ratio_units=None):
        return self.surface_fuel.get_crown_ratio(ratio_units)

    def get_canopy_base_height(self, height_units='ft'):
        cbh = self.crown_inputs['canopy_base_height']
        if isinstance(height_units, str) and 'ft' in height_units.lower():
            return cbh
        return LengthUnits.fromBaseUnits(cbh, height_units) if not isinstance(height_units, str) else cbh

    def get_canopy_bulk_density(self, density_units=None):
        cbd = self.crown_inputs['canopy_bulk_density']
        if density_units == DensityUnits.DensityUnitsEnum.KilogramsPerCubicMeter:
            return cbd * 16.0185
        return cbd

    def get_moisture_foliar(self, moisture_units=None):
        if moisture_units == FractionUnits.FractionUnitsEnum.Fraction:
            return self.crown_inputs['moisture_foliar'] / 100.0
        return self.crown_inputs['moisture_foliar']

    def get_wind_speed(self, speed_units=None, wind_height_mode=None):
        return self.surface_fuel.state.get('wind_speed', 0.0)

    def get_wind_direction(self):
        return self.surface_fuel.get_wind_direction()

    def get_slope(self, slope_units=None):
        return self.surface_fuel.get_slope(slope_units)

    def get_aspect(self):
        return self.surface_fuel.get_aspect()


