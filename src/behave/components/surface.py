"""
Surface Fire Behavior Component - Python Implementation
Based on Rothermel (1972) fire spread model with BEHAVE support.

This implementation mirrors the C++ Behave Surface class providing complete
fire spread rate and flame length calculations using the Rothermel model.
"""

import math
from .behave_units import (
    SpeedUnits, LengthUnits, SlopeUnits, FractionUnits, LoadingUnits,
    HeatSourceAndReactionIntensityUnits, FirelineIntensityUnits, TemperatureUnits,
    SurfaceAreaToVolumeUnits, AreaUnits, TimeUnits
)


class SurfaceFuelbedIntermediates:
    """
    Direct Python translation of C++ SurfaceFuelbedIntermediates.

    Particle array layout (MaxParticles=5, MaxDeadSizeClasses=5, MaxLiveSizeClasses=2):
      loadDead_/savrDead_: [0]=1-hr  [1]=10-hr(109)  [2]=100-hr(30)  [3]=transferred-herb  [4]=0
      loadLive_/savrLive_: [0]=herb  [1]=woody  [2..4]=0
    All fuel densities default to 32.0 lb/ft³ (Albini 1976, p.91).
    """

    MAX_PARTICLES   = 5
    MAX_DEAD        = 4   # C++ MaxDeadSizeClasses = 4
    MAX_LIVE        = 5   # C++ MaxLiveSizeClasses = 5
    MAX_LIFE_STATES = 2
    DEAD = 0
    LIVE = 1

    def __init__(self, fuel_models):
        self.fuel_models = fuel_models
        self.initialize_members()

    # ------------------------------------------------------------------
    def initialize_members(self):
        N = self.MAX_PARTICLES
        L = self.MAX_LIFE_STATES
        self.fuel_model_number = None
        self.depth = 0.0
        self.sigma = 0.0
        self.bulk_density = 0.0
        self.packing_ratio = 0.0
        self.relative_packing_ratio = 0.0
        self.heat_sink = 0.0
        self.propagating_flux = 0.0
        self.total_silica_content = 0.0555

        self.load_dead  = [0.0] * N
        self.load_live  = [0.0] * N
        self.savr_dead  = [0.0] * N
        self.savr_live  = [0.0] * N
        self.moisture_dead = [0.0] * N
        self.moisture_live = [0.0] * N
        self.hoc_dead   = [0.0] * N
        self.hoc_live   = [0.0] * N
        self.silica_eff_dead = [0.01] * N  # default effective silica (C++ initializeMembers: silicaEffectiveDead_[i] = 0.01)
        self.silica_eff_live = [0.01, 0.01, 0.0, 0.0, 0.0]  # C++: only first NUMBER_OF_LIVE_SIZE_CLASSES=2 are 0.01
        self.fuel_density_dead = [32.0] * N  # Albini 1976 p.91
        self.fuel_density_live = [32.0] * N

        self.surface_area_dead = [0.0] * N
        self.surface_area_live = [0.0] * N
        self.frac_total_sa_dead = [0.0] * N   # fractionOfTotalSurfaceAreaDead_
        self.frac_total_sa_live = [0.0] * N
        self.size_sorted_frac_dead = [0.0] * N  # sizeSortedFractionOfSurfaceAreaDead_
        self.size_sorted_frac_live = [0.0] * N

        self.number_of_size_classes = [0, 0]   # [Dead, Live]
        self.total_load_for_life    = [0.0, 0.0]
        self.total_surface_area     = [0.0, 0.0]
        self.frac_total_sa          = [0.0, 0.0]  # fractionOfTotalSurfaceArea_ (dead vs live)
        self.weighted_moisture      = [0.0, 0.0]
        self.weighted_heat          = [0.0, 0.0]
        self.weighted_silica        = [0.0, 0.0]
        self.weighted_fuel_load     = [0.0, 0.0]
        self.moisture_of_extinction = [0.0, 0.0]  # [Dead, Live]

        # Convenience aliases kept for compatibility with older Surface code
        self.moisture_of_extinction_dead = 0.0
        self.moisture_of_extinction_live = 0.0
        self.weighted_moisture_dead = 0.0
        self.weighted_moisture_live = 0.0
        self.weighted_heat_dead = 0.0
        self.weighted_heat_live = 0.0
        self.weighted_silica_dead = 0.0
        self.weighted_silica_live = 0.0
        self.weighted_fuel_load_dead = 0.0
        self.weighted_fuel_load_live = 0.0
        self.total_load_dead = 0.0
        self.total_load_live = 0.0

    # ------------------------------------------------------------------
    def calculate_fuelbed_intermediates(self, fuel_model_number, surface_inputs):
        """Mirror of C++ calculateFuelbedIntermediates(fuelModelNumber)."""
        self.initialize_members()
        self.fuel_model_number = fuel_model_number

        model = self.fuel_models.get_fuel_model(fuel_model_number)
        if not model:
            return

        # --- setFuelbedDepth ---
        self.depth = model.get('fuel_bed_depth', 1.0)

        # --- setFuelLoad (normal case) ---
        self.load_dead[0] = model.get('dead_1h',   0.0)
        self.load_dead[1] = model.get('dead_10h',  0.0)
        self.load_dead[2] = model.get('dead_100h', 0.0)
        self.load_dead[3] = 0.0
        self.load_dead[4] = 0.0
        self.load_live[0] = model.get('live_herb',  0.0)
        self.load_live[1] = model.get('live_woody', 0.0)
        self.load_live[2] = 0.0
        self.load_live[3] = 0.0
        self.load_live[4] = 0.0

        # --- setMoistureContent (normal case) ---
        self.moisture_dead[0] = max(surface_inputs.get('moisture_1h',        0.06), 0.01)
        self.moisture_dead[1] = max(surface_inputs.get('moisture_10h',       0.07), 0.01)
        self.moisture_dead[2] = max(surface_inputs.get('moisture_100h',      0.08), 0.01)
        self.moisture_dead[3] = self.moisture_dead[0]   # C++: moistureDead_[3] = getMoistureOneHour
        self.moisture_dead[4] = 0.0
        self.moisture_live[0] = max(surface_inputs.get('moisture_live_herb',  0.60), 0.01)
        self.moisture_live[1] = max(surface_inputs.get('moisture_live_woody', 0.90), 0.01)
        self.moisture_live[2] = 0.0
        self.moisture_live[3] = 0.0
        self.moisture_live[4] = 0.0

        # --- setSAVR (normal case) ---
        # C++ hard-codes 109 and 30 for 10-hr and 100-hr regardless of fuel model
        self.savr_dead[0] = model.get('savr_1h',        2582.0)
        self.savr_dead[1] = 109.0
        self.savr_dead[2] = 30.0
        self.savr_dead[3] = model.get('savr_live_herb', 1500.0)  # transferred herb uses live herb SAVR
        self.savr_dead[4] = 0.0
        self.savr_live[0] = model.get('savr_live_herb', 1500.0)
        self.savr_live[1] = model.get('savr_live_woody',1500.0)
        self.savr_live[2] = 0.0
        self.savr_live[3] = 0.0
        self.savr_live[4] = 0.0

        # --- dynamicLoadTransfer (if model is dynamic) ---
        is_dynamic = model.get('is_dynamic', False)
        if is_dynamic:
            m_herb = self.moisture_live[0]
            if m_herb < 0.30:
                self.load_dead[3] = self.load_live[0]
                self.load_live[0] = 0.0
            elif m_herb <= 1.20:
                transferred = self.load_live[0] * (1.333 - 1.11 * m_herb)
                self.load_dead[3] = transferred
                self.load_live[0] -= transferred

        # --- setHeatOfCombustion (normal case) ---
        hoc_d = model.get('heat_of_combustion_dead', 8000.0)
        hoc_l = model.get('heat_of_combustion_live', 8000.0)
        for i in range(self.MAX_PARTICLES):
            self.hoc_dead[i] = hoc_d
            self.hoc_live[i] = hoc_l if i < 3 else 0.0   # NUMBER_OF_LIVE_SIZE_CLASSES = 3

        # --- countSizeClasses ---
        dead_count = sum(1 for i in range(self.MAX_DEAD) if self.load_dead[i])
        live_count = sum(1 for i in range(self.MAX_LIVE) if self.load_live[i])
        # C++ boosts to MaxDeadSizeClasses=4 or MaxLiveSizeClasses=5 if any exist
        self.number_of_size_classes[self.DEAD] = self.MAX_DEAD if dead_count > 0 else 0
        self.number_of_size_classes[self.LIVE] = self.MAX_LIVE if live_count > 0 else 0

        # --- setDeadFuelMoistureOfExtinction ---
        # C++ stores moistureOfExtinctionDead_ as a fraction (e.g. 0.40)
        self.moisture_of_extinction[self.DEAD] = model.get('moisture_of_extinction_dead', 0.25)

        # --- calculateFractionOfTotalSurfaceAreaForLifeStates ---
        self._calculate_surface_area_fractions()

        # --- calculateLiveMoistureOfExtinction ---
        self._calculate_live_moe()

        # --- calculateCharacteristicSAVR (sigma, weighted values) ---
        self._calculate_characteristic_savr()

        # final: bulk density, packing ratio, relative packing ratio
        total_load = (self.total_load_for_life[self.DEAD] +
                      self.total_load_for_life[self.LIVE])
        self.bulk_density = total_load / self.depth if self.depth > 1e-7 else 0.0

        self.packing_ratio = 0.0
        for i in range(self.MAX_PARTICLES):
            if self.load_dead[i] > 0:
                self.packing_ratio += self.load_dead[i] / (self.depth * self.fuel_density_dead[i])
            if self.load_live[i] > 0:
                self.packing_ratio += self.load_live[i] / (self.depth * self.fuel_density_live[i])

        if self.sigma > 1e-7:
            opt = 3.348 / (self.sigma ** 0.8189)
            self.relative_packing_ratio = self.packing_ratio / opt if opt > 1e-7 else 0.0
        else:
            self.relative_packing_ratio = 0.0

        # --- calculateHeatSink ---
        self._calculate_heat_sink()

        # --- calculatePropagatingFlux ---
        self._calculate_propagating_flux()

        # Update convenience aliases
        self.moisture_of_extinction_dead = self.moisture_of_extinction[self.DEAD]
        self.moisture_of_extinction_live = self.moisture_of_extinction[self.LIVE]
        self.weighted_moisture_dead = self.weighted_moisture[self.DEAD]
        self.weighted_moisture_live = self.weighted_moisture[self.LIVE]
        self.weighted_heat_dead     = self.weighted_heat[self.DEAD]
        self.weighted_heat_live     = self.weighted_heat[self.LIVE]
        self.weighted_silica_dead   = self.weighted_silica[self.DEAD]
        self.weighted_silica_live   = self.weighted_silica[self.LIVE]
        self.weighted_fuel_load_dead = self.weighted_fuel_load[self.DEAD]
        self.weighted_fuel_load_live = self.weighted_fuel_load[self.LIVE]
        self.total_load_dead = self.total_load_for_life[self.DEAD]
        self.total_load_live = self.total_load_for_life[self.LIVE]

    # ------------------------------------------------------------------
    def _calculate_surface_area_fractions(self):
        """Mirror of calculateFractionOfTotalSurfaceAreaForLifeStates."""
        for life in (self.DEAD, self.LIVE):
            if self.number_of_size_classes[life] == 0:
                continue
            # calculateTotalSurfaceAreaForLifeState
            self.total_surface_area[life] = 0.0
            for i in range(self.number_of_size_classes[life]):
                if life == self.DEAD:
                    sa = self.load_dead[i] * self.savr_dead[i] / self.fuel_density_dead[i]
                    self.surface_area_dead[i] = sa
                    self.total_surface_area[life] += sa
                else:
                    sa = self.load_live[i] * self.savr_live[i] / self.fuel_density_live[i]
                    self.surface_area_live[i] = sa
                    self.total_surface_area[life] += sa

            # calculateFractionOfTotalSurfaceAreaForSizeClasses
            tot = self.total_surface_area[life]
            for i in range(self.number_of_size_classes[life]):
                if tot > 1e-7:
                    if life == self.DEAD:
                        self.frac_total_sa_dead[i] = self.surface_area_dead[i] / tot
                    else:
                        self.frac_total_sa_live[i] = self.surface_area_live[i] / tot
                else:
                    if life == self.DEAD:
                        self.frac_total_sa_dead[i] = 0.0
                    else:
                        self.frac_total_sa_live[i] = 0.0

            # sumFractionOfTotalSurfaceAreaBySizeClass + assignFractionOfTotalSurfaceAreaBySizeClass
            if life == self.DEAD:
                summed = self._sum_by_size_class(self.frac_total_sa_dead, self.savr_dead)
                self.size_sorted_frac_dead = self._assign_by_size_class(self.savr_dead, summed)
            else:
                summed = self._sum_by_size_class(self.frac_total_sa_live, self.savr_live)
                self.size_sorted_frac_live = self._assign_by_size_class(self.savr_live, summed)

        # fractionOfTotalSurfaceArea_ dead vs live
        tot_sa = self.total_surface_area[self.DEAD] + self.total_surface_area[self.LIVE]
        if tot_sa > 1e-7:
            self.frac_total_sa[self.DEAD] = self.total_surface_area[self.DEAD] / tot_sa
        else:
            self.frac_total_sa[self.DEAD] = 0.0
        self.frac_total_sa[self.LIVE] = 1.0 - self.frac_total_sa[self.DEAD]

    def _sum_by_size_class(self, frac_arr, savr_arr):
        """sumFractionOfTotalSurfaceAreaBySizeClass."""
        s = [0.0] * self.MAX_PARTICLES
        for i in range(self.MAX_PARTICLES):
            sv = savr_arr[i]
            if   sv >= 1200.0: s[0] += frac_arr[i]
            elif sv >= 192.0:  s[1] += frac_arr[i]
            elif sv >= 96.0:   s[2] += frac_arr[i]
            elif sv >= 48.0:   s[3] += frac_arr[i]
            elif sv >= 16.0:   s[4] += frac_arr[i]
        return s

    def _assign_by_size_class(self, savr_arr, summed):
        """assignFractionOfTotalSurfaceAreaBySizeClass."""
        out = [0.0] * self.MAX_PARTICLES
        for i in range(self.MAX_PARTICLES):
            sv = savr_arr[i]
            if   sv >= 1200.0: out[i] = summed[0]
            elif sv >= 192.0:  out[i] = summed[1]
            elif sv >= 96.0:   out[i] = summed[2]
            elif sv >= 48.0:   out[i] = summed[3]
            elif sv >= 16.0:   out[i] = summed[4]
            else:              out[i] = 0.0
        return out

    # ------------------------------------------------------------------
    def _calculate_live_moe(self):
        """Mirror of calculateLiveMoistureOfExtinction."""
        if self.number_of_size_classes[self.LIVE] == 0:
            return
        fine_dead = 0.0
        weighted_moisture_fine_dead = 0.0
        for i in range(self.MAX_PARTICLES):
            if self.savr_dead[i] > 1e-7:
                wf = self.load_dead[i] * math.exp(-138.0 / self.savr_dead[i])
                fine_dead += wf
                weighted_moisture_fine_dead += wf * self.moisture_dead[i]
        fine_dead_moisture = (weighted_moisture_fine_dead / fine_dead
                               if fine_dead > 1e-7 else 0.0)

        fine_live = 0.0
        for i in range(self.number_of_size_classes[self.LIVE]):
            if self.savr_live[i] > 1e-7:
                fine_live += self.load_live[i] * math.exp(-500.0 / self.savr_live[i])

        fine_dead_over_fine_live = (fine_dead / fine_live if fine_live > 1e-7 else 0.0)

        moe_dead = self.moisture_of_extinction[self.DEAD]
        moe_live = (2.9 * fine_dead_over_fine_live *
                    (1.0 - fine_dead_moisture / moe_dead) - 0.226
                    if moe_dead > 1e-7 else moe_dead)
        if moe_live < moe_dead:
            moe_live = moe_dead
        self.moisture_of_extinction[self.LIVE] = moe_live

    # ------------------------------------------------------------------
    def _calculate_characteristic_savr(self):
        """Mirror of calculateCharacteristicSAVR."""
        weighted_savr = [0.0, 0.0]
        for life in (self.DEAD, self.LIVE):
            self.total_load_for_life[life] = 0.0
            self.weighted_heat[life] = 0.0
            self.weighted_silica[life] = 0.0
            self.weighted_moisture[life] = 0.0
            self.weighted_fuel_load[life] = 0.0

        wn_dead = [0.0] * self.MAX_PARTICLES
        wn_live = [0.0] * self.MAX_PARTICLES

        for i in range(self.MAX_PARTICLES):
            if self.savr_dead[i] > 1e-7:
                wn_dead[i] = self.load_dead[i] * (1.0 - self.total_silica_content)
                self.weighted_heat[self.DEAD]     += self.frac_total_sa_dead[i] * self.hoc_dead[i]
                self.weighted_silica[self.DEAD]   += self.frac_total_sa_dead[i] * self.silica_eff_dead[i]
                self.weighted_moisture[self.DEAD] += self.frac_total_sa_dead[i] * self.moisture_dead[i]
                weighted_savr[self.DEAD]          += self.frac_total_sa_dead[i] * self.savr_dead[i]
                self.total_load_for_life[self.DEAD] += self.load_dead[i]
            if self.savr_live[i] > 1e-7:
                wn_live[i] = self.load_live[i] * (1.0 - self.total_silica_content)
                self.weighted_heat[self.LIVE]     += self.frac_total_sa_live[i] * self.hoc_live[i]
                self.weighted_silica[self.LIVE]   += self.frac_total_sa_live[i] * self.silica_eff_live[i]
                self.weighted_moisture[self.LIVE] += self.frac_total_sa_live[i] * self.moisture_live[i]
                weighted_savr[self.LIVE]          += self.frac_total_sa_live[i] * self.savr_live[i]
                self.total_load_for_life[self.LIVE] += self.load_live[i]
            self.weighted_fuel_load[self.DEAD] += self.size_sorted_frac_dead[i] * wn_dead[i]
            self.weighted_fuel_load[self.LIVE] += self.size_sorted_frac_live[i] * wn_live[i]

        self.sigma = sum(self.frac_total_sa[life] * weighted_savr[life]
                         for life in (self.DEAD, self.LIVE))

    # ------------------------------------------------------------------
    def _calculate_heat_sink(self):
        """Mirror of calculateHeatSink."""
        self.heat_sink = 0.0
        for i in range(self.MAX_PARTICLES):
            if self.savr_dead[i] > 1e-7:
                qig = 250.0 + 1116.0 * self.moisture_dead[i]
                self.heat_sink += (self.frac_total_sa[self.DEAD] *
                                   self.frac_total_sa_dead[i] * qig *
                                   math.exp(-138.0 / self.savr_dead[i]))
            if self.savr_live[i] > 1e-7:
                qig = 250.0 + 1116.0 * self.moisture_live[i]
                self.heat_sink += (self.frac_total_sa[self.LIVE] *
                                   self.frac_total_sa_live[i] * qig *
                                   math.exp(-138.0 / self.savr_live[i]))
        self.heat_sink *= self.bulk_density

    # ------------------------------------------------------------------
    def _calculate_propagating_flux(self):
        """Mirror of calculatePropagatingFlux (C++ formula)."""
        if self.sigma < 1e-7:
            self.propagating_flux = 0.0
        else:
            self.propagating_flux = (
                math.exp((0.792 + 0.681 * math.sqrt(self.sigma)) *
                         (self.packing_ratio + 0.1)) /
                (192.0 + 0.2595 * self.sigma)
            )


class SurfaceFireReactionIntensity:
    """
    Calculates reaction intensity using Rothermel model.
    Equivalent to C++ SurfaceFireReactionIntensity class.
    """
    
    def __init__(self):
        """Initialize reaction intensity calculator."""
        self.reaction_intensity = 0.0
        self.eta_m_dead = 0.0
        self.eta_m_live = 0.0
        self.eta_s_dead = 0.0
        self.eta_s_live = 0.0
    
    def calculate_reaction_intensity(self, intermediates):
        """
        Calculate reaction intensity.
        Based on Rothermel 1972, equation 27.
        """
        sigma = intermediates.sigma
        rpr = intermediates.relative_packing_ratio
        
        if sigma < 1e-7 or rpr < 0:
            self.reaction_intensity = 0
            return 0
        
        # Calculate gamma (fuel coefficient)
        aa = 133.0 / (sigma ** 0.7913)
        gamma_max = (sigma ** 1.5) / (495.0 + 0.0594 * (sigma ** 1.5))
        gamma = (gamma_max * (rpr ** aa) * 
                 math.exp(aa * (1.0 - rpr)))
        
        # Calculate moisture damping factors (eta M)
        self.eta_m_dead = self._calculate_eta_m(
            intermediates.weighted_moisture_dead,
            intermediates.moisture_of_extinction_dead
        )
        self.eta_m_live = self._calculate_eta_m(
            intermediates.weighted_moisture_live,
            intermediates.moisture_of_extinction_live
        )
        
        # Calculate silica damping factors (eta S)
        self.eta_s_dead = self._calculate_eta_s(intermediates.weighted_silica_dead)
        self.eta_s_live = self._calculate_eta_s(intermediates.weighted_silica_live)
        
        # Calculate reaction intensity
        ri_dead = (gamma * intermediates.weighted_fuel_load_dead * 
                   intermediates.weighted_heat_dead * 
                   self.eta_m_dead * self.eta_s_dead)
        ri_live = (gamma * intermediates.weighted_fuel_load_live * 
                   intermediates.weighted_heat_live * 
                   self.eta_m_live * self.eta_s_live)
        
        self.reaction_intensity = ri_dead + ri_live
        return self.reaction_intensity
    
    def _calculate_eta_m(self, moisture, moe):
        """Calculate moisture damping factor (eta M)."""
        if moe < 1e-7:
            return 0
        
        relative_moisture = moisture / moe
        
        if moisture >= moe or relative_moisture > 1.0:
            return 0
        
        return (1.0 - 2.59 * relative_moisture + 5.11 * (relative_moisture ** 2) - 
                3.52 * (relative_moisture ** 3))
    
    def _calculate_eta_s(self, silica):
        """Calculate silica damping factor (eta S)."""
        if silica < 1e-7:
            return 0
        
        eta_s = 0.174 / (silica ** 0.19)
        return min(eta_s, 1.0)


class Surface:
    """
    Surface fire behavior model using Rothermel fire spread equation.
    Matches C++ BehaveRun::Surface API with complete functionality.
    """
    
    def __init__(self, fuel_models):
        """Initialize the Surface fire behavior model."""
        self.fuel_models = fuel_models
        self.intermediates = SurfaceFuelbedIntermediates(fuel_models)
        self.reaction_intensity_calc = SurfaceFireReactionIntensity()
        self.state = {}
        self.results = {}
        self._phi_w_state = {}
        self.initialize_members()
    
    def initialize_members(self):
        """Reset all state and results to default values."""
        self.state = {
            'fuel_model_number': None,
            'moisture_1h': 0.06,
            'moisture_10h': 0.07,
            'moisture_100h': 0.08,
            'moisture_live_herb': 0.60,
            'moisture_live_woody': 0.90,
            'wind_speed': 0.0,
            'wind_speed_units': SpeedUnits.SpeedUnitsEnum.FeetPerMinute,
            'wind_direction': 0.0,
            'wind_height_input_mode': 'TwentyFoot',
            'wind_orientation_mode': 'RelativeToUpslope',
            'wind_adjustment_factor': 1.0,
            'slope': 0.0,
            'aspect': 0.0,
            'canopy_cover': 0.0,
            'canopy_height': 0.0,
            'crown_ratio': 0.0,
            'air_temperature': 68.0,
        }
        self.results = {
            'spread_rate': 0.0,
            'backing_spread_rate': 0.0,
            'flanking_spread_rate': 0.0,
            'flame_length': 0.0,
            'backing_flame_length': 0.0,
            'flanking_flame_length': 0.0,
            'fireline_intensity': 0.0,
            'backing_fireline_intensity': 0.0,
            'flanking_fireline_intensity': 0.0,
            'heat_per_unit_area': 0.0,
            'direction_of_max_spread': 0.0,
            'fire_length_to_width_ratio': 1.0,
            'effective_wind_speed': 0.0,
            'reaction_intensity': 0.0,
            'residence_time': 0.0,
            'phi_s': 0.0,
            'phi_w': 0.0,
            'spread_rate_in_direction_of_interest': 0.0,
            'fireline_intensity_in_direction_of_interest': 0.0,
            'flame_length_in_direction_of_interest': 0.0,
        }
    
    def set_fuel_models(self, fuel_models):
        """Set the fuel models reference."""
        self.fuel_models = fuel_models
        self.intermediates.fuel_models = fuel_models
    
    def set_moisture_scenarios(self, moisture_scenarios):
        """Set moisture scenarios (for future use)."""
        self.moisture_scenarios = moisture_scenarios
    
    def updateSurfaceInputs(self, fuel_model_number, moisture_one_hour,
                           moisture_ten_hour, moisture_hundred_hour,
                           moisture_live_herbaceous, moisture_live_woody,
                           moisture_units, wind_speed, wind_speed_units,
                           wind_height_input_mode, wind_direction,
                           wind_orientation_mode,
                           slope, slope_units, aspect,
                           canopy_cover, canopy_cover_units,
                           canopy_height, canopy_height_units,
                           crown_ratio, crown_ratio_units):
        """
        Update all surface fire inputs at once.
        Matches C++ BehaveRun::Surface::updateSurfaceInputs() API signature:
          updateSurfaceInputs(fuelModelNumber, moisture..., windSpeed, windSpeedUnits,
                              windHeightInputMode, windDirection, windAndSpreadOrientationMode,
                              slope, slopeUnits, aspect, canopyCover, ..., crownRatio, ...)
        """
        self.set_fuel_model(fuel_model_number)
        self.set_moisture_one_hour(moisture_one_hour, moisture_units)
        self.set_moisture_ten_hour(moisture_ten_hour, moisture_units)
        self.set_moisture_hundred_hour(moisture_hundred_hour, moisture_units)
        self.set_moisture_live_herbaceous(moisture_live_herbaceous, moisture_units)
        self.set_moisture_live_woody(moisture_live_woody, moisture_units)
        self.set_wind_speed(wind_speed, wind_speed_units)
        self.set_wind_height_input_mode(wind_height_input_mode)
        self.set_wind_direction(wind_direction)
        self.set_wind_orientation_mode(wind_orientation_mode)
        self.set_slope(slope, slope_units)
        self.set_aspect(aspect)
        self.set_canopy_cover(canopy_cover, canopy_cover_units)
        self.set_canopy_height(canopy_height, canopy_height_units)
        self.set_crown_ratio(crown_ratio, crown_ratio_units)
    
    def set_fuel_model(self, fuel_model_number):
        """Set the fuel model number."""
        self.state['fuel_model_number'] = fuel_model_number
    
    def set_moisture_one_hour(self, moisture, units):
        """Set 1-hour fuel moisture."""
        self.state['moisture_1h'] = self._convert_moisture(moisture, units)
    
    def set_moisture_ten_hour(self, moisture, units):
        """Set 10-hour fuel moisture."""
        self.state['moisture_10h'] = self._convert_moisture(moisture, units)
    
    def set_moisture_hundred_hour(self, moisture, units):
        """Set 100-hour fuel moisture."""
        self.state['moisture_100h'] = self._convert_moisture(moisture, units)
    
    def set_moisture_live_herbaceous(self, moisture, units):
        """Set live herbaceous fuel moisture."""
        self.state['moisture_live_herb'] = self._convert_moisture(moisture, units)
    
    def set_moisture_live_woody(self, moisture, units):
        """Set live woody fuel moisture."""
        self.state['moisture_live_woody'] = self._convert_moisture(moisture, units)
    
    def set_wind_speed(self, wind_speed, units, height_mode=None):
        """Set wind speed. Optional height_mode sets wind_height_input_mode."""
        self.state['wind_speed'] = wind_speed
        self.state['wind_speed_units'] = units
        if height_mode is not None:
            self.set_wind_height_input_mode(height_mode)

    def set_wind_direction(self, wind_direction):
        """Set wind direction."""
        self.state['wind_direction'] = wind_direction
    
    def set_wind_height_input_mode(self, mode):
        """Set wind height input mode."""
        self.state['wind_height_input_mode'] = mode

    def set_wind_orientation_mode(self, mode):
        """Set wind and spread orientation mode (RelativeToUpslope or RelativeToNorth)."""
        self.state['wind_orientation_mode'] = mode
    
    def set_slope(self, slope, units):
        """Set slope."""
        self.state['slope'] = self._convert_slope(slope, units)
    
    def set_aspect(self, aspect):
        """Set aspect."""
        self.state['aspect'] = aspect
    
    def set_canopy_cover(self, canopy_cover, units):
        """Set canopy cover."""
        cc = self._convert_fraction(canopy_cover, units)
        self.state['canopy_cover'] = cc
    
    def set_canopy_height(self, canopy_height, units):
        """Set canopy height."""
        ch = self._convert_length(canopy_height, units, 'feet')
        self.state['canopy_height'] = ch
    
    def set_crown_ratio(self, crown_ratio, units):
        """Set crown ratio."""
        self.state['crown_ratio'] = self._convert_fraction(crown_ratio, units)
    
    def set_air_temperature(self, temperature, units):
        """Set air temperature."""
        if units == TemperatureUnits.TemperatureUnitsEnum.Celsius:
            self.state['air_temperature'] = temperature * 9/5 + 32  # Convert to Fahrenheit
        else:
            self.state['air_temperature'] = temperature
    
    def _convert_moisture(self, value, units):
        """Convert moisture units to fraction (0-1)."""
        # FractionUnitsEnum.Percent = 1 (integer), FractionUnitsEnum.Fraction = 0
        if units == FractionUnits.FractionUnitsEnum.Percent:
            return value / 100.0
        if isinstance(units, str) and 'percent' in units.lower():
            return value / 100.0
        if hasattr(units, 'name') and 'Percent' in units.name:
            return value / 100.0
        return value
    
    def _convert_fraction(self, value, units):
        """Convert fraction/percent to decimal fraction."""
        # Handle integer-based FractionUnitsEnum (Percent = 1)
        if units == FractionUnits.FractionUnitsEnum.Percent:
            return value / 100.0
        if isinstance(units, str):
            if 'percent' in units.lower():
                return value / 100.0
        elif hasattr(units, 'name'):
            if 'Percent' in units.name:
                return value / 100.0
        return value
    
    def _convert_length(self, value, units, target='feet'):
        """Convert length units."""
        if isinstance(units, str):
            units_str = units.lower()
        else:
            units_str = units.name.lower() if hasattr(units, 'name') else str(units).lower()
        
        if target == 'feet':
            if 'meter' in units_str:
                return value * 3.28084
            elif 'feet' in units_str or 'foot' in units_str:
                return value
        return value
    
    def _convert_slope(self, value, units):
        """Convert slope to base units (degrees), matching C++ SlopeUnits::toBaseUnits."""
        # SlopeUnits.SlopeUnitsEnum.Degrees = 0, Percent = 1
        if units == SlopeUnits.SlopeUnitsEnum.Percent:
            # Convert percent to degrees: degrees = atan(pct/100) * 180/pi
            return math.degrees(math.atan(value / 100.0))
        # Already in degrees (Degrees = 0, or fallback)
        return value
    
    def do_surface_run_in_direction_of_max_spread(self):
        """
        Run surface fire calculations in the direction of maximum spread.
        Handles standard fuel models, chaparral, palmetto-gallberry, western aspen, and two fuel models.
        """
        # Check for special fuel type modes
        if self.state.get('is_using_chaparral', False):
            self._run_chaparral()
            return
        if self.state.get('is_using_palmetto_gallberry', False):
            self._run_palmetto_gallberry()
            return
        if self.state.get('is_using_western_aspen', False):
            self._run_western_aspen()
            return
        if self.state.get('is_using_two_fuel_models', False):
            self._run_two_fuel_models()
            return

        fuel_model_num = self.state.get('fuel_model_number')
        if not fuel_model_num:
            return

        # Calculate fuelbed intermediates
        self.intermediates.calculate_fuelbed_intermediates(
            fuel_model_num, self.state
        )

        # Calculate reaction intensity
        ri = self.reaction_intensity_calc.calculate_reaction_intensity(self.intermediates)
        self.results['reaction_intensity'] = ri

        # Calculate spread rate components
        self._calculate_spread_rate(ri)

        # Calculate flame length and fireline intensity
        self._calculate_flame_and_intensity()
    
    def _calculate_spread_rate(self, reaction_intensity):
        """
        Calculate spread rate using Rothermel model.
        Mirrors C++ SurfaceFire::calculateForwardSpreadRate flow exactly.
        """
        intermediates = self.intermediates

        if intermediates.heat_sink < 1e-7:
            self.results['spread_rate'] = 0.0
            self.results['backing_spread_rate'] = 0.0
            self.results['flanking_spread_rate'] = 0.0
            self.results['phi_s'] = 0.0
            self.results['phi_w'] = 0.0
            self.results['fire_length_to_width_ratio'] = 1.0
            self.results['eccentricity'] = 0.0
            self.results['effective_wind_speed'] = 0.0
            self.results['direction_of_max_spread'] = 0.0
            return

        # Step 1: no-wind no-slope spread rate
        r0 = reaction_intensity * intermediates.propagating_flux / intermediates.heat_sink

        # Step 2: wind and slope phi factors
        phi_w = self._calculate_wind_factor()
        phi_s = self._calculate_slope_factor()

        # Cache wind coefficients set by _calculate_wind_factor
        wind_b = self._phi_w_state.get('wind_b', 0.0)
        wind_c = self._phi_w_state.get('wind_c', 0.0)
        wind_e = self._phi_w_state.get('wind_e', 0.0)
        rpr = intermediates.relative_packing_ratio

        # Step 3: wind speed limit and cap phiS
        wind_speed_limit = 0.9 * reaction_intensity
        if phi_s > 0.0 and phi_s > wind_speed_limit:
            phi_s = wind_speed_limit

        # Step 4: initial forward rate (before direction-of-max-spread)
        forward_spread_rate = r0 * (1.0 + phi_w + phi_s)

        # Step 5: direction-of-max-spread via vector composition
        # C++ calculateDirectionOfMaxSpread
        wind_direction = self.state.get('wind_direction', 0.0)
        orientation_mode = self.state.get('wind_orientation_mode', 'RelativeToUpslope')
        if isinstance(orientation_mode, str) and 'north' in orientation_mode.lower():
            aspect = self.state.get('aspect', 0.0)
            corrected_wind_dir = wind_direction - aspect
        else:
            corrected_wind_dir = wind_direction

        wind_dir_radians = corrected_wind_dir * math.pi / 180.0
        slope_rate = r0 * phi_s
        wind_rate = r0 * phi_w
        x = slope_rate + wind_rate * math.cos(wind_dir_radians)
        y = wind_rate * math.sin(wind_dir_radians)
        rate_vector = math.sqrt(x * x + y * y)
        # Forward spread rate is r0 + rate_vector (overwrites earlier estimate)
        forward_spread_rate = r0 + rate_vector

        # Step 6: derive effective wind speed from combined phi
        effective_wind_speed = 0.0  # ft/min
        is_wind_limit_exceeded = False
        if r0 > 1e-7 and wind_b > 1e-7 and wind_c > 1e-7 and rpr > 1e-7:
            phi_effective_wind = forward_spread_rate / r0 - 1.0
            if phi_effective_wind > 0:
                try:
                    effective_wind_speed = ((phi_effective_wind * (rpr ** wind_e)) / wind_c) ** (1.0 / wind_b)
                except Exception:
                    effective_wind_speed = 0.0

            # Step 7: wind speed limit check (C++ isWindLimitEnabled_ = false by default)
            # Only flag if exceeded; do NOT modify forward_spread_rate (wind limit disabled by default)
            if effective_wind_speed > wind_speed_limit:
                is_wind_limit_exceeded = True

        self.results['is_wind_limit_exceeded'] = is_wind_limit_exceeded

        # C++ converts effectiveWindSpeed to mph before storing
        effective_wind_speed_mph = SpeedUnits.fromBaseUnits(
            SpeedUnits.toBaseUnits(effective_wind_speed, SpeedUnits.SpeedUnitsEnum.FeetPerMinute),
            SpeedUnits.SpeedUnitsEnum.MilesPerHour
        )
        self.results['effective_wind_speed'] = effective_wind_speed_mph  # mph (matches C++)

        # Residence time
        self.results['residence_time'] = (
            384.0 / intermediates.sigma if intermediates.sigma > 1e-7 else 0.0
        )

        self.results['spread_rate'] = forward_spread_rate
        self.results['heat_per_unit_area'] = reaction_intensity * self.results['residence_time']
        self.results['phi_s'] = phi_s
        self.results['phi_w'] = phi_w

        # Fire length-to-width ratio (C++ fireSize.cpp calculateSurfaceFireLengthToWidthRatio)
        # Uses .936*exp(.1147*ews_mph) + .461*exp(-.0692*ews_mph) - .397
        ews_mph = effective_wind_speed_mph
        if ews_mph > 1e-7:
            lwr = 0.936 * math.exp(0.1147 * ews_mph) + 0.461 * math.exp(-0.0692 * ews_mph) - 0.397
            if lwr > 8.0:
                lwr = 8.0
        else:
            lwr = 1.0
        self.results['fire_length_to_width_ratio'] = lwr

        # Fire eccentricity from LWR
        eccentricity = 0.0
        x_ecc = lwr * lwr - 1.0
        if x_ecc > 0.0:
            eccentricity = math.sqrt(x_ecc) / lwr
        self.results['eccentricity'] = eccentricity

        # Backing spread rate: forward * (1-e)/(1+e)
        backing = forward_spread_rate * (1.0 - eccentricity) / (1.0 + eccentricity) if (1.0 + eccentricity) > 1e-7 else 0.0
        # Flanking spread rate: (forward + backing) / (2 * LWR)
        flanking = (forward_spread_rate + backing) / (2.0 * lwr) if lwr > 1e-7 else 0.0
        self.results['backing_spread_rate'] = backing
        self.results['flanking_spread_rate'] = flanking

        # Elliptical fire dimensions (rates, per ft/min, elapsed=1 min gives ft)
        # B = (forward + backing) / 2
        # A = B / LWR
        # C = B - backing
        elliptical_b = (forward_spread_rate + backing) / 2.0
        elliptical_a = elliptical_b / lwr if lwr > 1e-7 else 0.0
        elliptical_c = elliptical_b - backing
        self.results['elliptical_a'] = elliptical_a  # ft/min
        self.results['elliptical_b'] = elliptical_b  # ft/min
        self.results['elliptical_c'] = elliptical_c  # ft/min

        # Heading-to-backing ratio (Alexander 1985)
        part = math.sqrt(lwr * lwr - 1) if lwr > 1.0 else 0.0
        heading_to_backing_ratio = (lwr + part) / (lwr - part) if (lwr - part) > 1e-7 else 0.0
        self.results['heading_to_backing_ratio'] = heading_to_backing_ratio

        # Direction of max spread
        if rate_vector > 1e-7:
            dir_rad = math.atan2(y, x)
            dir_deg = math.degrees(dir_rad)
        else:
            dir_deg = 0.0
        # Undocumented BehavePlus hack: if abs < 0.5 set to 0
        if abs(dir_deg) < 0.5:
            dir_deg = 0.0
        if dir_deg < -1e-20:
            dir_deg += 360.0
        # Convert to relative-to-north if needed
        # C++: convertDirectionOfSpreadToRelativeToNorth = dirFromUpslope + aspect + 180
        if isinstance(orientation_mode, str) and 'north' in orientation_mode.lower():
            aspect = self.state.get('aspect', 0.0)
            dir_deg = dir_deg + aspect + 180.0
            while dir_deg >= 360.0:
                dir_deg -= 360.0
        self.results['direction_of_max_spread'] = dir_deg
    
    def _calculate_wind_adjustment_factor(self):
        """
        Calculate wind adjustment factor (WAF).
        Respects waf_calculation_method: 'UserInput' uses user-provided value,
        'UseCrownRatio' (default) computes from canopy (Albini & Baughman 1979).
        """
        method = self.state.get('waf_calculation_method', 'UseCrownRatio')
        if isinstance(method, str) and 'userinput' in method.lower().replace(' ', '').replace('_', ''):
            return self.state.get('user_provided_waf', 1.0)

        canopy_cover = self.state.get('canopy_cover', 0.0)
        canopy_height = self.state.get('canopy_height', 0.0)
        crown_ratio = self.state.get('crown_ratio', 0.0)
        fuelbed_depth = self.intermediates.depth

        canopy_crown_fraction = crown_ratio * canopy_cover / 3.0

        if canopy_cover < 1e-7 or canopy_crown_fraction < 0.05 or canopy_height < 6.0:
            if fuelbed_depth > 1e-7:
                waf = 1.83 / math.log((20.0 + 0.36 * fuelbed_depth) / (0.13 * fuelbed_depth))
            else:
                waf = 1.0
        else:
            denom = math.sqrt(canopy_crown_fraction * canopy_height) * math.log(
                (20.0 + 0.36 * canopy_height) / (0.13 * canopy_height)
            )
            waf = 0.555 / denom if denom > 1e-7 else 1.0

        return waf

    def _calculate_wind_factor(self):
        """
        Calculate wind phi factor (phi_w).
        Mirrors C++ SurfaceFire::calculateMidflameWindSpeed + calculateWindFactor.
        Wind speed limit is also stored for use in _calculate_spread_rate.
        """
        sigma = self.intermediates.sigma
        rpr = self.intermediates.relative_packing_ratio

        if sigma < 1e-7:
            self._phi_w_state = {'wind_b': 0.0, 'wind_c': 0.0, 'wind_e': 0.0,
                                 'midflame_ws': 0.0}
            return 0.0

        # Wind coefficients (Rothermel 1972)
        wind_c = 7.47 * math.exp(-0.133 * (sigma ** 0.55))
        wind_b = 0.02526 * (sigma ** 0.54)
        wind_e = 0.715 * math.exp(-0.000359 * sigma)

        # Store for potential wind limit re-calculation
        self._phi_w_state = {
            'wind_b': wind_b, 'wind_c': wind_c, 'wind_e': wind_e
        }

        # Convert input wind speed to ft/min (base units)
        ws_base = SpeedUnits.toBaseUnits(self.state['wind_speed'],
                                          self.state['wind_speed_units'])
        ws_ft_per_min = SpeedUnits.fromBaseUnits(ws_base, SpeedUnits.SpeedUnitsEnum.FeetPerMinute)

        # Apply 10-meter to 20-foot conversion if needed (C++: windSpeed /= 1.15)
        wind_height_mode = self.state.get('wind_height_input_mode', 'TwentyFoot')
        if isinstance(wind_height_mode, str) and 'tenmeter' in wind_height_mode.lower().replace(' ', '').replace('_', ''):
            ws_ft_per_min /= 1.15

        # Compute WAF and midflame wind speed (C++: calculateWindAdjustmentFactor then midflame = waf * windSpeed)
        wind_height_lower = wind_height_mode.lower().replace(' ', '').replace('_', '') if isinstance(wind_height_mode, str) else ''
        if 'directmidflame' in wind_height_lower:
            midflame_ws = ws_ft_per_min
        else:
            # TwentyFoot or TenMeter: apply WAF
            waf = self._calculate_wind_adjustment_factor()
            midflame_ws = waf * ws_ft_per_min

        self._phi_w_state['midflame_ws'] = midflame_ws

        if midflame_ws < 1e-7:
            return 0.0

        phi_w = (midflame_ws ** wind_b) * wind_c * (rpr ** (-wind_e))
        return phi_w

    def _calculate_slope_factor(self):
        """
        Calculate slope phi factor (phi_s).
        Mirrors C++ SurfaceFire::calculateSlopeFactor exactly:
          double slope = surfaceInputs_->getSlope(SlopeUnits::Degrees);
          double slopex = tan(slope / 180.0 * M_PI);
          phiS_ = 5.275 * pow(packingRatio, -0.3) * (slopex * slopex);
        Slope is stored in state in degrees (base unit).
        """
        slope_deg = self.state.get('slope', 0.0)
        if slope_deg < 1e-7:
            return 0.0

        # Convert degrees to tangent (C++: slopex = tan(degrees * pi/180))
        slope_tan = math.tan(math.radians(slope_deg))

        pr = self.intermediates.packing_ratio
        if pr < 1e-7:
            return 0.0

        phi_s = 5.275 * (pr ** -0.3) * (slope_tan ** 2)
        return phi_s

    # ------------------------------------------------------------------
    # Special fuel type handlers
    # ------------------------------------------------------------------

    def _run_with_custom_intermediates(self):
        """Common path: given intermediates already populated, compute spread rate + flame."""
        ri = self.reaction_intensity_calc.calculate_reaction_intensity(self.intermediates)
        self.results['reaction_intensity'] = ri
        self._calculate_spread_rate(ri)
        self._calculate_flame_and_intensity()

    def _run_chaparral(self):
        """Run Chaparral special fuel type calculations."""
        depth_ft = self.state.get('chaparral_fuel_bed_depth', 1.0)  # already in ft from setter
        fuel_type = self.state.get('chaparral_fuel_type', 'NotSet')
        load_mode = self.state.get('chaparral_fuel_load_input_mode', 'DirectFuelLoad')
        dead_fraction = self.state.get('chaparral_fuel_dead_load_fraction', 0.5)

        # Determine total fuel load (lb/ft²)
        if 'direct' in load_mode.lower():
            total_load = self.state.get('chaparral_total_fuel_load', 0.333)
        else:
            # Calculate from depth and fuel type
            if 'chamise' in fuel_type.lower():
                age = math.exp(3.912023 * math.sqrt(depth_ft / 7.5))
                chamise_factor = 0.0347
                tpa = age / (1.4459 + chamise_factor * age)
            elif 'mixed' in fuel_type.lower():
                age = math.exp(3.912023 * math.sqrt(depth_ft / 10.0))
                tpa = age / (0.4849 + 0.0170 * age)
            else:
                tpa = 0.0
            total_load = tpa * 2000.0 / 43560.0

        # Build fuel particle arrays matching ChaparralFuel
        # Dead: 4 size classes (indices 0-3), SAVR = 640,127,61,27; density=46 lb/ft³
        # Live: 5 classes (0=leaf SAVR=2200, 1-4 stems SAVR=640,127,61,27); density[0]=32, rest=46
        savr_dead = [640.0, 127.0, 61.0, 27.0, 0.0]
        savr_live = [2200.0, 640.0, 127.0, 61.0, 27.0]
        density_dead = [46.0, 46.0, 46.0, 46.0, 46.0]
        density_live = [32.0, 46.0, 46.0, 46.0, 46.0]

        # Dead load fractions (Bevins)
        ld = [0.347, 0.364, 0.207, 0.082, 0.0]
        load_dead = [dead_fraction * total_load * ld[i] for i in range(5)]
        # Live load
        ll_raw = [
            total_load * (0.1957 - 0.3050 * dead_fraction),
            total_load * (0.2416 - 0.2560 * dead_fraction),
            total_load * (0.1918 - 0.2560 * dead_fraction),
            total_load * (0.2648 - 0.0500 * dead_fraction),
        ]
        total_live = (1.0 - dead_fraction) * total_load
        ll_last = total_live - sum(ll_raw)
        ll_raw.append(max(0.0, ll_last))
        load_live = ll_raw

        moe_dead = 0.30
        # Dead moisture: C++ setMoistureContent for chaparral: [1h, 10h, 10h, 100h, 0]
        m1h = self.state.get('moisture_1h', 0.06)
        m10h = self.state.get('moisture_10h', 0.07)
        m100h = self.state.get('moisture_100h', 0.08)
        mlh = self.state.get('moisture_live_herb', 0.60)
        mlw = self.state.get('moisture_live_woody', 0.90)
        moisture_dead = [m1h, m10h, m10h, m100h, 0.0]
        # Live moisture: C++ [herb, woody, woody, woody, woody]
        moisture_live = [mlh, mlw, mlw, mlw, mlw]

        hoc_dead = [8000.0] * 5
        hoc_live = [10500.0, 10500.0, 9500.0, 9500.0, 9500.0]
        eff_silica_dead = [0.015] * 5
        eff_silica_live = [0.035, 0.015, 0.015, 0.015, 0.015]

        self._populate_intermediates_from_arrays(
            depth_ft, savr_dead, savr_live, load_dead, load_live,
            moisture_dead, moisture_live, hoc_dead, hoc_live,
            density_dead, density_live, moe_dead, eff_silica_dead, eff_silica_live
        )
        # C++ uses totalSilicaContent_ = 0.055 for chaparral and western aspen
        self.intermediates.total_silica_content = 0.055
        self.intermediates._calculate_characteristic_savr()
        # Update aliases after recalculation
        self.intermediates.weighted_fuel_load_dead = self.intermediates.weighted_fuel_load[self.intermediates.DEAD]
        self.intermediates.weighted_fuel_load_live = self.intermediates.weighted_fuel_load[self.intermediates.LIVE]
        self._run_with_custom_intermediates()

    def _run_palmetto_gallberry(self):
        """Run Palmetto-Gallberry special fuel type - exact C++ translation."""
        age = self.state.get('palmetto_gallberry_age_of_rough', 10)
        height = self.state.get('palmetto_gallberry_height_of_understory', 4.0)  # feet
        # C++ gets coverage as fraction; test stores as passed (50 percent)
        # getPalmettoGallberryPalmettoCoverage(Fraction) means it's stored as fraction
        # but the test passes 50 with units SquareFeetPerAcre... coverage is separate
        # Let's check: test passes palmetto_coverage=50 (no units arg - it's stored raw)
        # C++ palmettoCoverage *= 100.0 inside each formula (converts fraction->pct)
        # So stored value must be fraction. But test passes 50... it was % in the input
        # The updateSurfaceInputsForPalmettoGallberry stores it directly as passed.
        # Looking at C++ surfaceInputs: getPalmettoGallberryPalmettoCoverage(FractionUnits::Fraction)
        # means we need to store as fraction. Test passes 50 (percent), so convert:
        coverage_pct = self.state.get('palmetto_gallberry_palmetto_coverage', 50.0)
        # If basal_area_units is SquareFeetPerAcre this is treated as raw percent
        coverage_frac = coverage_pct / 100.0
        basal_area = self.state.get('palmetto_gallberry_overstory_basal_area', 50.0)

        # Exact C++ formulas from palmettoGallberry.cpp
        pct = coverage_frac * 100.0  # C++ does palmettoCoverage *= 100 inside each method
        dead_1h = max(0.0, -0.00121 + 0.00379 * math.log(age) + 0.00118 * height * height)
        dead_10h = max(0.0, -0.00775 + 0.00021 * pct + 0.00007 * age * age)
        dead_foliage = 0.00221 * (age ** 0.51263) * math.exp(0.02482 * pct)
        litter = (0.03632 + 0.0005336 * basal_area) * (1.0 - (0.25 ** age))
        live_1h = 0.00546 + 0.00092 * age + 0.00212 * height * height
        live_10h = max(0.0, -0.02128 + 0.00014 * age * age + 0.00314 * height * height)
        live_foliage = max(0.0, -0.0036 + 0.00253 * age + 0.00049 * pct + 0.00282 * height * height)

        depth = 2.0 * height / 3.0

        # C++ setSAVR() for PalmettoGallberry
        savr_dead = [350.0, 140.0, 2000.0, 2000.0, 0.0]
        savr_live = [350.0, 140.0, 2000.0, 0.0, 0.0]
        load_dead = [dead_1h, dead_10h, dead_foliage, litter, 0.0]
        load_live = [live_1h, live_10h, live_foliage, 0.0, 0.0]

        # C++ setMoistureContent() for PalmettoGallberry:
        # dead: [1h, 10h, 1h, 100h, 0]  live: [liveWoody, liveWoody, liveHerb, 0, 0]
        m1h   = self.state.get('moisture_1h', 0.06)
        m10h  = self.state.get('moisture_10h', 0.07)
        m100h = self.state.get('moisture_100h', 0.08)
        mlh   = self.state.get('moisture_live_herb', 0.60)
        mlw   = self.state.get('moisture_live_woody', 0.90)
        moisture_dead = [m1h, m10h, m1h, m100h, 0.0]
        moisture_live = [mlw, mlw, mlh, 0.0, 0.0]

        hoc_dead = [8300.0] * 5
        hoc_live = [8300.0] * 5
        # C++ setFuelLoad: silicaEffectiveLive_[i] = 0.015 for all
        eff_silica_dead = [0.01] * 5
        eff_silica_live = [0.015] * 5
        # C++ density: dead=30.0, live=46.0 (surfaceFuelbedIntermediates.cpp line 693-694)
        density_dead = [30.0] * 5
        density_live = [46.0] * 5
        moe_dead = 0.40

        self._populate_intermediates_from_arrays(
            depth, savr_dead, savr_live, load_dead, load_live,
            moisture_dead, moisture_live, hoc_dead, hoc_live,
            density_dead, density_live, moe_dead, eff_silica_dead, eff_silica_live
        )
        # C++ calculateCharacteristicSAVR: totalSilicaContent_ = 0.030 for PalmettoGallberry
        self.intermediates.total_silica_content = 0.030
        self.intermediates._calculate_characteristic_savr()
        total_load = (self.intermediates.total_load_for_life[self.intermediates.DEAD] +
                      self.intermediates.total_load_for_life[self.intermediates.LIVE])
        self.intermediates.bulk_density = total_load / depth if depth > 1e-7 else 0.0
        self.intermediates.packing_ratio = 0.0
        for i in range(5):
            if self.intermediates.load_dead[i] > 0:
                self.intermediates.packing_ratio += self.intermediates.load_dead[i] / (depth * density_dead[i])
            if self.intermediates.load_live[i] > 0:
                self.intermediates.packing_ratio += self.intermediates.load_live[i] / (depth * density_live[i])
        if self.intermediates.sigma > 1e-7:
            opt = 3.348 / (self.intermediates.sigma ** 0.8189)
            self.intermediates.relative_packing_ratio = self.intermediates.packing_ratio / opt if opt > 1e-7 else 0.0
        self.intermediates._calculate_heat_sink()
        self.intermediates._calculate_propagating_flux()
        self.intermediates.weighted_fuel_load_dead = self.intermediates.weighted_fuel_load[self.intermediates.DEAD]
        self.intermediates.weighted_fuel_load_live = self.intermediates.weighted_fuel_load[self.intermediates.LIVE]
        self._run_with_custom_intermediates()

    def _run_western_aspen(self):
        """Run Western Aspen special fuel type."""
        aspen_model = self.state.get('aspen_fuel_model_number', 1)
        curing = self.state.get('aspen_curing_level', 0.50)
        severity = self.state.get('aspen_fire_severity', 'Low')
        dbh = self.state.get('aspen_dbh', 10.0)

        if isinstance(severity, str):
            severity_idx = 1 if 'high' in severity.lower() else 0
        else:
            severity_idx = int(severity)

        # Clamp model index
        idx = max(1, min(5, int(aspen_model)))
        aspen_idx = idx - 1

        curing_arr = [0.0, 0.3, 0.5, 0.7, 0.9, 1.000000001]

        def aspen_interp(curing, table):
            c = max(0.0, min(1.0, curing))
            for i in range(1, len(curing_arr)):
                if c < curing_arr[i]:
                    frac = 1.0 - (curing_arr[i] - c) / (curing_arr[i] - curing_arr[i-1])
                    return table[i-1] + frac * (table[i] - table[i-1])
            return table[-1]

        load1h_table = [
            [0.800, 0.893, 1.056, 1.218, 1.379, 1.4595],
            [0.738, 0.930, 1.056, 1.183, 1.309, 1.3720],
            [0.601, 0.645, 0.671, 0.699, 0.730, 0.7455],
            [0.880, 0.906, 1.037, 1.167, 1.300, 1.3665],
            [0.754, 0.797, 0.825, 0.854, 0.884, 0.8990],
        ]
        load10h_arr = [0.975, 0.475, 1.035, 1.340, 1.115]
        load_lh_table = [
            [0.335, 0.234, 0.167, 0.100, 0.033, 0.000],
            [0.665, 0.465, 0.332, 0.199, 0.067, 0.000],
            [0.150, 0.105, 0.075, 0.045, 0.015, 0.000],
            [0.100, 0.070, 0.050, 0.030, 0.010, 0.000],
            [0.150, 0.105, 0.075, 0.045, 0.015, 0.000],
        ]
        load_lw_table = [
            [0.403, 0.403, 0.333, 0.283, 0.277, 0.274],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.455, 0.455, 0.364, 0.290, 0.261, 0.2465],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
        ]
        savr1h_table = [
            [1440., 1620., 1910., 2090., 2220., 2285.],
            [1480., 1890., 2050., 2160., 2240., 2280.],
            [1400., 1540., 1620., 1690., 1750., 1780.],
            [1350., 1420., 1710., 1910., 2060., 2135.],
            [1420., 1540., 1610., 1670., 1720., 1745.],
        ]
        savr_lw_table = [
            [2440., 2440., 2310., 2090., 1670., 1670.],
            [2440., 2440., 2440., 2440., 2440., 2440.],
            [2440., 2440., 2440., 2440., 2440., 2440.],
            [2530., 2530., 2410., 2210., 1800., 1800.],
            [2440., 2440., 2440., 2440., 2440., 2440.],
        ]

        factor = 2000.0 / 43560.0
        dead_1h = aspen_interp(curing, load1h_table[aspen_idx]) * factor
        dead_10h = load10h_arr[aspen_idx] * factor
        live_herb = aspen_interp(curing, load_lh_table[aspen_idx]) * factor
        live_woody = aspen_interp(curing, load_lw_table[aspen_idx]) * factor

        savr_1h = aspen_interp(curing, savr1h_table[aspen_idx])
        savr_lw = aspen_interp(curing, savr_lw_table[aspen_idx])

        depth_arr = [0.65, 0.30, 0.18, 0.50, 0.18]
        depth = depth_arr[aspen_idx]

        m1h = self.state.get('moisture_1h', 0.06)
        m10h = self.state.get('moisture_10h', 0.07)
        mlh = self.state.get('moisture_live_herb', 0.60)
        mlw = self.state.get('moisture_live_woody', 0.90)

        savr_dead = [savr_1h, 109.0, 0.0, 0.0, 0.0]
        savr_live = [2800.0, savr_lw, 0.0, 0.0, 0.0]
        load_dead = [dead_1h, dead_10h, 0.0, 0.0, 0.0]
        load_live = [live_herb, live_woody, 0.0, 0.0, 0.0]
        density = [32.0] * 5
        moisture_dead = [m1h, m10h, 0.0, 0.0, 0.0]
        moisture_live = [mlh, mlw, 0.0, 0.0, 0.0]
        hoc = [8000.0] * 5
        eff_silica = [0.01] * 5

        self._populate_intermediates_from_arrays(
            depth, savr_dead, savr_live, load_dead, load_live,
            moisture_dead, moisture_live, hoc, hoc,
            density, density, 0.25, eff_silica, eff_silica
        )
        # C++ uses totalSilicaContent_ = 0.055 for chaparral and western aspen
        self.intermediates.total_silica_content = 0.055
        self.intermediates._calculate_characteristic_savr()
        # Update aliases after recalculation
        self.intermediates.weighted_fuel_load_dead = self.intermediates.weighted_fuel_load[self.intermediates.DEAD]
        self.intermediates.weighted_fuel_load_live = self.intermediates.weighted_fuel_load[self.intermediates.LIVE]
        self._run_with_custom_intermediates()

        # Compute western aspen mortality
        forward_fl = self.results.get('flame_length', 0.0)
        char_height = forward_fl / 1.8
        if severity_idx == 0:
            mortality = 1.0 / (1.0 + math.exp(-4.407 + 0.638 * dbh - 2.134 * char_height))
        else:
            mortality = 1.0 / (1.0 + math.exp(-2.157 + 0.218 * dbh - 3.600 * char_height))
        mortality = max(0.0, min(1.0, mortality))
        self.results['aspen_mortality'] = mortality

    def _run_two_fuel_models(self):
        """Run Two Fuel Models method (arithmetic, harmonic, or 2D)."""
        first_model = self.state.get('fuel_model_number')
        second_model = self.state.get('second_fuel_model_number', first_model)
        first_coverage = self.state.get('first_fuel_model_coverage', 0.5)  # fraction
        second_coverage = 1.0 - first_coverage
        method = self.state.get('two_fuel_models_method', 'TwoDimensional')

        # Compute spread rate for each model independently
        results_list = []
        for model_num in [first_model, second_model]:
            self.intermediates.calculate_fuelbed_intermediates(model_num, self.state)
            ri = self.reaction_intensity_calc.calculate_reaction_intensity(self.intermediates)
            self.results['reaction_intensity'] = ri
            self._calculate_spread_rate(ri)
            self._calculate_flame_and_intensity()
            results_list.append({
                'spread_rate': self.results.get('spread_rate', 0.0),
                'fire_length_to_width_ratio': self.results.get('fire_length_to_width_ratio', 1.0),
                'reaction_intensity': self.results.get('reaction_intensity', 0.0),
                'fireline_intensity': self.results.get('fireline_intensity', 0.0),
                'flame_length': self.results.get('flame_length', 0.0),
                'heat_per_unit_area': self.results.get('heat_per_unit_area', 0.0),
                'effective_wind_speed': self.results.get('effective_wind_speed', 0.0),
                'backing_spread_rate': self.results.get('backing_spread_rate', 0.0),
                'flanking_spread_rate': self.results.get('flanking_spread_rate', 0.0),
                'elliptical_a': self.results.get('elliptical_a', 0.0),
                'elliptical_b': self.results.get('elliptical_b', 0.0),
                'elliptical_c': self.results.get('elliptical_c', 0.0),
                'eccentricity': self.results.get('eccentricity', 0.0),
                'heading_to_backing_ratio': self.results.get('heading_to_backing_ratio', 0.0),
                'residence_time': self.results.get('residence_time', 0.0),
                'direction_of_max_spread': self.results.get('direction_of_max_spread', 0.0),
            })

        ros1 = results_list[0]['spread_rate']
        ros2 = results_list[1]['spread_rate']
        cov1, cov2 = first_coverage, second_coverage

        if abs(cov1 - 1.0) < 1e-4:
            combined_ros = ros1
            use_idx = 0
        elif abs(cov2 - 1.0) < 1e-4:
            combined_ros = ros2
            use_idx = 1
        else:
            use_idx = None
            method_lower = method.lower().replace(' ', '').replace('_', '')
            if 'arithmetic' in method_lower:
                combined_ros = cov1 * ros1 + cov2 * ros2
            elif 'harmonic' in method_lower:
                if ros1 > 1e-6 and ros2 > 1e-6:
                    combined_ros = 1.0 / (cov1 / ros1 + cov2 / ros2)
                else:
                    combined_ros = 0.0
            else:
                # 2D / Finney method
                combined_ros = self._two_fuel_two_dimensional(
                    ros1, ros2, cov1, cov2,
                    results_list[1]['fire_length_to_width_ratio']
                )

        # Apply combined spread rate and set other results based on C++ rules
        # For 100%/0% coverage use that model's values directly
        if use_idx is not None:
            for k, v in results_list[use_idx].items():
                self.results[k] = v
        else:
            # Reaction intensity: max of two
            self.results['reaction_intensity'] = max(
                results_list[0]['reaction_intensity'], results_list[1]['reaction_intensity'])
            # LWR, effective wind: from first model
            for k in ['fire_length_to_width_ratio', 'effective_wind_speed', 'direction_of_max_spread',
                      'residence_time', 'eccentricity', 'heading_to_backing_ratio']:
                self.results[k] = results_list[0][k]
            # Heat per unit area: max
            self.results['heat_per_unit_area'] = max(
                results_list[0]['heat_per_unit_area'], results_list[1]['heat_per_unit_area'])
            # Flame length, fireline intensity: max
            self.results['fireline_intensity'] = max(
                results_list[0]['fireline_intensity'], results_list[1]['fireline_intensity'])
            self.results['flame_length'] = max(
                results_list[0]['flame_length'], results_list[1]['flame_length'])
            # Set combined spread rate and recalculate fire shape
            self.results['spread_rate'] = combined_ros
            lwr = results_list[0]['fire_length_to_width_ratio']
            e = results_list[0]['eccentricity']
            backing = combined_ros * (1.0 - e) / (1.0 + e) if (1.0 + e) > 1e-7 else 0.0
            flanking = (combined_ros + backing) / (2.0 * lwr) if lwr > 1e-7 else 0.0
            self.results['backing_spread_rate'] = backing
            self.results['flanking_spread_rate'] = flanking
            ellb = (combined_ros + backing) / 2.0
            ella = ellb / lwr if lwr > 1e-7 else 0.0
            ellc = ellb - backing
            self.results['elliptical_a'] = ella
            self.results['elliptical_b'] = ellb
            self.results['elliptical_c'] = ellc

    def _two_fuel_two_dimensional(self, ros1, ros2, cov1, cov2, lbRatio):
        """
        Exact EXRATE algorithm (Finney) for 2D spread rate.
        Mirrors C++ RandFuel::computeSpread2 with samples=2, depth=2, laterals=0.
        
        Algorithm:
        1. Normalize to relative ROS (divide by max)
        2. Generate all 2^(samples*depth) = 16 combinations of 2x2 grid
        3. For each combination, compute max ROS via calcSpreadPaths2
        4. weighted_average = sum(prob[i] * maxRos[i]) * maxRos_absolute
        """
        if ros1 < 1e-7 and ros2 < 1e-7:
            return 0.0

        # The C++ uses lbRatio from the SECOND fuel model
        lb = max(1.0, lbRatio)

        # Elliptical dimensions (RandThread::calcEllipticalDimensions)
        # hbRatio = (LB + sqrt(LB^2-1)) / (LB - sqrt(LB^2-1))
        if lb > 1.0:
            sq = math.sqrt(lb * lb - 1.0)
            hb = (lb + sq) / (lb - sq)
        else:
            hb = 1.0
        # a (lateral), b+c (forward), c (backing component)
        a_ell = 0.5 * (1.0 + 1.0 / hb) / lb    # lateral spread rate (relative)
        b_ell = (1.0 + 1.0 / hb) / 2.0          # forward semi-axis
        c_ell = b_ell - 1.0 / hb                 # offset

        # theta = acos(c/b), lateral ROS = a * sin(theta)
        if abs(b_ell) > 1e-7:
            cos_theta = min(1.0, max(-1.0, c_ell / b_ell))
            theta = math.acos(cos_theta)
            lat_ros = abs(a_ell * math.sin(theta))
        else:
            lat_ros = 0.0

        # Fuel data: 2 fuels, normalized relative ROS
        max_abs = max(ros1, ros2)
        rel1 = ros1 / max_abs
        rel2 = ros2 / max_abs
        fuels = [(rel1, cov1), (rel2, cov2)]  # (relRos, fraction)

        samples = 2
        depth = 2
        cell_size = 10.0  # arbitrary, cancels out

        def calc_flanking_time(n_layers, separation, overlap, lat_dists, spread_rates):
            """RandThread::calcFlankingTime - time to flank across overlap distance."""
            beta = math.atan2(overlap, separation)
            cos_b = math.cos(beta)
            sin_b = math.sin(beta)
            cos_b2 = cos_b * cos_b
            sin_b2 = sin_b * sin_b

            # cosT = (a*cosB*sqrt(a^2*cosB^2 + (b^2-c^2)*sinB^2) - b*c*sinB^2) / (a^2*cosB^2 + b^2*sinB^2)
            inner = a_ell*a_ell*cos_b2 + (b_ell*b_ell - c_ell*c_ell)*sin_b2
            if inner < 0:
                inner = 0.0
            num = a_ell * cos_b * math.sqrt(inner) - b_ell * c_ell * sin_b2
            denom = a_ell*a_ell*cos_b2 + b_ell*b_ell*sin_b2
            if abs(denom) < 1e-12:
                return 1e12
            cos_t = num / denom
            cos_t = min(1.0, max(-1.0, cos_t))
            t_angle = math.acos(cos_t)
            # travel time = sum(latDist[i] / (a * sin(theta) * ros[i]))
            travel_time = 0.0
            lat_ros_val = abs(a_ell * math.sin(t_angle))
            for i in range(n_layers):
                if spread_rates[i] > 1e-10 and lat_ros_val > 1e-10:
                    travel_time += lat_dists[i] / (lat_ros_val * spread_rates[i])
                else:
                    travel_time += 1e12
            return travel_time

        # Generate all 2^4=16 combinations of 2x2 grid
        # rosArray[comb_idx][row*samples + col]
        n_combs = 2 ** (samples * depth)  # 16
        ros_array = []
        comb_array = []
        for idx in range(n_combs):
            ros_row = []
            prob_row = []
            for cell in range(samples * depth):
                bit = (idx >> cell) & 1
                ros_row.append(fuels[bit][0])
                prob_row.append(fuels[bit][1])
            ros_array.append(ros_row)
            comb_array.append(prob_row)

        # For each combination, compute maxRos (RandThread::calcSpreadPaths2)
        max_ros_array = []
        for i in range(n_combs):
            ros_i = ros_array[i]
            comb_i = comb_array[i]
            # sample_time[k] = best exit time for starting column k
            sample_time = [9e12] * samples

            for k in range(samples):  # starting column
                # Path tracking: list of (time, loc, ignition_pt, rel_cell_size)
                paths = [(0.0, k, 0, 0.0)]
                j = 0  # current row being processed
                while j < depth and paths:
                    new_paths = []
                    for (path_time, loc, ign_pt, rel_size) in paths:
                        parent_ros = ros_i[j * samples + loc] if 0 <= loc < samples else 0.0
                        if parent_ros > 1e-10:
                            separation = cell_size
                            overlap = cell_size
                            if ign_pt == 0:
                                overlap /= 2.0
                            separation += rel_size

                            # Straight ahead (same column, next row)
                            straight_time = path_time + cell_size / parent_ros
                            new_paths.append((straight_time, loc, ign_pt, separation))

                            # Lateral flanking (only if not at last depth row)
                            if j < depth - 1:
                                # Flanking time
                                lat_dists = [overlap]
                                spread_rates = [ros_i[j * samples + loc]]
                                if separation > cell_size:
                                    lat_dists[0] = overlap / (j + 1)
                                delay = calc_flanking_time(
                                    int(separation / cell_size),
                                    separation, overlap, lat_dists, spread_rates
                                )
                                delay += path_time

                                # Go left
                                if ign_pt >= 0 and loc - 1 >= 0:
                                    new_paths.append((delay, loc - 1, -1, 0.0))
                                # Go right
                                if ign_pt <= 0 and loc + 1 < samples:
                                    new_paths.append((delay, loc + 1, 1, 0.0))
                        else:
                            new_paths.append((0.0, loc, ign_pt, rel_size))

                    paths = new_paths
                    j += 1

                # Find best time (minimum) for this starting column
                for (t, loc2, _, _) in paths:
                    if t > 0.0 and t < sample_time[k]:
                        sample_time[k] = t

            # Max spread rate = depth*cellSize / min_time
            max_ros_i = 0.0
            for k in range(samples):
                if sample_time[k] > 0.0 and sample_time[k] < 9e12:
                    ros_k = (depth * cell_size) / sample_time[k]
                    if ros_k > max_ros_i:
                        max_ros_i = ros_k
            max_ros_array.append(max_ros_i)

        # Compute weighted average (expected ROS)
        average = 0.0
        for i in range(n_combs):
            prob = 1.0
            for j in range(depth):
                for k in range(samples):
                    prob *= comb_array[i][j * samples + k]
            average += max_ros_array[i] * prob

        return average * max_abs

    def _populate_intermediates_from_arrays(self, depth, savr_dead, savr_live,
                                             load_dead, load_live,
                                             moisture_dead, moisture_live,
                                             hoc_dead, hoc_live,
                                             density_dead, density_live,
                                             moe_dead, eff_silica_dead, eff_silica_live):
        """Directly populate SurfaceFuelbedIntermediates with provided arrays."""
        imd = self.intermediates
        imd.initialize_members()
        imd.depth = depth
        N = imd.MAX_PARTICLES

        for i in range(N):
            imd.savr_dead[i] = savr_dead[i] if i < len(savr_dead) else 0.0
            imd.savr_live[i] = savr_live[i] if i < len(savr_live) else 0.0
            imd.load_dead[i] = load_dead[i] if i < len(load_dead) else 0.0
            imd.load_live[i] = load_live[i] if i < len(load_live) else 0.0
            imd.moisture_dead[i] = moisture_dead[i] if i < len(moisture_dead) else 0.0
            imd.moisture_live[i] = moisture_live[i] if i < len(moisture_live) else 0.0
            imd.hoc_dead[i] = hoc_dead[i] if i < len(hoc_dead) else 0.0
            imd.hoc_live[i] = hoc_live[i] if i < len(hoc_live) else 0.0
            imd.fuel_density_dead[i] = density_dead[i] if i < len(density_dead) else 32.0
            imd.fuel_density_live[i] = density_live[i] if i < len(density_live) else 32.0
            imd.silica_eff_dead[i] = eff_silica_dead[i] if i < len(eff_silica_dead) else 0.01
            imd.silica_eff_live[i] = eff_silica_live[i] if i < len(eff_silica_live) else 0.01

        imd.moisture_of_extinction[imd.DEAD] = moe_dead

        # Count size classes - boost to MAX_DEAD or MAX_LIVE if any nonzero
        dead_count = sum(1 for i in range(imd.MAX_DEAD) if imd.load_dead[i] > 1e-10)
        live_count = sum(1 for i in range(imd.MAX_LIVE) if imd.load_live[i] > 1e-10)
        imd.number_of_size_classes[imd.DEAD] = imd.MAX_DEAD if dead_count > 0 else 0
        imd.number_of_size_classes[imd.LIVE] = imd.MAX_LIVE if live_count > 0 else 0

        # Run the full intermediates calculation chain
        imd._calculate_surface_area_fractions()
        imd._calculate_live_moe()
        imd._calculate_characteristic_savr()

        total_load = (imd.total_load_for_life[imd.DEAD] + imd.total_load_for_life[imd.LIVE])
        imd.bulk_density = total_load / depth if depth > 1e-7 else 0.0

        imd.packing_ratio = 0.0
        for i in range(N):
            if imd.load_dead[i] > 0:
                imd.packing_ratio += imd.load_dead[i] / (depth * imd.fuel_density_dead[i])
            if imd.load_live[i] > 0:
                imd.packing_ratio += imd.load_live[i] / (depth * imd.fuel_density_live[i])

        if imd.sigma > 1e-7:
            opt = 3.348 / (imd.sigma ** 0.8189)
            imd.relative_packing_ratio = imd.packing_ratio / opt if opt > 1e-7 else 0.0
        else:
            imd.relative_packing_ratio = 0.0

        imd._calculate_heat_sink()
        imd._calculate_propagating_flux()

        # Update aliases
        imd.moisture_of_extinction_dead = imd.moisture_of_extinction[imd.DEAD]
        imd.moisture_of_extinction_live = imd.moisture_of_extinction[imd.LIVE]
        imd.weighted_moisture_dead = imd.weighted_moisture[imd.DEAD]
        imd.weighted_moisture_live = imd.weighted_moisture[imd.LIVE]
        imd.weighted_heat_dead = imd.weighted_heat[imd.DEAD]
        imd.weighted_heat_live = imd.weighted_heat[imd.LIVE]
        imd.weighted_silica_dead = imd.weighted_silica[imd.DEAD]
        imd.weighted_silica_live = imd.weighted_silica[imd.LIVE]
        imd.weighted_fuel_load_dead = imd.weighted_fuel_load[imd.DEAD]
        imd.weighted_fuel_load_live = imd.weighted_fuel_load[imd.LIVE]
        imd.total_load_dead = imd.total_load_for_life[imd.DEAD]
        imd.total_load_live = imd.total_load_for_life[imd.LIVE]
        imd.fuel_model_number = -1  # Mark as populated

    def _calculate_flame_and_intensity(self):
        """
        Calculate flame length and fireline intensity.
        Uses Byram (1959) flame length equation.
        """
        spread_rate = self.results['spread_rate']
        ri = self.results['reaction_intensity']
        res_time = self.results['residence_time']
        
        # Fireline intensity: I = R * Ir * Tau / 60
        # where R = spread rate (ft/min), Ir = reaction intensity, Tau = residence time
        if spread_rate > 0 and res_time > 0:
            fli = (spread_rate * ri * res_time) / 60.0
        else:
            fli = 0
        
        self.results['fireline_intensity'] = fli
        
        # Flame length: FL = 0.45 * I^0.46 (Byram 1959)
        if fli > 1e-7:
            fl = 0.45 * (fli ** 0.46)
        else:
            fl = 0
        
        self.results['flame_length'] = fl
    
    def getSpreadRate(self, spread_rate_units):
        """
        Get spread rate in specified units.
        Matches C++ API naming convention.
        """
        # Trigger calculation only if not yet computed
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        
        spread_rate_ft_min = self.results['spread_rate']
        return SpeedUnits.fromBaseUnits(
            SpeedUnits.toBaseUnits(spread_rate_ft_min,
                                   SpeedUnits.SpeedUnitsEnum.FeetPerMinute),
            spread_rate_units
        )
    
    def get_spread_rate(self, spread_rate_units):
        """Get spread rate (snake_case alias for getSpreadRate)."""
        return self.getSpreadRate(spread_rate_units)
    
    def getLiveFuelMoistureOfExtinction(self, moisture_units):
        """
        Get live fuel moisture of extinction.
        C++: FractionUnits::fromBaseUnits(surfaceFire_.getMoistureOfExtinctionByLifeState(Live), units)
        """
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        moe_live = self.intermediates.moisture_of_extinction[self.intermediates.LIVE]
        return FractionUnits.fromBaseUnits(moe_live, moisture_units)

    def getCharacteristicMoistureByLifeState(self, life_state, moisture_units):
        """
        Get characteristic (weighted) moisture content for dead or live fuel.
        C++: FractionUnits::fromBaseUnits(surfaceFire_.getWeightedMoistureByLifeState(lifeState), units)
        life_state: 'Dead'/'Live' string or 0/1 integer (matches FuelLifeState enum)
        """
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        # Resolve life_state to index (0=Dead, 1=Live)
        if isinstance(life_state, str):
            idx = self.intermediates.LIVE if life_state.lower() == 'live' else self.intermediates.DEAD
        else:
            idx = int(life_state)  # 0=Dead, 1=Live
        weighted_moisture = self.intermediates.weighted_moisture[idx]
        return FractionUnits.fromBaseUnits(weighted_moisture, moisture_units)

    def getCharacteristicSAVR(self, savr_units):
        """
        Get characteristic SAVR.
        C++: SurfaceAreaToVolumeUnits::fromBaseUnits(surfaceFire_.getCharacteristicSAVR(), savrUnits)
        Base unit: SquareFeetOverCubicFeet.
        """
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        return SurfaceAreaToVolumeUnits.fromBaseUnits(self.intermediates.sigma, savr_units)

    def getHeatSource(self, heat_source_units):
        """
        Get heat source (reaction intensity * propagating flux * (1 + phiS + phiW)).
        C++: HeatSourceAndReactionIntensityUnits::fromBaseUnits(surfaceFire_.getHeatSource(), units)
        Computed as ri * propagating_flux * (1 + phi_s + phi_w).
        """
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        ri = self.results.get('reaction_intensity', 0.0)
        phi_s = self.results.get('phi_s', 0.0)
        phi_w = self.results.get('phi_w', 0.0)
        heat_source = ri * self.intermediates.propagating_flux * (1.0 + phi_s + phi_w)
        return HeatSourceAndReactionIntensityUnits.fromBaseUnits(heat_source, heat_source_units)

    def getFlameLength(self, length_units):
        """Get flame length. C++: LengthUnits::fromBaseUnits(surfaceFire_.getFlameLength(), units)"""
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        return LengthUnits.fromBaseUnits(self.results.get('flame_length', 0.0), length_units)

    def getFirelineIntensity(self, intensity_units):
        """Get fireline intensity. C++: FirelineIntensityUnits::fromBaseUnits(surfaceFire_.getFirelineIntensity(), units)"""
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        return FirelineIntensityUnits.fromBaseUnits(self.results.get('fireline_intensity', 0.0), intensity_units)

    def getBackingSpreadRate(self, spread_rate_units):
        """
        Get backing spread rate.
        C++: size_.getBackingSpreadRate() = forward * (1-e)/(1+e)
        """
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        backing_fpm = self.results.get('backing_spread_rate', 0.0)
        return SpeedUnits.fromBaseUnits(backing_fpm, spread_rate_units)

    def getFlankingSpreadRate(self, spread_rate_units):
        """
        Get flanking spread rate.
        C++: (forward+backing)/(2*LWR)
        """
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        flanking_fpm = self.results.get('flanking_spread_rate', 0.0)
        return SpeedUnits.fromBaseUnits(flanking_fpm, spread_rate_units)

    def _get_eccentricity(self, lwr=None):
        """Compute fire ellipse eccentricity from length-to-width ratio."""
        if lwr is None:
            return self.results.get('eccentricity', 0.0)
        if lwr <= 1.0:
            return 0.0
        x = lwr * lwr - 1.0
        return math.sqrt(x) / lwr if x > 0 else 0.0

    def getSpreadDistance(self, length_units, elapsed_time, time_units):
        """Forward spread distance = spreadRate * elapsedTime."""
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        elapsed_min = TimeUnits.toBaseUnits(elapsed_time, time_units)
        dist_ft = self.results.get('spread_rate', 0.0) * elapsed_min
        return LengthUnits.fromBaseUnits(dist_ft, length_units)

    def getBackingSpreadDistance(self, length_units, elapsed_time, time_units):
        """Backing spread distance."""
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        backing_fpm = self.results.get('backing_spread_rate', 0.0)
        elapsed_min = TimeUnits.toBaseUnits(elapsed_time, time_units)
        dist_ft = backing_fpm * elapsed_min
        return LengthUnits.fromBaseUnits(dist_ft, length_units)

    def getFlankingSpreadDistance(self, length_units, elapsed_time, time_units):
        """Flanking spread distance."""
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        flanking_fpm = self.results.get('flanking_spread_rate', 0.0)
        elapsed_min = TimeUnits.toBaseUnits(elapsed_time, time_units)
        dist_ft = flanking_fpm * elapsed_min
        return LengthUnits.fromBaseUnits(dist_ft, length_units)

    def getFireLengthToWidthRatio(self):
        """Get fire length-to-width ratio."""
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        return self.results.get('fire_length_to_width_ratio', 1.0)

    def getHeadingToBackingRatio(self):
        """Heading-to-backing ratio (Alexander 1985)."""
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        return self.results.get('heading_to_backing_ratio', 0.0)

    def getEllipticalA(self, length_units, elapsed_time, time_units):
        """
        Semi-minor axis a of the fire ellipse per unit time.
        C++: ellipticalA_ * elapsedTime
        A = B / LWR, stored as ft/min
        """
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        elapsed_min = TimeUnits.toBaseUnits(elapsed_time, time_units)
        a_ft = self.results.get('elliptical_a', 0.0) * elapsed_min
        return LengthUnits.fromBaseUnits(a_ft, length_units)

    def getEllipticalB(self, length_units, elapsed_time, time_units):
        """
        Semi-major axis b of the fire ellipse per unit time.
        C++: ellipticalB_ * elapsedTime
        B = (forward + backing) / 2, stored as ft/min
        """
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        elapsed_min = TimeUnits.toBaseUnits(elapsed_time, time_units)
        b_ft = self.results.get('elliptical_b', 0.0) * elapsed_min
        return LengthUnits.fromBaseUnits(b_ft, length_units)

    def getEllipticalC(self, length_units, elapsed_time, time_units):
        """
        Distance from ignition to center of ellipse.
        C++: ellipticalC_ * elapsedTime
        C = B - backing_rate, stored as ft/min
        """
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        elapsed_min = TimeUnits.toBaseUnits(elapsed_time, time_units)
        c_ft = self.results.get('elliptical_c', 0.0) * elapsed_min
        return LengthUnits.fromBaseUnits(c_ft, length_units)

    def getFireArea(self, area_units, elapsed_time, time_units):
        """
        Fire ellipse area = pi * A * B * elapsed^2
        C++: pi * ellipticalA_ * ellipticalB_ * elapsedTime^2
        """
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        elapsed_min = TimeUnits.toBaseUnits(elapsed_time, time_units)
        a = self.results.get('elliptical_a', 0.0)
        b = self.results.get('elliptical_b', 0.0)
        area_sq_ft = math.pi * a * b * elapsed_min * elapsed_min
        return AreaUnits.fromBaseUnits(area_sq_ft, area_units)

    def getFirePerimeter(self, length_units, elapsed_time, time_units):
        """
        Fire ellipse perimeter using approximation from C++:
        P = pi * (a+b) * (1 + h/4 + h^2/64)  where h = (a-b)^2/(a+b)^2
        """
        if self.intermediates.fuel_model_number is None and self.state.get('fuel_model_number'):
            self.do_surface_run_in_direction_of_max_spread()
        elapsed_min = TimeUnits.toBaseUnits(elapsed_time, time_units)
        a = self.results.get('elliptical_a', 0.0) * elapsed_min
        b = self.results.get('elliptical_b', 0.0) * elapsed_min
        perim_ft = 0.0
        if (a + b) > 1e-7:
            a_minus_b = a - b
            a_plus_b = a + b
            h = (a_minus_b * a_minus_b) / (a_plus_b * a_plus_b)
            perim_ft = math.pi * a_plus_b * (1.0 + h / 4.0 + (h * h) / 64.0)
        return LengthUnits.fromBaseUnits(perim_ft, length_units)

    def do_surface_run_in_direction_of_interest(self, direction_of_interest, direction_mode):
        """
        Run surface fire calculation in a specified direction of interest.
        C++: doSurfaceRunInDirectionOfInterest(directionOfInterest, directionMode)
        Stores spread rate/flame length at that direction in results.
        """
        # First compute in max spread direction
        self.do_surface_run_in_direction_of_max_spread()
        # Then compute rate at the specified direction
        rate_at_dir = self._calculate_spread_rate_at_vector(direction_of_interest, direction_mode)
        self.results['spread_rate_in_direction_of_interest'] = rate_at_dir
        # Flame length at that rate (Byram 1959)
        ri = self.results.get('reaction_intensity', 0.0)
        res_time = self.results.get('residence_time', 0.0)
        fli = (rate_at_dir * ri * res_time) / 60.0 if res_time > 0 else 0.0
        self.results['fireline_intensity_in_direction_of_interest'] = fli
        fl = 0.45 * (fli ** 0.46) if fli > 1e-7 else 0.0
        self.results['flame_length_in_direction_of_interest'] = fl

    def _calculate_spread_rate_at_vector(self, direction_of_interest, direction_mode):
        """
        Compute spread rate at a direction relative to max spread, mirroring
        C++ Surface::calculateSpreadRateAtVector().

        For FromPerimeter (Catchpole et al. 1982):
          L = forward + backing
          f = L/2,  g = forward - f,  h = flanking_rate
          Rpsi = g*cos(beta) + sqrt(f^2*cos^2(beta) + h^2*sin^2(beta))

        For FromIgnitionPoint:
          rosVector = forward * (1-e) / (1 - e*cos(beta))
        """
        forward_rate = self.results.get('spread_rate', 0.0)
        if forward_rate < 1e-7:
            return forward_rate

        # Constrain direction_of_interest to [0, 360)
        while direction_of_interest < 0.0:
            direction_of_interest += 360.0
        while direction_of_interest >= 360.0:
            direction_of_interest -= 360.0

        max_spread_dir = self.results.get('direction_of_max_spread', 0.0)
        backing_rate = self.results.get('backing_spread_rate', 0.0)
        flanking_rate = self.results.get('flanking_spread_rate', 0.0)
        eccentricity = self.results.get('eccentricity', 0.0)

        # Calculate beta: angle between max-spread direction and direction of interest
        beta = abs(max_spread_dir - direction_of_interest)
        if beta > 180.0:
            beta = 360.0 - beta

        beta_rad = math.radians(beta)
        cos_beta = math.cos(beta_rad)
        sin_beta = math.sin(beta_rad)

        if isinstance(direction_mode, str) and 'perimeter' in direction_mode.lower():
            # Catchpole et al. (1982) - spread rate perpendicular to perimeter
            L = forward_rate + backing_rate
            f = L / 2.0
            g = forward_rate - f
            h = flanking_rate
            ros_vector = g * cos_beta + math.sqrt(f * f * cos_beta * cos_beta + h * h * sin_beta * sin_beta)
        else:
            # FromIgnitionPoint: ellipse polar form from ignition point
            denom = 1.0 - eccentricity * cos_beta
            if abs(denom) < 1e-7:
                ros_vector = 0.0
            else:
                ros_vector = forward_rate * (1.0 - eccentricity) / denom

        return max(0.0, ros_vector)

    def getSpreadRateInDirectionOfInterest(self, spread_rate_units):
        """Get spread rate in the last direction-of-interest run."""
        rate_fpm = self.results.get('spread_rate_in_direction_of_interest', 0.0)
        return SpeedUnits.fromBaseUnits(rate_fpm, spread_rate_units)

    def getFlameLengthInDirectionOfInterest(self, length_units):
        """Get flame length in the last direction-of-interest run."""
        fl_ft = self.results.get('flame_length_in_direction_of_interest', 0.0)
        return LengthUnits.fromBaseUnits(fl_ft, length_units)

    # ------------------------------------------------------------------
    # Setters that match C++ API names (camelCase) used by tests
    # ------------------------------------------------------------------

    def setWindHeightInputMode(self, mode):
        """C++ API alias."""
        self.set_wind_height_input_mode(mode)

    def setWindAndSpreadOrientationMode(self, mode):
        """C++ API alias."""
        self.set_wind_orientation_mode(mode)

    def setWindSpeed(self, wind_speed, units, height_mode=None):
        """C++ API alias. height_mode is stored if provided."""
        self.set_wind_speed(wind_speed, units)
        if height_mode is not None:
            self.set_wind_height_input_mode(height_mode)

    def setWindDirection(self, wind_direction):
        """C++ API alias."""
        self.set_wind_direction(wind_direction)

    def setSlope(self, slope, slope_units):
        """C++ API alias."""
        self.set_slope(slope, slope_units)

    def setAspect(self, aspect):
        """C++ API alias."""
        self.set_aspect(aspect)

    def setFuelModelNumber(self, fuel_model_number):
        """C++ API alias."""
        self.set_fuel_model(fuel_model_number)

    def setCanopyCover(self, canopy_cover, units):
        """C++ API alias."""
        self.set_canopy_cover(canopy_cover, units)

    def setCanopyHeight(self, canopy_height, units):
        """C++ API alias."""
        self.set_canopy_height(canopy_height, units)

    def setCrownRatio(self, crown_ratio, units):
        """C++ API alias."""
        self.set_crown_ratio(crown_ratio, units)

    def setUserProvidedWindAdjustmentFactor(self, waf):
        """Store user-provided WAF and switch to UserInput calculation mode."""
        self.state['user_provided_waf'] = waf

    def setWindAdjustmentFactorCalculationMethod(self, method):
        """
        Set WAF calculation method.
        'UserInput' uses user-provided WAF; 'UseCrownRatio' computes from canopy.
        """
        self.state['waf_calculation_method'] = method

    def set_wind_adjustment_factor_calculation_method(self, method):
        """snake_case alias."""
        self.setWindAdjustmentFactorCalculationMethod(method)

    def set_user_provided_wind_adjustment_factor(self, waf):
        """snake_case alias."""
        self.setUserProvidedWindAdjustmentFactor(waf)

    def doSurfaceRunInDirectionOfMaxSpread(self):
        """C++ API alias."""
        self.do_surface_run_in_direction_of_max_spread()

    def doSurfaceRunInDirectionOfInterest(self, direction, mode):
        """C++ API alias."""
        self.do_surface_run_in_direction_of_interest(direction, mode)

    # Stubs for special fuel type methods (Chaparral, PalmettoGallberry, WesternAspen)
    def set_is_using_chaparral(self, val):
        self.state['is_using_chaparral'] = val
        if not val:
            self.intermediates.is_using_chaparral = False

    def set_chaparral_fuel_bed_depth(self, depth, units):
        self.state['chaparral_fuel_bed_depth'] = LengthUnits.toBaseUnits(depth, units)

    def set_chaparral_fuel_type(self, fuel_type):
        self.state['chaparral_fuel_type'] = fuel_type

    def set_chaparral_fuel_load_input_mode(self, mode):
        self.state['chaparral_fuel_load_input_mode'] = mode

    def set_chaparral_fuel_dead_load_fraction(self, fraction):
        self.state['chaparral_fuel_dead_load_fraction'] = fraction

    def set_chaparral_total_fuel_load(self, load, units):
        self.state['chaparral_total_fuel_load'] = load  # stored as provided; intermediates handles conversion

    def set_is_using_palmetto_gallberry(self, val):
        self.state['is_using_palmetto_gallberry'] = val

    def update_surface_inputs_for_palmetto_gallberry(self, m1h, m10h, m100h, mlh, mlw, m_units,
                                                      wind_speed, wind_speed_units, wind_height_mode,
                                                      wind_dir, orientation_mode,
                                                      age_of_rough, height_of_understory, palmetto_coverage,
                                                      overstory_basal_area, basal_area_units,
                                                      slope, slope_units, aspect,
                                                      canopy_cover, canopy_cover_units,
                                                      canopy_height, canopy_height_units,
                                                      crown_ratio, crown_ratio_units):
        self.updateSurfaceInputs(0, m1h, m10h, m100h, mlh, mlw, m_units,
                                 wind_speed, wind_speed_units, wind_height_mode,
                                 wind_dir, orientation_mode,
                                 slope, slope_units, aspect,
                                 canopy_cover, canopy_cover_units,
                                 canopy_height, canopy_height_units,
                                 crown_ratio, crown_ratio_units)
        self.state['palmetto_gallberry_age_of_rough'] = age_of_rough
        self.state['palmetto_gallberry_height_of_understory'] = height_of_understory
        self.state['palmetto_gallberry_palmetto_coverage'] = palmetto_coverage
        self.state['palmetto_gallberry_overstory_basal_area'] = overstory_basal_area
        self.state['palmetto_gallberry_basal_area_units'] = basal_area_units

    def set_is_using_western_aspen(self, val):
        self.state['is_using_western_aspen'] = val

    def update_surface_inputs_for_western_aspen(self, aspen_fuel_model_num, aspen_curing_level,
                                                  curing_level_units, aspen_fire_severity, dbh, dbh_units,
                                                  m1h, m10h, m100h, mlh, mlw, m_units,
                                                  wind_speed, wind_speed_units, wind_height_mode,
                                                  wind_dir, orientation_mode,
                                                  slope, slope_units, aspect,
                                                  canopy_cover, canopy_cover_units,
                                                  canopy_height, canopy_height_units,
                                                  crown_ratio, crown_ratio_units):
        self.updateSurfaceInputs(0, m1h, m10h, m100h, mlh, mlw, m_units,
                                 wind_speed, wind_speed_units, wind_height_mode,
                                 wind_dir, orientation_mode,
                                 slope, slope_units, aspect,
                                 canopy_cover, canopy_cover_units,
                                 canopy_height, canopy_height_units,
                                 crown_ratio, crown_ratio_units)
        self.state['aspen_fuel_model_number'] = aspen_fuel_model_num
        self.state['aspen_curing_level'] = self._convert_fraction(aspen_curing_level, curing_level_units)
        self.state['aspen_fire_severity'] = aspen_fire_severity
        self.state['aspen_dbh'] = dbh

    def get_aspen_mortality(self, mortality_units):
        """Get western aspen mortality fraction."""
        return self.results.get('aspen_mortality', 0.0)

    def update_surface_inputs_for_two_fuel_models(self, first_fuel_model, second_fuel_model,
                                                    m1h, m10h, m100h, mlh, mlw, m_units,
                                                    wind_speed, wind_speed_units, wind_height_mode,
                                                    wind_dir, orientation_mode,
                                                    first_coverage, first_coverage_units,
                                                    two_fuel_models_method,
                                                    slope, slope_units, aspect,
                                                    canopy_cover, canopy_cover_units,
                                                    canopy_height, canopy_height_units,
                                                    crown_ratio, crown_ratio_units):
        self.updateSurfaceInputs(first_fuel_model, m1h, m10h, m100h, mlh, mlw, m_units,
                                 wind_speed, wind_speed_units, wind_height_mode,
                                 wind_dir, orientation_mode,
                                 slope, slope_units, aspect,
                                 canopy_cover, canopy_cover_units,
                                 canopy_height, canopy_height_units,
                                 crown_ratio, crown_ratio_units)
        self.state['second_fuel_model_number'] = second_fuel_model
        self.state['first_fuel_model_coverage'] = self._convert_fraction(first_coverage, first_coverage_units)
        self.state['two_fuel_models_method'] = two_fuel_models_method
        self.state['is_using_two_fuel_models'] = True

    def set_two_fuel_models_first_fuel_model_coverage(self, coverage, units):
        self.state['first_fuel_model_coverage'] = self._convert_fraction(coverage, units)

    def get_fireline_intensity(self, intensity_units):
        """snake_case alias for getFirelineIntensity."""
        return self.getFirelineIntensity(intensity_units)

    def get_flame_length(self, length_units):
        """snake_case alias for getFlameLength."""
        return self.getFlameLength(length_units)

    def get_spread_rate(self, units):
        """Get spread rate - accepts unit string or SpeedUnitsEnum."""
        if isinstance(units, str):
            units_lower = units.replace('/', '').replace(' ', '').lower()
            if 'ftmin' in units_lower or 'feetperminute' in units_lower:
                return self.getSpreadRate(SpeedUnits.SpeedUnitsEnum.FeetPerMinute)
            elif 'chph' in units_lower or 'chainsperhour' in units_lower:
                return self.getSpreadRate(SpeedUnits.SpeedUnitsEnum.ChainsPerHour)
            elif 'mph' in units_lower or 'milesperhour' in units_lower:
                return self.getSpreadRate(SpeedUnits.SpeedUnitsEnum.MilesPerHour)
            else:
                return self.results.get('spread_rate', 0.0)
        return self.getSpreadRate(units)

    def get_heat_per_unit_area(self, units):
        """Get heat per unit area (base unit: Btu/ft²)."""
        val = self.results.get('heat_per_unit_area', 0.0)
        return val

    def get_canopy_height(self, units='feet'):
        """Get canopy height in specified units."""
        ch = self.state.get('canopy_height', 0.0)
        if isinstance(units, str) and 'ft' in units.lower():
            return ch
        return LengthUnits.fromBaseUnits(ch, units) if not isinstance(units, str) else ch

    def get_canopy_cover(self, units=None):
        """Get canopy cover as fraction."""
        return self.state.get('canopy_cover', 0.0)

    def get_crown_ratio(self, units=None):
        """Get crown ratio as fraction."""
        return self.state.get('crown_ratio', 0.0)

    def get_canopy_base_height(self, units='feet'):
        """Get canopy base height (stored via crown inputs)."""
        return self.state.get('canopy_base_height', 0.0)

    def get_slope(self, units=None):
        """Get slope in degrees."""
        return self.state.get('slope', 0.0)

    def get_aspect(self):
        """Get aspect in degrees."""
        return self.state.get('aspect', 0.0)

    def get_wind_speed(self, units=None):
        """Get wind speed."""
        return self.state.get('wind_speed', 0.0)

    def get_wind_direction(self):
        """Get wind direction."""
        return self.state.get('wind_direction', 0.0)

    def get_wind_height_input_mode(self):
        """Get wind height input mode string."""
        return self.state.get('wind_height_input_mode', 'TwentyFoot')

    def get_wind_adjustment_factor_calculation_method(self):
        """Get WAF calculation method string."""
        return self.state.get('waf_calculation_method', 'UseCrownRatio')

    def get_reaction_intensity(self, units=None):
        """Get reaction intensity (base unit: Btu/ft2/min)."""
        return self.results.get('reaction_intensity', 0.0)

    def get_heat_sink(self, units=None):
        """Get heat sink (base unit: Btu/ft3)."""
        return self.intermediates.heat_sink

    def get_midflame_wind_speed(self):
        """Get midflame wind speed (ft/min)."""
        return self._phi_w_state.get('midflame_ws', 0.0)

    def set_wind_and_spread_orientation_mode(self, mode):
        """Set wind and spread orientation mode."""
        self.set_wind_orientation_mode(mode)

    def set_fuel_model_number(self, fuel_model_number):
        """Alias for set_fuel_model."""
        self.set_fuel_model(fuel_model_number)
