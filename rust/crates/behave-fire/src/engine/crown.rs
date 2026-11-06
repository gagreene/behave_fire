//! Crown composition with borrowed model-10 fuel and independent Surface values.
use super::{CrownTweaks, PreparedFuel, SurfaceOptions, SurfaceTweaks};
use crate::crown::{self, CrownResult, CrownSurfaceResult, SurfaceFireInputs};

/// Base-unit Crown inputs without string controls or synthetic Surface records.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrownCell {
    /// One-hour dead moisture fraction.
    pub m1h: f64,
    /// Ten-hour dead moisture fraction.
    pub m10h: f64,
    /// Hundred-hour dead moisture fraction.
    pub m100h: f64,
    /// Live herbaceous moisture fraction.
    pub mlh: f64,
    /// Live woody moisture fraction.
    pub mlw: f64,
    /// Wind speed in ft/min at the configured height.
    pub wind_speed: f64,
    /// Wind direction in degrees in the configured coordinate system.
    pub wind_direction: f64,
    /// Supplied slope in degrees, retained in the first model-10 run.
    pub slope: f64,
    /// Aspect in degrees clockwise from north.
    pub aspect: f64,
    /// Canopy base height in feet.
    pub canopy_base_height: f64,
    /// Canopy height in feet.
    pub canopy_height: f64,
    /// Canopy bulk density in lb/ft³.
    pub canopy_bulk_density: f64,
    /// Foliar moisture in percent.
    pub moisture_foliar: f64,
}

impl Default for CrownCell {
    /// Construct calm, flat inputs with 100 percent foliar moisture.
    fn default() -> Self {
        Self {
            m1h: 0.06,
            m10h: 0.07,
            m100h: 0.08,
            mlh: 0.60,
            mlw: 0.90,
            wind_speed: 0.0,
            wind_direction: 0.0,
            slope: 0.0,
            aspect: 0.0,
            canopy_base_height: 0.0,
            canopy_height: 0.0,
            canopy_bulk_density: 0.0,
            moisture_foliar: 100.0,
        }
    }
}

/// Immutable Crown configuration, with model-10 fuel supplied explicitly per call.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CrownPlan {
    options: SurfaceOptions,
}

impl CrownPlan {
    /// Construct controls; model-10 WAF remains 0.4 unless a stage is overridden.
    pub const fn new(options: SurfaceOptions) -> Self {
        Self { options }
    }

    /// Borrow the input wind controls.
    pub const fn options(&self) -> &SurfaceOptions {
        &self.options
    }

    /// Evaluate model-10 particles, fuelbed, reaction, and spread exactly once.
    pub fn prepare_surface(
        &self,
        cell: &CrownCell,
        fm10: &PreparedFuel,
        tweaks: SurfaceTweaks,
    ) -> CrownSurfaceResult {
        crown::calculate_crown_surface_fire(cell, fm10, &self.options, tweaks)
    }

    /// Evaluate Crown from independent supplied Surface fields and typed overrides.
    pub fn run_cell(
        &self,
        cell: &CrownCell,
        supplied: &SurfaceFireInputs<f64>,
        fm10: &PreparedFuel,
        tweaks: CrownTweaks,
    ) -> CrownResult {
        let c = self.prepare_surface(cell, fm10, tweaks.fm10);
        let h = crown::calculate_crown_heat_and_intensity(
            supplied.surface_hpua,
            c.crown_fire_spread_rate,
            cell.canopy_base_height,
            cell.canopy_height,
            cell.canopy_bulk_density,
        );
        let critical_intensity = tweaks.critical_surface_intensity.unwrap_or_else(|| {
            crown::calculate_critical_surface_fireline_intensity(
                cell.canopy_base_height,
                cell.moisture_foliar,
            )
        });
        let critical_rate = tweaks.critical_crown_rate.unwrap_or_else(|| {
            crown::calculate_critical_crown_fire_spread_rate(cell.canopy_bulk_density)
        });
        let ratios = crown::calculate_crown_fire_ratios(
            supplied.surface_fli,
            c.crown_fire_spread_rate,
            critical_intensity,
            critical_rate,
        );
        let class = crown::classify_crown_fire_type(
            ratios.crown_fire_transition_ratio,
            ratios.crown_fire_active_ratio,
        );
        let surface_critical = crown::calculate_surface_fire_critical_spread_rate(
            critical_intensity,
            supplied.surface_hpua,
        );
        let crowning = crown::calculate_crowning_surface_fire_spread_rate(
            c.ri_crown,
            &c.ib_crown,
            critical_rate,
            cell.canopy_height,
        );
        let fraction = tweaks.crown_fraction_burned.unwrap_or_else(|| {
            crown::calculate_crown_fraction_burned(supplied.surface_ros, surface_critical, crowning)
        });
        let passive = crown::calculate_passive_crown_fire_behavior(
            supplied.surface_ros,
            supplied.surface_hpua,
            c.crown_fire_spread_rate,
            h.canopy_heat_per_unit_area,
            fraction,
        );
        let final_behavior = crown::assign_final_crown_fire_behavior(
            class.is_surface_fire,
            class.is_passive_crown,
            supplied.surface_ros,
            supplied.surface_hpua,
            supplied.surface_fli,
            passive.passive_crown_fire_spread_rate,
            passive.passive_crown_fire_heat_per_unit_area,
            passive.passive_crown_fire_line_intensity,
            passive.passive_crown_fire_flame_length,
            c.crown_fire_spread_rate,
            h.crown_fire_heat_per_unit_area,
            h.crown_fire_line_intensity,
            h.crown_flame_length,
        );
        CrownResult {
            crown_fire_spread_rate: c.crown_fire_spread_rate,
            crown_flame_length: h.crown_flame_length,
            crown_fire_line_intensity: h.crown_fire_line_intensity,
            crown_fire_heat_per_unit_area: h.crown_fire_heat_per_unit_area,
            canopy_heat_per_unit_area: h.canopy_heat_per_unit_area,
            crown_critical_surface_fire_line_intensity: critical_intensity,
            crown_critical_fire_spread_rate: critical_rate,
            crown_fire_transition_ratio: ratios.crown_fire_transition_ratio,
            crown_fire_active_ratio: ratios.crown_fire_active_ratio,
            crown_length_to_width_ratio: c.crown_length_to_width_ratio,
            fire_type: class.fire_type,
            surface_fire_critical_spread_rate: surface_critical,
            crowning_surface_fire_spread_rate: crowning,
            crown_fraction_burned: fraction,
            passive_crown_fire_spread_rate: passive.passive_crown_fire_spread_rate,
            passive_crown_fire_heat_per_unit_area: passive.passive_crown_fire_heat_per_unit_area,
            passive_crown_fire_line_intensity: passive.passive_crown_fire_line_intensity,
            passive_crown_fire_flame_length: passive.passive_crown_fire_flame_length,
            final_spread_rate: final_behavior.final_spread_rate,
            final_heat_per_unit_area: final_behavior.final_heat_per_unit_area,
            final_fireline_intensity: final_behavior.final_fireline_intensity,
            final_flame_length: final_behavior.final_flame_length,
        }
    }
}
