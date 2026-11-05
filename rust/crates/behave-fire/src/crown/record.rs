//! Full Crown runner retaining independently typed supplied surface fields.
use super::{CrownInput, SurfaceFireInputs};
use crate::{fuel_models::FuelModels, surface::masked::MaskedValue, BehaveError};

macro_rules! result {
    ($($field:ident),+) => {
        /// Complete Crown outputs with independent raw-data, dtype, mask, class, and rank metadata.
        #[derive(Clone,Copy,Debug,Default,PartialEq)]
        #[non_exhaustive]
        pub struct MaskedCrownResult {
            $(#[doc=concat!("Python `",stringify!($field),"` field in component base units.")] pub $field:MaskedValue,)+
            /// Exact classification: 0=surface, 1=torching, 2=conditional crown, 3=crowning.
            pub fire_type:i32,
        }
        impl MaskedCrownResult {
            /// Construct an empty result with ordinary f64 array defaults.
            pub fn new()->Self {Self::default()}
            /// Return all twenty-one numerical fields under their exact Python keys.
            pub fn named_values(&self)->[(&'static str,MaskedValue);21] {[$((stringify!($field),self.$field)),+]}
        }
    }
}
result!(
    crown_fire_spread_rate,
    crown_flame_length,
    crown_fire_line_intensity,
    crown_fire_heat_per_unit_area,
    canopy_heat_per_unit_area,
    crown_critical_surface_fire_line_intensity,
    crown_critical_fire_spread_rate,
    crown_fire_transition_ratio,
    crown_fire_active_ratio,
    crown_length_to_width_ratio,
    surface_fire_critical_spread_rate,
    crowning_surface_fire_spread_rate,
    crown_fraction_burned,
    passive_crown_fire_spread_rate,
    passive_crown_fire_heat_per_unit_area,
    passive_crown_fire_line_intensity,
    passive_crown_fire_flame_length,
    final_spread_rate,
    final_heat_per_unit_area,
    final_fireline_intensity,
    final_flame_length
);

/// Run Crown from three supplied surface fields without requiring unused SurfaceResult entries.
pub fn run_with_surface_inputs(
    surface: &SurfaceFireInputs<MaskedValue>,
    input: &CrownInput,
    fuels: &FuelModels,
) -> Result<MaskedCrownResult, BehaveError> {
    let normalized = super::coerce_crown_fire_inputs(input);
    let crown = super::calculate_crown_surface_fire(input, fuels)?;
    // Python normalizes Crown spatial inputs with np.atleast_1d(..., dtype=float).
    // Supplied surface dictionary values remain independently typed until each stage handles them.
    let plain = MaskedValue::plain;
    let crown_rate = plain(crown.crown_fire_spread_rate);
    let heat = super::masked::calculate_crown_heat_and_intensity(
        surface.surface_hpua,
        crown_rate,
        plain(normalized.cbh),
        plain(normalized.ch),
        plain(normalized.cbd),
    );
    let critical_intensity = plain(super::calculate_critical_surface_fireline_intensity(
        normalized.cbh,
        normalized.mf_pct,
    ));
    let critical_rate = plain(super::calculate_critical_crown_fire_spread_rate(
        normalized.cbd,
    ));
    let ratios = super::masked::calculate_crown_fire_ratios(
        surface.surface_fli,
        crown_rate,
        critical_intensity,
        critical_rate,
    );
    let class = super::masked::classify_crown_fire_type(
        ratios.crown_fire_transition_ratio,
        ratios.crown_fire_active_ratio,
    );
    let surface_critical = super::masked::calculate_surface_fire_critical_spread_rate(
        critical_intensity,
        surface.surface_hpua,
    );
    let crowning = plain(super::calculate_crowning_surface_fire_spread_rate(
        crown.ri_crown,
        &crown.ib_crown,
        critical_rate.value,
        normalized.ch,
    ));
    let fraction = super::masked::calculate_crown_fraction_burned(
        surface.surface_ros,
        surface_critical,
        crowning,
    );
    let passive = super::masked::calculate_passive_crown_fire_behavior(
        surface.surface_ros,
        surface.surface_hpua,
        crown_rate,
        heat.canopy_heat_per_unit_area,
        fraction,
    );
    let final_behavior = super::masked::assign_final_crown_fire_behavior(
        class.is_surface_fire,
        class.is_passive_crown,
        surface.surface_ros,
        surface.surface_hpua,
        surface.surface_fli,
        passive.passive_crown_fire_spread_rate,
        passive.passive_crown_fire_heat_per_unit_area,
        passive.passive_crown_fire_line_intensity,
        passive.passive_crown_fire_flame_length,
        crown_rate,
        heat.crown_fire_heat_per_unit_area,
        heat.crown_fire_line_intensity,
        heat.crown_flame_length,
    );
    Ok(MaskedCrownResult {
        crown_fire_spread_rate: crown_rate,
        crown_flame_length: heat.crown_flame_length,
        crown_fire_line_intensity: heat.crown_fire_line_intensity,
        crown_fire_heat_per_unit_area: heat.crown_fire_heat_per_unit_area,
        canopy_heat_per_unit_area: heat.canopy_heat_per_unit_area,
        crown_critical_surface_fire_line_intensity: critical_intensity,
        crown_critical_fire_spread_rate: critical_rate,
        crown_fire_transition_ratio: ratios.crown_fire_transition_ratio,
        crown_fire_active_ratio: ratios.crown_fire_active_ratio,
        crown_length_to_width_ratio: plain(crown.crown_length_to_width_ratio),
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
    })
}
