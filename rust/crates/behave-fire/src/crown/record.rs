//! Full Crown runner retaining independently typed supplied surface fields.
use super::{arithmetic::CrownNumber, SurfaceFireInputs};
use crate::engine::{CrownCell, PreparedFuel, SurfaceOptions, SurfaceTweaks};
use crate::{surface::masked::MaskedValue, BehaveError};

macro_rules! result {
    ($($field:ident),+) => {
        /// Complete Crown outputs with independent raw-data, dtype, mask, class, and rank metadata.
        #[derive(Clone,Copy,Debug,Default,PartialEq)]
        #[non_exhaustive]
        pub struct MaskedCrownResult<T=MaskedValue> {
            $(#[doc=concat!("Python `",stringify!($field),"` field in component base units.")] pub $field:T,)+
            /// Exact classification: 0=surface, 1=torching, 2=conditional crown, 3=crowning.
            pub fire_type:i32,
        }
        impl<T:Copy+Default> MaskedCrownResult<T> {
            /// Construct an empty result with ordinary f64 array defaults.
            pub fn new()->Self {Self::default()}
            /// Transform every numerical output while retaining exact classification.
            pub(crate) fn map<U:Copy+Default>(self, convert:impl Fn(T)->U)->MaskedCrownResult<U> {
                MaskedCrownResult {$($field:convert(self.$field),)+fire_type:self.fire_type}
            }
            /// Return all twenty-one numerical fields under their exact Python keys.
            pub fn named_values(&self)->[(&'static str,T);21] {[$((stringify!($field),self.$field)),+]}
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
pub fn run_with_surface_inputs<T: CrownNumber>(
    surface: &SurfaceFireInputs<T>,
    input: &CrownCell,
    fuel: &PreparedFuel,
    options: &SurfaceOptions,
) -> Result<MaskedCrownResult<T>, BehaveError> {
    let crown = super::calculate_crown_spread(input, fuel, options, SurfaceTweaks::default());
    // Python normalizes Crown spatial inputs with np.atleast_1d(..., dtype=float).
    // Supplied surface dictionary values remain independently typed until each stage handles them.
    let plain = T::plain;
    let crown_rate = plain(crown.crown_fire_spread_rate);
    let heat = super::masked::calculate_crown_heat_and_intensity(
        surface.surface_hpua,
        crown_rate,
        plain(input.canopy_base_height),
        plain(input.canopy_height),
        plain(input.canopy_bulk_density),
    );
    let critical_intensity = plain(super::calculate_critical_surface_fireline_intensity(
        input.canopy_base_height,
        input.moisture_foliar,
    ));
    let critical_rate = plain(super::calculate_critical_crown_fire_spread_rate(
        input.canopy_bulk_density,
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
        critical_rate.value(),
        input.canopy_height,
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

/// Evaluate ordinary real layouts with compact metadata and the shared Crown equations.
///
/// `None` requests the checked native path for unsupported payload traits or an
/// exceptional intermediate representation. No input values or dtype tuples
/// identify the fast path; precision and promotion are retained at each operation.
pub fn try_run_compact_surface_inputs(
    surface: &SurfaceFireInputs<MaskedValue>,
    input: &CrownCell,
    fuel: &PreparedFuel,
    options: &SurfaceOptions,
) -> Option<Result<MaskedCrownResult, BehaveError>> {
    use crate::surface::masked::CompactValue;
    let mut supplied = SurfaceFireInputs::new();
    supplied.surface_ros = CompactValue::from_checked(surface.surface_ros)?;
    supplied.surface_fli = CompactValue::from_checked(surface.surface_fli)?;
    supplied.surface_hpua = CompactValue::from_checked(surface.surface_hpua)?;
    match run_with_surface_inputs(&supplied, input, fuel, options) {
        Ok(result) => {
            if result
                .named_values()
                .iter()
                .any(|(_, value)| value.unsupported())
            {
                None
            } else {
                Some(Ok(result.map(CompactValue::checked)))
            }
        }
        Err(error) => Some(Err(error)),
    }
}
