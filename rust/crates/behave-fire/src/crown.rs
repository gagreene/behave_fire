//! Crown fire stages in reference base units: ft, ft/min, lb/ftÂ³, BTU/ftÂ², BTU/ft/s.
use crate::math::{np_max, np_min};
pub mod masked;
mod record;
use crate::{
    engine::{
        CrownCell, PreparedFuel, SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceTweaks, WafMethod,
    },
    surface::{self, FuelbedIntermediates, Particles, SurfaceResult},
};
pub use record::{run_with_surface_inputs, MaskedCrownResult};

macro_rules! record {
    ($name:ident { $($field:ident),+ $(,)? }) => {
        #[doc = concat!("Named Python fields for ", stringify!($name), ".")]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        #[non_exhaustive]
        pub struct $name<T = f64> { $(#[doc = concat!("Python `", stringify!($field), "` field in component base units.")] pub $field:T),+ }
        impl<T:Copy+Default> $name<T> {
            /// Construct an empty typed stage result.
            pub fn new() -> Self { Self::default() }
        }
        impl<T:Copy> $name<T> {
            /// Return fields under their exact Python keys.
            pub fn named_values(&self) -> [(&'static str,T); [$(stringify!($field)),+].len()] { [$((stringify!($field),self.$field)),+] }
        }
    };
}
record!(CrownHeat {
    crown_fuel_load,
    canopy_heat_per_unit_area,
    crown_fire_heat_per_unit_area,
    crown_fire_line_intensity,
    crown_flame_length
});
record!(CrownRatios {
    crown_fire_transition_ratio,
    crown_fire_active_ratio
});
record!(PassiveCrownBehavior {
    passive_crown_fire_spread_rate,
    passive_crown_fire_heat_per_unit_area,
    passive_crown_fire_line_intensity,
    passive_crown_fire_flame_length
});
record!(FinalCrownBehavior {
    final_spread_rate,
    final_heat_per_unit_area,
    final_fireline_intensity,
    final_flame_length
});
record!(SurfaceFireInputs {
    surface_ros,
    surface_fli,
    surface_hpua
});
/// Exact Python integer classification and flags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct FireClassification {
    /// 0=surface, 1=torching, 2=conditional crown, 3=crowning.
    pub fire_type: i32,
    /// True for types 0 and 2.
    pub is_surface_fire: bool,
    /// True for type 1.
    pub is_passive_crown: bool,
    /// True for type 3.
    pub is_active_crown: bool,
}
macro_rules! numeric_stages {
    ($t:ty) => {
        /// Calculate Van Wagner critical active spread rate in ft/min.
        pub fn calculate_critical_crown_fire_spread_rate(canopy_bulk_density: $t) -> $t {
            let cbd = canopy_bulk_density * 16.0185;
            if cbd > 1e-7 {
                (3.0 / cbd) * 3.28084
            } else {
                0.0
            }
        }
        /// Calculate critical surface intensity from feet and foliar moisture percent.
        pub fn calculate_critical_surface_fireline_intensity(
            canopy_base_height: $t,
            moisture_foliar: $t,
        ) -> $t {
            let mf = np_max(moisture_foliar, 30.0);
            let cbh = np_max(canopy_base_height * 0.3048, 0.1);
            (0.010 * cbh * (460.0 + 25.9 * mf)).powf(1.5) * 0.2886719
        }
        /// Calculate guarded transition and active ratios.
        pub fn calculate_crown_fire_ratios(
            surface_fli: $t,
            crown_ros: $t,
            critical_surface_fli: $t,
            critical_crown_ros: $t,
        ) -> CrownRatios<$t> {
            CrownRatios {
                crown_fire_transition_ratio: if critical_surface_fli > 1e-7 {
                    surface_fli / critical_surface_fli
                } else {
                    0.0
                },
                crown_fire_active_ratio: if critical_crown_ros > 1e-7 {
                    crown_ros / critical_crown_ros
                } else {
                    0.0
                },
            }
        }
        /// Calculate and clip the linear Scott and Reinhardt crown fraction.
        pub fn calculate_crown_fraction_burned(
            surface_ros: $t,
            surface_critical_ros: $t,
            crowning_surface_ros: $t,
        ) -> $t {
            let d = crowning_surface_ros - surface_critical_ros;
            let v = if d > 1e-7 {
                (surface_ros - surface_critical_ros) / d
            } else {
                0.0
            };
            np_min(np_max(v, 0.0), 1.0)
        }
        /// Calculate canopy loading, heat, active crown intensity, and flame length.
        pub fn calculate_crown_heat_and_intensity(
            surface_hpua: $t,
            crown_ros: $t,
            canopy_base_height: $t,
            canopy_height: $t,
            canopy_bulk_density: $t,
        ) -> CrownHeat<$t> {
            let load = np_max(
                canopy_bulk_density * (canopy_height - canopy_base_height),
                0.0,
            );
            let canopy = load * 8000.0;
            let hpua = surface_hpua + canopy;
            let fli = (crown_ros / 60.0) * hpua;
            CrownHeat {
                crown_fuel_load: load,
                canopy_heat_per_unit_area: canopy,
                crown_fire_heat_per_unit_area: hpua,
                crown_fire_line_intensity: fli,
                crown_flame_length: flame(fli),
            }
        }
        /// Interpolate passive spread and heat, then calculate intensity and flame length.
        pub fn calculate_passive_crown_fire_behavior(
            surface_ros: $t,
            surface_hpua: $t,
            crown_ros: $t,
            canopy_hpua: $t,
            cfb: $t,
        ) -> PassiveCrownBehavior<$t> {
            let ros = surface_ros + cfb * (crown_ros - surface_ros);
            let hpua = surface_hpua + cfb * canopy_hpua;
            let fli = hpua * ros / 60.0;
            PassiveCrownBehavior {
                passive_crown_fire_spread_rate: ros,
                passive_crown_fire_heat_per_unit_area: hpua,
                passive_crown_fire_line_intensity: fli,
                passive_crown_fire_flame_length: flame(fli),
            }
        }
        /// Calculate surface spread required for torching, in ft/min.
        pub fn calculate_surface_fire_critical_spread_rate(
            critical_surface_fli: $t,
            surface_hpua: $t,
        ) -> $t {
            if surface_hpua > 1e-7 {
                (60.0 * critical_surface_fli) / surface_hpua
            } else {
                0.0
            }
        }
        /// Classify strict less-than ratios; exact unity enters the higher class.
        pub fn classify_crown_fire_type(
            transition_ratio: $t,
            active_ratio: $t,
        ) -> FireClassification {
            let fire_type = if transition_ratio < 1.0 {
                if active_ratio < 1.0 {
                    0
                } else {
                    2
                }
            } else if active_ratio < 1.0 {
                1
            } else {
                3
            };
            FireClassification {
                fire_type,
                is_surface_fire: fire_type == 0 || fire_type == 2,
                is_passive_crown: fire_type == 1,
                is_active_crown: fire_type == 3,
            }
        }
        /// Select final behavior, recalculating the surface flame from surface intensity.
        #[allow(clippy::too_many_arguments)]
        pub fn assign_final_crown_fire_behavior(
            is_surface_fire: bool,
            is_passive_crown: bool,
            surface_ros: $t,
            surface_hpua: $t,
            surface_fli: $t,
            passive_ros: $t,
            passive_hpua: $t,
            passive_fli: $t,
            passive_fl: $t,
            crown_ros: $t,
            crown_hpua: $t,
            crown_fli: $t,
            crown_fl: $t,
        ) -> FinalCrownBehavior<$t> {
            if is_surface_fire {
                FinalCrownBehavior {
                    final_spread_rate: surface_ros,
                    final_heat_per_unit_area: surface_hpua,
                    final_fireline_intensity: surface_fli,
                    final_flame_length: flame(surface_fli),
                }
            } else if is_passive_crown {
                FinalCrownBehavior {
                    final_spread_rate: passive_ros,
                    final_heat_per_unit_area: passive_hpua,
                    final_fireline_intensity: passive_fli,
                    final_flame_length: passive_fl,
                }
            } else {
                FinalCrownBehavior {
                    final_spread_rate: crown_ros,
                    final_heat_per_unit_area: crown_hpua,
                    final_fireline_intensity: crown_fli,
                    final_flame_length: crown_fl,
                }
            }
        }
        /// Evaluate the reference crown flame expression in native precision.
        fn flame(fli: $t) -> $t {
            if fli > 0.0 {
                0.2 * fli.powf(2.0 / 3.0)
            } else {
                0.0
            }
        }
    };
}
numeric_stages!(f64);
/// Native f32 arithmetic for helpers retaining Python single precision.
pub mod single {
    use super::{
        CrownHeat, CrownRatios, FinalCrownBehavior, FireClassification, PassiveCrownBehavior,
    };
    use crate::math::single::{np_max, np_min};
    numeric_stages!(f32);
}

/// Retain f32 height conversion and coefficient multiplication before f64 foliar-moisture promotion.
pub fn calculate_critical_surface_fireline_intensity_single_height(
    canopy_base_height: f32,
    moisture_foliar: f64,
) -> f64 {
    let height = crate::math::single::np_max(canopy_base_height * 0.3048, 0.1);
    ((0.010_f32 * height) as f64 * (460.0 + 25.9 * np_max(moisture_foliar, 30.0))).powf(1.5)
        * 0.2886719
}

/// Retain the f32 intensity multiplication before division by f64 surface heat.
pub fn calculate_surface_fire_critical_spread_rate_single_intensity(
    critical_surface_fli: f32,
    surface_hpua: f64,
) -> f64 {
    if surface_hpua > 1e-7 {
        ((60.0_f32 * critical_surface_fli) as f64) / surface_hpua
    } else {
        0.0
    }
}

/// Retain a single-precision surface flame before final selection promotes against f64 crown branches.
#[allow(clippy::too_many_arguments)]
pub fn assign_final_crown_fire_behavior_single_surface(
    is_surface_fire: bool,
    is_passive_crown: bool,
    surface_ros: f32,
    surface_hpua: f32,
    surface_fli: f32,
    passive_ros: f64,
    passive_hpua: f64,
    passive_fli: f64,
    passive_fl: f64,
    crown_ros: f64,
    crown_hpua: f64,
    crown_fli: f64,
    crown_fl: f64,
) -> FinalCrownBehavior {
    let mut result = assign_final_crown_fire_behavior(
        is_surface_fire,
        is_passive_crown,
        surface_ros as f64,
        surface_hpua as f64,
        surface_fli as f64,
        passive_ros,
        passive_hpua,
        passive_fli,
        passive_fl,
        crown_ros,
        crown_hpua,
        crown_fli,
        crown_fl,
    );
    if is_surface_fire {
        result.final_flame_length = single::assign_final_crown_fire_behavior(
            true,
            false,
            surface_ros,
            surface_hpua,
            surface_fli,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        )
        .final_flame_length as f64;
    }
    result
}

/// Extract the three supplied base-unit surface fields used by crown calculations.
pub fn get_surface_fire_inputs(s: &SurfaceResult) -> SurfaceFireInputs {
    SurfaceFireInputs {
        surface_ros: s.spread_rate,
        surface_fli: s.fireline_intensity,
        surface_hpua: s.heat_per_unit_area,
    }
}
/// Model-10 surface results and every source-equivalent intermediate record.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CrownSurfaceResult {
    /// Model-10 particles.
    pub p_crown: Particles,
    /// Model-10 fuelbed intermediates.
    pub ib_crown: FuelbedIntermediates,
    /// Reaction intensity BTU/ftÂ²/min.
    pub ri_crown: f64,
    /// Model-10 surface results with WAF 0.4.
    pub crown_surface: SurfaceResult,
    /// Crown spread ft/min.
    pub crown_fire_spread_rate: f64,
    /// Crown length-to-width ratio.
    pub crown_length_to_width_ratio: f64,
}
/// Prepare model-10 science once with WAF 0.4 and the supplied slope.
pub fn calculate_crown_surface_fire(
    input: &CrownCell,
    fuel: &PreparedFuel,
    options: &SurfaceOptions,
    tweaks: SurfaceTweaks,
) -> CrownSurfaceResult {
    calculate_crown_surface_selected::<0>(input, fuel, options, tweaks)
}

/// Minimal model-10 stages consumed by complete Crown calculations.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CrownSpreadResult {
    /// Prepared particles available for explicit stage inspection.
    pub p_crown: Particles,
    /// Complete fuelbed state used by spread and backsolve.
    pub ib_crown: FuelbedIntermediates,
    /// Reaction intensity in BTU/ftÂ²/min.
    pub ri_crown: f64,
    /// Crown spread in ft/min.
    pub crown_fire_spread_rate: f64,
    /// Crown length-to-width ratio.
    pub crown_length_to_width_ratio: f64,
}

/// Compute Crown's consumed model-10 fields without unrelated Surface finishing.
pub fn calculate_crown_spread(
    input: &CrownCell,
    fuel: &PreparedFuel,
    options: &SurfaceOptions,
    tweaks: SurfaceTweaks,
) -> CrownSpreadResult {
    let result = calculate_crown_surface_selected::<2>(input, fuel, options, tweaks);
    CrownSpreadResult {
        p_crown: result.p_crown,
        ib_crown: result.ib_crown,
        ri_crown: result.ri_crown,
        crown_fire_spread_rate: result.crown_fire_spread_rate,
        crown_length_to_width_ratio: result.crown_length_to_width_ratio,
    }
}

/// Share preparation and spread drivers while retaining full public-helper output.
fn calculate_crown_surface_selected<const FINISH: u8>(
    input: &CrownCell,
    fuel: &PreparedFuel,
    options: &SurfaceOptions,
    tweaks: SurfaceTweaks,
) -> CrownSurfaceResult {
    let x = SurfaceCell {
        fuel_model_grid: 10,
        m1h: input.m1h,
        m10h: input.m10h,
        m100h: input.m100h,
        mlh: input.mlh,
        mlw: input.mlw,
        wind_speed: input.wind_speed,
        wind_direction: input.wind_direction,
        slope: input.slope,
        aspect: input.aspect,
        canopy_height: input.canopy_height,
        user_waf: Some(0.4),
        ..SurfaceCell::default()
    };
    let p_crown =
        surface::build_particle_arrays_from_prepared(fuel, x.m1h, x.m10h, x.m100h, x.mlh, x.mlw);
    let ib_crown = fuel.prepare_fuelbed(&p_crown);
    let ri_crown = tweaks
        .reaction_intensity
        .unwrap_or_else(|| fuel.reaction_intensity(&ib_crown));
    let controls = SurfaceOptions {
        waf_method: WafMethod::UserInput,
        ..*options
    };
    let crown_surface = SurfacePlan::new(controls).run_selected_from_fuelbed::<FINISH>(
        &x,
        &ib_crown,
        SurfaceTweaks {
            reaction_intensity: Some(ri_crown),
            ..tweaks
        },
    );
    let ews = crown_surface.effective_wind_speed;
    CrownSurfaceResult {
        p_crown,
        ib_crown,
        ri_crown,
        crown_fire_spread_rate: 3.34 * crown_surface.spread_rate,
        crown_length_to_width_ratio: if ews > 1e-7 { 1.0 + 0.125 * ews } else { 1.0 },
        crown_surface,
    }
}

/// Back-solve active-crown wind, then calculate model-10 surface spread on flat terrain.
pub fn calculate_crowning_surface_fire_spread_rate(
    ri_crown: f64,
    ib_crown: &FuelbedIntermediates,
    critical_crown_ros: f64,
    canopy_height: f64,
) -> f64 {
    crowning_surface_rate(ri_crown, ib_crown, critical_crown_ros / 3.34, canopy_height)
}
/// Retain the f32 critical-rate division before its promotion against f64 fuelbed intermediates.
pub fn calculate_crowning_surface_fire_spread_rate_single_critical_rate(
    ri_crown: f64,
    ib_crown: &FuelbedIntermediates,
    critical_crown_ros: f32,
    canopy_height: f64,
) -> f64 {
    crowning_surface_rate(
        ri_crown,
        ib_crown,
        (critical_crown_ros / 3.34) as f64,
        canopy_height,
    )
}
/// Calculate the crowning surface rate from a precision-normalized model-10 target rate.
fn crowning_surface_rate(
    ri_crown: f64,
    ib_crown: &FuelbedIntermediates,
    r10: f64,
    canopy_height: f64,
) -> f64 {
    let ros0 = ri_crown * ib_crown.propagating_flux
        / if ib_crown.heat_sink > 1e-7 {
            ib_crown.heat_sink
        } else {
            1.0
        };
    // Keep the reference coefficient's decimal spelling for traceability.
    #[allow(clippy::excessive_precision)]
    let a = if ros0 > 1e-7 {
        crate::math::np_max(r10 / ros0 - 1.0, 0.0) / 0.0016102128596515481
    } else {
        0.0
    };
    let u_mid = if a > 0.0 {
        a.powf(1.0 / 1.4308256324729873)
    } else {
        0.0
    };
    let mut x = SurfaceCell::new(10);
    x.wind_speed = u_mid / 0.4;
    x.canopy_height = canopy_height;
    x.user_waf = Some(0.4);
    let options = SurfaceOptions {
        waf_method: WafMethod::UserInput,
        ..SurfaceOptions::default()
    };
    SurfacePlan::new(options)
        .run_selected_from_fuelbed::<1>(
            &x,
            ib_crown,
            SurfaceTweaks {
                reaction_intensity: Some(ri_crown),
                ..SurfaceTweaks::default()
            },
        )
        .spread_rate
}

macro_rules! crown_result {
    ($($field:ident),+) => {
        /// Complete Python crown result, preserving the classification as an integer.
        #[derive(Clone,Copy,Debug,Default,PartialEq)]
        #[non_exhaustive]
        pub struct CrownResult {
            $(#[doc=concat!("Python `",stringify!($field),"` result in component base units.")] pub $field:f64,)+
            /// Exact classification: 0=surface, 1=torching, 2=conditional crown, 3=crowning.
            pub fire_type:i32,
        }
        impl CrownResult {
            /// Construct an empty result for explicit native stage use.
            pub fn new()->Self {Self::default()}
            /// Return all floating-point fields under their exact Python keys, excluding integer fire_type.
            pub fn named_values(&self)->[(&'static str,f64);21] {[$((stringify!($field),self.$field)),+]}
        }
    }
}
crown_result!(
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
