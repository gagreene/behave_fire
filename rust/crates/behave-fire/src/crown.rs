//! Crown fire stages in reference base units: ft, ft/min, lb/ft³, BTU/ft², BTU/ft/s.
use crate::math::{np_max, np_min};
pub mod masked;
mod record;
use crate::{
    fuel_models::FuelModels,
    surface::{self, FuelbedIntermediates, Particles, SurfaceInput, SurfaceResult},
    units::{SlopeUnit, SpeedUnit},
    BehaveError,
};
pub use record::{run_with_surface_inputs, MaskedCrownResult};

/// Scalar crown inputs with an already calculated component-base-unit surface result.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct CrownInput {
    /// Supplied base-unit surface results.
    pub surface_results: SurfaceResult,
    /// Original grid model identifier; crown fuel always uses model 10.
    pub fuel_model_grid: i32,
    /// One-hour moisture fraction.
    pub m1h: f64,
    /// Ten-hour moisture fraction.
    pub m10h: f64,
    /// Hundred-hour moisture fraction.
    pub m100h: f64,
    /// Live herb moisture fraction.
    pub mlh: f64,
    /// Live woody moisture fraction.
    pub mlw: f64,
    /// Wind speed in selected units.
    pub wind_speed: f64,
    /// Wind speed units.
    pub wind_speed_units: SpeedUnit,
    /// Wind direction in degrees.
    pub wind_direction: f64,
    /// Wind orientation name.
    pub wind_orientation_mode: String,
    /// Slope in selected units.
    pub slope: f64,
    /// Slope units.
    pub slope_units: SlopeUnit,
    /// Aspect in degrees.
    pub aspect: f64,
    /// Canopy base height in feet.
    pub canopy_base_height: f64,
    /// Canopy height in feet.
    pub canopy_height: f64,
    /// Canopy bulk density in lb/ft³.
    pub canopy_bulk_density: f64,
    /// Foliar moisture percent, default 100.
    pub moisture_foliar: f64,
}
impl Default for CrownInput {
    /// Construct calm, flat inputs with zero canopy and 100 percent foliar moisture.
    fn default() -> Self {
        let s = SurfaceInput::default();
        Self {
            surface_results: SurfaceResult::default(),
            fuel_model_grid: s.fuel_model_grid,
            m1h: s.m1h,
            m10h: s.m10h,
            m100h: s.m100h,
            mlh: s.mlh,
            mlw: s.mlw,
            wind_speed: s.wind_speed,
            wind_speed_units: s.wind_speed_units,
            wind_direction: s.wind_direction,
            wind_orientation_mode: s.wind_orientation_mode,
            slope: s.slope,
            slope_units: s.slope_units,
            aspect: s.aspect,
            canopy_base_height: 0.0,
            canopy_height: 0.0,
            canopy_bulk_density: 0.0,
            moisture_foliar: 100.0,
        }
    }
}
impl CrownInput {
    /// Construct inputs around a supplied base-unit surface result.
    pub fn new(surface_results: SurfaceResult) -> Self {
        Self {
            surface_results,
            ..Self::default()
        }
    }
}
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
            pub fn named_values(&self) -> Vec<(&'static str,T)> { vec![$((stringify!($field),self.$field)),+] }
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

/// Scalar normalized input names matching the Python coercion mapping.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct CoercedCrownInputs {
    /// Original grid identifier.
    pub fm_grid: i32,
    /// Slope degrees.
    pub slope_deg: f64,
    /// Aspect degrees.
    pub aspect: f64,
    /// Wind direction degrees.
    pub wind_dir: f64,
    /// Canopy base height feet.
    pub cbh: f64,
    /// Canopy height feet.
    pub ch: f64,
    /// Canopy density lb/ft³.
    pub cbd: f64,
    /// Foliar moisture percent.
    pub mf_pct: f64,
    /// Wind speed in selected units.
    pub wind_speed: f64,
}
/// Normalize scalar input slope and retain the other base-unit values.
pub fn coerce_crown_fire_inputs(input: &CrownInput) -> CoercedCrownInputs {
    CoercedCrownInputs {
        fm_grid: input.fuel_model_grid,
        slope_deg: input.slope_units.to_base(input.slope),
        aspect: input.aspect,
        wind_dir: input.wind_direction,
        cbh: input.canopy_base_height,
        ch: input.canopy_height,
        cbd: input.canopy_bulk_density,
        mf_pct: input.moisture_foliar,
        wind_speed: input.wind_speed,
    }
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
    /// Reaction intensity BTU/ft²/min.
    pub ri_crown: f64,
    /// Model-10 surface results with WAF 0.4.
    pub crown_surface: SurfaceResult,
    /// Crown spread ft/min.
    pub crown_fire_spread_rate: f64,
    /// Crown length-to-width ratio.
    pub crown_length_to_width_ratio: f64,
}
/// Run model 10 with WAF 0.4 and supplied slope, as current Python does.
pub fn calculate_crown_surface_fire(
    input: &CrownInput,
    fuels: &FuelModels,
) -> Result<CrownSurfaceResult, BehaveError> {
    let mut x = SurfaceInput::new(10);
    x.m1h = input.m1h;
    x.m10h = input.m10h;
    x.m100h = input.m100h;
    x.mlh = input.mlh;
    x.mlw = input.mlw;
    x.wind_speed = input.wind_speed;
    x.wind_speed_units = input.wind_speed_units;
    x.wind_direction = input.wind_direction;
    x.wind_orientation_mode = input.wind_orientation_mode.clone();
    x.slope = input.slope_units.to_base(input.slope);
    x.slope_units = SlopeUnit::Degrees;
    x.aspect = input.aspect;
    x.canopy_height = input.canopy_height;
    x.waf_method = "UserInput".into();
    x.user_waf = Some(0.4);
    let p_crown = surface::build_particle_arrays(fuels, 10, x.m1h, x.m10h, x.m100h, x.mlh, x.mlw);
    let ib_crown = surface::calculate_fuelbed_intermediates(&p_crown);
    let ri_crown = surface::calculate_reaction_intensity(&ib_crown);
    let crown_surface = surface::run(&x, fuels)?;
    let ews = crown_surface.effective_wind_speed;
    Ok(CrownSurfaceResult {
        p_crown,
        ib_crown,
        ri_crown,
        crown_fire_spread_rate: 3.34 * crown_surface.spread_rate,
        crown_length_to_width_ratio: if ews > 1e-7 { 1.0 + 0.125 * ews } else { 1.0 },
        crown_surface,
    })
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
    let mut x = SurfaceInput::new(10);
    x.wind_speed = u_mid / 0.4;
    x.wind_speed_units = SpeedUnit::FeetPerMinute;
    x.canopy_height = canopy_height;
    x.waf_method = "UserInput".into();
    x.user_waf = Some(0.4);
    surface::run_surface_fire(ri_crown, ib_crown, &x).spread_rate
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

/// Run every crown stage from the supplied base-unit surface result and fuel-table snapshot.
pub fn run_crown_fire(
    surface_results: &SurfaceResult,
    input: &CrownInput,
    fuels: &FuelModels,
) -> Result<CrownResult, BehaveError> {
    let i = coerce_crown_fire_inputs(input);
    let s = get_surface_fire_inputs(surface_results);
    let c = calculate_crown_surface_fire(input, fuels)?;
    let h = calculate_crown_heat_and_intensity(
        s.surface_hpua,
        c.crown_fire_spread_rate,
        i.cbh,
        i.ch,
        i.cbd,
    );
    let crit_fli = calculate_critical_surface_fireline_intensity(i.cbh, i.mf_pct);
    let crit_ros = calculate_critical_crown_fire_spread_rate(i.cbd);
    let ratios =
        calculate_crown_fire_ratios(s.surface_fli, c.crown_fire_spread_rate, crit_fli, crit_ros);
    let class = classify_crown_fire_type(
        ratios.crown_fire_transition_ratio,
        ratios.crown_fire_active_ratio,
    );
    let surface_crit = calculate_surface_fire_critical_spread_rate(crit_fli, s.surface_hpua);
    let crowning =
        calculate_crowning_surface_fire_spread_rate(c.ri_crown, &c.ib_crown, crit_ros, i.ch);
    let cfb = calculate_crown_fraction_burned(s.surface_ros, surface_crit, crowning);
    let p = calculate_passive_crown_fire_behavior(
        s.surface_ros,
        s.surface_hpua,
        c.crown_fire_spread_rate,
        h.canopy_heat_per_unit_area,
        cfb,
    );
    let f = assign_final_crown_fire_behavior(
        class.is_surface_fire,
        class.is_passive_crown,
        s.surface_ros,
        s.surface_hpua,
        s.surface_fli,
        p.passive_crown_fire_spread_rate,
        p.passive_crown_fire_heat_per_unit_area,
        p.passive_crown_fire_line_intensity,
        p.passive_crown_fire_flame_length,
        c.crown_fire_spread_rate,
        h.crown_fire_heat_per_unit_area,
        h.crown_fire_line_intensity,
        h.crown_flame_length,
    );
    Ok(CrownResult {
        crown_fire_spread_rate: c.crown_fire_spread_rate,
        crown_flame_length: h.crown_flame_length,
        crown_fire_line_intensity: h.crown_fire_line_intensity,
        crown_fire_heat_per_unit_area: h.crown_fire_heat_per_unit_area,
        canopy_heat_per_unit_area: h.canopy_heat_per_unit_area,
        crown_critical_surface_fire_line_intensity: crit_fli,
        crown_critical_fire_spread_rate: crit_ros,
        crown_fire_transition_ratio: ratios.crown_fire_transition_ratio,
        crown_fire_active_ratio: ratios.crown_fire_active_ratio,
        crown_length_to_width_ratio: c.crown_length_to_width_ratio,
        fire_type: class.fire_type,
        surface_fire_critical_spread_rate: surface_crit,
        crowning_surface_fire_spread_rate: crowning,
        crown_fraction_burned: cfb,
        passive_crown_fire_spread_rate: p.passive_crown_fire_spread_rate,
        passive_crown_fire_heat_per_unit_area: p.passive_crown_fire_heat_per_unit_area,
        passive_crown_fire_line_intensity: p.passive_crown_fire_line_intensity,
        passive_crown_fire_flame_length: p.passive_crown_fire_flame_length,
        final_spread_rate: f.final_spread_rate,
        final_heat_per_unit_area: f.final_heat_per_unit_area,
        final_fireline_intensity: f.final_fireline_intensity,
        final_flame_length: f.final_flame_length,
    })
}
/// Run the crown pipeline using the typed input's supplied surface result.
pub fn run(input: &CrownInput, fuels: &FuelModels) -> Result<CrownResult, BehaveError> {
    run_crown_fire(&input.surface_results, input, fuels)
}
