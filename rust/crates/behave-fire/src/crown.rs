//! Crown fire stages in reference base units: ft, ft/min, lb/ftÂ³, BTU/ftÂ², BTU/ft/s.
use crate::math::np_max;

use crate::{
    engine::{
        CrownCell, PreparedFuel, SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceTweaks, WafMethod,
    },
    surface::{self, FuelbedIntermediates, Particles, SurfaceResult},
};

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
        /// Calculate canopy fuel load without supplied Surface or spread inputs.
        pub fn calculate_canopy_fuel_load(base: $t, height: $t, density: $t) -> $t {
            np_max(density * (height - base), 0.0)
        }
        /// Calculate canopy heat from actual fuel load in base units.
        pub fn calculate_canopy_heat_per_unit_area(load: $t) -> $t {
            load * 8000.0
        }
        /// Calculate Van Wagner critical active spread rate in ft/min.
        pub fn calculate_critical_crown_fire_spread_rate(canopy_bulk_density: $t) -> $t {
            if canopy_bulk_density.is_nan() {
                return <$t>::NAN;
            }
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
                crown_fire_transition_ratio: if critical_surface_fli.is_nan() {
                    <$t>::NAN
                } else if critical_surface_fli > 1e-7 {
                    surface_fli / critical_surface_fli
                } else {
                    0.0
                },
                crown_fire_active_ratio: if critical_crown_ros.is_nan() {
                    <$t>::NAN
                } else if critical_crown_ros > 1e-7 {
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
            if d.is_nan() {
                return <$t>::NAN;
            }
            let v = if d > 1e-7 {
                (surface_ros - surface_critical_ros) / d
            } else {
                0.0
            };
            v.clamp(0.0, 1.0)
        }
        /// Calculate canopy loading, heat, active crown intensity, and flame length.
        pub fn calculate_crown_heat_and_intensity(
            surface_hpua: $t,
            crown_ros: $t,
            canopy_base_height: $t,
            canopy_height: $t,
            canopy_bulk_density: $t,
        ) -> CrownHeat<$t> {
            let load =
                calculate_canopy_fuel_load(canopy_base_height, canopy_height, canopy_bulk_density);
            let canopy = calculate_canopy_heat_per_unit_area(load);
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
        /// Interpolate passive spread without unrelated heat inputs.
        pub fn calculate_passive_spread_rate(surface: $t, crown: $t, fraction: $t) -> $t {
            surface + fraction * (crown - surface)
        }
        /// Interpolate heat without a spread or flame dependency.
        pub fn calculate_passive_heat_per_unit_area(surface: $t, canopy: $t, fraction: $t) -> $t {
            surface + fraction * canopy
        }
        /// Preserve the reference passive-intensity product order.
        pub fn calculate_passive_intensity(heat: $t, rate: $t) -> $t {
            heat * rate / 60.0
        }
        /// Interpolate passive spread and heat, then calculate intensity and flame length.
        pub fn calculate_passive_crown_fire_behavior(
            surface_ros: $t,
            surface_hpua: $t,
            crown_ros: $t,
            canopy_hpua: $t,
            cfb: $t,
        ) -> PassiveCrownBehavior<$t> {
            let ros = calculate_passive_spread_rate(surface_ros, crown_ros, cfb);
            let hpua = calculate_passive_heat_per_unit_area(surface_hpua, canopy_hpua, cfb);
            let fli = calculate_passive_intensity(hpua, ros);
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
            if surface_hpua.is_nan() {
                return <$t>::NAN;
            }
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
            if transition_ratio.is_nan() || active_ratio.is_nan() {
                return FireClassification {
                    fire_type: -1,
                    is_surface_fire: false,
                    is_passive_crown: false,
                    is_active_crown: false,
                };
            }
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
            if fli.is_nan() {
                <$t>::NAN
            } else if fli > 0.0 {
                0.2 * fli.powf(2.0 / 3.0)
            } else {
                0.0
            }
        }
    };
}
numeric_stages!(f64);

/// F64 availability rules shared by complete and restricted evaluation.
pub mod available {
    use super::*;
    /// Guard a critical denominator without converting missingness to zero.
    pub fn ratio(numerator: f64, denominator: f64) -> f64 {
        if denominator.is_nan() {
            f64::NAN
        } else if denominator > 1e-7 {
            numerator / denominator
        } else {
            0.
        }
    }
    /// Keep missing canopy density distinct from valid zero density.
    pub fn critical_rate(density: f64) -> f64 {
        if density.is_nan() {
            f64::NAN
        } else {
            calculate_critical_crown_fire_spread_rate(density)
        }
    }
    /// Keep missing Surface heat distinct from the finite zero threshold.
    pub fn surface_critical(intensity: f64, heat: f64) -> f64 {
        if heat.is_nan() {
            f64::NAN
        } else {
            calculate_surface_fire_critical_spread_rate(intensity, heat)
        }
    }
    /// Preserve the finite fraction threshold and unknown denominator.
    pub fn fraction(surface: f64, critical: f64, crowning: f64) -> f64 {
        if (crowning - critical).is_nan() {
            f64::NAN
        } else {
            calculate_crown_fraction_burned(surface, critical, crowning)
        }
    }
    /// Classify only when both actual ratios are available.
    pub fn classify(transition: f64, active: f64) -> FireClassification {
        if transition.is_nan() || active.is_nan() {
            FireClassification {
                fire_type: -1,
                is_surface_fire: false,
                is_passive_crown: false,
                is_active_crown: false,
            }
        } else {
            classify_crown_fire_type(transition, active)
        }
    }
    /// Preserve missing intensity rather than the finite nonpositive literal.
    pub fn flame_length(intensity: f64) -> f64 {
        if intensity.is_nan() {
            f64::NAN
        } else {
            flame(intensity)
        }
    }
    /// Prepare actual model-10 moisture/fuel dependencies for fixed-WAF backsolve.
    pub fn crowning_rate(moisture: [f64; 5], fuel: &PreparedFuel, critical: f64) -> f64 {
        let p = surface::build_particle_arrays_from_prepared(
            fuel,
            moisture[0],
            moisture[1],
            moisture[2],
            moisture[3],
            moisture[4],
        );
        let bed = fuel.prepare_fuelbed_available(&p);
        let reaction = fuel.reaction_intensity_available(&bed);
        if critical.is_nan()
            || reaction.is_nan()
            || bed.heat_sink.is_nan()
            || bed.propagating_flux.is_nan()
        {
            return f64::NAN;
        }
        let wind = fuel.wind_terms();
        crowning_without_geometry(reaction, &bed, critical, wind.as_ref())
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
    calculate_crown_surface_selected::<0, true>(
        input,
        fuel,
        options,
        tweaks,
        fuel.wind_terms().as_ref(),
    )
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
    crown_spread_with_policy::<true>(input, fuel, options, tweaks, fuel.wind_terms().as_ref())
}

/// Compute model-10 spread under the f64 missing-data contract.
pub(crate) fn calculate_crown_spread_available(
    input: &CrownCell,
    fuel: &PreparedFuel,
    options: &SurfaceOptions,
    tweaks: SurfaceTweaks,
    wind: Option<&surface::PreparedWind>,
) -> CrownSpreadResult {
    crown_spread_with_policy::<true>(input, fuel, options, tweaks, wind)
}
fn crown_spread_with_policy<const STRICT: bool>(
    input: &CrownCell,
    fuel: &PreparedFuel,
    options: &SurfaceOptions,
    tweaks: SurfaceTweaks,
    wind: Option<&surface::PreparedWind>,
) -> CrownSpreadResult {
    let result = calculate_crown_surface_selected::<2, STRICT>(input, fuel, options, tweaks, wind);
    CrownSpreadResult {
        p_crown: result.p_crown,
        ib_crown: result.ib_crown,
        ri_crown: result.ri_crown,
        crown_fire_spread_rate: result.crown_fire_spread_rate,
        crown_length_to_width_ratio: result.crown_length_to_width_ratio,
    }
}
/// Evaluate only real spatial/moisture dependencies under the fixed Crown WAF.
pub(crate) fn calculate_crown_spread_spatial(
    input: &crate::engine::CrownSpatial,
    fuel: &PreparedFuel,
    options: &SurfaceOptions,
) -> CrownSpreadResult {
    // Fixed FM10 configuration has no canopy-geometry dependency: WAF=0.4.
    // SurfaceCell's remaining defaults are model configuration, never substitute
    // caller values; this path does not read those unused geometry fields.
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
        user_waf: Some(0.4),
        ..SurfaceCell::default()
    };
    let wind = fuel.wind_terms();
    let result = calculate_fm10_surface::<2, true>(
        &x,
        fuel,
        options,
        SurfaceTweaks::default(),
        wind.as_ref(),
    );
    CrownSpreadResult {
        p_crown: result.p_crown,
        ib_crown: result.ib_crown,
        ri_crown: result.ri_crown,
        crown_fire_spread_rate: result.crown_fire_spread_rate,
        crown_length_to_width_ratio: result.crown_length_to_width_ratio,
    }
}

/// Share preparation and spread drivers while retaining full public-helper output.
fn calculate_crown_surface_selected<const FINISH: u8, const STRICT: bool>(
    input: &CrownCell,
    fuel: &PreparedFuel,
    options: &SurfaceOptions,
    tweaks: SurfaceTweaks,
    wind: Option<&surface::PreparedWind>,
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
    calculate_fm10_surface::<FINISH, STRICT>(&x, fuel, options, tweaks, wind)
}

/// Share the model-10 equations for complete and dependency-restricted callers.
fn calculate_fm10_surface<const FINISH: u8, const STRICT: bool>(
    x: &SurfaceCell,
    fuel: &PreparedFuel,
    options: &SurfaceOptions,
    tweaks: SurfaceTweaks,
    wind: Option<&surface::PreparedWind>,
) -> CrownSurfaceResult {
    let p_crown =
        surface::build_particle_arrays_from_prepared(fuel, x.m1h, x.m10h, x.m100h, x.mlh, x.mlw);
    let ib_crown = if STRICT {
        fuel.prepare_fuelbed_available(&p_crown)
    } else {
        fuel.prepare_fuelbed(&p_crown)
    };
    let ri_crown = tweaks.reaction_intensity.unwrap_or_else(|| {
        if STRICT {
            fuel.reaction_intensity_available(&ib_crown)
        } else {
            fuel.reaction_intensity(&ib_crown)
        }
    });
    let controls = SurfaceOptions {
        waf_method: WafMethod::UserInput,
        ..*options
    };
    let configured = SurfaceTweaks {
        reaction_intensity: Some(ri_crown),
        ..tweaks
    };
    let plan = SurfacePlan::new(controls);
    let crown_surface = plan.run_with_wind::<FINISH, true>(x, &ib_crown, configured, wind);
    let ews = crown_surface.effective_wind_speed;
    CrownSurfaceResult {
        p_crown,
        ib_crown,
        ri_crown,
        crown_fire_spread_rate: 3.34 * crown_surface.spread_rate,
        crown_length_to_width_ratio: if STRICT && ews.is_nan() {
            f64::NAN
        } else if ews > 1e-7 {
            1.0 + 0.125 * ews
        } else {
            1.0
        },
        crown_surface,
    }
}

/// Back-solve active-crown wind, then calculate model-10 surface spread on flat terrain.
pub fn calculate_crowning_surface_fire_spread_rate(
    ri_crown: f64,
    ib_crown: &FuelbedIntermediates,
    critical_crown_ros: f64,
    _canopy_height: f64,
) -> f64 {
    crowning_surface_rate(ri_crown, ib_crown, critical_crown_ros / 3.34, None)
}

/// Reuse the same physical wind context for the backsolved Surface calculation.
pub(crate) fn crowning_with_wind(
    ri: f64,
    bed: &FuelbedIntermediates,
    critical: f64,
    _height: f64,
    wind: Option<&surface::PreparedWind>,
) -> f64 {
    crowning_without_geometry(ri, bed, critical, wind)
}
/// Fixed WAF is the entire wind-height configuration of this backsolve.
fn crowning_without_geometry(
    ri: f64,
    bed: &FuelbedIntermediates,
    critical: f64,
    wind: Option<&surface::PreparedWind>,
) -> f64 {
    crowning_surface_rate(ri, bed, critical / 3.34, wind)
}
/// Calculate the crowning surface rate from a precision-normalized model-10 target rate.
fn crowning_surface_rate(
    ri_crown: f64,
    ib_crown: &FuelbedIntermediates,
    r10: f64,
    wind: Option<&surface::PreparedWind>,
) -> f64 {
    let wind = wind.filter(|context| context.matches(ib_crown));
    let coefficients = wind
        .map(|v| v.coefficients)
        .unwrap_or_else(|| surface::calculate_surface_wind_coefficients(ib_crown.sigma));
    crowning_from_inputs(
        &CrowningInputs {
            reaction: ri_crown,
            sigma: ib_crown.sigma,
            packing: ib_crown.packing_ratio,
            relative_packing: ib_crown.relative_packing_ratio,
            flux: ib_crown.propagating_flux,
            heat_sink: ib_crown.heat_sink,
            defined: ib_crown.is_defined,
            target: r10,
            wind_b: coefficients.wind_b,
            wind_c: coefficients.wind_c,
            wind_e: coefficients.wind_e,
        },
        wind,
    )
}

/// Only the actual declared physical inputs of the fixed-WAF backsolve.
pub(crate) struct CrowningInputs {
    pub reaction: f64,
    pub sigma: f64,
    pub packing: f64,
    pub relative_packing: f64,
    pub flux: f64,
    pub heat_sink: f64,
    pub defined: bool,
    pub target: f64,
    pub wind_b: f64,
    pub wind_c: f64,
    pub wind_e: f64,
}

/// Share backsolve equations with staged, potentially injected wind coefficients.
pub(crate) fn crowning_from_inputs(
    v: &CrowningInputs,
    wind: Option<&surface::PreparedWind>,
) -> f64 {
    let ri_crown = v.reaction;
    let r10 = v.target;
    let ros0 = ri_crown * v.flux / if v.heat_sink > 1e-7 { v.heat_sink } else { 1.0 };
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
    let r0 = surface::calculate_no_wind_no_slope_spread_rate(ri_crown, v.flux, v.heat_sink);
    let mid = surface::calculate_midflame_wind_speed(0.4, u_mid / 0.4);
    let phi = wind
        .map(|context| context.wind_phi(mid))
        .unwrap_or_else(|| {
            surface::calculate_wind_phi(
                v.sigma,
                mid,
                v.wind_b,
                v.wind_c,
                v.relative_packing,
                v.wind_e,
            )
        });
    let slope =
        surface::calculate_slope_phi(0., crate::units::SlopeUnit::Degrees, v.packing, ri_crown);
    let vector = surface::calculate_surface_rate_vector(
        r0,
        phi,
        slope,
        0.,
        crate::engine::WindOrientation::Upslope,
        0.,
    );
    if v.defined {
        surface::calculate_forward_spread_rate(r0, vector.rate_vector)
    } else {
        0.
    }
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
