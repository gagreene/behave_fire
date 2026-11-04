//! Reaction and residence stages, including native float32 arithmetic.
use super::FuelbedIntermediates;
use crate::math::np_min;

macro_rules! retaining_reaction {
    ($t:ty) => {
        /// Calculate no-wind, no-slope spread in ft/min from RI, flux, and heat sink.
        pub fn calculate_no_wind_no_slope_spread_rate(
            ri: $t,
            propagating_flux: $t,
            heat_sink: $t,
        ) -> $t {
            let safe = if heat_sink > 1e-7 { heat_sink } else { 1.0 };
            if heat_sink > 1e-7 {
                ri * propagating_flux / safe
            } else {
                0.0
            }
        }
        /// Calculate residence time in minutes from sigma in ft²/ft³.
        pub fn calculate_residence_time(sigma: $t) -> $t {
            let safe = if sigma > 1e-7 { sigma } else { 1.0 };
            if sigma > 1e-7 {
                384.0 / safe
            } else {
                0.0
            }
        }
        /// Moisture damping coefficient; preserve the strict extinction threshold and polynomial order.
        fn eta_m(moisture: $t, moe: $t) -> $t {
            let rm = if moe > 1e-7 { moisture / moe } else { 1.0 };
            if moe < 1e-7 || rm >= 1.0 {
                0.0
            } else {
                1.0 - 2.59 * rm + 5.11 * rm.powi(2) - 3.52 * rm.powi(3)
            }
        }
        /// Mineral damping coefficient with NumPy NaN/threshold behavior.
        fn eta_s(silica: $t) -> $t {
            let safe = if silica > 1e-7 { silica } else { 1.0 };
            if silica > 1e-7 {
                np_min(0.174 / safe.powf(0.19), 1.0)
            } else {
                0.0
            }
        }
    };
}
retaining_reaction!(f64);

/// Calculate reaction intensity in BTU/ft²/min using all weighted fuelbed fields.
pub fn calculate_reaction_intensity(ib: &FuelbedIntermediates) -> f64 {
    let valid = ib.sigma > 1e-7 && ib.relative_packing_ratio > 0.0;
    let sigma = if valid { ib.sigma } else { 1.0 };
    let rpr = if valid {
        ib.relative_packing_ratio
    } else {
        1.0
    };
    let aa = 133.0 / sigma.powf(0.7913);
    let gamma_max = sigma.powf(1.5) / (495.0 + 0.0594 * sigma.powf(1.5));
    let gamma = gamma_max * rpr.powf(aa) * (aa * (1.0 - rpr)).exp();
    let dead = gamma
        * ib.wfl_dead
        * ib.w_heat_dead
        * eta_m(ib.w_moist_dead, ib.moe_dead)
        * eta_s(ib.w_silica_dead);
    let live = gamma
        * ib.wfl_live
        * ib.w_heat_live
        * eta_m(ib.w_moist_live, ib.moe_live)
        * eta_s(ib.w_silica_live);
    if valid {
        dead + live
    } else {
        0.0
    }
}
/// Return reaction intensity, normalized to f64 as in Python.
pub fn calculate_reaction_intensity_output(reaction_intensity: f64) -> f64 {
    reaction_intensity
}
/// Calculate heat per unit area in BTU/ft² from reaction intensity and residence minutes.
pub fn calculate_heat_per_unit_area(reaction_intensity: f64, residence_time: f64) -> f64 {
    reaction_intensity * residence_time
}

pub(super) mod single {
    use crate::math::single::np_min;
    retaining_reaction!(f32);
    /// Calculate reaction intensity retaining float32 arithmetic in every stage.
    ///
    /// The typed f64 record is a transport container; each value is narrowed before arithmetic.
    pub fn calculate_reaction_intensity(ib: &super::FuelbedIntermediates) -> f32 {
        let valid = (ib.sigma as f32) > 1e-7 && (ib.relative_packing_ratio as f32) > 0.0;
        let sigma = if valid { ib.sigma as f32 } else { 1.0 };
        let rpr = if valid {
            ib.relative_packing_ratio as f32
        } else {
            1.0
        };
        let aa = 133.0 / sigma.powf(0.7913);
        let gm = sigma.powf(1.5) / (495.0 + 0.0594 * sigma.powf(1.5));
        let gamma = gm * rpr.powf(aa) * (aa * (1.0 - rpr)).exp();
        let dead = gamma
            * (ib.wfl_dead as f32)
            * (ib.w_heat_dead as f32)
            * eta_m(ib.w_moist_dead as f32, ib.moe_dead as f32)
            * eta_s(ib.w_silica_dead as f32);
        let live = gamma
            * (ib.wfl_live as f32)
            * (ib.w_heat_live as f32)
            * eta_m(ib.w_moist_live as f32, ib.moe_live as f32)
            * eta_s(ib.w_silica_live as f32);
        if valid {
            dead + live
        } else {
            0.0
        }
    }
}
