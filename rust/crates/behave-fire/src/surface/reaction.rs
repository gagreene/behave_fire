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
    ReactionCoefficients::new(ib).run(ib)
}

/// Calculate reaction using the availability policy of configured f64 models.
pub(crate) fn calculate_reaction_intensity_available(ib: &FuelbedIntermediates) -> f64 {
    ReactionCoefficients::new(ib).run_available(ib)
}

/// Moisture-independent reaction coefficients, retaining left-to-right products.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReactionInputs {
    pub sigma: f64,
    pub relative_packing_ratio: f64,
    pub wfl_dead: f64,
    pub wfl_live: f64,
    pub w_heat_dead: f64,
    pub w_heat_live: f64,
    pub w_silica_dead: f64,
    pub w_silica_live: f64,
    pub w_moist_dead: f64,
    pub w_moist_live: f64,
    pub moe_dead: f64,
    pub moe_live: f64,
}
impl From<&FuelbedIntermediates> for ReactionInputs {
    fn from(v: &FuelbedIntermediates) -> Self {
        Self {
            sigma: v.sigma,
            relative_packing_ratio: v.relative_packing_ratio,
            wfl_dead: v.wfl_dead,
            wfl_live: v.wfl_live,
            w_heat_dead: v.w_heat_dead,
            w_heat_live: v.w_heat_live,
            w_silica_dead: v.w_silica_dead,
            w_silica_live: v.w_silica_live,
            w_moist_dead: v.w_moist_dead,
            w_moist_live: v.w_moist_live,
            moe_dead: v.moe_dead,
            moe_live: v.moe_live,
        }
    }
}
/// Evaluate only the declared reaction dependencies, with no invented fuelbed fields.
pub(crate) fn reaction_from_inputs(inputs: &ReactionInputs) -> f64 {
    ReactionCoefficients::from_inputs(inputs).run_inputs::<true>(inputs)
}

/// Moisture-independent reaction coefficients, retaining left-to-right products.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ReactionCoefficients {
    valid: bool,
    dead: f64,
    live: f64,
    silica_dead: f64,
    silica_live: f64,
}
impl ReactionCoefficients {
    pub(crate) fn new(ib: &FuelbedIntermediates) -> Self {
        Self::from_inputs(&ib.into())
    }
    fn from_inputs(ib: &ReactionInputs) -> Self {
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
        Self {
            valid,
            dead: gamma * ib.wfl_dead * ib.w_heat_dead,
            live: gamma * ib.wfl_live * ib.w_heat_live,
            silica_dead: eta_s(ib.w_silica_dead),
            silica_live: eta_s(ib.w_silica_live),
        }
    }
    pub(crate) fn run(&self, ib: &FuelbedIntermediates) -> f64 {
        self.run_selected::<false>(ib)
    }
    /// Retain cached finite coefficients without treating unknown guards as false.
    pub(crate) fn run_available(&self, ib: &FuelbedIntermediates) -> f64 {
        self.run_selected::<true>(ib)
    }
    fn run_selected<const STRICT: bool>(&self, ib: &FuelbedIntermediates) -> f64 {
        self.run_inputs::<STRICT>(&ib.into())
    }
    fn run_inputs<const STRICT: bool>(&self, ib: &ReactionInputs) -> f64 {
        if STRICT {
            match crate::engine::missing::all_above(&[
                (ib.sigma, 1e-7),
                (ib.relative_packing_ratio, 0.),
            ]) {
                None => return f64::NAN,
                Some(false) => return 0.,
                Some(true) => {}
            }
        }
        let moisture_dead = if STRICT && ib.moe_dead.is_nan() {
            f64::NAN
        } else {
            eta_m(ib.w_moist_dead, ib.moe_dead)
        };
        let moisture_live = if STRICT && ib.moe_live.is_nan() {
            f64::NAN
        } else {
            eta_m(ib.w_moist_live, ib.moe_live)
        };
        let silica_dead = if STRICT && ib.w_silica_dead.is_nan() {
            f64::NAN
        } else {
            self.silica_dead
        };
        let silica_live = if STRICT && ib.w_silica_live.is_nan() {
            f64::NAN
        } else {
            self.silica_live
        };
        let dead = self.dead * moisture_dead * silica_dead;
        let live = self.live * moisture_live * silica_live;
        if self.valid {
            dead + live
        } else {
            0.0
        }
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
