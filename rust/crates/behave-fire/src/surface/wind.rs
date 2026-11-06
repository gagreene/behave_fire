//! Wind and slope stages with enum-valued direction, height, and adjustment controls.
use super::{RateVector, WindCoefficients};
use crate::{
    math::np_min,
    units::{SlopeUnit, SpeedUnit},
};

macro_rules! retaining_wind {
    ($t:ty, $speed:ty) => {
        /// Calculate effective wind speed in mph for ellipse shape.
        pub fn calculate_effective_wind_speed(
            spread_rate: $t,
            no_wind_no_slope_spread_rate: $t,
            wind_b: $t,
            wind_c: $t,
            relative_packing_ratio: $t,
            wind_e: $t,
        ) -> $speed {
            let r0 = if no_wind_no_slope_spread_rate > 1e-7 {
                no_wind_no_slope_spread_rate
            } else {
                1.0
            };
            let phi = if no_wind_no_slope_spread_rate > 1e-7 {
                spread_rate / r0 - 1.0
            } else {
                0.0
            };
            let rpr = if relative_packing_ratio > 1e-7 {
                relative_packing_ratio
            } else {
                1.0
            };
            let wc = if wind_c > 1e-7 { wind_c } else { 1.0 };
            let wb = if wind_b > 1e-7 { wind_b } else { 1.0 };
            let fpm =
                if phi > 0.0 && wind_b > 1e-7 && wind_c > 1e-7 && relative_packing_ratio > 1e-7 {
                    ((phi * rpr.powf(wind_e)) / wc).powf(1.0 / wb)
                } else {
                    0.0
                };
            // Exact independent Python speed_from_base factor (unit code 5).
            (fpm as $speed) * 0.01136363636
        }
        /// Calculate wind amplification from midflame ft/min and dimensionless coefficients.
        pub fn calculate_wind_phi(
            sigma: $t,
            midflame_wind_speed: $t,
            wind_b: $t,
            wind_c: $t,
            relative_packing_ratio: $t,
            wind_e: $t,
        ) -> $t {
            let rpr = if relative_packing_ratio > 1e-7 {
                relative_packing_ratio
            } else {
                1.0
            };
            if sigma > 1e-7 && midflame_wind_speed > 1e-7 {
                midflame_wind_speed.powf(wind_b) * wind_c * rpr.powf(-wind_e)
            } else {
                0.0
            }
        }
    };
}
retaining_wind!(f64, f64);

/// Calculate midflame wind speed in ft/min, normalizing inputs to f64 like Python.
pub fn calculate_midflame_wind_speed(waf: f64, twenty_foot_wind_speed: f64) -> f64 {
    waf * twenty_foot_wind_speed
}
/// Calculate slope phi with slope units, packing ratio, and reaction-intensity limit.
pub fn calculate_slope_phi(
    slope: f64,
    slope_units: SlopeUnit,
    packing_ratio: f64,
    reaction_intensity: f64,
) -> f64 {
    let slope_tan = slope_units.to_base(slope).to_radians().tan();
    let pr = if packing_ratio > 1e-7 {
        packing_ratio
    } else {
        1.0
    };
    let phi = if slope_tan > 1e-7 && packing_ratio > 1e-7 {
        5.275 * pr.powf(-0.3) * slope_tan.powi(2)
    } else {
        0.0
    };
    np_min(phi, 0.9 * reaction_intensity)
}
/// Calculate the spread vector in ft/min, correcting north-oriented wind by aspect.
pub fn calculate_surface_rate_vector(
    no_wind_no_slope_spread_rate: f64,
    wind_phi: f64,
    slope_phi: f64,
    wind_direction: f64,
    wind_orientation_mode: crate::engine::WindOrientation,
    aspect: f64,
) -> RateVector {
    let corrected = if wind_orientation_mode == crate::engine::WindOrientation::North {
        wind_direction - aspect
    } else {
        wind_direction
    };
    let rad = corrected.to_radians();
    let slope_rate = no_wind_no_slope_spread_rate * slope_phi;
    let wind_rate = no_wind_no_slope_spread_rate * wind_phi;
    let x_component = slope_rate + wind_rate * rad.cos();
    let y_component = wind_rate * rad.sin();
    let rate_vector = (x_component.powi(2) + y_component.powi(2)).sqrt();
    RateVector {
        x_component,
        y_component,
        rate_vector,
    }
}
/// Calculate WAF, using user input only when the method matches and a value is present.
pub fn calculate_surface_wind_adjustment_factor(
    canopy_cover: f64,
    canopy_height: f64,
    crown_ratio: f64,
    depth: f64,
    waf_method: crate::engine::WafMethod,
    user_waf: Option<f64>,
) -> f64 {
    if waf_method == crate::engine::WafMethod::UserInput {
        if let Some(value) = user_waf {
            return value;
        }
    }
    calculate_wind_adjustment_factor(canopy_cover, canopy_height, crown_ratio, depth)
}
/// Calculate Albini/Rothermel wind coefficients from sigma in ft²/ft³.
pub fn calculate_surface_wind_coefficients(sigma: f64) -> WindCoefficients {
    let safe_sigma = if sigma > 1e-7 { sigma } else { 1.0 };
    WindCoefficients {
        safe_sigma,
        wind_c: 7.47 * (-0.133 * safe_sigma.powf(0.55)).exp(),
        wind_b: 0.02526 * safe_sigma.powf(0.54),
        wind_e: 0.715 * (-0.000359 * safe_sigma).exp(),
    }
}
/// Convert open wind to twenty-foot ft/min; 10-meter mode divides by 1.15.
pub fn calculate_twenty_foot_wind_speed(
    wind_speed: f64,
    wind_speed_units: SpeedUnit,
    wind_height_mode: crate::engine::WindHeight,
) -> f64 {
    let value = wind_speed_units.to_base(wind_speed);
    if wind_height_mode == crate::engine::WindHeight::TenMetre {
        value / 1.15
    } else {
        value
    }
}
/// Calculate open or sheltered wind adjustment from canopy fractions, canopy feet, and fuel depth feet.
pub fn calculate_wind_adjustment_factor(
    canopy_cover: f64,
    canopy_height: f64,
    crown_ratio: f64,
    depth: f64,
) -> f64 {
    let ccf = crown_ratio * canopy_cover / 3.0;
    let no_canopy = canopy_cover < 1e-7 || ccf < 0.05 || canopy_height < 6.0;
    let safe_depth = if depth > 1e-7 { depth } else { 1.0 };
    let open = if depth > 1e-7 {
        1.83 / ((20.0 + 0.36 * safe_depth) / (0.13 * safe_depth)).ln()
    } else {
        1.0
    };
    let ch = if canopy_height > 1e-7 {
        canopy_height
    } else {
        1.0
    };
    let cc = if ccf > 1e-7 { ccf } else { 1.0 };
    let denom = (cc * ch).sqrt() * ((20.0 + 0.36 * ch) / (0.13 * ch)).ln();
    let canopy = if denom > 1e-7 { 0.555 / denom } else { 1.0 };
    if no_canopy {
        open
    } else {
        canopy
    }
}

pub(super) mod single {
    retaining_wind!(f32, f64);
    /// Preserve float32 packing and reaction products before f64-normalized slope trigonometry.
    pub fn calculate_slope_phi(
        slope: f64,
        slope_units: crate::units::SlopeUnit,
        packing_ratio: f32,
        reaction_intensity: f32,
    ) -> f64 {
        let tan = slope_units.to_base(slope).to_radians().tan();
        let pr = if packing_ratio > 1e-7 {
            packing_ratio
        } else {
            1.0
        };
        let phi = if tan > 1e-7 && packing_ratio > 1e-7 {
            (5.275 * pr.powf(-0.3)) as f64 * tan.powi(2)
        } else {
            0.0
        };
        crate::math::np_min(phi, (0.9 * reaction_intensity) as f64)
    }
    /// Retain float32 rate products before Python's normalized direction trigonometry promotes to f64.
    pub fn calculate_surface_rate_vector(
        r0: f32,
        wind_phi: f32,
        slope_phi: f32,
        wind_direction: f64,
        orientation: crate::engine::WindOrientation,
        aspect: f64,
    ) -> super::RateVector {
        let direction = if orientation == crate::engine::WindOrientation::North {
            wind_direction - aspect
        } else {
            wind_direction
        };
        let rad = direction.to_radians();
        let sr = (r0 * slope_phi) as f64;
        let wr = (r0 * wind_phi) as f64;
        let x_component = sr + wr * rad.cos();
        let y_component = wr * rad.sin();
        super::RateVector {
            x_component,
            y_component,
            rate_vector: (x_component.powi(2) + y_component.powi(2)).sqrt(),
        }
    }
    /// Calculate wind coefficients retaining float32 powers and exponentials; widen only the returned transport fields.
    pub fn calculate_surface_wind_coefficients(sigma: f32) -> super::WindCoefficients {
        let s = if sigma > 1e-7 { sigma } else { 1.0 };
        super::WindCoefficients {
            safe_sigma: s as f64,
            wind_c: (7.47 * (-0.133 * s.powf(0.55)).exp()) as f64,
            wind_b: (0.02526 * s.powf(0.54)) as f64,
            wind_e: (0.715 * (-0.000359 * s).exp()) as f64,
        }
    }
}
