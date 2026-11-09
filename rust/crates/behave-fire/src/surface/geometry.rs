//! Ellipse geometry, spread, and intensity helpers in Python operation order.
use crate::math::{np_max, np_min, np_remainder};

macro_rules! retaining_geometry {
    ($t:ty) => {
        /// Calculate backing spread rate in ft/min.
        pub fn calculate_backing_spread_rate(spread_rate: $t, eccentricity: $t) -> $t {
            let denom = 1.0 + eccentricity;
            if denom.is_nan() {
                return <$t>::NAN;
            }
            if denom > 1e-7 {
                spread_rate * (1.0 - eccentricity) / denom
            } else {
                0.0
            }
        }
        /// Calculate ellipse eccentricity (dimensionless).
        pub fn calculate_eccentricity(fire_length_to_width_ratio: $t) -> $t {
            if fire_length_to_width_ratio.is_nan() {
                return <$t>::NAN;
            }
            let x = fire_length_to_width_ratio.powi(2) - 1.0;
            if x > 0.0 {
                np_max(x, 0.0).sqrt() / fire_length_to_width_ratio
            } else {
                0.0
            }
        }
        /// Calculate fire ellipse length in feet after elapsed minutes.
        pub fn calculate_fire_length(forward_ros: $t, backing_ros: $t, elapsed_min: $t) -> $t {
            (forward_ros + backing_ros) * elapsed_min
        }
        /// Calculate surface ellipse length-to-width ratio from effective wind in mph.
        pub fn calculate_fire_length_to_width_ratio(effective_wind_speed: $t) -> $t {
            if effective_wind_speed.is_nan() {
                return <$t>::NAN;
            }
            if effective_wind_speed > 1e-7 {
                np_min(
                    0.936 * (0.1147 * effective_wind_speed).exp()
                        + 0.461 * (-0.0692 * effective_wind_speed).exp()
                        - 0.397,
                    8.0,
                )
            } else {
                1.0
            }
        }
        /// Calculate flame length in feet from fireline intensity in BTU/ft/s.
        pub fn calculate_flame_length(fireline_intensity: $t) -> $t {
            if fireline_intensity.is_nan() {
                return <$t>::NAN;
            }
            if fireline_intensity > 0.0 {
                0.45 * fireline_intensity.powf(0.46)
            } else {
                0.0
            }
        }
        /// Calculate flank spread rate in ft/min.
        pub fn calculate_flanking_spread_rate(
            spread_rate: $t,
            backing_spread_rate: $t,
            fire_length_to_width_ratio: $t,
        ) -> $t {
            if fire_length_to_width_ratio.is_nan() {
                return <$t>::NAN;
            }
            let safe = if fire_length_to_width_ratio > 1e-7 {
                fire_length_to_width_ratio
            } else {
                1.0
            };
            if fire_length_to_width_ratio > 1e-7 {
                (spread_rate + backing_spread_rate) / (2.0 * safe)
            } else {
                0.0
            }
        }
    };
}
retaining_geometry!(f64);

/// Preserve NumPy boolean addition (logical OR) for the fire-length helper's boolean-array inputs.
pub fn calculate_fire_length_bool(forward_ros: bool, backing_ros: bool, elapsed_min: f64) -> f64 {
    f64::from(u8::from(forward_ros || backing_ros)) * elapsed_min
}
/// Preserve NumPy boolean addition for boolean spread-rate inputs in flank spread.
pub fn calculate_flanking_spread_rate_bool(
    spread_rate: bool,
    backing_spread_rate: bool,
    lwr: f64,
) -> f64 {
    if lwr > 1e-7 {
        f64::from(u8::from(spread_rate || backing_spread_rate)) / (2.0 * lwr)
    } else {
        0.0
    }
}

/// Calculate maximum spread direction in degrees; north orientation includes aspect and 180°.
pub fn calculate_direction_of_max_spread(
    x_component: f64,
    y_component: f64,
    rate_vector: f64,
    wind_orientation_mode: crate::engine::WindOrientation,
    aspect: f64,
) -> f64 {
    if rate_vector.is_nan() {
        return f64::NAN;
    }
    let mut dir = if rate_vector > 1e-7 {
        y_component.atan2(x_component).to_degrees()
    } else {
        0.0
    };
    if dir.abs() < 0.5 {
        dir = 0.0;
    }
    if dir < -1e-20 {
        dir += 360.0;
    }
    if wind_orientation_mode == crate::engine::WindOrientation::North {
        dir = np_remainder(dir + aspect + 180.0, 360.0);
    }
    dir
}

/// Calculate area in square feet, using the Python surface ellipse or crown approximation.
pub fn calculate_fire_area(
    forward_ros: f64,
    backing_ros: f64,
    lwr: f64,
    elapsed_min: f64,
    is_crown: bool,
) -> f64 {
    if lwr.is_nan() {
        return f64::NAN;
    }
    let safe = if lwr > 1e-7 { lwr } else { 1.0 };
    if is_crown {
        let d = forward_ros * elapsed_min;
        if lwr > 1e-7 {
            std::f64::consts::PI * d.powi(2) / (4.0 * safe)
        } else {
            0.0
        }
    } else {
        let b = (forward_ros + backing_ros) / 2.0 * elapsed_min;
        let a = if lwr > 1e-7 { b / safe } else { 0.0 };
        std::f64::consts::PI * a * b
    }
}
/// Calculate perimeter in feet with the reference truncated Ramanujan series.
pub fn calculate_fire_perimeter(
    forward_ros: f64,
    backing_ros: f64,
    lwr: f64,
    elapsed_min: f64,
    is_crown: bool,
) -> f64 {
    if lwr.is_nan() {
        return f64::NAN;
    }
    let safe = if lwr > 1e-7 { lwr } else { 1.0 };
    if is_crown {
        let d = forward_ros * elapsed_min;
        if lwr > 1e-7 {
            0.5 * std::f64::consts::PI * d * (1.0 + 1.0 / safe)
        } else {
            0.0
        }
    } else {
        let b = (forward_ros + backing_ros) / 2.0 * elapsed_min;
        let a = if lwr > 1e-7 { b / safe } else { 0.0 };
        let apb = a + b;
        let safe2 = if apb > 1e-7 { apb.powi(2) } else { 1.0 };
        let h = if apb > 1e-7 {
            (a - b).powi(2) / safe2
        } else {
            0.0
        };
        if apb > 1e-7 {
            std::f64::consts::PI * apb * (1.0 + h / 4.0 + h.powi(2) / 64.0)
        } else {
            0.0
        }
    }
}
/// Calculate ellipse width in feet after elapsed minutes.
pub fn calculate_fire_width(forward_ros: f64, backing_ros: f64, lwr: f64, elapsed_min: f64) -> f64 {
    if lwr.is_nan() {
        return f64::NAN;
    }
    let length = (forward_ros + backing_ros) * elapsed_min;
    let safe = if lwr > 1e-7 { lwr } else { 1.0 };
    if lwr > 1e-7 {
        length / safe
    } else {
        0.0
    }
}
/// Calculate fireline intensity in BTU/ft/s, normalizing inputs to f64 like Python.
pub fn calculate_fireline_intensity(spread_rate: f64, heat_per_unit_area: f64) -> f64 {
    spread_rate * heat_per_unit_area / 60.0
}
/// Calculate forward spread rate in ft/min, normalizing inputs to f64 like Python.
pub fn calculate_forward_spread_rate(no_wind_no_slope_spread_rate: f64, rate_vector: f64) -> f64 {
    no_wind_no_slope_spread_rate + rate_vector
}
