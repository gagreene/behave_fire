//! Stateful fire ellipse helper, preserving all Python FireSize methods.
use crate::units::{AreaUnit, LengthUnit, SpeedUnit, TimeUnit};
use std::f64::consts::PI;

/// Exceptions raised by the reference's scalar Python arithmetic and math calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GeometryError {
    /// Invalid argument to scalar math, mapped to Python ValueError.
    Domain,
    /// Finite scalar exponent overflow, mapped to Python OverflowError.
    Overflow,
    /// Scalar float division by zero, mapped to Python ZeroDivisionError.
    DivisionByZero,
}
impl std::fmt::Display for GeometryError {
    /// Describe the original Python exception category.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Domain => "math domain error",
            Self::Overflow => "math range error",
            Self::DivisionByZero => "float division by zero",
        })
    }
}
impl std::error::Error for GeometryError {}

/// Numeric semantics of an unconverted scalar Python input or stored value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericKind {
    /// Python float arithmetic raises on scalar power overflow and zero division.
    PythonFloat,
    /// NumPy float64 arithmetic propagates IEEE overflow and division results.
    NumpyFloat64,
    /// NumPy float32 scalar arithmetic retains single-precision intermediate values.
    NumpyFloat32,
}

impl NumericKind {
    /// Compare a stored scalar with a Python float using NumPy scalar promotion.
    fn greater_than(self, value: f64, threshold: f64) -> bool {
        if self == Self::NumpyFloat32 {
            value as f32 > threshold as f32
        } else {
            value > threshold
        }
    }
}

/// Evaluate Python math.exp, allowing infinite inputs but rejecting finite overflow.
fn scalar_exp(value: f64) -> Result<f64, GeometryError> {
    let result = value.exp();
    if value.is_finite() && result.is_infinite() {
        Err(GeometryError::Overflow)
    } else {
        Ok(result)
    }
}

/// Stateful ellipse dimensions. Public fields preserve the serialized Python state.
#[derive(Debug, Clone, PartialEq)]
pub struct FireSize {
    /// Numeric kind of the stored effective wind, including unconverted MPH inputs.
    pub effective_wind_numeric_kind_: NumericKind,
    /// Numeric kind of the stored length-to-width ratio.
    pub ratio_numeric_kind_: NumericKind,
    /// Numeric kind of the stored eccentricity.
    pub eccentricity_numeric_kind_: NumericKind,
    /// Whether rate arithmetic currently follows NumPy division semantics.
    /// Set by basic-dimension input conversion, and retained in snapshots.
    pub spread_rates_are_numpy_: bool,
    /// Whether wind was converted through NumPy-based unit conversion.
    pub effective_wind_is_numpy_: bool,
    /// Whether the current ratio derives from NumPy crown-wind arithmetic.
    pub ratio_is_numpy_: bool,
    /// Effective wind in miles per hour.
    pub effective_wind_speed_: f64,
    /// Forward spread in feet per minute.
    pub forward_spread_rate_: f64,
    /// Retained elapsed-time field from the Python state.
    pub elapsed_time_: f64,
    /// Semi-minor axis rate in feet per minute.
    pub elliptical_a_: f64,
    /// Semi-major axis rate in feet per minute.
    pub elliptical_b_: f64,
    /// Center-to-focus rate in feet per minute.
    pub elliptical_c_: f64,
    /// Ellipse eccentricity.
    pub eccentricity_: f64,
    /// Backing spread rate in feet per minute.
    pub backing_spread_rate_: f64,
    /// Flanking spread rate in feet per minute.
    pub flanking_spread_rate_: f64,
    /// Length-to-width ratio.
    pub fire_length_to_width_ratio_: f64,
    /// Heading-to-backing ratio.
    pub heading_to_backing_ratio_: f64,
}
impl Default for FireSize {
    /// Create the same initial state as Python FireSize.
    fn default() -> Self {
        Self {
            effective_wind_numeric_kind_: NumericKind::PythonFloat,
            ratio_numeric_kind_: NumericKind::PythonFloat,
            eccentricity_numeric_kind_: NumericKind::PythonFloat,
            spread_rates_are_numpy_: false,
            effective_wind_is_numpy_: false,
            ratio_is_numpy_: false,
            effective_wind_speed_: 0.0,
            forward_spread_rate_: 0.0,
            elapsed_time_: 0.0,
            elliptical_a_: 0.0,
            elliptical_b_: 0.0,
            elliptical_c_: 0.0,
            eccentricity_: 0.0,
            backing_spread_rate_: 0.0,
            flanking_spread_rate_: 0.0,
            fire_length_to_width_ratio_: 1.0,
            heading_to_backing_ratio_: 0.0,
        }
    }
}
impl FireSize {
    /// Create default state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Calculate backing spread from the current forward rate and eccentricity.
    pub fn calculate_backing_spread_rate(&mut self) -> Result<(), GeometryError> {
        if !self.spread_rates_are_numpy_ && 1.0 + self.eccentricity_ == 0.0 {
            return Err(GeometryError::DivisionByZero);
        }
        let (minus, plus) = if self.eccentricity_numeric_kind_ == NumericKind::NumpyFloat32 {
            (
                f64::from(1.0f32 - self.eccentricity_ as f32),
                f64::from(1.0f32 + self.eccentricity_ as f32),
            )
        } else {
            (1.0 - self.eccentricity_, 1.0 + self.eccentricity_)
        };
        self.backing_spread_rate_ = self.forward_spread_rate_ * minus / plus;
        Ok(())
    }

    /// Calculate crown length-to-width ratio from the current wind.
    pub fn calculate_crown_fire_length_to_width_ratio(&mut self) {
        let active = self
            .effective_wind_numeric_kind_
            .greater_than(self.effective_wind_speed_, 1e-7);
        self.fire_length_to_width_ratio_ = if active {
            if self.effective_wind_numeric_kind_ == NumericKind::NumpyFloat32 {
                f64::from(1.0f32 + 0.125f32 * self.effective_wind_speed_ as f32)
            } else {
                1.0 + 0.125 * self.effective_wind_speed_
            }
        } else {
            1.0
        };
        self.ratio_is_numpy_ = active && self.effective_wind_is_numpy_;
        self.ratio_numeric_kind_ = if active {
            self.effective_wind_numeric_kind_
        } else {
            NumericKind::PythonFloat
        };
    }

    /// Calculate ellipse rates and Alexander heading-to-backing ratio.
    pub fn calculate_elliptical_dimensions(&mut self) -> Result<(), GeometryError> {
        self.elliptical_a_ = 0.0;
        self.elliptical_b_ = (self.forward_spread_rate_ + self.backing_spread_rate_) / 2.0;
        self.elliptical_c_ = 0.0;
        self.heading_to_backing_ratio_ = 0.0;
        if self
            .ratio_numeric_kind_
            .greater_than(self.fire_length_to_width_ratio_, 1e-7)
        {
            let single = self.ratio_numeric_kind_ == NumericKind::NumpyFloat32;
            let squared = if single {
                f64::from((self.fire_length_to_width_ratio_ as f32).powi(2))
            } else {
                self.fire_length_to_width_ratio_.powi(2)
            };
            if !self.ratio_is_numpy_
                && self.fire_length_to_width_ratio_.is_finite()
                && squared.is_infinite()
            {
                return Err(GeometryError::Overflow);
            }
            if squared - 1.0 < 0.0 {
                return Err(GeometryError::Domain);
            }
            let difference = if single {
                f64::from(squared as f32 - 1.0f32)
            } else {
                squared - 1.0
            };
            let part = difference.sqrt();
            if !self.ratio_is_numpy_ && self.fire_length_to_width_ratio_ - part == 0.0 {
                return Err(GeometryError::DivisionByZero);
            }
            self.heading_to_backing_ratio_ = if single {
                let ratio = self.fire_length_to_width_ratio_ as f32;
                let part = part as f32;
                f64::from((ratio + part) / (ratio - part))
            } else {
                (self.fire_length_to_width_ratio_ + part)
                    / (self.fire_length_to_width_ratio_ - part)
            };
            self.elliptical_a_ = self.elliptical_b_ / self.fire_length_to_width_ratio_;
        }
        self.elliptical_c_ = self.elliptical_b_ - self.backing_spread_rate_;
        Ok(())
    }

    /// Convert inputs and update all basic fire dimensions in Python call order.
    pub fn calculate_fire_basic_dimensions(
        &mut self,
        is_crown: bool,
        effective_wind_speed: f64,
        wind_speed_units: SpeedUnit,
        forward_spread_rate: f64,
        spread_rate_units: SpeedUnit,
    ) -> Result<(), GeometryError> {
        self.calculate_fire_basic_dimensions_with_wind_kind(
            is_crown,
            effective_wind_speed,
            wind_speed_units,
            forward_spread_rate,
            spread_rate_units,
            NumericKind::PythonFloat,
        )
    }

    /// Convert inputs while retaining the original numeric kind of an MPH wind.
    /// Other wind units normalize through float64 NumPy conversion as in Python.
    #[allow(clippy::too_many_arguments)]
    pub fn calculate_fire_basic_dimensions_with_wind_kind(
        &mut self,
        is_crown: bool,
        effective_wind_speed: f64,
        wind_speed_units: SpeedUnit,
        forward_spread_rate: f64,
        spread_rate_units: SpeedUnit,
        wind_kind: NumericKind,
    ) -> Result<(), GeometryError> {
        let effective_wind_speed = if wind_kind == NumericKind::NumpyFloat32 {
            f64::from(effective_wind_speed as f32)
        } else {
            effective_wind_speed
        };
        self.forward_spread_rate_ = spread_rate_units.to_base(forward_spread_rate);
        self.spread_rates_are_numpy_ = true;
        self.effective_wind_speed_ = if wind_speed_units == SpeedUnit::MilesPerHour {
            effective_wind_speed
        } else {
            SpeedUnit::MilesPerHour.from_base(wind_speed_units.to_base(effective_wind_speed))
        };
        self.effective_wind_numeric_kind_ = if wind_speed_units == SpeedUnit::MilesPerHour {
            wind_kind
        } else {
            NumericKind::NumpyFloat64
        };
        self.effective_wind_is_numpy_ =
            self.effective_wind_numeric_kind_ != NumericKind::PythonFloat;
        if is_crown {
            self.calculate_crown_fire_length_to_width_ratio();
        } else {
            self.calculate_surface_fire_length_to_width_ratio()?;
        }
        self.calculate_fire_eccentricity();
        self.calculate_backing_spread_rate()?;
        self.calculate_flanking_spread_rate()?;
        self.calculate_elliptical_dimensions()?;
        Ok(())
    }

    /// Calculate eccentricity from the current length-to-width ratio.
    pub fn calculate_fire_eccentricity(&mut self) {
        self.eccentricity_ = 0.0;
        self.eccentricity_numeric_kind_ = NumericKind::PythonFloat;
        let single = self.ratio_numeric_kind_ == NumericKind::NumpyFloat32;
        let x = if single {
            let ratio = self.fire_length_to_width_ratio_ as f32;
            f64::from(ratio * ratio - 1.0f32)
        } else {
            self.fire_length_to_width_ratio_ * self.fire_length_to_width_ratio_ - 1.0
        };
        if x > 0.0 {
            self.eccentricity_ = if single {
                f64::from(x.sqrt() as f32 / self.fire_length_to_width_ratio_ as f32)
            } else {
                x.sqrt() / self.fire_length_to_width_ratio_
            };
            self.eccentricity_numeric_kind_ = self.ratio_numeric_kind_;
        }
    }

    /// Calculate flanking spread from the current forward and backing rates.
    pub fn calculate_flanking_spread_rate(&mut self) -> Result<(), GeometryError> {
        let length = self.backing_spread_rate_ + self.forward_spread_rate_;
        if !self.spread_rates_are_numpy_
            && !self.ratio_is_numpy_
            && self.fire_length_to_width_ratio_ == 0.0
        {
            return Err(GeometryError::DivisionByZero);
        }
        let width = length / self.fire_length_to_width_ratio_;
        self.flanking_spread_rate_ = width * 0.5;
        Ok(())
    }

    /// Calculate surface length-to-width ratio, capped at eight.
    pub fn calculate_surface_fire_length_to_width_ratio(&mut self) -> Result<(), GeometryError> {
        self.fire_length_to_width_ratio_ = if self
            .effective_wind_numeric_kind_
            .greater_than(self.effective_wind_speed_, 1e-7)
        {
            let (positive, negative) =
                if self.effective_wind_numeric_kind_ == NumericKind::NumpyFloat32 {
                    (
                        f64::from(0.1147f32 * self.effective_wind_speed_ as f32),
                        f64::from(-0.0692f32 * self.effective_wind_speed_ as f32),
                    )
                } else {
                    (
                        0.1147 * self.effective_wind_speed_,
                        -0.0692 * self.effective_wind_speed_,
                    )
                };
            0.936 * scalar_exp(positive)? + 0.461 * scalar_exp(negative)? - 0.397
        } else {
            1.0
        };
        if self.fire_length_to_width_ratio_ > 8.0 {
            self.fire_length_to_width_ratio_ = 8.0;
        }
        self.ratio_is_numpy_ = false;
        self.ratio_numeric_kind_ = NumericKind::PythonFloat;
        Ok(())
    }

    /// Return backing spread in requested units.
    pub fn get_backing_spread_rate(&self, spread_rate_units: SpeedUnit) -> f64 {
        spread_rate_units.from_base(self.backing_spread_rate_)
    }
    /// Return eccentricity.
    pub fn get_eccentricity(&self) -> f64 {
        self.eccentricity_
    }
    /// Return semi-minor axis for the requested elapsed time.
    pub fn get_elliptical_a(
        &self,
        length_units: LengthUnit,
        elapsed_time: f64,
        time_units: TimeUnit,
    ) -> f64 {
        length_units.from_base(self.elliptical_a_ * time_units.to_base(elapsed_time))
    }
    /// Return semi-major axis for the requested elapsed time.
    pub fn get_elliptical_b(
        &self,
        length_units: LengthUnit,
        elapsed_time: f64,
        time_units: TimeUnit,
    ) -> f64 {
        length_units.from_base(self.elliptical_b_ * time_units.to_base(elapsed_time))
    }
    /// Return center-to-focus distance for the requested elapsed time.
    pub fn get_elliptical_c(
        &self,
        length_units: LengthUnit,
        elapsed_time: f64,
        time_units: TimeUnit,
    ) -> f64 {
        length_units.from_base(self.elliptical_c_ * time_units.to_base(elapsed_time))
    }

    /// Return surface or crown area, preserving the distinct reference formulas.
    pub fn get_fire_area(
        &self,
        is_crown: bool,
        area_units: AreaUnit,
        elapsed_time: f64,
        time_units: TimeUnit,
    ) -> f64 {
        let time = time_units.to_base(elapsed_time);
        let area = if is_crown {
            let spread_distance = self.forward_spread_rate_ * time;
            let denominator = if self.ratio_numeric_kind_ == NumericKind::NumpyFloat32 {
                f64::from(4.0f32 * self.fire_length_to_width_ratio_ as f32)
            } else {
                4.0 * self.fire_length_to_width_ratio_
            };
            PI * spread_distance * spread_distance / denominator
        } else {
            PI * self.elliptical_a_ * self.elliptical_b_ * time * time
        };
        area_units.from_base(area)
    }
    /// Return full head-to-tail length at the requested elapsed time.
    pub fn get_fire_length(
        &self,
        length_units: LengthUnit,
        elapsed_time: f64,
        time_units: TimeUnit,
    ) -> f64 {
        length_units.from_base(self.elliptical_b_ * time_units.to_base(elapsed_time) * 2.0)
    }
    /// Return the current length-to-width ratio.
    pub fn get_fire_length_to_width_ratio(&self) -> f64 {
        self.fire_length_to_width_ratio_
    }

    /// Return crown or surface perimeter using the reference approximation.
    pub fn get_fire_perimeter(
        &self,
        is_crown: bool,
        length_units: LengthUnit,
        elapsed_time: f64,
        time_units: TimeUnit,
    ) -> Result<f64, GeometryError> {
        let time = time_units.to_base(elapsed_time);
        let perimeter = if is_crown {
            if !self.ratio_is_numpy_ && self.fire_length_to_width_ratio_ == 0.0 {
                return Err(GeometryError::DivisionByZero);
            }
            let spread_distance = self.forward_spread_rate_ * time;
            let factor = if self.ratio_numeric_kind_ == NumericKind::NumpyFloat32 {
                f64::from(1.0f32 + 1.0f32 / self.fire_length_to_width_ratio_ as f32)
            } else {
                1.0 + 1.0 / self.fire_length_to_width_ratio_
            };
            0.5 * PI * spread_distance * factor
        } else {
            let a = self.elliptical_a_ * time;
            let b = self.elliptical_b_ * time;
            if a + b > 1e-7 {
                let difference = a - b;
                let sum = a + b;
                let h = difference * difference / (sum * sum);
                PI * sum * (1.0 + h / 4.0 + h * h / 64.0)
            } else {
                0.0
            }
        };
        Ok(length_units.from_base(perimeter))
    }
    /// Return flanking spread in requested units.
    pub fn get_flanking_spread_rate(&self, spread_rate_units: SpeedUnit) -> f64 {
        spread_rate_units.from_base(self.flanking_spread_rate_)
    }
    /// Return Alexander heading-to-backing ratio.
    pub fn get_heading_to_backing_ratio(&self) -> f64 {
        self.heading_to_backing_ratio_
    }
    /// Return maximum width at the requested elapsed time.
    pub fn get_max_fire_width(
        &self,
        length_units: LengthUnit,
        elapsed_time: f64,
        time_units: TimeUnit,
    ) -> f64 {
        length_units.from_base(self.elliptical_a_ * time_units.to_base(elapsed_time) * 2.0)
    }
}
