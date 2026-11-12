//! Stateful fire ellipse helper using base-unit float64 values and IEEE arithmetic.
use crate::units::{AreaUnit, LengthUnit, SpeedUnit, TimeUnit};
use std::f64::consts::PI;

/// Stateful ellipse dimensions, including independently editable physical fields.
#[derive(Debug, Clone, PartialEq)]
pub struct FireSize {
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
    /// Create the reference's initial physical state without numeric provenance.
    fn default() -> Self {
        Self {
            effective_wind_speed_: 0.,
            forward_spread_rate_: 0.,
            elapsed_time_: 0.,
            elliptical_a_: 0.,
            elliptical_b_: 0.,
            elliptical_c_: 0.,
            eccentricity_: 0.,
            backing_spread_rate_: 0.,
            flanking_spread_rate_: 0.,
            fire_length_to_width_ratio_: 1.,
            heading_to_backing_ratio_: 0.,
        }
    }
}
impl FireSize {
    /// Create default state.
    pub fn new() -> Self {
        Self::default()
    }
    /// Calculate backing spread from the actual forward rate and eccentricity.
    pub fn calculate_backing_spread_rate(&mut self) {
        self.backing_spread_rate_ =
            self.forward_spread_rate_ * (1. - self.eccentricity_) / (1. + self.eccentricity_);
    }
    /// Convert inputs and update all basic fire dimensions in the reference order.
    pub fn calculate_fire_basic_dimensions(
        &mut self,
        is_crown: bool,
        effective_wind_speed: f64,
        wind_speed_units: SpeedUnit,
        forward_spread_rate: f64,
        spread_rate_units: SpeedUnit,
    ) {
        self.forward_spread_rate_ = spread_rate_units.to_base(forward_spread_rate);
        self.effective_wind_speed_ = if wind_speed_units == SpeedUnit::MilesPerHour {
            effective_wind_speed
        } else {
            SpeedUnit::MilesPerHour.from_base(wind_speed_units.to_base(effective_wind_speed))
        };
        if is_crown {
            self.calculate_crown_fire_length_to_width_ratio();
        } else {
            self.calculate_surface_fire_length_to_width_ratio();
        }
        self.calculate_fire_eccentricity();
        self.calculate_backing_spread_rate();
        self.calculate_flanking_spread_rate();
        self.calculate_elliptical_dimensions();
    }
    /// Calculate crown length-to-width ratio from the current wind.
    pub fn calculate_crown_fire_length_to_width_ratio(&mut self) {
        self.fire_length_to_width_ratio_ = if self.effective_wind_speed_.is_nan() {
            f64::NAN
        } else if self.effective_wind_speed_ > 1e-7 {
            1. + 0.125 * self.effective_wind_speed_
        } else {
            1.
        };
    }
    /// Calculate ellipse rates and Alexander heading-to-backing ratio.
    pub fn calculate_elliptical_dimensions(&mut self) {
        self.elliptical_b_ = (self.forward_spread_rate_ + self.backing_spread_rate_) / 2.;
        let ratio = self.fire_length_to_width_ratio_;
        if ratio.is_nan() {
            self.elliptical_a_ = f64::NAN;
            self.heading_to_backing_ratio_ = f64::NAN;
        } else if ratio > 1e-7 {
            let part = (ratio.powi(2) - 1.).sqrt();
            self.heading_to_backing_ratio_ = (ratio + part) / (ratio - part);
            self.elliptical_a_ = self.elliptical_b_ / ratio;
        } else {
            self.elliptical_a_ = 0.;
            self.heading_to_backing_ratio_ = 0.;
        }
        self.elliptical_c_ = self.elliptical_b_ - self.backing_spread_rate_;
    }
    /// Calculate eccentricity from the current length-to-width ratio.
    pub fn calculate_fire_eccentricity(&mut self) {
        self.eccentricity_ =
            crate::surface::calculate_eccentricity(self.fire_length_to_width_ratio_);
    }
    /// Calculate flanking spread from the current forward and backing rates.
    pub fn calculate_flanking_spread_rate(&mut self) {
        let length = self.backing_spread_rate_ + self.forward_spread_rate_;
        let width = length / self.fire_length_to_width_ratio_;
        self.flanking_spread_rate_ = width * 0.5;
    }
    /// Calculate surface length-to-width ratio, capped at eight.
    pub fn calculate_surface_fire_length_to_width_ratio(&mut self) {
        self.fire_length_to_width_ratio_ =
            crate::surface::calculate_fire_length_to_width_ratio(self.effective_wind_speed_);
    }
    /// Return backing spread in requested units.
    pub fn get_backing_spread_rate(&self, units: SpeedUnit) -> f64 {
        units.from_base(self.backing_spread_rate_)
    }
    /// Return eccentricity.
    pub fn get_eccentricity(&self) -> f64 {
        self.eccentricity_
    }
    /// Return semi-minor axis for the requested elapsed time.
    pub fn get_elliptical_a(&self, units: LengthUnit, elapsed: f64, time: TimeUnit) -> f64 {
        units.from_base(self.elliptical_a_ * time.to_base(elapsed))
    }
    /// Return semi-major axis for the requested elapsed time.
    pub fn get_elliptical_b(&self, units: LengthUnit, elapsed: f64, time: TimeUnit) -> f64 {
        units.from_base(self.elliptical_b_ * time.to_base(elapsed))
    }
    /// Return center-to-focus distance for the requested elapsed time.
    pub fn get_elliptical_c(&self, units: LengthUnit, elapsed: f64, time: TimeUnit) -> f64 {
        units.from_base(self.elliptical_c_ * time.to_base(elapsed))
    }
    /// Return surface or crown area, preserving the distinct reference formulas.
    pub fn get_fire_area(
        &self,
        is_crown: bool,
        units: AreaUnit,
        elapsed: f64,
        time: TimeUnit,
    ) -> f64 {
        let time = time.to_base(elapsed);
        let area = if is_crown {
            let distance = self.forward_spread_rate_ * time;
            PI * distance * distance / (4. * self.fire_length_to_width_ratio_)
        } else {
            PI * self.elliptical_a_ * self.elliptical_b_ * time * time
        };
        units.from_base(area)
    }
    /// Return full head-to-tail length at the requested elapsed time.
    pub fn get_fire_length(&self, units: LengthUnit, elapsed: f64, time: TimeUnit) -> f64 {
        units.from_base(self.elliptical_b_ * time.to_base(elapsed) * 2.)
    }
    /// Return the current length-to-width ratio.
    pub fn get_fire_length_to_width_ratio(&self) -> f64 {
        self.fire_length_to_width_ratio_
    }
    /// Return crown or surface perimeter using the reference approximation.
    pub fn get_fire_perimeter(
        &self,
        is_crown: bool,
        units: LengthUnit,
        elapsed: f64,
        time: TimeUnit,
    ) -> f64 {
        let time = time.to_base(elapsed);
        let perimeter = if is_crown {
            let distance = self.forward_spread_rate_ * time;
            0.5 * PI * distance * (1. + 1. / self.fire_length_to_width_ratio_)
        } else {
            let a = self.elliptical_a_ * time;
            let b = self.elliptical_b_ * time;
            if (a + b).is_nan() {
                f64::NAN
            } else if a + b > 1e-7 {
                let difference = a - b;
                let sum = a + b;
                let h = difference * difference / (sum * sum);
                PI * sum * (1. + h / 4. + h * h / 64.)
            } else {
                0.
            }
        };
        units.from_base(perimeter)
    }
    /// Return flanking spread in requested units.
    pub fn get_flanking_spread_rate(&self, units: SpeedUnit) -> f64 {
        units.from_base(self.flanking_spread_rate_)
    }
    /// Return Alexander heading-to-backing ratio.
    pub fn get_heading_to_backing_ratio(&self) -> f64 {
        self.heading_to_backing_ratio_
    }
    /// Return maximum width at the requested elapsed time.
    pub fn get_max_fire_width(&self, units: LengthUnit, elapsed: f64, time: TimeUnit) -> f64 {
        units.from_base(self.elliptical_a_ * time.to_base(elapsed) * 2.)
    }
}
