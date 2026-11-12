//! Unit-aware Tetens vapor pressure calculations.
use crate::units::{FractionUnit, PressureUnit, TemperatureUnit};

/// Vapor pressures in the caller's requested output unit.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct VpdResult {
    /// Vapor pressure deficit.
    pub vpd: f64,
    /// Actual vapor pressure.
    pub actual_vp: f64,
    /// Saturated vapor pressure.
    pub saturated_vp: f64,
}
impl VpdResult {
    /// Output names matching the Python dictionary.
    pub fn named_values(self) -> [(&'static str, f64); 3] {
        [
            ("vpd", self.vpd),
            ("actual_vp", self.actual_vp),
            ("saturated_vp", self.saturated_vp),
        ]
    }
}
/// Calculate vapor pressure deficit with the Python reference's conversion order.
pub fn calculate_vpd(
    temperature: f64,
    temp_units: TemperatureUnit,
    relative_humidity: f64,
    rh_units: FractionUnit,
    output_units: PressureUnit,
) -> VpdResult {
    let sat = saturated_vapor_pressure_hpa(temperature, temp_units);
    let rh = rh_units.to_base(relative_humidity);
    let act = rh * sat;
    VpdResult {
        vpd: output_units.from_base((sat - act) * 100.0),
        actual_vp: output_units.from_base(act * 100.0),
        saturated_vp: output_units.from_base(sat * 100.0),
    }
}

/// Calculate saturated pressure in hPa without a humidity dependency.
pub fn saturated_vapor_pressure_hpa(temperature: f64, temp_units: TemperatureUnit) -> f64 {
    let temp_f = temp_units.to_base(temperature);
    let temp_c = (temp_f - 32.0) * 5.0 / 9.0;
    let tk = temp_c + 237.3;
    let safe_tk = if tk.abs() > 1e-7 { tk } else { 1.0 };
    6.11 * 10.0f64.powf(7.5 * temp_c / safe_tk)
}
