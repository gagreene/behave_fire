//! Butler and Cohen safety-zone geometry.
use std::f64::consts::PI;

/// Safety-zone dimensions in feet and square feet.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SafetyZoneResult {
    /// Minimum separation in feet.
    pub separation_distance: f64,
    /// Total radius including personnel and equipment core.
    pub radius: f64,
    /// Total area in square feet.
    pub area: f64,
}
impl SafetyZoneResult {
    /// Output names matching the Python dictionary.
    pub fn named_values(self) -> [(&'static str, f64); 3] {
        [
            ("separation_distance", self.separation_distance),
            ("radius", self.radius),
            ("area", self.area),
        ]
    }
}
/// Calculate the safety-zone dimensions, preserving negative and nonfinite inputs.
pub fn calculate_safety_zone(
    flame_height_ft: f64,
    number_of_personnel: f64,
    area_per_person_sqft: f64,
    number_of_equipment: f64,
    area_per_equipment_sqft: f64,
) -> SafetyZoneResult {
    let separation_distance = calculate_separation_distance(flame_height_ft);
    let core =
        area_per_person_sqft * number_of_personnel + area_per_equipment_sqft * number_of_equipment;
    let core_radius = if core.is_nan() {
        f64::NAN
    } else if core > 1e-7 {
        (core / PI).sqrt()
    } else {
        0.0
    };
    let radius = separation_distance + core_radius;
    SafetyZoneResult {
        separation_distance,
        radius,
        area: PI * radius.powi(2),
    }
}

/// Calculate separation from the actual flame height, independent of zone occupancy.
pub fn calculate_separation_distance(flame_height_ft: f64) -> f64 {
    4.0 * flame_height_ft
}
