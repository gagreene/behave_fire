//! Stateful slope helper, preserving raw-distance conversion and table quirks.
pub use crate::fire_size::GeometryError;
use crate::units::{LengthUnit, SlopeUnit};
use std::f64::consts::PI;

/// Stateful map-slope calculations and map-scale tables.
#[derive(Debug, Clone, PartialEq)]
pub struct SlopeTool {
    /// Horizontal distances in feet for seven directions.
    pub horizontal_distances_: [f64; 7],
    /// Maximum slope in degrees.
    pub max_slope_in_degrees_: f64,
    /// Maximum slope in percent.
    pub max_slope_in_percent_: f64,
    /// Map slope in degrees.
    pub slope_from_map_measurements_: f64,
    /// Map horizontal distance in feet.
    pub slope_horizontal_distance_: f64,
    /// Map elevation change in feet.
    pub slope_elevation_change_: f64,
    /// Reference representative fractions.
    pub representative_fractions_: [i32; 18],
    /// Reference inches per mile.
    pub inches_per_mile_: [f64; 18],
    /// Reference miles per inch.
    pub miles_per_inch_: [f64; 18],
    /// Reference centimeters per kilometer.
    pub centimeters_per_kilometer_: [f64; 18],
    /// Reference kilometers per centimeter.
    pub kilometers_per_centimeter_: [f64; 18],
}
impl Default for SlopeTool {
    /// Create the complete Python initial state and fixed map tables.
    fn default() -> Self {
        Self {
            horizontal_distances_: [0.0; 7],
            max_slope_in_degrees_: -1.0,
            max_slope_in_percent_: -1.0,
            slope_from_map_measurements_: -1.0,
            slope_horizontal_distance_: -1.0,
            slope_elevation_change_: -1.0,
            representative_fractions_: [
                1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000,
                126720, 250000, 253440, 506880, 1000000, 1013760,
            ],
            inches_per_mile_: [
                32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5,
                0.2534, 0.25, 0.125, 0.0634, 0.0625,
            ],
            miles_per_inch_: [
                0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.33330, 0.37879, 0.5, 0.78914, 0.98643,
                1.0, 1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
            ],
            centimeters_per_kilometer_: [
                50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783,
                1.0, 0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
            ],
            kilometers_per_centimeter_: [
                0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0,
                1.2672, 2.50, 2.5344, 5.0688, 10.0, 10.1376,
            ],
        }
    }
}

/// Return the indexed value or the reference invalid-index sentinel.
fn indexed(values: &[f64], index: i64) -> f64 {
    if index >= 0 {
        values.get(index as usize).copied().unwrap_or(-1.0)
    } else {
        -1.0
    }
}

impl SlopeTool {
    /// Create default state.
    pub fn new() -> Self {
        Self::default()
    }
    /// Calculate seven directions while preserving the raw-input-as-feet quirk.
    pub fn calculate_horizontal_distance(
        &mut self,
        calculated_map_distance: f64,
        distance_units: LengthUnit,
        max_slope_steepness: f64,
        slope_units: SlopeUnit,
    ) -> Result<(), GeometryError> {
        self.max_slope_in_degrees_ = slope_units.to_base(max_slope_steepness);
        self.max_slope_in_percent_ = SlopeUnit::Percent.from_base(self.max_slope_in_degrees_);
        let _ground_distance_in_feet = distance_units.to_base(calculated_map_distance);
        // The explicit reference converts the RAW input from feet to inches here.
        let inches = LengthUnit::Inches.from_base(calculated_map_distance);
        let slope_angle = self.max_slope_in_degrees_ * PI / 180.0;
        if slope_angle.is_infinite() {
            return Err(GeometryError::Domain);
        }
        for (index, distance) in self.horizontal_distances_.iter_mut().enumerate() {
            let direction = 15.0 * index as f64;
            let a = inches * (direction * PI / 180.0).cos();
            let b = inches * (direction * PI / 180.0).sin();
            let c = a * slope_angle.cos();
            let d = (c * c + b * b).sqrt();
            *distance = LengthUnit::Inches.to_base(d);
        }
        Ok(())
    }
    /// Calculate map slope from map scale, contours, and unit-aware distances.
    pub fn calculate_slope_from_map_measurements(
        &mut self,
        map_representative_fraction: f64,
        map_distance: f64,
        distance_units: LengthUnit,
        contour_interval: f64,
        number_of_contours: f64,
        contour_units: LengthUnit,
    ) {
        let feet = distance_units.to_base(map_distance);
        let inches = LengthUnit::Inches.from_base(feet);
        self.slope_elevation_change_ = contour_units.to_base(contour_interval * number_of_contours);
        self.slope_horizontal_distance_ =
            LengthUnit::Inches.to_base(map_representative_fraction * inches);
        self.slope_from_map_measurements_ = if self.slope_horizontal_distance_ < 0.01 {
            0.0
        } else {
            (self.slope_elevation_change_ / self.slope_horizontal_distance_).atan() * 180.0 / PI
        };
    }
    /// Return centimeters per kilometer or -1 for invalid index.
    pub fn get_centimeters_per_kilometer_at_index(&self, index: i64) -> f64 {
        indexed(&self.centimeters_per_kilometer_, index)
    }
    /// Return horizontal distance in requested units or -1 for invalid index.
    pub fn get_horizontal_distance_at_index(
        &self,
        index: i64,
        map_distance_units: LengthUnit,
    ) -> f64 {
        if (0..7).contains(&index) {
            map_distance_units.from_base(self.horizontal_distances_[index as usize])
        } else {
            -1.0
        }
    }
    /// Return inches per mile or -1 for invalid index.
    pub fn get_inches_per_mile_at_index(&self, index: i64) -> f64 {
        indexed(&self.inches_per_mile_, index)
    }
    /// Return kilometers per centimeter or -1 for invalid index.
    pub fn get_kilometers_per_centimeter_at_index(&self, index: i64) -> f64 {
        indexed(&self.kilometers_per_centimeter_, index)
    }
    /// Return miles per inch or -1 for invalid index.
    pub fn get_miles_per_inch_at_index(&self, index: i64) -> f64 {
        indexed(&self.miles_per_inch_, index)
    }
    /// Return the number of direction distances.
    pub fn get_number_of_horizontal_distances(&self) -> usize {
        self.horizontal_distances_.len()
    }
    /// Return the number of scale-table rows.
    pub fn get_number_of_representative_fractions(&self) -> usize {
        self.representative_fractions_.len()
    }
    /// Return representative fraction or -1 for invalid index.
    pub fn get_representative_fraction_at_index(&self, index: i64) -> i32 {
        if index >= 0 {
            self.representative_fractions_
                .get(index as usize)
                .copied()
                .unwrap_or(-1)
        } else {
            -1
        }
    }
    /// Return map elevation change in requested units.
    pub fn get_slope_elevation_change_from_map_measurements(
        &self,
        elevation_units: LengthUnit,
    ) -> f64 {
        elevation_units.from_base(self.slope_elevation_change_)
    }
    /// Return map slope in requested units.
    pub fn get_slope_from_map_measurements(&self, slope_units: SlopeUnit) -> f64 {
        slope_units.from_base(self.slope_from_map_measurements_)
    }
    /// Return map slope in degrees.
    pub fn get_slope_from_map_measurements_in_degrees(&self) -> f64 {
        SlopeUnit::Degrees.from_base(self.slope_from_map_measurements_)
    }
    /// Return map slope in percent.
    pub fn get_slope_from_map_measurements_in_percent(&self) -> f64 {
        SlopeUnit::Percent.from_base(self.slope_from_map_measurements_)
    }
    /// Return map horizontal distance in requested units.
    pub fn get_slope_horizontal_distance_from_map_measurements(
        &self,
        distance_units: LengthUnit,
    ) -> f64 {
        distance_units.from_base(self.slope_horizontal_distance_)
    }
}
