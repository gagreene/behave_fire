//! Owned registry state shared by the standalone model's calculation methods.

use crate::FuelModels;

/// A model owns the fuel snapshot used by its calculations.
///
/// Later mutations of the registry supplied by a caller do not affect this model.
#[derive(Clone, Debug)]
pub struct BehaveRun {
    fuel_models: FuelModels,
    mortality_coefficients: crate::mortality::MortalityLookup,
}

impl BehaveRun {
    /// Calculate crown scorch fractions from feet, fractional crown ratio, and DBH inches.
    pub fn calculate_crown_scorch_mortality(
        &self,
        scorch_height_ft: f64,
        tree_height_ft: f64,
        crown_ratio: f64,
        dbh_inches: f64,
        equation_number: i32,
    ) -> crate::mortality::MortalityResult {
        crate::mortality::calculate_crown_scorch_mortality(
            scorch_height_ft,
            tree_height_ft,
            crown_ratio,
            dbh_inches,
            equation_number,
            &self.mortality_coefficients,
        )
    }

    /// Calculate area in square feet from spread rates in ft/min and elapsed minutes.
    pub fn calculate_fire_area(
        forward_ros: f64,
        backing_ros: f64,
        length_to_width_ratio: f64,
        elapsed_min: f64,
        is_crown: bool,
    ) -> f64 {
        crate::surface::calculate_fire_area(
            forward_ros,
            backing_ros,
            length_to_width_ratio,
            elapsed_min,
            is_crown,
        )
    }

    /// Calculate length in feet from spread rates in ft/min and elapsed minutes.
    pub fn calculate_fire_length(forward_ros: f64, backing_ros: f64, elapsed_min: f64) -> f64 {
        crate::surface::calculate_fire_length(forward_ros, backing_ros, elapsed_min)
    }

    /// Calculate perimeter in feet from spread rates in ft/min and elapsed minutes.
    pub fn calculate_fire_perimeter(
        forward_ros: f64,
        backing_ros: f64,
        length_to_width_ratio: f64,
        elapsed_min: f64,
        is_crown: bool,
    ) -> f64 {
        crate::surface::calculate_fire_perimeter(
            forward_ros,
            backing_ros,
            length_to_width_ratio,
            elapsed_min,
            is_crown,
        )
    }

    /// Calculate width in feet from spread rates in ft/min and elapsed minutes.
    pub fn calculate_fire_width(
        forward_ros: f64,
        backing_ros: f64,
        length_to_width_ratio: f64,
        elapsed_min: f64,
    ) -> f64 {
        crate::surface::calculate_fire_width(
            forward_ros,
            backing_ros,
            length_to_width_ratio,
            elapsed_min,
        )
    }

    /// Calculate scorch height in feet from BTU/ft/s, wind mph, and Fahrenheit.
    pub fn calculate_scorch_height(
        fireline_intensity_btu_ft_s: f64,
        midflame_wind_mph: f64,
        air_temp_f: f64,
    ) -> f64 {
        crate::mortality::calculate_scorch_height(
            fireline_intensity_btu_ft_s,
            midflame_wind_mph,
            air_temp_f,
        )
    }

    /// Calculate pile spotting distance in feet from flame/cover feet and wind mph.
    pub fn calculate_spotting_from_burning_pile(
        flame_height_ft: f64,
        wind_mph: f64,
        cover_height_ft: f64,
    ) -> f64 {
        crate::spot::calculate_spotting_from_burning_pile(
            flame_height_ft,
            wind_mph,
            cover_height_ft,
        )
    }

    /// Calculate surface spotting distance in feet from flame/cover feet and wind mph.
    pub fn calculate_spotting_from_surface_fire(
        flame_length_ft: f64,
        wind_mph: f64,
        cover_height_ft: f64,
    ) -> f64 {
        crate::spot::calculate_spotting_from_surface_fire(
            flame_length_ft,
            wind_mph,
            cover_height_ft,
        )
    }

    /// Calculate tree spotting distance in feet from DBH inches, heights feet, and wind mph.
    pub fn calculate_spotting_from_torching_trees(
        dbh_inches: f64,
        tree_height_ft: f64,
        count: f64,
        wind_mph: f64,
        cover_height_ft: f64,
    ) -> f64 {
        crate::spot::calculate_spotting_from_torching_trees(
            dbh_inches,
            tree_height_ft,
            count,
            wind_mph,
            cover_height_ft,
        )
    }

    /// Run crown calculations using the supplied surface result and owned fuels.
    pub fn do_crown_run(
        &self,
        input: &crate::crown::CrownInput,
    ) -> Result<crate::crown::CrownResult, crate::BehaveError> {
        crate::crown::run(input, &self.fuel_models)
    }

    /// Run the surface pipeline using this model's fuel snapshot.
    pub fn do_surface_run(
        &self,
        input: &crate::surface::SurfaceInput,
    ) -> Result<crate::surface::SurfaceResult, crate::BehaveError> {
        crate::surface::run(input, &self.fuel_models)
    }

    /// Borrow the immutable registry snapshot used by this model.
    pub fn fuel_models(&self) -> &FuelModels {
        &self.fuel_models
    }

    /// Borrow the model's immutable mortality coefficient snapshot.
    pub fn mortality_coefficients(&self) -> &crate::mortality::MortalityLookup {
        &self.mortality_coefficients
    }

    /// Take ownership of a registry snapshot.
    pub fn new(fuel_models: FuelModels) -> Self {
        Self {
            fuel_models,
            mortality_coefficients: crate::mortality::build_mortality_lookup(None),
        }
    }
}

/// Derive a fractional crown ratio from canopy heights in the same length unit.
///
/// The reference uses a denominator of one for nonpositive heights and preserves
/// NaN propagation while clipping the fraction to zero through one.
pub fn canopy_crown_ratio(height: f64, base: f64) -> f64 {
    let safe_height = if height > 0.0 { height } else { 1.0 };
    crate::math::np_clip((height - base) / safe_height, 0.0, 1.0)
}
