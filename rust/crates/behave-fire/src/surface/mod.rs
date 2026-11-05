//! Scalar surface fire stages in reference base units.
mod fuelbed;
mod geometry;
/// NumPy-compatible masked-array scalar stages, including raw masked data and precision.
pub mod masked;
mod particles;
mod reaction;
mod types;
mod wind;
pub use fuelbed::*;
pub use geometry::*;
pub use particles::*;
pub use reaction::*;
pub use types::*;
pub use wind::*;

use crate::{
    fuel_models::FuelModels,
    units::{SlopeUnit, SpeedUnit},
    BehaveError,
};

/// Surface inputs use moisture and canopy fractions, canopy height in feet, and selected wind/slope units.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct SurfaceInput {
    /// Grid lookup model number, clamped to 0..260 like Python.
    pub fuel_model_grid: i32,
    /// One-hour dead moisture fraction.
    pub m1h: f64,
    /// Ten-hour dead moisture fraction.
    pub m10h: f64,
    /// Hundred-hour dead moisture fraction.
    pub m100h: f64,
    /// Live herb moisture fraction.
    pub mlh: f64,
    /// Live woody moisture fraction.
    pub mlw: f64,
    /// Open wind speed in wind_speed_units.
    pub wind_speed: f64,
    /// Wind units.
    pub wind_speed_units: SpeedUnit,
    /// Direction in degrees, interpreted using wind_orientation_mode.
    pub wind_direction: f64,
    /// Names containing north use north orientation, case-insensitively.
    pub wind_orientation_mode: String,
    /// Slope in slope_units.
    pub slope: f64,
    /// Slope units.
    pub slope_units: SlopeUnit,
    /// Aspect clockwise from north in degrees.
    pub aspect: f64,
    /// Canopy cover fraction.
    pub canopy_cover: f64,
    /// Canopy height in feet.
    pub canopy_height: f64,
    /// Crown ratio fraction.
    pub crown_ratio: f64,
    /// Names containing 10 or ten apply the 10-meter conversion.
    pub wind_height_mode: String,
    /// UserInput (ignoring underscores and case) enables user_waf.
    pub waf_method: String,
    /// Optional user wind adjustment factor.
    pub user_waf: Option<f64>,
}

impl Default for SurfaceInput {
    /// Construct a calm, flat, open fuel bed with representative fractional moisture inputs.
    fn default() -> Self {
        Self {
            fuel_model_grid: 1,
            m1h: 0.06,
            m10h: 0.07,
            m100h: 0.08,
            mlh: 0.60,
            mlw: 0.90,
            wind_speed: 0.0,
            wind_speed_units: SpeedUnit::MilesPerHour,
            wind_direction: 0.0,
            wind_orientation_mode: "RelativeToUpslope".into(),
            slope: 0.0,
            slope_units: SlopeUnit::Degrees,
            aspect: 0.0,
            canopy_cover: 0.0,
            canopy_height: 0.0,
            crown_ratio: 0.0,
            wind_height_mode: "TwentyFoot".into(),
            waf_method: "UseCrownRatio".into(),
            user_waf: None,
        }
    }
}
impl SurfaceInput {
    /// Construct conventional base-unit inputs for a selected fuel model.
    pub fn new(fuel_model_grid: i32) -> Self {
        Self {
            fuel_model_grid,
            ..Self::default()
        }
    }
}

/// Run particle construction, fuelbed weighting, reaction intensity, and spread locally for one cell.
pub fn run(input: &SurfaceInput, fuels: &FuelModels) -> Result<SurfaceResult, BehaveError> {
    run_with_fuel(input, fuels.for_grid(input.fuel_model_grid))
}

/// Run all scalar stages from an exact raw fuel lookup record.
pub fn run_with_fuel(
    input: &SurfaceInput,
    fuel: &crate::FuelModel,
) -> Result<SurfaceResult, BehaveError> {
    let p = build_particle_arrays_from_fuel(
        fuel,
        input.m1h,
        input.m10h,
        input.m100h,
        input.mlh,
        input.mlw,
    );
    let ib = calculate_fuelbed_intermediates(&p);
    Ok(run_surface_fire(
        calculate_reaction_intensity(&ib),
        &ib,
        input,
    ))
}

/// Run the spread and shape stages from supplied reaction intensity (BTU/ft²/min) and fuelbed intermediates.
pub fn run_surface_fire(ri: f64, ib: &FuelbedIntermediates, input: &SurfaceInput) -> SurfaceResult {
    run_surface_fire_with_slope_limit(ri, ib, input, 0.9 * ri)
}

/// Run supplied f32 reaction intensity with f64 intermediates, retaining Python's f32 slope-limit product.
pub fn run_surface_fire_single_reaction(
    ri: f32,
    ib: &FuelbedIntermediates,
    input: &SurfaceInput,
) -> SurfaceResult {
    run_surface_fire_with_slope_limit(ri as f64, ib, input, (0.9 * ri) as f64)
}

/// Compose spread stages with the reaction input's already-rounded slope cap.
fn run_surface_fire_with_slope_limit(
    ri: f64,
    ib: &FuelbedIntermediates,
    input: &SurfaceInput,
    slope_limit: f64,
) -> SurfaceResult {
    let wc = calculate_surface_wind_coefficients(ib.sigma);
    let r0 = calculate_no_wind_no_slope_spread_rate(ri, ib.propagating_flux, ib.heat_sink);
    let ws = calculate_twenty_foot_wind_speed(
        input.wind_speed,
        input.wind_speed_units,
        &input.wind_height_mode,
    );
    let waf = calculate_surface_wind_adjustment_factor(
        input.canopy_cover,
        input.canopy_height,
        input.crown_ratio,
        ib.depth,
        &input.waf_method,
        input.user_waf,
    );
    let midflame = calculate_midflame_wind_speed(waf, ws);
    let wp = calculate_wind_phi(
        ib.sigma,
        midflame,
        wc.wind_b,
        wc.wind_c,
        ib.relative_packing_ratio,
        wc.wind_e,
    );
    let sp = crate::math::np_min(
        calculate_slope_phi(
            input.slope,
            input.slope_units,
            ib.packing_ratio,
            f64::INFINITY,
        ),
        slope_limit,
    );
    let vector = calculate_surface_rate_vector(
        r0,
        wp,
        sp,
        input.wind_direction,
        &input.wind_orientation_mode,
        input.aspect,
    );
    let spread_rate = calculate_forward_spread_rate(r0, vector.rate_vector);
    let effective_wind_speed = calculate_effective_wind_speed(
        spread_rate,
        r0,
        wc.wind_b,
        wc.wind_c,
        ib.relative_packing_ratio,
        wc.wind_e,
    );
    let fire_length_to_width_ratio = calculate_fire_length_to_width_ratio(effective_wind_speed);
    let eccentricity = calculate_eccentricity(fire_length_to_width_ratio);
    let backing_spread_rate = calculate_backing_spread_rate(spread_rate, eccentricity);
    let flanking_spread_rate = calculate_flanking_spread_rate(
        spread_rate,
        backing_spread_rate,
        fire_length_to_width_ratio,
    );
    let residence_time = calculate_residence_time(ib.sigma);
    let heat_per_unit_area = calculate_heat_per_unit_area(ri, residence_time);
    let fireline_intensity = calculate_fireline_intensity(spread_rate, heat_per_unit_area);
    let flame_length = calculate_flame_length(fireline_intensity);
    let direction_of_max_spread = calculate_direction_of_max_spread(
        vector.x_component,
        vector.y_component,
        vector.rate_vector,
        &input.wind_orientation_mode,
        input.aspect,
    );
    if !ib.is_defined {
        return SurfaceResult {
            fire_length_to_width_ratio: 1.0,
            ..SurfaceResult::default()
        };
    }
    SurfaceResult {
        spread_rate,
        backing_spread_rate,
        flanking_spread_rate,
        flame_length,
        fireline_intensity,
        heat_per_unit_area,
        effective_wind_speed,
        fire_length_to_width_ratio,
        eccentricity,
        direction_of_max_spread,
        residence_time,
        reaction_intensity: calculate_reaction_intensity_output(ri),
        midflame_wind_speed: midflame,
        no_wind_no_slope_spread_rate: r0,
    }
}

/// Native single-precision variants for helpers whose Python arithmetic retains float32.
pub mod single {
    pub use super::fuelbed::single::*;
    pub use super::geometry::single::*;
    pub use super::reaction::single::*;
    pub use super::wind::single::*;
}
