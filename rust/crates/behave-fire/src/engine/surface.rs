//! Surface stage composition over numerical cells and enum-valued controls.
use super::{PreparedFuel, SurfaceOptions, SurfaceTweaks};
use crate::surface::{self, FuelbedIntermediates, SurfaceResult};
use crate::units::{SlopeUnit, SpeedUnit};

/// Numerical inputs in fractions, feet, ft/min, and degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceCell {
    /// Fuel lookup identifier, clamped by PreparedFuels.
    pub fuel_model_grid: i32,
    /// One-hour dead moisture fraction.
    pub m1h: f64,
    /// Ten-hour dead moisture fraction.
    pub m10h: f64,
    /// Hundred-hour dead moisture fraction.
    pub m100h: f64,
    /// Live herbaceous moisture fraction.
    pub mlh: f64,
    /// Live woody moisture fraction.
    pub mlw: f64,
    /// Open wind speed in ft/min at the plan's measurement height.
    pub wind_speed: f64,
    /// Wind bearing in degrees in the plan's coordinate reference.
    pub wind_direction: f64,
    /// Slope in degrees.
    pub slope: f64,
    /// Aspect in degrees clockwise from north.
    pub aspect: f64,
    /// Fractional canopy cover.
    pub canopy_cover: f64,
    /// Canopy height in feet.
    pub canopy_height: f64,
    /// Fractional crown ratio.
    pub crown_ratio: f64,
    /// Optional adjustment factor used by UserInput mode.
    pub user_waf: Option<f64>,
}

impl Default for SurfaceCell {
    /// Construct a calm, flat cell with representative moisture fractions.
    fn default() -> Self {
        Self {
            fuel_model_grid: 1,
            m1h: 0.06,
            m10h: 0.07,
            m100h: 0.08,
            mlh: 0.60,
            mlw: 0.90,
            wind_speed: 0.0,
            wind_direction: 0.0,
            slope: 0.0,
            aspect: 0.0,
            canopy_cover: 0.0,
            canopy_height: 0.0,
            crown_ratio: 0.0,
            user_waf: None,
        }
    }
}

impl SurfaceCell {
    /// Construct base-unit inputs for a selected grid identifier.
    pub fn new(fuel_model_grid: i32) -> Self {
        Self {
            fuel_model_grid,
            ..Self::default()
        }
    }
}

/// Immutable Surface configuration reusable across cells and raster tiles.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfacePlan {
    options: SurfaceOptions,
}

impl SurfacePlan {
    /// Construct an enum-configured plan without allocating state.
    pub const fn new(options: SurfaceOptions) -> Self {
        Self { options }
    }

    /// Borrow the controls used for every cell.
    pub const fn options(&self) -> &SurfaceOptions {
        &self.options
    }

    /// Build moisture-dependent particles and fuelbed state from a numeric fuel.
    pub fn prepare_fuelbed(&self, cell: &SurfaceCell, fuel: &PreparedFuel) -> FuelbedIntermediates {
        let particles = surface::build_particle_arrays_from_prepared(
            fuel, cell.m1h, cell.m10h, cell.m100h, cell.mlh, cell.mlw,
        );
        surface::calculate_fuelbed_intermediates(&particles)
    }

    /// Evaluate a cell with explicit stage overrides and no per-cell ownership.
    pub fn run_cell(
        &self,
        cell: &SurfaceCell,
        fuel: &PreparedFuel,
        tweaks: SurfaceTweaks,
    ) -> SurfaceResult {
        self.run_from_fuelbed(cell, &self.prepare_fuelbed(cell, fuel), tweaks)
    }

    /// Resume from supplied fuelbed state without rebuilding or replacing it.
    pub fn run_from_fuelbed(
        &self,
        cell: &SurfaceCell,
        fuelbed: &FuelbedIntermediates,
        tweaks: SurfaceTweaks,
    ) -> SurfaceResult {
        let reaction = tweaks
            .reaction_intensity
            .unwrap_or_else(|| surface::calculate_reaction_intensity(fuelbed));
        run_stages(
            reaction,
            0.9 * reaction,
            fuelbed,
            cell,
            &self.options,
            tweaks,
        )
    }
}

/// Compose shared spread stages with an already-rounded reaction slope limit.
pub(crate) fn run_stages(
    reaction: f64,
    slope_limit: f64,
    ib: &FuelbedIntermediates,
    cell: &SurfaceCell,
    options: &SurfaceOptions,
    tweaks: SurfaceTweaks,
) -> SurfaceResult {
    let wc = surface::calculate_surface_wind_coefficients(ib.sigma);
    let r0 = tweaks.no_wind_no_slope_spread_rate.unwrap_or_else(|| {
        surface::calculate_no_wind_no_slope_spread_rate(reaction, ib.propagating_flux, ib.heat_sink)
    });
    let wind20 = surface::calculate_twenty_foot_wind_speed(
        cell.wind_speed,
        SpeedUnit::FeetPerMinute,
        options.wind_height,
    );
    let waf = surface::calculate_surface_wind_adjustment_factor(
        cell.canopy_cover,
        cell.canopy_height,
        cell.crown_ratio,
        ib.depth,
        options.waf_method,
        cell.user_waf,
    );
    let midflame = tweaks
        .midflame_wind_speed
        .unwrap_or_else(|| surface::calculate_midflame_wind_speed(waf, wind20));
    let wind_phi = tweaks.wind_phi.unwrap_or_else(|| {
        surface::calculate_wind_phi(
            ib.sigma,
            midflame,
            wc.wind_b,
            wc.wind_c,
            ib.relative_packing_ratio,
            wc.wind_e,
        )
    });
    let slope_phi = tweaks.slope_phi.unwrap_or_else(|| {
        crate::math::np_min(
            surface::calculate_slope_phi(
                cell.slope,
                SlopeUnit::Degrees,
                ib.packing_ratio,
                f64::INFINITY,
            ),
            slope_limit,
        )
    });
    let vector = surface::calculate_surface_rate_vector(
        r0,
        wind_phi,
        slope_phi,
        cell.wind_direction,
        options.wind_orientation,
        cell.aspect,
    );
    let spread_rate = tweaks
        .spread_rate
        .unwrap_or_else(|| surface::calculate_forward_spread_rate(r0, vector.rate_vector));
    let effective_wind_speed = surface::calculate_effective_wind_speed(
        spread_rate,
        r0,
        wc.wind_b,
        wc.wind_c,
        ib.relative_packing_ratio,
        wc.wind_e,
    );
    let lwr = surface::calculate_fire_length_to_width_ratio(effective_wind_speed);
    let eccentricity = surface::calculate_eccentricity(lwr);
    let backing = surface::calculate_backing_spread_rate(spread_rate, eccentricity);
    let flanking = surface::calculate_flanking_spread_rate(spread_rate, backing, lwr);
    let residence = surface::calculate_residence_time(ib.sigma);
    let heat = surface::calculate_heat_per_unit_area(reaction, residence);
    let intensity = surface::calculate_fireline_intensity(spread_rate, heat);
    let direction = surface::calculate_direction_of_max_spread(
        vector.x_component,
        vector.y_component,
        vector.rate_vector,
        options.wind_orientation,
        cell.aspect,
    );
    if !ib.is_defined {
        return SurfaceResult {
            fire_length_to_width_ratio: 1.0,
            ..SurfaceResult::default()
        };
    }
    SurfaceResult {
        spread_rate,
        backing_spread_rate: backing,
        flanking_spread_rate: flanking,
        flame_length: surface::calculate_flame_length(intensity),
        fireline_intensity: intensity,
        heat_per_unit_area: heat,
        effective_wind_speed,
        fire_length_to_width_ratio: lwr,
        eccentricity,
        direction_of_max_spread: direction,
        residence_time: residence,
        reaction_intensity: surface::calculate_reaction_intensity_output(reaction),
        midflame_wind_speed: midflame,
        no_wind_no_slope_spread_rate: r0,
    }
}
