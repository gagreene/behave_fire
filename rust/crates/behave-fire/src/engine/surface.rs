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
        fuel.prepare_fuelbed(&particles)
    }

    /// Evaluate a cell with explicit stage overrides and no per-cell ownership.
    pub fn run_cell(
        &self,
        cell: &SurfaceCell,
        fuel: &PreparedFuel,
        tweaks: SurfaceTweaks,
    ) -> SurfaceResult {
        let fuelbed = self.prepare_fuelbed(cell, fuel);
        let reaction = tweaks
            .reaction_intensity
            .unwrap_or_else(|| fuel.reaction_intensity(&fuelbed));
        run_selected_stages::<0, false>(
            reaction,
            0.9 * reaction,
            &fuelbed,
            cell,
            &self.options,
            tweaks,
        )
    }

    /// Resume from supplied fuelbed state without rebuilding or replacing it.
    pub fn run_from_fuelbed(
        &self,
        cell: &SurfaceCell,
        fuelbed: &FuelbedIntermediates,
        tweaks: SurfaceTweaks,
    ) -> SurfaceResult {
        self.run_selected_from_fuelbed::<0>(cell, fuelbed, tweaks)
    }

    /// Evaluate only the stages consumed by the native caller.
    pub(crate) fn run_selected_from_fuelbed<const FINISH: u8>(
        &self,
        cell: &SurfaceCell,
        fuelbed: &FuelbedIntermediates,
        tweaks: SurfaceTweaks,
    ) -> SurfaceResult {
        let reaction = tweaks
            .reaction_intensity
            .unwrap_or_else(|| surface::calculate_reaction_intensity(fuelbed));
        run_selected_stages::<FINISH, false>(
            reaction,
            0.9 * reaction,
            fuelbed,
            cell,
            &self.options,
            tweaks,
        )
    }
    /// Evaluate the availability-aware model contract with cached physical fuel data.
    pub(crate) fn run_available_cell(
        &self,
        cell: &SurfaceCell,
        fuel: &PreparedFuel,
        tweaks: SurfaceTweaks,
    ) -> SurfaceResult {
        let fuelbed = self.prepare_fuelbed(cell, fuel);
        let reaction = tweaks
            .reaction_intensity
            .unwrap_or_else(|| fuel.reaction_intensity(&fuelbed));
        run_selected_stages::<0, true>(
            reaction,
            0.9 * reaction,
            &fuelbed,
            cell,
            &self.options,
            tweaks,
        )
    }
    /// Resume an availability-aware run from supplied fuelbed state.
    pub(crate) fn run_available_from_fuelbed<const FINISH: u8>(
        &self,
        cell: &SurfaceCell,
        fuelbed: &FuelbedIntermediates,
        tweaks: SurfaceTweaks,
    ) -> SurfaceResult {
        let reaction = tweaks
            .reaction_intensity
            .unwrap_or_else(|| surface::calculate_reaction_intensity(fuelbed));
        run_selected_stages::<FINISH, true>(
            reaction,
            0.9 * reaction,
            fuelbed,
            cell,
            &self.options,
            tweaks,
        )
    }
}

/// Share spread equations while omitting unused Crown finishing stages.
fn run_selected_stages<const FINISH: u8, const STRICT: bool>(
    reaction: f64,
    slope_limit: f64,
    ib: &FuelbedIntermediates,
    cell: &SurfaceCell,
    options: &SurfaceOptions,
    tweaks: SurfaceTweaks,
) -> SurfaceResult {
    let mut wc = surface::calculate_surface_wind_coefficients(ib.sigma);
    if STRICT && ib.sigma.is_nan() {
        wc.safe_sigma = f64::NAN;
        wc.wind_b = f64::NAN;
        wc.wind_c = f64::NAN;
        wc.wind_e = f64::NAN;
    }
    let mut r0 = tweaks.no_wind_no_slope_spread_rate.unwrap_or_else(|| {
        surface::calculate_no_wind_no_slope_spread_rate(reaction, ib.propagating_flux, ib.heat_sink)
    });
    if STRICT && tweaks.no_wind_no_slope_spread_rate.is_none() && ib.heat_sink.is_nan() {
        r0 = f64::NAN;
    }
    let wind20 = surface::calculate_twenty_foot_wind_speed(
        cell.wind_speed,
        SpeedUnit::FeetPerMinute,
        options.wind_height,
    );
    let waf = if STRICT {
        super::missing::waf(cell, ib.depth, options)
    } else {
        surface::calculate_surface_wind_adjustment_factor(
            cell.canopy_cover,
            cell.canopy_height,
            cell.crown_ratio,
            ib.depth,
            options.waf_method,
            cell.user_waf,
        )
    };
    let midflame = tweaks
        .midflame_wind_speed
        .unwrap_or_else(|| surface::calculate_midflame_wind_speed(waf, wind20));
    let mut wind_phi = tweaks.wind_phi.unwrap_or_else(|| {
        surface::calculate_wind_phi(
            ib.sigma,
            midflame,
            wc.wind_b,
            wc.wind_c,
            ib.relative_packing_ratio,
            wc.wind_e,
        )
    });
    if STRICT && tweaks.wind_phi.is_none() {
        match super::missing::all_above(&[(ib.sigma, 1e-7), (midflame, 1e-7)]) {
            None => wind_phi = f64::NAN,
            Some(true)
                if wc.wind_b.is_nan()
                    || wc.wind_c.is_nan()
                    || wc.wind_e.is_nan()
                    || ib.relative_packing_ratio.is_nan() =>
            {
                wind_phi = f64::NAN
            }
            _ => {}
        }
    }
    let mut slope_phi = tweaks.slope_phi.unwrap_or_else(|| {
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
    if STRICT
        && tweaks.slope_phi.is_none()
        && super::missing::all_above(&[
            (cell.slope.to_radians().tan(), 1e-7),
            (ib.packing_ratio, 1e-7),
        ])
        .is_none()
    {
        slope_phi = f64::NAN;
    }
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
    if FINISH == 1 {
        return SurfaceResult {
            spread_rate: if ib.is_defined { spread_rate } else { 0.0 },
            ..SurfaceResult::default()
        };
    }
    let mut effective_wind_speed = surface::calculate_effective_wind_speed(
        spread_rate,
        r0,
        wc.wind_b,
        wc.wind_c,
        ib.relative_packing_ratio,
        wc.wind_e,
    );
    if STRICT {
        let phi = if r0.is_nan() {
            f64::NAN
        } else if r0 > 1e-7 {
            spread_rate / r0 - 1.0
        } else {
            0.0
        };
        if super::missing::all_above(&[
            (phi, 0.0),
            (wc.wind_b, 1e-7),
            (wc.wind_c, 1e-7),
            (ib.relative_packing_ratio, 1e-7),
        ])
        .is_none()
        {
            effective_wind_speed = f64::NAN;
        }
    }
    if FINISH == 2 {
        return SurfaceResult {
            spread_rate: if ib.is_defined { spread_rate } else { 0.0 },
            effective_wind_speed: if ib.is_defined {
                effective_wind_speed
            } else {
                0.0
            },
            ..SurfaceResult::default()
        };
    }
    let lwr = if STRICT && effective_wind_speed.is_nan() {
        f64::NAN
    } else {
        surface::calculate_fire_length_to_width_ratio(effective_wind_speed)
    };
    let eccentricity = if STRICT && lwr.is_nan() {
        f64::NAN
    } else {
        surface::calculate_eccentricity(lwr)
    };
    let backing = if STRICT && eccentricity.is_nan() {
        f64::NAN
    } else {
        surface::calculate_backing_spread_rate(spread_rate, eccentricity)
    };
    let flanking = if STRICT && lwr.is_nan() {
        f64::NAN
    } else {
        surface::calculate_flanking_spread_rate(spread_rate, backing, lwr)
    };
    let residence = if STRICT && ib.sigma.is_nan() {
        f64::NAN
    } else {
        surface::calculate_residence_time(ib.sigma)
    };
    let heat = surface::calculate_heat_per_unit_area(reaction, residence);
    let intensity = surface::calculate_fireline_intensity(spread_rate, heat);
    let direction = if STRICT && vector.rate_vector.is_nan() {
        f64::NAN
    } else {
        surface::calculate_direction_of_max_spread(
            vector.x_component,
            vector.y_component,
            vector.rate_vector,
            options.wind_orientation,
            cell.aspect,
        )
    };
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
        flame_length: if STRICT && intensity.is_nan() {
            f64::NAN
        } else {
            surface::calculate_flame_length(intensity)
        },
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

#[cfg(test)]
mod spread_tests {
    use super::*;
    use crate::{
        engine::{PreparedFuel, WindOrientation},
        FuelModels,
    };
    #[test]
    fn crown_spread_stages_preserve_first_run_backsolve_and_overrides() {
        let models = FuelModels::default();
        for id in [0, 10, 101, 124, 145, 201] {
            let fuel = PreparedFuel::from_model(models.for_grid(id));
            let plan = SurfacePlan::new(SurfaceOptions {
                wind_orientation: WindOrientation::North,
                ..SurfaceOptions::default()
            });
            let cell = SurfaceCell {
                slope: 27.0,
                aspect: 113.0,
                wind_direction: 271.0,
                wind_speed: 551.0,
                ..SurfaceCell::new(id)
            };
            let bed = plan.prepare_fuelbed(&cell, &fuel);
            for tweaks in [
                SurfaceTweaks::default(),
                SurfaceTweaks {
                    reaction_intensity: Some(913.0),
                    no_wind_no_slope_spread_rate: Some(3.1),
                    midflame_wind_speed: Some(61.0),
                    wind_phi: Some(2.7),
                    slope_phi: Some(0.9),
                    spread_rate: Some(13.7),
                },
            ] {
                let full = plan.run_from_fuelbed(&cell, &bed, tweaks);
                let rate = plan.run_selected_from_fuelbed::<1>(&cell, &bed, tweaks);
                let wind = plan.run_selected_from_fuelbed::<2>(&cell, &bed, tweaks);
                assert_eq!(rate.spread_rate.to_bits(), full.spread_rate.to_bits());
                assert_eq!(wind.spread_rate.to_bits(), full.spread_rate.to_bits());
                assert_eq!(
                    wind.effective_wind_speed.to_bits(),
                    full.effective_wind_speed.to_bits()
                );
            }
        }
    }
}
