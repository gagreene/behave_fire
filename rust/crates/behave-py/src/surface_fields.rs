//! Actual fuel-only and wind-only closures for independent Surface output groups.
use behave_fire::engine::{
    available_wind_adjustment_factor, Field as F, PreparedFuels, SurfaceOptions, WafMethod,
    WindHeight,
};
use behave_fire::surface::{self, SurfaceResult};
use behave_fire::PerCell;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::BTreeMap;

struct Physics<'a> {
    moisture: [PerCell<'a, f64>; 5],
}
enum Wind<'a> {
    User {
        speed: PerCell<'a, f64>,
        waf: PerCell<'a, f64>,
    },
    Computed {
        speed: PerCell<'a, f64>,
        cover: PerCell<'a, f64>,
        height: PerCell<'a, f64>,
        ratio: PerCell<'a, f64>,
    },
}
pub(crate) struct Program<'a> {
    requested: Vec<F>,
    ids: PerCell<'a, i32>,
    fuels: &'a PreparedFuels,
    physics: Option<Physics<'a>>,
    wind: Option<Wind<'a>>,
    options: SurfaceOptions,
}
impl<'a> Program<'a> {
    pub(crate) fn new(
        source: &'a BTreeMap<String, Vec<f64>>,
        ids: PerCell<'a, i32>,
        fields: &[F],
        fuels: &'a PreparedFuels,
        options: SurfaceOptions,
    ) -> PyResult<Self> {
        let col = |name| crate::model_f64::column(source, name);
        let mut need_physics = false;
        let mut need_wind = false;
        for field in fields {
            match field {
                F::ReactionIntensity
                | F::ResidenceTime
                | F::HeatPerUnitArea
                | F::NoWindNoSlopeSpreadRate => need_physics = true,
                F::MidflameWindSpeed => need_wind = true,
                _ => {
                    return Err(PyValueError::new_err(format!(
                        "restricted Surface evaluator not implemented for {}",
                        field.name()
                    )))
                }
            }
        }
        let physics = if need_physics {
            Some(Physics {
                moisture: [
                    col("m1h")?,
                    col("m10h")?,
                    col("m100h")?,
                    col("mlh")?,
                    col("mlw")?,
                ],
            })
        } else {
            None
        };
        let wind = if need_wind {
            Some(
                if options.waf_method == WafMethod::UserInput && source.contains_key("user_waf") {
                    Wind::User {
                        speed: col("wind_speed")?,
                        waf: col("user_waf")?,
                    }
                } else {
                    Wind::Computed {
                        speed: col("wind_speed")?,
                        cover: col("canopy_cover")?,
                        height: col("canopy_height")?,
                        ratio: col("crown_ratio")?,
                    }
                },
            )
        } else {
            None
        };
        Ok(Self {
            requested: fields.to_vec(),
            ids,
            fuels,
            physics,
            wind,
            options,
        })
    }

    pub(crate) fn evaluate(&self, index: usize) -> SurfaceResult {
        let id = *self.ids.get(index).expect("validated Surface ID");
        let fuel = self.fuels.for_grid(id);
        if !fuel.is_defined {
            return SurfaceResult::new();
        }
        let at = |v: &PerCell<'_, f64>| *v.get(index).expect("validated Surface closure");
        let physics = self.physics.as_ref().map(|values| {
            let m = values.moisture.map(|v| at(&v));
            let particles =
                surface::build_particle_arrays_from_prepared(fuel, m[0], m[1], m[2], m[3], m[4]);
            let bed = fuel.prepare_fuelbed_available(&particles);
            let reaction = fuel.reaction_intensity_available(&bed);
            let residence = if bed.sigma.is_nan() {
                f64::NAN
            } else {
                surface::calculate_residence_time(bed.sigma)
            };
            let rate = if bed.heat_sink.is_nan() {
                f64::NAN
            } else {
                surface::calculate_no_wind_no_slope_spread_rate(
                    reaction,
                    bed.propagating_flux,
                    bed.heat_sink,
                )
            };
            (reaction, residence, rate)
        });
        let mid = self.wind.as_ref().map(|wind| {
            let (speed, waf) = match wind {
                Wind::User { speed, waf } => (at(speed), at(waf)),
                Wind::Computed {
                    speed,
                    cover,
                    height,
                    ratio,
                } => (
                    at(speed),
                    available_wind_adjustment_factor(
                        at(cover),
                        at(height),
                        at(ratio),
                        fuel.fuel_bed_depth,
                        self.options.waf_method,
                        None,
                    ),
                ),
            };
            let twenty = if self.options.wind_height == WindHeight::TenMetre {
                speed / 1.15
            } else {
                speed
            };
            surface::calculate_midflame_wind_speed(waf, twenty)
        });
        let mut result = SurfaceResult::new();
        for field in &self.requested {
            match field {
                F::ReactionIntensity => {
                    result.reaction_intensity = physics.expect("compiled physics").0
                }
                F::ResidenceTime => result.residence_time = physics.expect("compiled physics").1,
                F::HeatPerUnitArea => {
                    let (r, t, _) = physics.expect("compiled physics");
                    result.heat_per_unit_area = surface::calculate_heat_per_unit_area(r, t);
                }
                F::NoWindNoSlopeSpreadRate => {
                    result.no_wind_no_slope_spread_rate = physics.expect("compiled physics").2
                }
                F::MidflameWindSpeed => result.midflame_wind_speed = mid.expect("compiled wind"),
                _ => unreachable!("validated Surface field"),
            }
        }
        result
    }

    pub(crate) fn missing(&self) -> SurfaceResult {
        let mut result = SurfaceResult::new();
        for field in &self.requested {
            match field {
                F::ReactionIntensity => result.reaction_intensity = f64::NAN,
                F::ResidenceTime => result.residence_time = f64::NAN,
                F::HeatPerUnitArea => result.heat_per_unit_area = f64::NAN,
                F::NoWindNoSlopeSpreadRate => result.no_wind_no_slope_spread_rate = f64::NAN,
                F::MidflameWindSpeed => result.midflame_wind_speed = f64::NAN,
                _ => unreachable!("validated Surface field"),
            }
        }
        result
    }
}
