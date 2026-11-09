//! Scalar Surface stage composition using only declared, retained fields.
use super::*;
use crate::surface as s;
use crate::units::{SlopeUnit, SpeedUnit};

impl GridSession {
    pub(super) fn surface_physical(&self, field: Field, index: usize) -> Result<f64, BehaveError> {
        let p = |field| self.physical(field, index);
        let aspect = || {
            if self.options.wind_orientation == super::super::WindOrientation::North {
                p(Field::Aspect)
            } else {
                Ok(0.)
            }
        };
        Ok(match field {
            Field::WindB | Field::WindC | Field::WindE => {
                let sigma = p(Field::FuelbedSigma)?;
                if sigma.is_nan() {
                    f64::NAN
                } else {
                    let c = s::calculate_surface_wind_coefficients(sigma);
                    match field {
                        Field::WindB => c.wind_b,
                        Field::WindC => c.wind_c,
                        _ => c.wind_e,
                    }
                }
            }
            Field::Wind20 => s::calculate_twenty_foot_wind_speed(
                p(Field::WindSpeed)?,
                SpeedUnit::FeetPerMinute,
                self.options.wind_height,
            ),
            Field::Waf if self.pipeline == Pipeline::Crown => 0.4,
            Field::Waf
                if self.options.waf_method == super::super::WafMethod::UserInput
                    && self.slots[Field::UserWaf as usize].active().is_some() =>
            {
                p(Field::UserWaf)?
            }
            Field::Waf => super::super::available_wind_adjustment_factor(
                p(Field::CanopyCover)?,
                p(Field::CanopyHeight)?,
                p(Field::CrownRatio)?,
                p(Field::FuelbedDepth)?,
                self.options.waf_method,
                None,
            ),
            Field::MidflameWindSpeed => {
                s::calculate_midflame_wind_speed(p(Field::Waf)?, p(Field::Wind20)?)
            }
            Field::WindPhi => {
                let sigma = p(Field::FuelbedSigma)?;
                let mid = p(Field::MidflameWindSpeed)?;
                let b = p(Field::WindB)?;
                let c = p(Field::WindC)?;
                let e = p(Field::WindE)?;
                let rpr = p(Field::FuelbedRelativePackingRatio)?;
                match super::super::missing::all_above(&[(sigma, 1e-7), (mid, 1e-7)]) {
                    None => f64::NAN,
                    Some(true) if b.is_nan() || c.is_nan() || e.is_nan() || rpr.is_nan() => {
                        f64::NAN
                    }
                    _ => s::calculate_wind_phi(sigma, mid, b, c, rpr, e),
                }
            }
            Field::SlopePhi => {
                let slope = p(Field::Slope)?;
                let packing = p(Field::FuelbedPackingRatio)?;
                if super::super::missing::all_above(&[
                    (slope.to_radians().tan(), 1e-7),
                    (packing, 1e-7),
                ])
                .is_none()
                {
                    f64::NAN
                } else {
                    s::calculate_slope_phi(
                        slope,
                        SlopeUnit::Degrees,
                        packing,
                        p(Field::ReactionIntensity)?,
                    )
                }
            }
            Field::NoWindNoSlopeSpreadRate => {
                let sink = p(Field::FuelbedHeatSink)?;
                if sink.is_nan() {
                    f64::NAN
                } else {
                    s::calculate_no_wind_no_slope_spread_rate(
                        p(Field::ReactionIntensity)?,
                        p(Field::FuelbedPropagatingFlux)?,
                        sink,
                    )
                }
            }
            Field::VectorX | Field::VectorY => {
                let vector = s::calculate_surface_rate_vector(
                    p(Field::NoWindNoSlopeSpreadRate)?,
                    p(Field::WindPhi)?,
                    p(Field::SlopePhi)?,
                    p(Field::WindDirection)?,
                    self.options.wind_orientation,
                    aspect()?,
                );
                if field == Field::VectorX {
                    vector.x_component
                } else {
                    vector.y_component
                }
            }
            Field::RateVector => (p(Field::VectorX)?.powi(2) + p(Field::VectorY)?.powi(2)).sqrt(),
            Field::SpreadRate => s::calculate_forward_spread_rate(
                p(Field::NoWindNoSlopeSpreadRate)?,
                p(Field::RateVector)?,
            ),
            Field::EffectiveWindSpeed => {
                let spread = p(Field::SpreadRate)?;
                let r0 = p(Field::NoWindNoSlopeSpreadRate)?;
                let b = p(Field::WindB)?;
                let c = p(Field::WindC)?;
                let e = p(Field::WindE)?;
                let rpr = p(Field::FuelbedRelativePackingRatio)?;
                let phi = if r0.is_nan() {
                    f64::NAN
                } else if r0 > 1e-7 {
                    spread / r0 - 1.
                } else {
                    0.
                };
                if super::super::missing::all_above(&[(phi, 0.), (b, 1e-7), (c, 1e-7), (rpr, 1e-7)])
                    .is_none()
                {
                    f64::NAN
                } else {
                    s::calculate_effective_wind_speed(spread, r0, b, c, rpr, e)
                }
            }
            Field::DirectionOfMaxSpread => {
                let rate = p(Field::RateVector)?;
                if rate.is_nan() {
                    f64::NAN
                } else {
                    s::calculate_direction_of_max_spread(
                        p(Field::VectorX)?,
                        p(Field::VectorY)?,
                        rate,
                        self.options.wind_orientation,
                        aspect()?,
                    )
                }
            }
            Field::FireLengthToWidthRatio => {
                let v = p(Field::EffectiveWindSpeed)?;
                if v.is_nan() {
                    f64::NAN
                } else {
                    s::calculate_fire_length_to_width_ratio(v)
                }
            }
            Field::Eccentricity => {
                let v = p(Field::FireLengthToWidthRatio)?;
                if v.is_nan() {
                    f64::NAN
                } else {
                    s::calculate_eccentricity(v)
                }
            }
            Field::BackingSpreadRate => {
                let v = p(Field::Eccentricity)?;
                if v.is_nan() {
                    f64::NAN
                } else {
                    s::calculate_backing_spread_rate(p(Field::SpreadRate)?, v)
                }
            }
            Field::FlankingSpreadRate => {
                let v = p(Field::FireLengthToWidthRatio)?;
                if v.is_nan() {
                    f64::NAN
                } else {
                    s::calculate_flanking_spread_rate(
                        p(Field::SpreadRate)?,
                        p(Field::BackingSpreadRate)?,
                        v,
                    )
                }
            }
            Field::ResidenceTime => {
                let v = p(Field::FuelbedSigma)?;
                if v.is_nan() {
                    f64::NAN
                } else {
                    s::calculate_residence_time(v)
                }
            }
            Field::HeatPerUnitArea => s::calculate_heat_per_unit_area(
                p(Field::ReactionIntensity)?,
                p(Field::ResidenceTime)?,
            ),
            Field::FirelineIntensity => {
                s::calculate_fireline_intensity(p(Field::SpreadRate)?, p(Field::HeatPerUnitArea)?)
            }
            Field::FlameLength => {
                let v = p(Field::FirelineIntensity)?;
                if v.is_nan() {
                    f64::NAN
                } else {
                    s::calculate_flame_length(v)
                }
            }
            _ => return self.crown_physical(field, index),
        })
    }
}
