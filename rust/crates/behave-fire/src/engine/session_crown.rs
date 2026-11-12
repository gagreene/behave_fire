//! Independent Crown fields and selected final branches from actual staged values.
use super::*;
use crate::crown as c;

impl GridSession {
    pub(super) fn crown_physical(&self, field: Field, index: usize) -> Result<f64, BehaveError> {
        let p = |field| self.physical(field, index);
        Ok(match field {
            Field::CrownFireSpreadRate => 3.34 * p(Field::SpreadRate)?,
            Field::CrownLengthToWidthRatio => {
                let v = p(Field::EffectiveWindSpeed)?;
                if v.is_nan() {
                    f64::NAN
                } else if v > 1e-7 {
                    1. + 0.125 * v
                } else {
                    1.
                }
            }
            Field::CrownCriticalSurfaceFireLineIntensity => {
                c::calculate_critical_surface_fireline_intensity(
                    p(Field::CanopyBaseHeight)?,
                    p(Field::MoistureFoliar)?,
                )
            }
            Field::CrownCriticalFireSpreadRate => {
                c::available::critical_rate(p(Field::CanopyBulkDensity)?)
            }
            Field::CrownFireLineIntensity => {
                (p(Field::CrownFireSpreadRate)? / 60.) * p(Field::CrownFireHeatPerUnitArea)?
            }
            Field::CrownFlameLength => {
                c::available::flame_length(p(Field::CrownFireLineIntensity)?)
            }
            Field::CrownFireTransitionRatio => c::available::ratio(
                p(Field::SurfaceFli)?,
                p(Field::CrownCriticalSurfaceFireLineIntensity)?,
            ),
            Field::CrownFireActiveRatio => c::available::ratio(
                p(Field::CrownFireSpreadRate)?,
                p(Field::CrownCriticalFireSpreadRate)?,
            ),
            Field::SurfaceFireCriticalSpreadRate => c::available::surface_critical(
                p(Field::CrownCriticalSurfaceFireLineIntensity)?,
                p(Field::SurfaceHpua)?,
            ),
            Field::CrowningSurfaceFireSpreadRate => {
                let critical = p(Field::CrownCriticalFireSpreadRate)?;
                let reaction = p(Field::ReactionIntensity)?;
                let sink = p(Field::FuelbedHeatSink)?;
                let flux = p(Field::FuelbedPropagatingFlux)?;
                if critical.is_nan() || reaction.is_nan() || sink.is_nan() || flux.is_nan() {
                    f64::NAN
                } else {
                    c::crowning_from_inputs(
                        &c::CrowningInputs {
                            reaction,
                            sigma: p(Field::FuelbedSigma)?,
                            packing: p(Field::FuelbedPackingRatio)?,
                            relative_packing: p(Field::FuelbedRelativePackingRatio)?,
                            flux,
                            heat_sink: sink,
                            defined: self.flag_value(Field::FuelbedDefined, index)?,
                            target: critical / 3.34,
                            wind_b: p(Field::WindB)?,
                            wind_c: p(Field::WindC)?,
                            wind_e: p(Field::WindE)?,
                        },
                        None,
                    )
                }
            }
            Field::CrownFractionBurned => c::available::fraction(
                p(Field::SurfaceRos)?,
                p(Field::SurfaceFireCriticalSpreadRate)?,
                p(Field::CrowningSurfaceFireSpreadRate)?,
            ),
            Field::PassiveCrownFireSpreadRate => c::calculate_passive_spread_rate(
                p(Field::SurfaceRos)?,
                p(Field::CrownFireSpreadRate)?,
                p(Field::CrownFractionBurned)?,
            ),
            Field::PassiveCrownFireHeatPerUnitArea => c::calculate_passive_heat_per_unit_area(
                p(Field::SurfaceHpua)?,
                p(Field::CanopyHeatPerUnitArea)?,
                p(Field::CrownFractionBurned)?,
            ),
            Field::PassiveCrownFireLineIntensity => c::calculate_passive_intensity(
                p(Field::PassiveCrownFireHeatPerUnitArea)?,
                p(Field::PassiveCrownFireSpreadRate)?,
            ),
            Field::PassiveCrownFireFlameLength => {
                c::available::flame_length(p(Field::PassiveCrownFireLineIntensity)?)
            }
            Field::FinalSpreadRate
            | Field::FinalHeatPerUnitArea
            | Field::FinalFirelineIntensity
            | Field::FinalFlameLength => {
                // Explicit flags select final behavior; Surface wins over passive,
                // then active. All-false flags retain unavailable classification.
                let code = if self.flag_value(Field::IsSurfaceFire, index)? {
                    0
                } else if self.flag_value(Field::IsPassiveCrown, index)? {
                    1
                } else if self.flag_value(Field::IsActiveCrown, index)? {
                    3
                } else {
                    -1
                };
                let sources = match field {
                    Field::FinalSpreadRate => (
                        Field::SurfaceRos,
                        Field::PassiveCrownFireSpreadRate,
                        Field::CrownFireSpreadRate,
                    ),
                    Field::FinalHeatPerUnitArea => (
                        Field::SurfaceHpua,
                        Field::PassiveCrownFireHeatPerUnitArea,
                        Field::CrownFireHeatPerUnitArea,
                    ),
                    Field::FinalFirelineIntensity => (
                        Field::SurfaceFli,
                        Field::PassiveCrownFireLineIntensity,
                        Field::CrownFireLineIntensity,
                    ),
                    _ => (
                        Field::SurfaceFli,
                        Field::PassiveCrownFireFlameLength,
                        Field::CrownFlameLength,
                    ),
                };
                match code {
                    0 | 2 if field == Field::FinalFlameLength => {
                        c::available::flame_length(p(sources.0)?)
                    }
                    0 | 2 => p(sources.0)?,
                    1 => p(sources.1)?,
                    3 => p(sources.2)?,
                    _ => f64::NAN,
                }
            }
            _ => return Err(access(field, "staged field evaluator is not implemented")),
        })
    }
}
