//! Native crown_dispatch marshalling and scalar entry-point dispatch.
use crate::column_output::RowWriter;
use crate::dispatch::{classification, named, scalar, widen};
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
use behave_fire::crown;
#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    single: bool,
    out: &mut RowWriter<'_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::CrownAssignFinalCrownFireBehavior => {
            if !single && a.optional(Field::GlobalMixed, i).is_some() {
                named(
                    out,
                    crown::assign_final_crown_fire_behavior_single_surface(
                        a.n(Field::IsSurfaceFire, i)? != 0.0,
                        a.n(Field::IsPassiveCrown, i)? != 0.0,
                        a.n(Field::SurfaceRos, i)? as f32,
                        a.n(Field::SurfaceHpua, i)? as f32,
                        a.n(Field::SurfaceFli, i)? as f32,
                        a.n(Field::PassiveRos, i)?,
                        a.n(Field::PassiveHpua, i)?,
                        a.n(Field::PassiveFli, i)?,
                        a.n(Field::PassiveFl, i)?,
                        a.n(Field::CrownRos, i)?,
                        a.n(Field::CrownHpua, i)?,
                        a.n(Field::CrownFli, i)?,
                        a.n(Field::CrownFl, i)?,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            } else {
                if single {
                    named(
                        out,
                        crown::single::assign_final_crown_fire_behavior(
                            a.n(Field::IsSurfaceFire, i)? != 0.0,
                            a.n(Field::IsPassiveCrown, i)? != 0.0,
                            a.n(Field::SurfaceRos, i)? as f32,
                            a.n(Field::SurfaceHpua, i)? as f32,
                            a.n(Field::SurfaceFli, i)? as f32,
                            a.n(Field::PassiveRos, i)? as f32,
                            a.n(Field::PassiveHpua, i)? as f32,
                            a.n(Field::PassiveFli, i)? as f32,
                            a.n(Field::PassiveFl, i)? as f32,
                            a.n(Field::CrownRos, i)? as f32,
                            a.n(Field::CrownHpua, i)? as f32,
                            a.n(Field::CrownFli, i)? as f32,
                            a.n(Field::CrownFl, i)? as f32,
                        )
                        .named_values()
                        .into_iter()
                        .map(|(k, v)| (k, widen(v))),
                    )
                } else {
                    named(
                        out,
                        crown::assign_final_crown_fire_behavior(
                            a.n(Field::IsSurfaceFire, i)? != 0.0,
                            a.n(Field::IsPassiveCrown, i)? != 0.0,
                            a.n(Field::SurfaceRos, i)?,
                            a.n(Field::SurfaceHpua, i)?,
                            a.n(Field::SurfaceFli, i)?,
                            a.n(Field::PassiveRos, i)?,
                            a.n(Field::PassiveHpua, i)?,
                            a.n(Field::PassiveFli, i)?,
                            a.n(Field::PassiveFl, i)?,
                            a.n(Field::CrownRos, i)?,
                            a.n(Field::CrownHpua, i)?,
                            a.n(Field::CrownFli, i)?,
                            a.n(Field::CrownFl, i)?,
                        )
                        .named_values()
                        .into_iter()
                        .map(|(k, v)| (k, widen(v))),
                    )
                }
            }
        }
        Operation::CrownCalculateCriticalCrownFireSpreadRate => {
            if single {
                scalar(
                    out,
                    widen(crown::single::calculate_critical_crown_fire_spread_rate(
                        a.n(Field::CanopyBulkDensity, i)? as f32,
                    )),
                )
            } else {
                scalar(
                    out,
                    widen(crown::calculate_critical_crown_fire_spread_rate(
                        a.n(Field::CanopyBulkDensity, i)?,
                    )),
                )
            }
        }
        Operation::CrownCalculateCriticalSurfaceFirelineIntensity => {
            if !single && a.optional(Field::GlobalMixed, i).is_some() {
                scalar(
                    out,
                    widen(
                        crown::calculate_critical_surface_fireline_intensity_single_height(
                            a.n(Field::CanopyBaseHeight, i)? as f32,
                            a.n(Field::MoistureFoliar, i)?,
                        ),
                    ),
                )
            } else {
                if single {
                    scalar(
                        out,
                        widen(
                            crown::single::calculate_critical_surface_fireline_intensity(
                                a.n(Field::CanopyBaseHeight, i)? as f32,
                                a.n(Field::MoistureFoliar, i)? as f32,
                            ),
                        ),
                    )
                } else {
                    scalar(
                        out,
                        widen(crown::calculate_critical_surface_fireline_intensity(
                            a.n(Field::CanopyBaseHeight, i)?,
                            a.n(Field::MoistureFoliar, i)?,
                        )),
                    )
                }
            }
        }
        Operation::CrownCalculateCrownFireRatios => {
            if single {
                named(
                    out,
                    crown::single::calculate_crown_fire_ratios(
                        a.n(Field::SurfaceFli, i)? as f32,
                        a.n(Field::CrownRos, i)? as f32,
                        a.n(Field::CriticalSurfaceFli, i)? as f32,
                        a.n(Field::CriticalCrownRos, i)? as f32,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            } else {
                named(
                    out,
                    crown::calculate_crown_fire_ratios(
                        a.n(Field::SurfaceFli, i)?,
                        a.n(Field::CrownRos, i)?,
                        a.n(Field::CriticalSurfaceFli, i)?,
                        a.n(Field::CriticalCrownRos, i)?,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            }
        }
        Operation::CrownCalculateCrownFractionBurned => {
            if single {
                scalar(
                    out,
                    widen(crown::single::calculate_crown_fraction_burned(
                        a.n(Field::SurfaceRos, i)? as f32,
                        a.n(Field::SurfaceCriticalRos, i)? as f32,
                        a.n(Field::CrowningSurfaceRos, i)? as f32,
                    )),
                )
            } else {
                scalar(
                    out,
                    widen(crown::calculate_crown_fraction_burned(
                        a.n(Field::SurfaceRos, i)?,
                        a.n(Field::SurfaceCriticalRos, i)?,
                        a.n(Field::CrowningSurfaceRos, i)?,
                    )),
                )
            }
        }
        Operation::CrownCalculateCrownHeatAndIntensity => {
            if single {
                named(
                    out,
                    crown::single::calculate_crown_heat_and_intensity(
                        a.n(Field::SurfaceHpua, i)? as f32,
                        a.n(Field::CrownRos, i)? as f32,
                        a.n(Field::CanopyBaseHeight, i)? as f32,
                        a.n(Field::CanopyHeight, i)? as f32,
                        a.n(Field::CanopyBulkDensity, i)? as f32,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            } else {
                named(
                    out,
                    crown::calculate_crown_heat_and_intensity(
                        a.n(Field::SurfaceHpua, i)?,
                        a.n(Field::CrownRos, i)?,
                        a.n(Field::CanopyBaseHeight, i)?,
                        a.n(Field::CanopyHeight, i)?,
                        a.n(Field::CanopyBulkDensity, i)?,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            }
        }
        Operation::CrownCalculatePassiveCrownFireBehavior => {
            if single {
                named(
                    out,
                    crown::single::calculate_passive_crown_fire_behavior(
                        a.n(Field::SurfaceRos, i)? as f32,
                        a.n(Field::SurfaceHpua, i)? as f32,
                        a.n(Field::CrownRos, i)? as f32,
                        a.n(Field::CanopyHpua, i)? as f32,
                        a.n(Field::Cfb, i)? as f32,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            } else {
                named(
                    out,
                    crown::calculate_passive_crown_fire_behavior(
                        a.n(Field::SurfaceRos, i)?,
                        a.n(Field::SurfaceHpua, i)?,
                        a.n(Field::CrownRos, i)?,
                        a.n(Field::CanopyHpua, i)?,
                        a.n(Field::Cfb, i)?,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            }
        }
        Operation::CrownCalculateSurfaceFireCriticalSpreadRate => {
            if !single && a.optional(Field::GlobalMixed, i).is_some() {
                scalar(
                    out,
                    widen(
                        crown::calculate_surface_fire_critical_spread_rate_single_intensity(
                            a.n(Field::CriticalSurfaceFli, i)? as f32,
                            a.n(Field::SurfaceHpua, i)?,
                        ),
                    ),
                )
            } else {
                if single {
                    scalar(
                        out,
                        widen(crown::single::calculate_surface_fire_critical_spread_rate(
                            a.n(Field::CriticalSurfaceFli, i)? as f32,
                            a.n(Field::SurfaceHpua, i)? as f32,
                        )),
                    )
                } else {
                    scalar(
                        out,
                        widen(crown::calculate_surface_fire_critical_spread_rate(
                            a.n(Field::CriticalSurfaceFli, i)?,
                            a.n(Field::SurfaceHpua, i)?,
                        )),
                    )
                }
            }
        }
        Operation::CrownClassifyCrownFireType => {
            if single {
                classification(
                    out,
                    crown::single::classify_crown_fire_type(
                        a.n(Field::TransitionRatio, i)? as f32,
                        a.n(Field::ActiveRatio, i)? as f32,
                    ),
                )
            } else {
                classification(
                    out,
                    crown::classify_crown_fire_type(
                        a.n(Field::TransitionRatio, i)?,
                        a.n(Field::ActiveRatio, i)?,
                    ),
                )
            }
        }
        Operation::CrownCalculateCrownSurfaceFire => crate::crown::surface_output(
            out,
            crown::calculate_crown_surface_fire(
                &crate::crown::input(a, i)?,
                &crate::crown::fuel(a, i)?,
                &crate::prepared::controls::crown(a),
                behave_fire::engine::SurfaceTweaks::default(),
            ),
        ),
        Operation::CrownCalculateCrowningSurfaceFireSpreadRate => scalar(
            out,
            if a.optional(Field::GlobalMixed, i).is_some() {
                crown::calculate_crowning_surface_fire_spread_rate_single_critical_rate(
                    a.n(Field::RiCrown, i)?,
                    &crate::structures::ib(a, i)?,
                    a.n(Field::CriticalCrownRos, i)? as f32,
                    a.n(Field::CanopyHeight, i)?,
                )
            } else {
                crown::calculate_crowning_surface_fire_spread_rate(
                    a.n(Field::RiCrown, i)?,
                    &crate::structures::ib(a, i)?,
                    a.n(Field::CriticalCrownRos, i)?,
                    a.n(Field::CanopyHeight, i)?,
                )
            },
        ),
        Operation::CrownRunCrownFire => {
            let r = crate::parallel::crown(a, i)?;
            named(out, r.named_values());
            out.i32("fire_type", r.fire_type);
        }
        _ => return Ok(None),
    }
    Ok(Some(()))
}
