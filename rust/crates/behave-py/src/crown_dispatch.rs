//! Native crown_dispatch marshalling and scalar entry-point dispatch.
use crate::dispatch::{classification, named, scalar, widen, Cell};
use crate::prepared::{Field, PreparedArguments};
use behave_fire::crown;
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &PreparedArguments<'_>,
    i: usize,
    single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "crown.assign_final_crown_fire_behavior" => {
            if !single && a.optional(Field::GlobalMixed, i).is_some() {
                named(
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
        "crown.calculate_critical_crown_fire_spread_rate" => {
            if single {
                scalar(widen(
                    crown::single::calculate_critical_crown_fire_spread_rate(
                        a.n(Field::CanopyBulkDensity, i)? as f32,
                    ),
                ))
            } else {
                scalar(widen(crown::calculate_critical_crown_fire_spread_rate(
                    a.n(Field::CanopyBulkDensity, i)?,
                )))
            }
        }
        "crown.calculate_critical_surface_fireline_intensity" => {
            if !single && a.optional(Field::GlobalMixed, i).is_some() {
                scalar(widen(
                    crown::calculate_critical_surface_fireline_intensity_single_height(
                        a.n(Field::CanopyBaseHeight, i)? as f32,
                        a.n(Field::MoistureFoliar, i)?,
                    ),
                ))
            } else {
                if single {
                    scalar(widen(
                        crown::single::calculate_critical_surface_fireline_intensity(
                            a.n(Field::CanopyBaseHeight, i)? as f32,
                            a.n(Field::MoistureFoliar, i)? as f32,
                        ),
                    ))
                } else {
                    scalar(widen(crown::calculate_critical_surface_fireline_intensity(
                        a.n(Field::CanopyBaseHeight, i)?,
                        a.n(Field::MoistureFoliar, i)?,
                    )))
                }
            }
        }
        "crown.calculate_crown_fire_ratios" => {
            if single {
                named(
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
        "crown.calculate_crown_fraction_burned" => {
            if single {
                scalar(widen(crown::single::calculate_crown_fraction_burned(
                    a.n(Field::SurfaceRos, i)? as f32,
                    a.n(Field::SurfaceCriticalRos, i)? as f32,
                    a.n(Field::CrowningSurfaceRos, i)? as f32,
                )))
            } else {
                scalar(widen(crown::calculate_crown_fraction_burned(
                    a.n(Field::SurfaceRos, i)?,
                    a.n(Field::SurfaceCriticalRos, i)?,
                    a.n(Field::CrowningSurfaceRos, i)?,
                )))
            }
        }
        "crown.calculate_crown_heat_and_intensity" => {
            if single {
                named(
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
        "crown.calculate_passive_crown_fire_behavior" => {
            if single {
                named(
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
        "crown.calculate_surface_fire_critical_spread_rate" => {
            if !single && a.optional(Field::GlobalMixed, i).is_some() {
                scalar(widen(
                    crown::calculate_surface_fire_critical_spread_rate_single_intensity(
                        a.n(Field::CriticalSurfaceFli, i)? as f32,
                        a.n(Field::SurfaceHpua, i)?,
                    ),
                ))
            } else {
                if single {
                    scalar(widen(
                        crown::single::calculate_surface_fire_critical_spread_rate(
                            a.n(Field::CriticalSurfaceFli, i)? as f32,
                            a.n(Field::SurfaceHpua, i)? as f32,
                        ),
                    ))
                } else {
                    scalar(widen(crown::calculate_surface_fire_critical_spread_rate(
                        a.n(Field::CriticalSurfaceFli, i)?,
                        a.n(Field::SurfaceHpua, i)?,
                    )))
                }
            }
        }
        "crown.classify_crown_fire_type" => {
            if single {
                classification(crown::single::classify_crown_fire_type(
                    a.n(Field::TransitionRatio, i)? as f32,
                    a.n(Field::ActiveRatio, i)? as f32,
                ))
            } else {
                classification(crown::classify_crown_fire_type(
                    a.n(Field::TransitionRatio, i)?,
                    a.n(Field::ActiveRatio, i)?,
                ))
            }
        }
        "crown.calculate_crown_surface_fire" => {
            crate::crown::surface_output(crown::calculate_crown_surface_fire(
                &crate::crown::input(a, i)?,
                &crate::crown::fuel(a, i)?,
                &crate::prepared::controls::crown(a),
                behave_fire::engine::SurfaceTweaks::default(),
            ))
        }
        "crown.calculate_crowning_surface_fire_spread_rate" => {
            scalar(if a.optional(Field::GlobalMixed, i).is_some() {
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
            })
        }
        "crown.run_crown_fire" => {
            let x = crate::crown::input(a, i)?;
            let r = behave_fire::engine::CrownPlan::new(crate::prepared::controls::crown(a))
                .run_cell(
                    &x,
                    &crate::crown::surface_values(a, i),
                    &crate::crown::fuel(a, i)?,
                    behave_fire::engine::CrownTweaks::default(),
                );
            let mut out = named(r.named_values());
            out.insert("fire_type".into(), r.fire_type as f64);
            out
        }
        _ => return Ok(None),
    }))
}
