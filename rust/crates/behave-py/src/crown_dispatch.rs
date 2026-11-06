//! Native crown_dispatch marshalling and scalar entry-point dispatch.
use crate::arrays::Arguments;
use crate::dispatch::{classification, named, scalar, widen, Cell};
use behave_fire::crown;
#[allow(deprecated)]
pub fn evaluate(
    operation: &str,
    a: &Arguments,
    i: usize,
    single: bool,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "crown.assign_final_crown_fire_behavior" => {
            if !single && a.optional("_mixed", i).is_some() {
                named(
                    crown::assign_final_crown_fire_behavior_single_surface(
                        a.n("is_surface_fire", i)? != 0.0,
                        a.n("is_passive_crown", i)? != 0.0,
                        a.n("surface_ros", i)? as f32,
                        a.n("surface_hpua", i)? as f32,
                        a.n("surface_fli", i)? as f32,
                        a.n("passive_ros", i)?,
                        a.n("passive_hpua", i)?,
                        a.n("passive_fli", i)?,
                        a.n("passive_fl", i)?,
                        a.n("crown_ros", i)?,
                        a.n("crown_hpua", i)?,
                        a.n("crown_fli", i)?,
                        a.n("crown_fl", i)?,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            } else {
                if single {
                    named(
                        crown::single::assign_final_crown_fire_behavior(
                            a.n("is_surface_fire", i)? != 0.0,
                            a.n("is_passive_crown", i)? != 0.0,
                            a.n("surface_ros", i)? as f32,
                            a.n("surface_hpua", i)? as f32,
                            a.n("surface_fli", i)? as f32,
                            a.n("passive_ros", i)? as f32,
                            a.n("passive_hpua", i)? as f32,
                            a.n("passive_fli", i)? as f32,
                            a.n("passive_fl", i)? as f32,
                            a.n("crown_ros", i)? as f32,
                            a.n("crown_hpua", i)? as f32,
                            a.n("crown_fli", i)? as f32,
                            a.n("crown_fl", i)? as f32,
                        )
                        .named_values()
                        .into_iter()
                        .map(|(k, v)| (k, widen(v))),
                    )
                } else {
                    named(
                        crown::assign_final_crown_fire_behavior(
                            a.n("is_surface_fire", i)? != 0.0,
                            a.n("is_passive_crown", i)? != 0.0,
                            a.n("surface_ros", i)?,
                            a.n("surface_hpua", i)?,
                            a.n("surface_fli", i)?,
                            a.n("passive_ros", i)?,
                            a.n("passive_hpua", i)?,
                            a.n("passive_fli", i)?,
                            a.n("passive_fl", i)?,
                            a.n("crown_ros", i)?,
                            a.n("crown_hpua", i)?,
                            a.n("crown_fli", i)?,
                            a.n("crown_fl", i)?,
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
                        a.n("canopy_bulk_density", i)? as f32,
                    ),
                ))
            } else {
                scalar(widen(crown::calculate_critical_crown_fire_spread_rate(
                    a.n("canopy_bulk_density", i)?,
                )))
            }
        }
        "crown.calculate_critical_surface_fireline_intensity" => {
            if !single && a.optional("_mixed", i).is_some() {
                scalar(widen(
                    crown::calculate_critical_surface_fireline_intensity_single_height(
                        a.n("canopy_base_height", i)? as f32,
                        a.n("moisture_foliar", i)?,
                    ),
                ))
            } else {
                if single {
                    scalar(widen(
                        crown::single::calculate_critical_surface_fireline_intensity(
                            a.n("canopy_base_height", i)? as f32,
                            a.n("moisture_foliar", i)? as f32,
                        ),
                    ))
                } else {
                    scalar(widen(crown::calculate_critical_surface_fireline_intensity(
                        a.n("canopy_base_height", i)?,
                        a.n("moisture_foliar", i)?,
                    )))
                }
            }
        }
        "crown.calculate_crown_fire_ratios" => {
            if single {
                named(
                    crown::single::calculate_crown_fire_ratios(
                        a.n("surface_fli", i)? as f32,
                        a.n("crown_ros", i)? as f32,
                        a.n("critical_surface_fli", i)? as f32,
                        a.n("critical_crown_ros", i)? as f32,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            } else {
                named(
                    crown::calculate_crown_fire_ratios(
                        a.n("surface_fli", i)?,
                        a.n("crown_ros", i)?,
                        a.n("critical_surface_fli", i)?,
                        a.n("critical_crown_ros", i)?,
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
                    a.n("surface_ros", i)? as f32,
                    a.n("surface_critical_ros", i)? as f32,
                    a.n("crowning_surface_ros", i)? as f32,
                )))
            } else {
                scalar(widen(crown::calculate_crown_fraction_burned(
                    a.n("surface_ros", i)?,
                    a.n("surface_critical_ros", i)?,
                    a.n("crowning_surface_ros", i)?,
                )))
            }
        }
        "crown.calculate_crown_heat_and_intensity" => {
            if single {
                named(
                    crown::single::calculate_crown_heat_and_intensity(
                        a.n("surface_hpua", i)? as f32,
                        a.n("crown_ros", i)? as f32,
                        a.n("canopy_base_height", i)? as f32,
                        a.n("canopy_height", i)? as f32,
                        a.n("canopy_bulk_density", i)? as f32,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            } else {
                named(
                    crown::calculate_crown_heat_and_intensity(
                        a.n("surface_hpua", i)?,
                        a.n("crown_ros", i)?,
                        a.n("canopy_base_height", i)?,
                        a.n("canopy_height", i)?,
                        a.n("canopy_bulk_density", i)?,
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
                        a.n("surface_ros", i)? as f32,
                        a.n("surface_hpua", i)? as f32,
                        a.n("crown_ros", i)? as f32,
                        a.n("canopy_hpua", i)? as f32,
                        a.n("cfb", i)? as f32,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            } else {
                named(
                    crown::calculate_passive_crown_fire_behavior(
                        a.n("surface_ros", i)?,
                        a.n("surface_hpua", i)?,
                        a.n("crown_ros", i)?,
                        a.n("canopy_hpua", i)?,
                        a.n("cfb", i)?,
                    )
                    .named_values()
                    .into_iter()
                    .map(|(k, v)| (k, widen(v))),
                )
            }
        }
        "crown.calculate_surface_fire_critical_spread_rate" => {
            if !single && a.optional("_mixed", i).is_some() {
                scalar(widen(
                    crown::calculate_surface_fire_critical_spread_rate_single_intensity(
                        a.n("critical_surface_fli", i)? as f32,
                        a.n("surface_hpua", i)?,
                    ),
                ))
            } else {
                if single {
                    scalar(widen(
                        crown::single::calculate_surface_fire_critical_spread_rate(
                            a.n("critical_surface_fli", i)? as f32,
                            a.n("surface_hpua", i)? as f32,
                        ),
                    ))
                } else {
                    scalar(widen(crown::calculate_surface_fire_critical_spread_rate(
                        a.n("critical_surface_fli", i)?,
                        a.n("surface_hpua", i)?,
                    )))
                }
            }
        }
        "crown.classify_crown_fire_type" => {
            if single {
                classification(crown::single::classify_crown_fire_type(
                    a.n("transition_ratio", i)? as f32,
                    a.n("active_ratio", i)? as f32,
                ))
            } else {
                classification(crown::classify_crown_fire_type(
                    a.n("transition_ratio", i)?,
                    a.n("active_ratio", i)?,
                ))
            }
        }
        "crown.calculate_crown_surface_fire" => {
            crate::crown::surface_output(crown::calculate_crown_surface_fire(
                &crate::crown::input(a, i)?,
                &crate::crown::fuel(a, i)?,
                &crate::controls::crown(a),
                behave_fire::engine::SurfaceTweaks::default(),
            ))
        }
        "crown.calculate_crowning_surface_fire_spread_rate" => {
            scalar(if a.optional("_mixed", i).is_some() {
                crown::calculate_crowning_surface_fire_spread_rate_single_critical_rate(
                    a.n("ri_crown", i)?,
                    &crate::structures::ib(a, i)?,
                    a.n("critical_crown_ros", i)? as f32,
                    a.n("canopy_height", i)?,
                )
            } else {
                crown::calculate_crowning_surface_fire_spread_rate(
                    a.n("ri_crown", i)?,
                    &crate::structures::ib(a, i)?,
                    a.n("critical_crown_ros", i)?,
                    a.n("canopy_height", i)?,
                )
            })
        }
        "crown.run_crown_fire" => {
            let x = crate::crown::input(a, i)?;
            let r = behave_fire::engine::CrownPlan::new(crate::controls::crown(a)).run_cell(
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
