//! Native crown_dispatch marshalling and scalar entry-point dispatch.
use crate::component_columns::RowWriter;
use crate::dispatch::{classification, named, scalar, widen};
use crate::operation::Operation;
use crate::prepared::{Field, PreparedArguments};
use behave_fire::crown;
#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    out: &mut RowWriter<'_, '_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::CrownAssignFinalCrownFireBehavior => named(
            out,
            crown::assign_final_crown_fire_behavior(
                a.flag(Field::IsSurfaceFire, i)?,
                a.flag(Field::IsPassiveCrown, i)?,
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
        ),
        Operation::CrownCalculateCriticalCrownFireSpreadRate => scalar(
            out,
            widen(crown::calculate_critical_crown_fire_spread_rate(
                a.n(Field::CanopyBulkDensity, i)?,
            )),
        ),
        Operation::CrownCalculateCriticalSurfaceFirelineIntensity => scalar(
            out,
            widen(crown::calculate_critical_surface_fireline_intensity(
                a.n(Field::CanopyBaseHeight, i)?,
                a.n(Field::MoistureFoliar, i)?,
            )),
        ),
        Operation::CrownCalculateCrownFireRatios => named(
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
        ),
        Operation::CrownCalculateCrownFractionBurned => scalar(
            out,
            widen(crown::calculate_crown_fraction_burned(
                a.n(Field::SurfaceRos, i)?,
                a.n(Field::SurfaceCriticalRos, i)?,
                a.n(Field::CrowningSurfaceRos, i)?,
            )),
        ),
        Operation::CrownCalculateCrownHeatAndIntensity => named(
            out,
            crown::calculate_crown_heat_and_intensity(
                a.n(Field::SurfaceHpua, i)?,
                a.n(Field::CrownRos, i)?,
                a.n(Field::CanopyBaseHeight, i)?,
                a.optional(Field::CanopyHeight, i).unwrap_or(f64::NAN),
                a.n(Field::CanopyBulkDensity, i)?,
            )
            .named_values()
            .into_iter()
            .map(|(k, v)| (k, widen(v))),
        ),
        Operation::CrownCalculatePassiveCrownFireBehavior => named(
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
        ),
        Operation::CrownCalculateSurfaceFireCriticalSpreadRate => scalar(
            out,
            widen(crown::calculate_surface_fire_critical_spread_rate(
                a.n(Field::CriticalSurfaceFli, i)?,
                a.n(Field::SurfaceHpua, i)?,
            )),
        ),
        Operation::CrownClassifyCrownFireType => classification(
            out,
            crown::classify_crown_fire_type(
                a.n(Field::TransitionRatio, i)?,
                a.n(Field::ActiveRatio, i)?,
            ),
        ),
        Operation::CrownCalculateCrownSurfaceFire => crate::crown::surface_output(
            out,
            crown::calculate_crown_surface_fire(
                &crate::crown::input(a, i)?,
                &crate::crown::fuel(a, i)?,
                &crate::prepared::controls::crown(a),
                behave_fire::engine::SurfaceTweaks::default(),
            ),
        ),
        Operation::CrownCalculateCrowningSurfaceFireSpreadRate => scalar(out, {
            crown::calculate_crowning_surface_fire_spread_rate(
                a.n(Field::RiCrown, i)?,
                &crate::structures::ib(a, i)?,
                a.n(Field::CriticalCrownRos, i)?,
                a.optional(Field::CanopyHeight, i).unwrap_or(f64::NAN),
            )
        }),
        _ => return Ok(None),
    }
    Ok(Some(()))
}
