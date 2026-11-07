//! Crown stage dtype and masks remain native per-input metadata.
use crate::column_output::RowWriter;
use crate::operation::Operation;
use crate::{
    masked::output,
    prepared::{Field, PreparedArguments},
};
use behave_fire::crown::masked::*;
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    i: usize,
    out: &mut RowWriter<'_>,
) -> Result<Option<()>, String> {
    match operation {
        Operation::CrownAssignFinalCrownFireBehavior => crate::masked::outputs(
            out,
            assign_final_crown_fire_behavior(
                a.n(Field::IsSurfaceFire, i)? != 0.0,
                a.n(Field::IsPassiveCrown, i)? != 0.0,
                a.masked(Field::SurfaceRos, i)?,
                a.masked(Field::SurfaceHpua, i)?,
                a.masked(Field::SurfaceFli, i)?,
                a.masked(Field::PassiveRos, i)?,
                a.masked(Field::PassiveHpua, i)?,
                a.masked(Field::PassiveFli, i)?,
                a.masked(Field::PassiveFl, i)?,
                a.masked(Field::CrownRos, i)?,
                a.masked(Field::CrownHpua, i)?,
                a.masked(Field::CrownFli, i)?,
                a.masked(Field::CrownFl, i)?,
            )
            .named_values(),
        )?,
        Operation::CrownCalculateCriticalCrownFireSpreadRate => output(
            out,
            calculate_critical_crown_fire_spread_rate(a.masked(Field::CanopyBulkDensity, i)?),
            "",
        )?,
        Operation::CrownCalculateCriticalSurfaceFirelineIntensity => output(
            out,
            calculate_critical_surface_fireline_intensity(
                a.masked(Field::CanopyBaseHeight, i)?,
                a.masked(Field::MoistureFoliar, i)?,
            ),
            "",
        )?,
        Operation::CrownCalculateCrownFireRatios => crate::masked::outputs(
            out,
            calculate_crown_fire_ratios(
                a.masked(Field::SurfaceFli, i)?,
                a.masked(Field::CrownRos, i)?,
                a.masked(Field::CriticalSurfaceFli, i)?,
                a.masked(Field::CriticalCrownRos, i)?,
            )
            .named_values(),
        )?,
        Operation::CrownCalculateCrownFractionBurned => output(
            out,
            calculate_crown_fraction_burned(
                a.masked(Field::SurfaceRos, i)?,
                a.masked(Field::SurfaceCriticalRos, i)?,
                a.masked(Field::CrowningSurfaceRos, i)?,
            ),
            "",
        )?,
        Operation::CrownCalculateCrownHeatAndIntensity => crate::masked::outputs(
            out,
            calculate_crown_heat_and_intensity(
                a.masked(Field::SurfaceHpua, i)?,
                a.masked(Field::CrownRos, i)?,
                a.masked(Field::CanopyBaseHeight, i)?,
                a.masked(Field::CanopyHeight, i)?,
                a.masked(Field::CanopyBulkDensity, i)?,
            )
            .named_values(),
        )?,
        Operation::CrownCalculatePassiveCrownFireBehavior => crate::masked::outputs(
            out,
            calculate_passive_crown_fire_behavior(
                a.masked(Field::SurfaceRos, i)?,
                a.masked(Field::SurfaceHpua, i)?,
                a.masked(Field::CrownRos, i)?,
                a.masked(Field::CanopyHpua, i)?,
                a.masked(Field::Cfb, i)?,
            )
            .named_values(),
        )?,
        Operation::CrownCalculateSurfaceFireCriticalSpreadRate => output(
            out,
            calculate_surface_fire_critical_spread_rate(
                a.masked(Field::CriticalSurfaceFli, i)?,
                a.masked(Field::SurfaceHpua, i)?,
            ),
            "",
        )?,
        Operation::CrownClassifyCrownFireType => crate::dispatch::classification(
            out,
            classify_crown_fire_type(
                a.masked(Field::TransitionRatio, i)?,
                a.masked(Field::ActiveRatio, i)?,
            ),
        ),
        Operation::CrownCalculateCrowningSurfaceFireSpreadRate => output(
            out,
            try_calculate_crowning_surface_fire_spread_rate_with_intermediates(
                a.masked(Field::RiCrown, i)?,
                &crate::masked_records::intermediates(a, i)?,
                a.masked(Field::CriticalCrownRos, i)?,
                a.masked(Field::CanopyHeight, i)?,
                a.optional(Field::GlobalScalarGrid, i).is_some(),
            )
            .map_err(crate::arrays::masked_error)?,
            "",
        )?,
        Operation::CrownRunCrownFire => {
            let mut surface = behave_fire::crown::SurfaceFireInputs::new();
            surface.surface_ros = a.masked(Field::SurfaceResultsSpreadRate, i)?;
            surface.surface_fli = a.masked(Field::SurfaceResultsFirelineIntensity, i)?;
            surface.surface_hpua = a.masked(Field::SurfaceResultsHeatPerUnitArea, i)?;
            let input = crate::crown::input(a, i)?;
            let fuel = crate::crown::fuel(a, i)?;
            let controls = crate::prepared::controls::crown(a);
            let r = {
                #[cfg(feature = "allocation-probe")]
                let _scope = crate::allocation_probe::Scope::enter(crate::allocation_probe::KERNEL);
                behave_fire::crown::run_with_surface_inputs(&surface, &input, &fuel, &controls)
            }
            .map_err(|e| e.to_string())?;
            crate::masked::outputs(out, r.named_values())?;
            out.i32("fire_type", r.fire_type);
        }
        _ => return Ok(None),
    }
    Ok(Some(()))
}
