//! Crown stage dtype and masks remain native per-input metadata.
use crate::{
    dispatch::Cell,
    masked::output,
    prepared::{Field, PreparedArguments},
};
use behave_fire::crown::masked::*;
pub fn evaluate(
    operation: &str,
    a: &PreparedArguments<'_>,
    i: usize,
) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "crown.assign_final_crown_fire_behavior" => crate::masked::outputs(
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
        "crown.calculate_critical_crown_fire_spread_rate" => output(
            calculate_critical_crown_fire_spread_rate(a.masked(Field::CanopyBulkDensity, i)?),
            "",
        )?,
        "crown.calculate_critical_surface_fireline_intensity" => output(
            calculate_critical_surface_fireline_intensity(
                a.masked(Field::CanopyBaseHeight, i)?,
                a.masked(Field::MoistureFoliar, i)?,
            ),
            "",
        )?,
        "crown.calculate_crown_fire_ratios" => crate::masked::outputs(
            calculate_crown_fire_ratios(
                a.masked(Field::SurfaceFli, i)?,
                a.masked(Field::CrownRos, i)?,
                a.masked(Field::CriticalSurfaceFli, i)?,
                a.masked(Field::CriticalCrownRos, i)?,
            )
            .named_values(),
        )?,
        "crown.calculate_crown_fraction_burned" => output(
            calculate_crown_fraction_burned(
                a.masked(Field::SurfaceRos, i)?,
                a.masked(Field::SurfaceCriticalRos, i)?,
                a.masked(Field::CrowningSurfaceRos, i)?,
            ),
            "",
        )?,
        "crown.calculate_crown_heat_and_intensity" => crate::masked::outputs(
            calculate_crown_heat_and_intensity(
                a.masked(Field::SurfaceHpua, i)?,
                a.masked(Field::CrownRos, i)?,
                a.masked(Field::CanopyBaseHeight, i)?,
                a.masked(Field::CanopyHeight, i)?,
                a.masked(Field::CanopyBulkDensity, i)?,
            )
            .named_values(),
        )?,
        "crown.calculate_passive_crown_fire_behavior" => crate::masked::outputs(
            calculate_passive_crown_fire_behavior(
                a.masked(Field::SurfaceRos, i)?,
                a.masked(Field::SurfaceHpua, i)?,
                a.masked(Field::CrownRos, i)?,
                a.masked(Field::CanopyHpua, i)?,
                a.masked(Field::Cfb, i)?,
            )
            .named_values(),
        )?,
        "crown.calculate_surface_fire_critical_spread_rate" => output(
            calculate_surface_fire_critical_spread_rate(
                a.masked(Field::CriticalSurfaceFli, i)?,
                a.masked(Field::SurfaceHpua, i)?,
            ),
            "",
        )?,
        "crown.classify_crown_fire_type" => {
            crate::dispatch::classification(classify_crown_fire_type(
                a.masked(Field::TransitionRatio, i)?,
                a.masked(Field::ActiveRatio, i)?,
            ))
        }
        "crown.calculate_crowning_surface_fire_spread_rate" => output(
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
        "crown.run_crown_fire" => {
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
            let mut out = crate::masked::outputs(r.named_values())?;
            out.insert("fire_type".into(), r.fire_type as f64);
            out
        }
        _ => return Ok(None),
    }))
}
