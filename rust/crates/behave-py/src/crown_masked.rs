//! Crown stage dtype and masks remain native per-input metadata.
use crate::{arrays::Arguments, dispatch::Cell, masked::output};
use behave_fire::crown::masked::*;
pub fn evaluate(operation: &str, a: &Arguments, i: usize) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "crown.assign_final_crown_fire_behavior" => crate::masked::outputs(
            assign_final_crown_fire_behavior(
                a.n("is_surface_fire", i)? != 0.0,
                a.n("is_passive_crown", i)? != 0.0,
                a.masked("surface_ros", i)?,
                a.masked("surface_hpua", i)?,
                a.masked("surface_fli", i)?,
                a.masked("passive_ros", i)?,
                a.masked("passive_hpua", i)?,
                a.masked("passive_fli", i)?,
                a.masked("passive_fl", i)?,
                a.masked("crown_ros", i)?,
                a.masked("crown_hpua", i)?,
                a.masked("crown_fli", i)?,
                a.masked("crown_fl", i)?,
            )
            .named_values(),
        )?,
        "crown.calculate_critical_crown_fire_spread_rate" => output(
            calculate_critical_crown_fire_spread_rate(a.masked("canopy_bulk_density", i)?),
            "",
        )?,
        "crown.calculate_critical_surface_fireline_intensity" => output(
            calculate_critical_surface_fireline_intensity(
                a.masked("canopy_base_height", i)?,
                a.masked("moisture_foliar", i)?,
            ),
            "",
        )?,
        "crown.calculate_crown_fire_ratios" => crate::masked::outputs(
            calculate_crown_fire_ratios(
                a.masked("surface_fli", i)?,
                a.masked("crown_ros", i)?,
                a.masked("critical_surface_fli", i)?,
                a.masked("critical_crown_ros", i)?,
            )
            .named_values(),
        )?,
        "crown.calculate_crown_fraction_burned" => output(
            calculate_crown_fraction_burned(
                a.masked("surface_ros", i)?,
                a.masked("surface_critical_ros", i)?,
                a.masked("crowning_surface_ros", i)?,
            ),
            "",
        )?,
        "crown.calculate_crown_heat_and_intensity" => crate::masked::outputs(
            calculate_crown_heat_and_intensity(
                a.masked("surface_hpua", i)?,
                a.masked("crown_ros", i)?,
                a.masked("canopy_base_height", i)?,
                a.masked("canopy_height", i)?,
                a.masked("canopy_bulk_density", i)?,
            )
            .named_values(),
        )?,
        "crown.calculate_passive_crown_fire_behavior" => crate::masked::outputs(
            calculate_passive_crown_fire_behavior(
                a.masked("surface_ros", i)?,
                a.masked("surface_hpua", i)?,
                a.masked("crown_ros", i)?,
                a.masked("canopy_hpua", i)?,
                a.masked("cfb", i)?,
            )
            .named_values(),
        )?,
        "crown.calculate_surface_fire_critical_spread_rate" => output(
            calculate_surface_fire_critical_spread_rate(
                a.masked("critical_surface_fli", i)?,
                a.masked("surface_hpua", i)?,
            ),
            "",
        )?,
        "crown.classify_crown_fire_type" => {
            crate::dispatch::classification(classify_crown_fire_type(
                a.masked("transition_ratio", i)?,
                a.masked("active_ratio", i)?,
            ))
        }
        "crown.calculate_crowning_surface_fire_spread_rate" => output(
            try_calculate_crowning_surface_fire_spread_rate_with_intermediates(
                a.masked("ri_crown", i)?,
                &crate::masked_records::intermediates(a, i)?,
                a.masked("critical_crown_ros", i)?,
                a.masked("canopy_height", i)?,
                a.optional("_scalar_grid", i).is_some(),
            )
            .map_err(crate::arrays::masked_error)?,
            "",
        )?,
        "crown.run_crown_fire" => {
            let mut surface = behave_fire::crown::SurfaceFireInputs::new();
            surface.surface_ros = a.masked("surface_results.spread_rate", i)?;
            surface.surface_fli = a.masked("surface_results.fireline_intensity", i)?;
            surface.surface_hpua = a.masked("surface_results.heat_per_unit_area", i)?;
            let input = crate::crown::input(a, i)?;
            let fuel = crate::crown::fuel(a, i)?;
            let controls = crate::controls::crown(a);
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
