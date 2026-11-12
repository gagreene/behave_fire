//! Actual f64 model raster attribution with complete independently specified outputs.
use super::{Arguments, Case};
use behave_fire::crown::{CrownResult, SurfaceFireInputs};
use behave_fire::engine::{
    run_model, CrownPlan, CrownTweaks, ModelBatch, ModelRun, OutputRequest, PreparedFuels,
    SurfacePlan, SurfaceSource, SurfaceTweaks,
};
use behave_fire::output::ColumnValues;
use behave_fire::raster::{
    CrownColumns, CrownRaster, CrownTweakColumns, SurfaceColumns, SurfaceRaster,
    SurfaceTweakColumns,
};
use behave_fire::{GridShape, PerCell};
use std::io::Write;

fn supplied(case: &Case<'_>, typed: bool, missing: bool) -> SurfaceFireInputs<f64> {
    if !typed {
        return case.supplied;
    }
    let mut result = SurfaceFireInputs::new();
    let value = |v: behave_fire::surface::masked::MaskedValue| {
        if missing && v.mask {
            f64::NAN
        } else {
            v.value
        }
    };
    result.surface_ros = value(case.typed.surface_ros);
    result.surface_fli = value(case.typed.surface_fli);
    result.surface_hpua = value(case.typed.surface_hpua);
    result
}

// These expected NoData overlays follow the reviewed truth table, independently
// of the new engine's availability implementation. Finite values use old scalar
// f64 science on the actual widened storage values, never the new raster result.
fn reference_crown(args: &Arguments, case: &Case<'_>, fuels: &PreparedFuels) -> CrownResult {
    let typed = args.operation == "typed-crown";
    let input = supplied(case, typed, false);
    let mut value = CrownPlan::default().run_cell(
        &case.crown,
        &input,
        fuels.for_grid(10),
        CrownTweaks {
            fm10: SurfaceTweaks {
                spread_rate: args.override_rate,
                ..SurfaceTweaks::default()
            },
            ..CrownTweaks::default()
        },
    );
    if typed && case.typed.surface_hpua.mask {
        value.crown_fire_heat_per_unit_area = f64::NAN;
        value.crown_fire_line_intensity = f64::NAN;
        value.crown_flame_length = f64::NAN;
        value.surface_fire_critical_spread_rate = f64::NAN;
        value.crown_fraction_burned = f64::NAN;
        value.passive_crown_fire_spread_rate = f64::NAN;
        value.passive_crown_fire_heat_per_unit_area = f64::NAN;
        value.passive_crown_fire_line_intensity = f64::NAN;
        value.passive_crown_fire_flame_length = f64::NAN;
        value.final_heat_per_unit_area = f64::NAN;
        if value.fire_type == 1 {
            value.final_spread_rate = f64::NAN;
        }
        if value.fire_type == 1 || value.fire_type == 3 {
            value.final_fireline_intensity = f64::NAN;
            value.final_flame_length = f64::NAN;
        }
    }
    if typed && case.typed.surface_ros.mask {
        let fraction_missing = value.crowning_surface_fire_spread_rate
            - value.surface_fire_critical_spread_rate
            > 1e-7;
        if fraction_missing {
            value.crown_fraction_burned = f64::NAN;
            value.passive_crown_fire_heat_per_unit_area = f64::NAN;
        }
        value.passive_crown_fire_spread_rate = f64::NAN;
        value.passive_crown_fire_line_intensity = f64::NAN;
        value.passive_crown_fire_flame_length = f64::NAN;
        if value.fire_type == 0 || value.fire_type == 2 {
            value.final_spread_rate = f64::NAN;
        }
        if value.fire_type == 1 {
            value.final_spread_rate = f64::NAN;
            value.final_heat_per_unit_area = value.passive_crown_fire_heat_per_unit_area;
            value.final_fireline_intensity = f64::NAN;
            value.final_flame_length = f64::NAN;
        }
    }
    value
}

fn validate(
    args: &Arguments,
    cases: &[Case<'_>],
    fuels: &PreparedFuels,
    output: ModelBatch,
) -> Result<(), String> {
    let reference = cases
        .iter()
        .map(|case| {
            if args.operation == "surface" {
                SurfacePlan::default()
                    .run_cell(
                        &case.surface,
                        case.fuel,
                        SurfaceTweaks {
                            spread_rate: args.override_rate,
                            ..SurfaceTweaks::default()
                        },
                    )
                    .named_values()
                    .to_vec()
            } else {
                let value = reference_crown(args, case, fuels);
                let mut fields = value.named_values().to_vec();
                fields.push(("fire_type", f64::from(value.fire_type)));
                fields
            }
        })
        .collect::<Vec<_>>();
    let columns = match output {
        ModelBatch::Surface(output) => output.into_named_columns(),
        ModelBatch::Crown(output) => output.into_named_columns(),
    };
    if columns.len() != reference[0].len() {
        return Err("incomplete f64 result inventory".into());
    }
    let export = std::env::var_os("BEHAVE_PROBE_OUTPUT").map(std::path::PathBuf::from);
    if let Some(path) = &export {
        std::fs::create_dir_all(path).map_err(|e| e.to_string())?;
    }
    let mut seen = vec![false; reference[0].len()];
    for (name, column) in columns {
        let slot = reference[0]
            .iter()
            .position(|(key, _)| *key == name)
            .ok_or_else(|| format!("unknown field{name}"))?;
        if seen[slot] {
            return Err(format!("duplicate field{name}"));
        }
        seen[slot] = true;
        match column {
            ColumnValues::Float64(values) => {
                if values.len() != args.cells || name == "fire_type" {
                    return Err(format!("wrong f64 field shape/type{name}"));
                }
                for (index, value) in values.iter().enumerate() {
                    if !super::raster_probe::check_value(
                        *value,
                        reference[index % reference.len()][slot].1,
                    ) {
                        return Err(format!("f64 scalar/truth mismatch {name}/{index}"));
                    }
                }
                if let Some(path) = &export {
                    let mut file = std::io::BufWriter::new(
                        std::fs::File::create(path.join(format!("{name}.f64")))
                            .map_err(|e| e.to_string())?,
                    );
                    for value in values {
                        file.write_all(&value.to_le_bytes())
                            .map_err(|e| e.to_string())?;
                    }
                    file.flush().map_err(|e| e.to_string())?;
                }
            }
            ColumnValues::Int32(values) => {
                if name != "fire_type" || values.len() != args.cells {
                    return Err("wrong categorical shape/type".into());
                }
                for (index, &value) in values.iter().enumerate() {
                    if f64::from(value) != reference[index % reference.len()][slot].1 {
                        return Err(format!("category mismatch{index}"));
                    }
                }
                if let Some(path) = &export {
                    let mut file = std::io::BufWriter::new(
                        std::fs::File::create(path.join(format!("{name}.i32")))
                            .map_err(|e| e.to_string())?,
                    );
                    for value in values {
                        file.write_all(&value.to_le_bytes())
                            .map_err(|e| e.to_string())?;
                    }
                    file.flush().map_err(|e| e.to_string())?;
                }
            }
            ColumnValues::Bool(_) => return Err("unexpected model flag column".into()),
        }
    }
    Ok(())
}

pub(super) fn run(
    args: &Arguments,
    cases: &[Case<'_>],
    fuels: &PreparedFuels,
) -> Result<(), String> {
    if args.operation == "surface" {
        let data = super::raster_probe::SurfaceData::new(args, cases);
        let models = (0..args.cells)
            .map(|i| cases[i % cases.len()].surface.fuel_model_grid)
            .collect::<Vec<_>>();
        let raster = SurfaceRaster::new(
            GridShape::new(vec![args.cells]).map_err(|e| e.to_string())?,
            SurfaceColumns {
                fuel_model_grid: PerCell::Cells(&models),
                ..data.columns()
            },
        )
        .map_err(|e| e.to_string())?;
        let run = ModelRun::Surface {
            raster,
            plan: SurfacePlan::default(),
            source: SurfaceSource::Fuels(fuels),
            tweaks: SurfaceTweakColumns {
                spread_rate: args.override_rate.map(PerCell::Scalar),
                ..SurfaceTweakColumns::default()
            },
        };
        super::raster_probe::measure(
            args,
            || run_model(&run, &OutputRequest::surface(), 1).map_err(|e| e.to_string()),
            |output| validate(args, cases, fuels, output),
        )
    } else {
        let data = super::raster_probe::CrownData::new(args, cases);
        let ros = (0..args.cells)
            .map(|i| {
                supplied(
                    &cases[i % cases.len()],
                    args.operation == "typed-crown",
                    true,
                )
                .surface_ros
            })
            .collect::<Vec<_>>();
        let fli = (0..args.cells)
            .map(|i| {
                supplied(
                    &cases[i % cases.len()],
                    args.operation == "typed-crown",
                    true,
                )
                .surface_fli
            })
            .collect::<Vec<_>>();
        let heat = (0..args.cells)
            .map(|i| {
                supplied(
                    &cases[i % cases.len()],
                    args.operation == "typed-crown",
                    true,
                )
                .surface_hpua
            })
            .collect::<Vec<_>>();
        let raster = CrownRaster::new(
            GridShape::new(vec![args.cells]).map_err(|e| e.to_string())?,
            CrownColumns {
                surface_ros: PerCell::Cells(&ros),
                surface_fli: PerCell::Cells(&fli),
                surface_hpua: PerCell::Cells(&heat),
                ..data.columns()
            },
        )
        .map_err(|e| e.to_string())?;
        let run = ModelRun::Crown {
            raster,
            plan: CrownPlan::default(),
            fuels,
            tweaks: CrownTweakColumns {
                fm10: SurfaceTweakColumns {
                    spread_rate: args.override_rate.map(PerCell::Scalar),
                    ..SurfaceTweakColumns::default()
                },
                ..CrownTweakColumns::default()
            },
        };
        super::raster_probe::measure(
            args,
            || run_model(&run, &OutputRequest::crown(), 1).map_err(|e| e.to_string()),
            |output| validate(args, cases, fuels, output),
        )
    }
}
