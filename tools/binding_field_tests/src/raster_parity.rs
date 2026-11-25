//! Float64 model raster columns match independently computed scalar results.
//!
//! Ported from the retired `perf_probe` crate without its timers or allocator.
//! Item order: the `data_columns!` macro is declared before its uses because
//! Rust macros must be defined before they are invoked.
use behave_fire::crown::{self, CrownResult, SurfaceFireInputs};
use behave_fire::engine::{
    run_model, CrownCell, CrownPlan, CrownTweaks, ModelBatch, ModelRun, OutputRequest,
    PreparedFuel, PreparedFuels, SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceSource,
    SurfaceTweaks,
};
use behave_fire::output::ColumnValues;
use behave_fire::raster::{
    CrownColumns, CrownRaster, CrownTweakColumns, SurfaceColumns, SurfaceRaster,
    SurfaceTweakColumns,
};
use behave_fire::{FuelModels, GridShape, PerCell};

struct Settings {
    operation: &'static str,
    layout: &'static str,
    dataset: &'static str,
    cells: usize,
    seed: usize,
}

struct Case<'a> {
    surface: SurfaceCell,
    crown: CrownCell,
    supplied: SurfaceFireInputs<f64>,
    typed: SurfaceFireInputs<f64>,
    missing_ros: bool,
    missing_hpua: bool,
    fuel: &'a PreparedFuel,
}

macro_rules! data_columns {
    ($name:ident,$member:ident,$columns:ident,$($field:ident),+) => {
        struct $name {$($field: Vec<f64>,)+}
        impl $name {
            fn columns(&self) -> $columns<'_> {
                $columns {$($field: PerCell::Cells(&self.$field),)+ ..$columns::default()}
            }
            fn new(settings: &Settings, cases: &[Case<'_>]) -> Self {
                Self {$($field: (0..settings.cells).map(|i| cases[i % cases.len()].$member.$field).collect(),)+}
            }
        }
    }
}
data_columns!(
    SurfaceData,
    surface,
    SurfaceColumns,
    m1h,
    m10h,
    m100h,
    mlh,
    mlw,
    wind_speed,
    wind_direction,
    slope,
    aspect,
    canopy_cover,
    canopy_height,
    crown_ratio
);
data_columns!(
    CrownData,
    crown,
    CrownColumns,
    m1h,
    m10h,
    m100h,
    mlh,
    mlw,
    wind_speed,
    wind_direction,
    slope,
    aspect,
    canopy_base_height,
    canopy_height,
    canopy_bulk_density,
    moisture_foliar
);

/// Prepare a bounded repeating input period, preserving each typed mask period.
fn build_cases<'a>(settings: &Settings, fuels: &'a PreparedFuels) -> Vec<Case<'a>> {
    let base = if settings.dataset == "uniform" {
        1
    } else if settings.dataset == "continuous" {
        settings.cells
    } else {
        136
    };
    let mask_period = if settings.operation != "typed-crown" {
        1
    } else if settings.layout == "a" {
        7
    } else {
        5
    };
    let period = (base * mask_period).min(settings.cells);
    let mut cases = Vec::with_capacity(period);
    for index in 0..period {
        let mut input = SurfaceCell::new(10);
        input.wind_speed = 8.0;
        input.wind_direction = 35.0;
        input.slope = 12.0;
        let phase = if settings.dataset == "continuous" {
            ((index + settings.seed) as f64 * ((5.0_f64.sqrt() - 1.0) / 2.0)).fract()
        } else {
            ((index + settings.seed) % 17) as f64 / 16.0
        };
        if settings.dataset != "uniform" {
            input.fuel_model_grid = [1, 4, 10, 101, 124, 145, 201, 0][index % 8];
            input.m1h = 0.03 + 0.07 * phase;
            input.m10h = 0.04 + 0.08 * phase;
            input.m100h = 0.05 + 0.09 * phase;
            input.mlh = 0.35 + 0.9 * phase;
            input.mlw = 0.6 + 0.8 * phase;
            input.wind_speed = 2.0 + 160.0 * phase;
            input.wind_direction = 350.0 * phase;
            input.slope = 35.0 * phase;
        }
        let fuel = fuels.for_grid(input.fuel_model_grid);
        let result = SurfacePlan::new(SurfaceOptions::default()).run_cell(
            &input,
            fuel,
            SurfaceTweaks::default(),
        );
        let mut crown = CrownCell {
            m1h: input.m1h,
            m10h: input.m10h,
            m100h: input.m100h,
            mlh: input.mlh,
            mlw: input.mlw,
            wind_speed: input.wind_speed,
            wind_direction: input.wind_direction,
            slope: input.slope,
            aspect: input.aspect,
            canopy_base_height: 4.0,
            canopy_height: 30.0,
            canopy_bulk_density: 0.01,
            moisture_foliar: 100.0,
        };
        if settings.dataset == "heterogeneous" {
            crown.canopy_base_height = 1.0 + 5.0 * phase;
            crown.canopy_height = 15.0 + 25.0 * phase;
            crown.canopy_bulk_density = 0.005 + 0.02 * phase;
            crown.moisture_foliar = 80.0 + 40.0 * phase;
        }
        // Typed storage is widened once, before the raster calculation.
        let mut typed = SurfaceFireInputs::<f64>::new();
        typed.surface_fli = 200.;
        let (missing_ros, missing_hpua) = if settings.layout == "a" {
            typed.surface_ros = result.spread_rate as f32 as f64;
            typed.surface_hpua = result.heat_per_unit_area;
            (false, index % 7 == 0)
        } else {
            typed.surface_ros = result.spread_rate;
            typed.surface_hpua = result.heat_per_unit_area as f32 as f64;
            (index % 5 == 0, false)
        };
        cases.push(Case {
            surface: input,
            crown,
            supplied: crown::get_surface_fire_inputs(&result),
            typed,
            missing_ros,
            missing_hpua,
            fuel,
        });
    }
    cases
}

/// Compare one value exactly, treating any two NaNs as equal.
fn check_value(actual: f64, expected: f64) -> bool {
    (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits()
}

/// Compute the scalar Crown result with the reviewed NoData overlays for typed inputs.
///
/// Finite values come from scalar float64 science on the widened storage values,
/// never from the raster result under test.
fn reference_crown(settings: &Settings, case: &Case<'_>, fuels: &PreparedFuels) -> CrownResult {
    let typed = settings.operation == "typed-crown";
    let input = supplied(case, typed, false);
    let mut value = CrownPlan::default().run_cell(
        &case.crown,
        &input,
        fuels.for_grid(10),
        CrownTweaks::default(),
    );
    if typed && case.missing_hpua {
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
    if typed && case.missing_ros {
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

/// Run one float64 model raster and validate every output column.
fn run(settings: &Settings, cases: &[Case<'_>], fuels: &PreparedFuels) -> Result<(), String> {
    let shape = GridShape::new(vec![settings.cells]).map_err(|e| e.to_string())?;
    if settings.operation == "surface" {
        let data = SurfaceData::new(settings, cases);
        let models = (0..settings.cells)
            .map(|i| cases[i % cases.len()].surface.fuel_model_grid)
            .collect::<Vec<_>>();
        let raster = SurfaceRaster::new(
            shape,
            SurfaceColumns {
                fuel_model_grid: PerCell::Cells(&models),
                ..data.columns()
            },
        )
        .map_err(|e| e.to_string())?;
        let model = ModelRun::Surface {
            raster,
            plan: SurfacePlan::default(),
            source: SurfaceSource::Fuels(fuels),
            tweaks: SurfaceTweakColumns::default(),
        };
        let output = run_model(&model, &OutputRequest::surface(), 1).map_err(|e| e.to_string())?;
        return validate(settings, cases, fuels, output);
    }
    let typed = settings.operation == "typed-crown";
    let data = CrownData::new(settings, cases);
    let column = |field: fn(&SurfaceFireInputs<f64>) -> f64| {
        (0..settings.cells)
            .map(|i| field(&supplied(&cases[i % cases.len()], typed, true)))
            .collect::<Vec<_>>()
    };
    let ros = column(|s| s.surface_ros);
    let fli = column(|s| s.surface_fli);
    let heat = column(|s| s.surface_hpua);
    let raster = CrownRaster::new(
        shape,
        CrownColumns {
            surface_ros: PerCell::Cells(&ros),
            surface_fli: PerCell::Cells(&fli),
            surface_hpua: PerCell::Cells(&heat),
            ..data.columns()
        },
    )
    .map_err(|e| e.to_string())?;
    let model = ModelRun::Crown {
        raster,
        plan: CrownPlan::default(),
        fuels,
        tweaks: CrownTweakColumns::default(),
    };
    let output = run_model(&model, &OutputRequest::crown(), 1).map_err(|e| e.to_string())?;
    validate(settings, cases, fuels, output)
}

/// Return the supplied Surface inputs, optionally with the typed layout's missing values.
fn supplied(case: &Case<'_>, typed: bool, missing: bool) -> SurfaceFireInputs<f64> {
    if !typed {
        return case.supplied;
    }
    let mut result = case.typed;
    if missing && case.missing_ros {
        result.surface_ros = f64::NAN;
    }
    if missing && case.missing_hpua {
        result.surface_hpua = f64::NAN;
    }
    result
}

/// Require a complete, unique column inventory whose every value matches the scalar reference.
fn validate(
    settings: &Settings,
    cases: &[Case<'_>],
    fuels: &PreparedFuels,
    output: ModelBatch,
) -> Result<(), String> {
    let reference = cases
        .iter()
        .map(|case| {
            if settings.operation == "surface" {
                SurfacePlan::default()
                    .run_cell(&case.surface, case.fuel, SurfaceTweaks::default())
                    .named_values()
                    .to_vec()
            } else {
                let value = reference_crown(settings, case, fuels);
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
    let mut seen = vec![false; reference[0].len()];
    for (name, column) in columns {
        let slot = reference[0]
            .iter()
            .position(|(key, _)| *key == name)
            .ok_or_else(|| format!("unknown field {name}"))?;
        if seen[slot] {
            return Err(format!("duplicate field {name}"));
        }
        seen[slot] = true;
        match column {
            ColumnValues::Float64(values) => {
                if values.len() != settings.cells || name == "fire_type" {
                    return Err(format!("wrong f64 field shape or type {name}"));
                }
                for (index, value) in values.iter().enumerate() {
                    if !check_value(*value, reference[index % reference.len()][slot].1) {
                        return Err(format!("f64 scalar/truth mismatch {name}/{index}"));
                    }
                }
            }
            ColumnValues::Int32(values) => {
                if name != "fire_type" || values.len() != settings.cells {
                    return Err("wrong categorical shape or type".into());
                }
                for (index, &value) in values.iter().enumerate() {
                    if f64::from(value) != reference[index % reference.len()][slot].1 {
                        return Err(format!("category mismatch {index}"));
                    }
                }
            }
            ColumnValues::Bool(_) => return Err("unexpected model flag column".into()),
        }
    }
    Ok(())
}

// ---- raster columns against scalar results ----
#[test]
fn raster_columns_match_scalar_results_for_every_dataset_and_layout() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    for dataset in ["uniform", "heterogeneous", "continuous"] {
        for (operation, layout) in [
            ("surface", "a"),
            ("crown", "a"),
            ("typed-crown", "a"),
            ("typed-crown", "b"),
        ] {
            let settings = Settings {
                operation,
                layout,
                dataset,
                cells: 64,
                seed: 20261105,
            };
            let cases = build_cases(&settings, &fuels);
            if let Err(error) = run(&settings, &cases, &fuels) {
                panic!("{dataset}/{operation}/{layout}: {error}");
            }
        }
    }
}
