//! Complete column-storage attribution and independent scalar replay checks.
use super::{Arguments, Case};
use behave_fire::crown::{self, MaskedCrownResult};
use behave_fire::engine::{
    CrownPlan, CrownTweaks, PreparedFuels, SurfaceOptions, SurfacePlan, SurfaceTweaks,
};
use behave_fire::raster::{
    CrownColumns, CrownRaster, CrownTweakColumns, SurfaceColumns, SurfaceRaster,
    SurfaceTweakColumns,
};
use behave_fire::surface::masked::MaskedValue;
use behave_fire::{GridShape, PerCell};
use std::hint::black_box;
#[cfg(not(feature = "count-alloc"))]
use std::time::Instant;

macro_rules! data_columns {
    ($name:ident,$member:ident,$columns:ident,$($field:ident),+) => {
        pub(super) struct $name {$($field:Vec<f64>,)+}
        impl $name {
            pub(super) fn new(args:&Arguments,cases:&[Case<'_>])->Self {
                Self {$($field:(0..args.cells).map(|i|cases[i%cases.len()].$member.$field).collect(),)+}
            }
            pub(super) fn columns(&self)->$columns<'_> {
                $columns {$($field:PerCell::Cells(&self.$field),)+..$columns::default()}
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

impl CrownData {
    pub(super) fn cell(&self, index: usize) -> behave_fire::engine::CrownCell {
        behave_fire::engine::CrownCell {
            m1h: self.m1h[index],
            m10h: self.m10h[index],
            m100h: self.m100h[index],
            mlh: self.mlh[index],
            mlw: self.mlw[index],
            wind_speed: self.wind_speed[index],
            wind_direction: self.wind_direction[index],
            slope: self.slope[index],
            aspect: self.aspect[index],
            canopy_base_height: self.canopy_base_height[index],
            canopy_height: self.canopy_height[index],
            canopy_bulk_density: self.canopy_bulk_density[index],
            moisture_foliar: self.moisture_foliar[index],
        }
    }
}
pub(super) fn check_value(actual: f64, expected: f64) -> bool {
    (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits()
}

fn check_columns(
    actual: Vec<(&'static str, behave_fire::output::ColumnValues)>,
    reference: &[Vec<(&'static str, f64)>],
    cells: usize,
) -> Result<(), String> {
    let actual = comparison_columns(actual);
    if reference.is_empty() || actual.len() != reference[0].len() {
        return Err("incomplete raster output inventory".into());
    }
    dump_plain(&actual)?;
    let mut seen = vec![false; reference[0].len()];
    for (name, column) in actual {
        if column.len() != cells {
            return Err(format!("wrong output length for {name}"));
        }
        let offset = reference[0]
            .iter()
            .position(|(key, _)| *key == name)
            .ok_or_else(|| format!("unknown field {name}"))?;
        if seen[offset] {
            return Err(format!("duplicate field {name}"));
        }
        seen[offset] = true;
        for (index, value) in column.into_iter().enumerate() {
            if !check_value(value, reference[index % reference.len()][offset].1) {
                return Err(format!("raster/scalar value mismatch {name}/{index}"));
            }
        }
    }
    Ok(())
}

pub(super) fn measure<T>(
    args: &Arguments,
    run: impl Fn() -> Result<T, String>,
    validate: impl Fn(T) -> Result<(), String>,
) -> Result<(), String> {
    #[cfg(feature = "count-alloc")]
    {
        use std::sync::atomic::Ordering;
        super::allocation::CALLS.store(0, Ordering::Relaxed);
        super::allocation::BYTES.store(0, Ordering::Relaxed);
        let output = run()?;
        let requests = super::allocation::CALLS.load(Ordering::Relaxed);
        let bytes = super::allocation::BYTES.load(Ordering::Relaxed);
        black_box(&output);
        validate(output)?;
        println!("{{\"operation\":\"{}\",\"dataset\":\"{}\",\"layout\":\"{}\",\"scope\":\"{}-core-with-storage\",\"cells\":{},\"batch_allocation_requests\":{requests},\"batch_requested_bytes\":{bytes},\"validated\":true}}",args.operation,args.dataset,args.layout,args.scope,args.cells);
    }
    #[cfg(not(feature = "count-alloc"))]
    {
        let mut times = Vec::with_capacity(args.repetitions);
        for _ in 0..args.repetitions {
            let start = Instant::now();
            let output = run()?;
            times.push(start.elapsed().as_secs_f64());
            black_box(&output);
            drop(output);
        }
        // Correctness work runs after all measurements, with a fresh complete output.
        validate(run()?)?;
        let first = times[0];
        let warmed = times[1..].to_vec();
        times.sort_by(f64::total_cmp);
        let median = if times.len() % 2 == 1 {
            times[times.len() / 2]
        } else {
            (times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.0
        };
        println!("{{\"operation\":\"{}\",\"dataset\":\"{}\",\"layout\":\"{}\",\"scope\":\"{}-core-with-storage\",\"cells\":{},\"first_seconds\":{first},\"warm_seconds\":{warmed:?},\"median_seconds\":{median},\"validated\":true}}",args.operation,args.dataset,args.layout,args.scope,args.cells);
    }
    Ok(())
}

fn surface(args: &Arguments, cases: &[Case<'_>], fuels: &PreparedFuels) -> Result<(), String> {
    let data = SurfaceData::new(args, cases);
    let models: Vec<_> = (0..args.cells)
        .map(|i| cases[i % cases.len()].surface.fuel_model_grid)
        .collect();
    let columns = SurfaceColumns {
        fuel_model_grid: PerCell::Cells(&models),
        ..data.columns()
    };
    let raster = SurfaceRaster::new(
        GridShape::new(vec![args.cells]).map_err(|e| e.to_string())?,
        columns,
    )
    .map_err(|e| e.to_string())?;
    let plan = SurfacePlan::new(SurfaceOptions::default());
    let overrides = SurfaceTweakColumns {
        spread_rate: args.override_rate.map(PerCell::Scalar),
        ..SurfaceTweakColumns::default()
    };
    let scalar_overrides = SurfaceTweaks {
        spread_rate: args.override_rate,
        ..SurfaceTweaks::default()
    };
    let reference: Vec<_> = cases
        .iter()
        .map(|case| {
            plan.run_cell(&case.surface, case.fuel, scalar_overrides)
                .named_values()
                .to_vec()
        })
        .collect();
    measure(
        args,
        || {
            raster
                .run(&plan, fuels, &overrides, 1)
                .map_err(|e| e.to_string())
        },
        |output| check_columns(output.into_named_columns(), &reference, args.cells),
    )
}

fn crown(args: &Arguments, cases: &[Case<'_>], fuels: &PreparedFuels) -> Result<(), String> {
    let data = CrownData::new(args, cases);
    let ros: Vec<_> = (0..args.cells)
        .map(|i| cases[i % cases.len()].supplied.surface_ros)
        .collect();
    let fli: Vec<_> = (0..args.cells)
        .map(|i| cases[i % cases.len()].supplied.surface_fli)
        .collect();
    let hpua: Vec<_> = (0..args.cells)
        .map(|i| cases[i % cases.len()].supplied.surface_hpua)
        .collect();
    let raster = CrownRaster::new(
        GridShape::new(vec![args.cells]).map_err(|e| e.to_string())?,
        CrownColumns {
            surface_ros: PerCell::Cells(&ros),
            surface_fli: PerCell::Cells(&fli),
            surface_hpua: PerCell::Cells(&hpua),
            ..data.columns()
        },
    )
    .map_err(|e| e.to_string())?;
    let plan = CrownPlan::new(SurfaceOptions::default());
    let overrides = CrownTweakColumns {
        fm10: SurfaceTweakColumns {
            spread_rate: args.override_rate.map(PerCell::Scalar),
            ..SurfaceTweakColumns::default()
        },
        ..CrownTweakColumns::default()
    };
    let scalar_overrides = CrownTweaks {
        fm10: SurfaceTweaks {
            spread_rate: args.override_rate,
            ..SurfaceTweaks::default()
        },
        ..CrownTweaks::default()
    };
    let reference: Vec<_> = cases
        .iter()
        .map(|case| {
            let result = plan.run_cell(
                &case.crown,
                &case.supplied,
                fuels.for_grid(10),
                scalar_overrides,
            );
            let mut values = result.named_values().to_vec();
            values.push(("fire_type", result.fire_type as f64));
            values
        })
        .collect();
    measure(
        args,
        || {
            raster
                .run(&plan, fuels, &overrides, 1)
                .map_err(|e| e.to_string())
        },
        |output| check_columns(output.into_named_columns(), &reference, args.cells),
    )
}

struct TypedColumns {
    values: Vec<Vec<f64>>,
    masks: Vec<Vec<bool>>,
    classes: Vec<i32>,
}

pub(super) fn schema(mut value: MaskedValue) -> MaskedValue {
    value.value = 0.0;
    value.mask = false;
    value
}

fn typed(args: &Arguments, cases: &[Case<'_>], fuels: &PreparedFuels) -> Result<(), String> {
    let options = SurfaceOptions::default();
    let reference: Vec<MaskedCrownResult> = cases
        .iter()
        .map(|case| {
            crown::run_with_surface_inputs(&case.typed, &case.crown, fuels.for_grid(10), &options)
                .map_err(|e| e.to_string())
        })
        .collect::<Result<_, _>>()?;
    let expected_schema = reference[0].named_values().map(|(_, value)| schema(value));
    for result in &reference {
        for ((name, value), expected) in result.named_values().into_iter().zip(&expected_schema) {
            if schema(value) != *expected {
                return Err(format!("nonconstant typed field schema: {name}"));
            }
        }
    }
    let run = || {
        let mut output = TypedColumns {
            values: Vec::with_capacity(21),
            masks: Vec::with_capacity(21),
            classes: Vec::new(),
        };
        for _ in 0..21 {
            let mut values = Vec::new();
            values
                .try_reserve_exact(args.cells)
                .map_err(|e| e.to_string())?;
            let mut masks = Vec::new();
            masks
                .try_reserve_exact(args.cells)
                .map_err(|e| e.to_string())?;
            output.values.push(values);
            output.masks.push(masks);
        }
        output
            .classes
            .try_reserve_exact(args.cells)
            .map_err(|e| e.to_string())?;
        for index in 0..args.cells {
            let case = &cases[index % cases.len()];
            let result = crown::run_with_surface_inputs(
                &case.typed,
                &case.crown,
                fuels.for_grid(10),
                &options,
            )
            .map_err(|e| e.to_string())?;
            for (field, (_, value)) in result.named_values().into_iter().enumerate() {
                output.values[field].push(value.value);
                output.masks[field].push(value.mask);
            }
            output.classes.push(result.fire_type);
        }
        Ok(output)
    };
    measure(args, run, |output| {
        dump_typed(
            &output.values,
            &output.masks,
            &output.classes,
            &reference[0],
        )?;
        for index in 0..args.cells {
            let expected = &reference[index % reference.len()];
            if output.classes[index] != expected.fire_type {
                return Err(format!("typed class mismatch/{index}"));
            }
            for (field, (name, value)) in expected.named_values().into_iter().enumerate() {
                if !check_value(output.values[field][index], value.value)
                    || output.masks[field][index] != value.mask
                {
                    return Err(format!("typed raster/scalar mismatch {name}/{index}"));
                }
            }
        }
        Ok(())
    })
}

pub(super) fn run(
    args: &Arguments,
    cases: &[Case<'_>],
    fuels: &PreparedFuels,
) -> Result<(), String> {
    match args.operation.as_str() {
        "surface" => surface(args, cases, fuels),
        "crown" => crown(args, cases, fuels),
        _ => typed(args, cases, fuels),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn full_output_gate_rejects_missing_or_duplicate_fields() {
        let reference = vec![vec![("a", 1.0), ("b", 2.0)]];
        assert!(super::check_columns(vec![("a", vec![1.0])], &reference, 1).is_err());
        assert!(
            super::check_columns(vec![("a", vec![1.0]), ("a", vec![1.0])], &reference, 1).is_err()
        );
    }
}

/// Export every stored result outside timing for comparison to unchanged science.
pub(super) fn dump_plain(columns: &[(&str, Vec<f64>)]) -> Result<(), String> {
    use std::io::Write;
    if let Some(path) = std::env::var_os("BEHAVE_PROBE_OUTPUT") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).map_err(|e| e.to_string())?;
        for (name, values) in columns {
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
    Ok(())
}

/// Export typed raw data, masks, class values, and complete constant metadata.
pub(super) fn dump_typed(
    values: &[Vec<f64>],
    masks: &[Vec<bool>],
    classes: &[i32],
    reference: &MaskedCrownResult,
) -> Result<(), String> {
    use std::io::Write;
    if let Some(path) = std::env::var_os("BEHAVE_PROBE_OUTPUT") {
        let path = std::path::PathBuf::from(path);
        let columns: Vec<_> = reference
            .named_values()
            .into_iter()
            .enumerate()
            .map(|(i, (name, _))| (name, values[i].clone()))
            .collect();
        dump_plain(&columns)?;
        for (i, (name, value)) in reference.named_values().into_iter().enumerate() {
            std::fs::write(
                path.join(format!("{name}.mask")),
                masks[i].iter().map(|m| u8::from(*m)).collect::<Vec<_>>(),
            )
            .map_err(|e| e.to_string())?;
            std::fs::write(
                path.join(format!("{name}.schema")),
                format!("{:?}", schema(value)),
            )
            .map_err(|e| e.to_string())?;
        }
        let mut file = std::io::BufWriter::new(
            std::fs::File::create(path.join("fire_type.i32")).map_err(|e| e.to_string())?,
        );
        for value in classes {
            file.write_all(&value.to_le_bytes())
                .map_err(|e| e.to_string())?;
        }
        file.flush().map_err(|e| e.to_string())?;
    }
    Ok(())
}

// Numerical fixture comparison only; production export retains typed columns.
fn comparison_columns(
    columns: Vec<(&'static str, behave_fire::output::ColumnValues)>,
) -> Vec<(&'static str, Vec<f64>)> {
    use behave_fire::output::ColumnValues;
    columns
        .into_iter()
        .map(|(name, values)| {
            (
                name,
                match values {
                    ColumnValues::Float64(values) => values,
                    ColumnValues::Int32(values) => values.into_iter().map(f64::from).collect(),
                    ColumnValues::Bool(values) => values
                        .into_iter()
                        .map(|value| f64::from(u8::from(value)))
                        .collect(),
                },
            )
        })
        .collect()
}
