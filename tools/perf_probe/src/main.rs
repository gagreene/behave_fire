//! Standalone core attribution, not end-to-end Python performance evidence.
//! Allocation instrumentation is confined to this unpublished diagnostic tool.

mod f64_raster;
mod raster_support;

use behave_fire::crown::{self, SurfaceFireInputs};
use behave_fire::engine::{
    CrownCell, CrownPlan, CrownTweaks, PreparedFuel, PreparedFuels, SurfaceCell, SurfaceOptions,
    SurfacePlan, SurfaceTweaks,
};

use behave_fire::FuelModels;
use std::hint::black_box;
#[cfg(not(feature = "count-alloc"))]
use std::time::Instant;

#[cfg(feature = "count-alloc")]
mod allocation {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub struct Counter;
    pub static CALLS: AtomicUsize = AtomicUsize::new(0);
    pub static BYTES: AtomicUsize = AtomicUsize::new(0);

    // SAFETY: every pointer/layout operation delegates unchanged to System.
    // Relaxed atomic counters observe requests without changing allocation ABI.
    unsafe impl GlobalAlloc for Counter {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            // SAFETY: the caller supplied the valid allocation layout.
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
            // SAFETY: the caller supplied the valid allocation layout.
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            // SAFETY: pointer and layout originate from the delegated allocator.
            unsafe { System.dealloc(pointer, layout) }
        }

        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            CALLS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size, Ordering::Relaxed);
            // SAFETY: all reallocation requirements are the caller's contract.
            unsafe { System.realloc(pointer, layout, size) }
        }
    }
}

#[cfg(feature = "count-alloc")]
#[global_allocator]
static ALLOCATOR: allocation::Counter = allocation::Counter;

struct Arguments {
    operation: String,
    layout: String,
    dataset: String,
    mode: String,
    cells: usize,
    repetitions: usize,
    seed: usize,
    override_rate: Option<f64>,
    scope: String,
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

/// Parse explicit attribution controls without a CLI runtime dependency.
fn arguments() -> Result<Arguments, String> {
    let mut result = Arguments {
        operation: "surface".into(),
        layout: "a".into(),
        dataset: "uniform".into(),
        mode: "timing".into(),
        cells: 10_000,
        repetitions: 3,
        seed: 20261105,
        override_rate: None,
        scope: "scalar".into(),
    };
    let mut args = std::env::args().skip(1);
    while let Some(name) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {name}"))?;
        match name.as_str() {
            "--case" => result.operation = value,
            "--layout" => result.layout = value,
            "--dataset" => result.dataset = value,
            "--mode" => result.mode = value,
            "--scope" => result.scope = value,
            "--cells" => result.cells = value.parse().map_err(|_| "invalid cells")?,
            "--repetitions" => {
                result.repetitions = value.parse().map_err(|_| "invalid repetitions")?
            }
            "--seed" => result.seed = value.parse().map_err(|_| "invalid seed")?,
            "--override-rate" => {
                result.override_rate = Some(value.parse().map_err(|_| "invalid override rate")?)
            }
            _ => return Err(format!("unknown option {name}")),
        }
    }
    if !matches!(
        result.operation.as_str(),
        "surface" | "crown" | "typed-crown"
    ) || !matches!(result.layout.as_str(), "a" | "b")
        || !matches!(
            result.dataset.as_str(),
            "uniform" | "heterogeneous" | "continuous"
        )
        || !matches!(result.mode.as_str(), "timing" | "allocations")
        || !matches!(result.scope.as_str(), "scalar" | "f64-raster")
        || result.cells == 0
        || result.repetitions == 0
    {
        return Err("expected a known case/layout/dataset/mode and positive work counts".into());
    }
    if (result.mode == "allocations") != cfg!(feature = "count-alloc") {
        return Err(
            "allocation mode requires count-alloc; timing requires an uninstrumented build".into(),
        );
    }
    if result.operation == "typed-crown" && result.override_rate.is_some() {
        return Err("The checked Python-layout probe has no fused native tweak argument".into());
    }
    Ok(result)
}

/// Prepare a bounded repeating input period, preserving each typed mask period.
fn build_cases<'a>(args: &Arguments, fuels: &'a PreparedFuels) -> Result<Vec<Case<'a>>, String> {
    let base = if args.dataset == "uniform" {
        1
    } else if args.dataset == "continuous" {
        args.cells
    } else {
        136
    };
    let mask_period = if args.operation != "typed-crown" {
        1
    } else if args.layout == "a" {
        7
    } else {
        5
    };
    let period = (base * mask_period).min(args.cells);
    let mut cases = Vec::with_capacity(period);
    for index in 0..period {
        let mut input = SurfaceCell::new(10);
        input.wind_speed = 8.0;
        input.wind_direction = 35.0;
        input.slope = 12.0;
        let phase = if args.dataset == "continuous" {
            ((index + args.seed) as f64 * ((5.0_f64.sqrt() - 1.0) / 2.0)).fract()
        } else {
            ((index + args.seed) % 17) as f64 / 16.0
        };
        if args.dataset != "uniform" {
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
        if args.dataset == "heterogeneous" {
            crown.canopy_base_height = 1.0 + 5.0 * phase;
            crown.canopy_height = 15.0 + 25.0 * phase;
            crown.canopy_bulk_density = 0.005 + 0.02 * phase;
            crown.moisture_foliar = 80.0 + 40.0 * phase;
        }
        // Fixture storage is widened once, before any timed core calculation.
        let mut typed = SurfaceFireInputs::<f64>::new();
        typed.surface_fli = 200.;
        let (missing_ros, missing_hpua) = if args.layout == "a" {
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
    Ok(cases)
}

/// Retain the complete result at a black-box boundary before selecting a checksum.
fn evaluate(args: &Arguments, case: &Case<'_>, fuels: &PreparedFuels) -> f64 {
    let surface_tweaks = SurfaceTweaks {
        spread_rate: args.override_rate,
        ..SurfaceTweaks::default()
    };
    let crown_tweaks = CrownTweaks {
        fm10: surface_tweaks,
        ..CrownTweaks::default()
    };
    match args.operation.as_str() {
        "surface" => {
            black_box(SurfacePlan::new(SurfaceOptions::default()).run_cell(
                black_box(&case.surface),
                black_box(case.fuel),
                surface_tweaks,
            ))
            .spread_rate
        }
        "crown" => {
            black_box(CrownPlan::new(SurfaceOptions::default()).run_cell(
                black_box(&case.crown),
                black_box(&case.supplied),
                black_box(fuels.for_grid(10)),
                crown_tweaks,
            ))
            .final_spread_rate
        }
        _ => {
            let supplied = f64_raster::supplied(case, true, true);
            black_box(CrownPlan::default().run_cell(
                &case.crown,
                &supplied,
                fuels.for_grid(10),
                crown_tweaks,
            ))
            .final_spread_rate
        }
    }
}
/// Emit core attribution with explicit limits on the measured scope.
fn main() -> Result<(), String> {
    let args = arguments()?;
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let cases = build_cases(&args, &fuels)?;
    if args.scope == "f64-raster" {
        return f64_raster::run(&args, &cases, &fuels);
    }
    black_box(evaluate(&args, &cases[0], &fuels));
    let operation = if args.operation == "typed-crown" && args.layout == "b" {
        "typed-crown-b"
    } else {
        &args.operation
    };
    #[cfg(feature = "count-alloc")]
    {
        use std::sync::atomic::Ordering;
        allocation::CALLS.store(0, Ordering::Relaxed);
        allocation::BYTES.store(0, Ordering::Relaxed);
        for index in 0..args.cells {
            black_box(evaluate(&args, &cases[index % cases.len()], &fuels));
        }
        let calls = allocation::CALLS.load(Ordering::Relaxed);
        let bytes = allocation::BYTES.load(Ordering::Relaxed);
        println!("{{\"operation\":\"{operation}\",\"dataset\":\"{}\",\"scope\":\"scalar-core-only\",\"cells\":{},\"kernel_per_cell\":{},\"requested_bytes_per_cell\":{},\"encoder_per_cell\":null,\"prepared_period\":{}}}",
                 args.dataset, args.cells, calls as f64 / args.cells as f64, bytes as f64 / args.cells as f64, cases.len());
    }
    #[cfg(not(feature = "count-alloc"))]
    {
        let mut times = Vec::with_capacity(args.repetitions);
        for _ in 0..args.repetitions {
            let start = Instant::now();
            for index in 0..args.cells {
                black_box(evaluate(&args, &cases[index % cases.len()], &fuels));
            }
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        let median = if times.len() % 2 == 1 {
            times[times.len() / 2]
        } else {
            (times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.0
        };
        println!("{{\"operation\":\"{operation}\",\"dataset\":\"{}\",\"scope\":\"scalar-core-only\",\"cells\":{},\"repetitions\":{},\"median_seconds\":{},\"prepared_period\":{}}}",
                 args.dataset, args.cells, args.repetitions, median, cases.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn actual_f64_probe_validates_all_columns_for_every_fixture_layout_and_dataset() {
        let fuels = PreparedFuels::from_registry(&FuelModels::default());
        for dataset in ["uniform", "heterogeneous", "continuous"] {
            for (operation, layout) in [
                ("surface", "a"),
                ("crown", "a"),
                ("typed-crown", "a"),
                ("typed-crown", "b"),
            ] {
                let args = Arguments {
                    operation: operation.into(),
                    layout: layout.into(),
                    dataset: dataset.into(),
                    mode: if cfg!(feature = "count-alloc") {
                        "allocations"
                    } else {
                        "timing"
                    }
                    .into(),
                    cells: 64,
                    repetitions: 1,
                    seed: 20261105,
                    override_rate: None,
                    scope: "f64-raster".into(),
                };
                let cases = build_cases(&args, &fuels).unwrap();
                f64_raster::run(&args, &cases, &fuels).unwrap();
            }
        }
    }
}
