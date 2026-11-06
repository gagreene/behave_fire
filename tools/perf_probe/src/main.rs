//! Standalone core attribution, not end-to-end Python performance evidence.
//! Allocation instrumentation is confined to this unpublished diagnostic tool.

use behave_fire::crown::{self, CrownInput, SurfaceFireInputs};
use behave_fire::surface::masked::{IntegerKind, IntegerValue, MaskedValue, Precision};
use behave_fire::surface::{self, SurfaceInput};
use behave_fire::units::SpeedUnit;
use behave_fire::{FuelModel, FuelModels};
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
}

struct Case<'a> {
    surface: SurfaceInput,
    crown: CrownInput,
    typed: SurfaceFireInputs<MaskedValue>,
    fuel: &'a FuelModel,
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
            "--cells" => result.cells = value.parse().map_err(|_| "invalid cells")?,
            "--repetitions" => {
                result.repetitions = value.parse().map_err(|_| "invalid repetitions")?
            }
            "--seed" => result.seed = value.parse().map_err(|_| "invalid seed")?,
            _ => return Err(format!("unknown option {name}")),
        }
    }
    if !matches!(
        result.operation.as_str(),
        "surface" | "crown" | "typed-crown"
    ) || !matches!(result.layout.as_str(), "a" | "b")
        || !matches!(result.dataset.as_str(), "uniform" | "heterogeneous")
        || !matches!(result.mode.as_str(), "timing" | "allocations")
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
    Ok(result)
}

/// Prepare a bounded repeating input period, preserving each typed mask period.
fn build_cases<'a>(args: &Arguments, fuels: &'a FuelModels) -> Result<Vec<Case<'a>>, String> {
    let base = if args.dataset == "uniform" { 1 } else { 136 };
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
        let mut input = SurfaceInput::new(10);
        input.wind_speed_units = SpeedUnit::FeetPerMinute;
        input.wind_speed = 8.0;
        input.wind_direction = 35.0;
        input.slope = 12.0;
        let phase = ((index + args.seed) % 17) as f64 / 16.0;
        if args.dataset == "heterogeneous" {
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
        let result = surface::run_with_fuel(&input, fuel).map_err(|error| error.to_string())?;
        let mut crown = CrownInput::new(result);
        crown.fuel_model_grid = input.fuel_model_grid;
        crown.m1h = input.m1h;
        crown.m10h = input.m10h;
        crown.m100h = input.m100h;
        crown.mlh = input.mlh;
        crown.mlw = input.mlw;
        crown.wind_speed = input.wind_speed;
        crown.wind_speed_units = SpeedUnit::FeetPerMinute;
        crown.wind_direction = input.wind_direction;
        crown.slope = input.slope;
        crown.canopy_base_height = 4.0;
        crown.canopy_height = 30.0;
        crown.canopy_bulk_density = 0.01;
        crown.moisture_foliar = 100.0;
        if args.dataset == "heterogeneous" {
            crown.canopy_base_height = 1.0 + 5.0 * phase;
            crown.canopy_height = 15.0 + 25.0 * phase;
            crown.canopy_bulk_density = 0.005 + 0.02 * phase;
            crown.moisture_foliar = 80.0 + 40.0 * phase;
        }
        let mut typed = SurfaceFireInputs::<MaskedValue>::new();
        if args.layout == "a" {
            typed.surface_ros = MaskedValue::new(
                result.spread_rate as f32 as f64,
                false,
                false,
                Precision::F32,
            );
            typed.surface_fli = MaskedValue::plain(200.0)
                .with_integer(IntegerValue::from_bits(IntegerKind::U16, 200));
            typed.surface_hpua = MaskedValue::new(
                result.heat_per_unit_area,
                index % 7 == 0,
                true,
                Precision::F64,
            );
        } else {
            typed.surface_ros =
                MaskedValue::new(result.spread_rate, index % 5 == 0, true, Precision::F64);
            typed.surface_fli = MaskedValue::plain(200.0)
                .with_integer(IntegerValue::from_bits(IntegerKind::I32, 200));
            typed.surface_hpua = MaskedValue::new(
                result.heat_per_unit_area as f32 as f64,
                false,
                false,
                Precision::F32,
            );
        }
        cases.push(Case {
            surface: input,
            crown,
            typed,
            fuel,
        });
    }
    Ok(cases)
}

/// Retain the complete result at a black-box boundary before selecting a checksum.
fn evaluate(args: &Arguments, case: &Case<'_>, fuels: &FuelModels) -> f64 {
    match args.operation.as_str() {
        "surface" => {
            black_box(
                surface::run_with_fuel(black_box(&case.surface), black_box(case.fuel)).unwrap(),
            )
            .spread_rate
        }
        "crown" => {
            black_box(crown::run(black_box(&case.crown), black_box(fuels)).unwrap())
                .final_spread_rate
        }
        _ => {
            black_box(
                crown::run_with_surface_inputs(
                    black_box(&case.typed),
                    black_box(&case.crown),
                    black_box(fuels),
                )
                .unwrap(),
            )
            .final_spread_rate
            .value
        }
    }
}

/// Emit core attribution with explicit limits on the measured scope.
fn main() -> Result<(), String> {
    let args = arguments()?;
    let fuels = FuelModels::default();
    let cases = build_cases(&args, &fuels)?;
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
        println!("{{\"operation\":\"{operation}\",\"dataset\":\"{}\",\"scope\":\"scalar-core-only\",\"cells\":{},\"repetitions\":{},\"median_seconds\":{},\"prepared_period\":{},\"masked_value_bytes\":{}}}",
                 args.dataset, args.cells, args.repetitions, median, cases.len(), std::mem::size_of::<MaskedValue>());
    }
    Ok(())
}
