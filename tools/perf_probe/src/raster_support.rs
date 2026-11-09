//! Prepared fixture columns and timers for the surviving float64 model probe.
use super::{Arguments, Case};
use behave_fire::raster::{CrownColumns, SurfaceColumns};
use behave_fire::PerCell;
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

pub(super) fn check_value(actual: f64, expected: f64) -> bool {
    (actual.is_nan() && expected.is_nan()) || actual.to_bits() == expected.to_bits()
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
