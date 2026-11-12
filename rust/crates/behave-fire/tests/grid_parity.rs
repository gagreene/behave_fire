use behave_fire::{map_chunks, ParallelError};
use std::sync::atomic::{AtomicUsize, Ordering};
#[test]
fn joins_all_workers_before_returning_an_error() {
    let finished = AtomicUsize::new(0);
    let result = map_chunks(8, 4, |range| {
        finished.fetch_add(1, Ordering::SeqCst);
        if range.start == 0 {
            Err(3)
        } else {
            Ok(range)
        }
    });
    assert!(matches!(result, Err(ParallelError::Work(3))));
    assert_eq!(finished.load(Ordering::SeqCst), 4);
}
#[test]
fn short_requests_are_bounded() {
    for count in [0, 1, 2, 3] {
        let calls = AtomicUsize::new(0);
        let values = map_chunks(count, 100, |range| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok::<_, ()>(range.collect::<Vec<_>>())
        })
        .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), count);
        assert_eq!(
            values.into_iter().flatten().collect::<Vec<_>>(),
            (0..count).collect::<Vec<_>>()
        );
    }
}
#[test]
fn surface_batches_match_scalar_cells_bit_for_bit() {
    use behave_fire::batch::SurfaceBatch;
    use behave_fire::engine::{
        PreparedFuels, SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceTweaks,
    };
    use behave_fire::units::SpeedUnit;
    use behave_fire::{BehaveError, FuelModels, GridShape};
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let plan = SurfacePlan::new(SurfaceOptions::default());
    let inputs: Vec<_> = [1, 10, 101, 0, 261]
        .into_iter()
        .map(|model| {
            let mut input = SurfaceCell::new(model);
            input.wind_speed = SpeedUnit::MilesPerHour.to_base(7.0);
            input.slope = 15.0;
            input
        })
        .collect();
    for threads in [0, 1, 2, 4] {
        let batch = SurfaceBatch::run_with(GridShape::new(vec![5]).unwrap(), threads, |i| {
            Ok::<_, BehaveError>(plan.run_cell(
                &inputs[i],
                fuels.for_grid(inputs[i].fuel_model_grid),
                SurfaceTweaks::default(),
            ))
        })
        .unwrap();
        for (name, values) in comparison_columns(batch.into_named_columns()) {
            for (i, value) in values.iter().enumerate() {
                let scalar = plan.run_cell(
                    &inputs[i],
                    fuels.for_grid(inputs[i].fuel_model_grid),
                    SurfaceTweaks::default(),
                );
                let expected = scalar
                    .named_values()
                    .into_iter()
                    .find(|(key, _)| *key == name)
                    .unwrap()
                    .1;
                assert_eq!(value.to_bits(), expected.to_bits(), "{name} {i} {threads}");
            }
        }
    }
}
#[test]
fn crown_batches_match_scalar_and_columns_transfer_without_copying() {
    use behave_fire::batch::{CrownBatch, SurfaceBatch};
    use behave_fire::crown::get_surface_fire_inputs;
    use behave_fire::engine::{
        CrownCell, CrownPlan, CrownTweaks, PreparedFuels, SurfaceCell, SurfaceOptions, SurfacePlan,
        SurfaceTweaks,
    };
    use behave_fire::units::SpeedUnit;
    use behave_fire::{BehaveError, FuelModels, GridShape};
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let surface_plan = SurfacePlan::new(SurfaceOptions::default());
    let supplied = get_surface_fire_inputs(&surface_plan.run_cell(
        &SurfaceCell::new(10),
        fuels.for_grid(10),
        SurfaceTweaks::default(),
    ));
    let plan = CrownPlan::new(SurfaceOptions::default());
    let input = CrownCell {
        canopy_height: 30.0,
        canopy_base_height: 4.0,
        canopy_bulk_density: 0.01,
        wind_speed: SpeedUnit::MilesPerHour.to_base(15.0),
        ..CrownCell::default()
    };
    for threads in [0, 1, 2, 4] {
        let batch = CrownBatch::run_with(GridShape::new(vec![2, 3]).unwrap(), threads, |_| {
            Ok::<_, BehaveError>(plan.run_cell(
                &input,
                &supplied,
                fuels.for_grid(10),
                CrownTweaks::default(),
            ))
        })
        .unwrap();
        let expected = plan.run_cell(
            &input,
            &supplied,
            fuels.for_grid(10),
            CrownTweaks::default(),
        );
        assert!(batch
            .fire_type
            .iter()
            .all(|value| *value == expected.fire_type));
        for (name, values) in comparison_columns(batch.into_named_columns()) {
            let expected = if name == "fire_type" {
                expected.fire_type as f64
            } else {
                expected
                    .named_values()
                    .into_iter()
                    .find(|(key, _)| *key == name)
                    .unwrap()
                    .1
            };
            assert!(values
                .iter()
                .all(|value| value.to_bits() == expected.to_bits()));
        }
    }
    let batch = SurfaceBatch::run_with(GridShape::new(vec![3]).unwrap(), 1, |_| {
        Ok::<_, BehaveError>(surface_plan.run_cell(
            &SurfaceCell::new(1),
            fuels.for_grid(1),
            SurfaceTweaks::default(),
        ))
    })
    .unwrap();
    let pointer = batch.spread_rate.as_ptr();
    let columns = comparison_columns(batch.into_named_columns());
    assert_eq!(
        columns
            .iter()
            .find(|(name, _)| *name == "spread_rate")
            .unwrap()
            .1
            .as_ptr(),
        pointer
    );
    let empty = SurfaceBatch::run_with(GridShape::new(vec![0]).unwrap(), 4, |_| {
        panic!("Empty batches must not evaluate a cell");
        #[allow(unreachable_code)]
        Ok::<_, BehaveError>(behave_fire::surface::SurfaceResult::default())
    })
    .unwrap();
    assert!(comparison_columns(empty.into_named_columns())
        .iter()
        .all(|(_, values)| values.is_empty()));
}

#[test]
fn requested_workers_execute_on_distinct_native_threads() {
    use std::sync::{Barrier, Mutex};
    let barrier = Barrier::new(4);
    let ids = Mutex::new(std::collections::HashSet::new());
    map_chunks(8, 4, |range| {
        ids.lock().unwrap().insert(std::thread::current().id());
        barrier.wait();
        Ok::<_, ()>(range)
    })
    .unwrap();
    assert_eq!(ids.lock().unwrap().len(), 4);
}
#[test]
fn allocation_failures_are_returned_before_work_starts() {
    let calls = AtomicUsize::new(0);
    let result = map_chunks(usize::MAX, usize::MAX, |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok::<u8, ()>(1)
    });
    assert!(matches!(result, Err(ParallelError::Allocation(_))));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}
#[test]
fn standalone_batch_validation_retains_structured_core_errors() {
    use behave_fire::{engine::SurfaceCell, BehaveError, GridShape, PerCell};
    let result =
        PerCell::<SurfaceCell>::Cells(&[]).validate(&GridShape::new(vec![2]).unwrap(), "inputs");
    assert!(matches!(
        result,
        Err(BehaveError::LengthMismatch {
            expected: 2,
            actual: 0,
            ..
        })
    ));
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
