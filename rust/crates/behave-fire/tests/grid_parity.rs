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
    use behave_fire::surface::{self, SurfaceInput};
    use behave_fire::{FuelModels, GridShape, PerCell};
    let fuels = FuelModels::default();
    let inputs: Vec<_> = [1, 10, 101, 0, 261]
        .into_iter()
        .map(|model| {
            let mut input = SurfaceInput::new(model);
            input.wind_speed = 7.0;
            input.slope = 15.0;
            input
        })
        .collect();
    for threads in [0, 1, 2, 4] {
        let batch = SurfaceBatch::run(
            GridShape::new(vec![5]).unwrap(),
            PerCell::Cells(&inputs),
            &fuels,
            threads,
        )
        .unwrap();
        for (name, values) in batch.into_named_vectors() {
            for (i, value) in values.iter().enumerate() {
                let scalar = surface::run(&inputs[i], &fuels).unwrap();
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
    use behave_fire::{
        crown::{self, CrownInput},
        surface::{self, SurfaceInput},
    };
    use behave_fire::{FuelModels, GridShape, PerCell};
    let fuels = FuelModels::default();
    let surface = surface::run(&SurfaceInput::new(10), &fuels).unwrap();
    let mut input = CrownInput::new(surface);
    input.canopy_height = 30.0;
    input.canopy_base_height = 4.0;
    input.canopy_bulk_density = 0.01;
    input.wind_speed = 15.0;
    for threads in [0, 1, 2, 4] {
        let batch = CrownBatch::run(
            GridShape::new(vec![2, 3]).unwrap(),
            PerCell::Scalar(input.clone()),
            &fuels,
            threads,
        )
        .unwrap();
        let expected = crown::run(&input, &fuels).unwrap();
        assert!(batch
            .fire_type
            .iter()
            .all(|value| *value == expected.fire_type));
        for (name, values) in batch.into_named_vectors() {
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
    let batch = SurfaceBatch::run(
        GridShape::new(vec![3]).unwrap(),
        PerCell::Scalar(SurfaceInput::new(1)),
        &fuels,
        1,
    )
    .unwrap();
    let pointer = batch.spread_rate.as_ptr();
    let columns = batch.into_named_vectors();
    assert_eq!(
        columns
            .iter()
            .find(|(name, _)| *name == "spread_rate")
            .unwrap()
            .1
            .as_ptr(),
        pointer
    );
    let empty = SurfaceBatch::run(
        GridShape::new(vec![0]).unwrap(),
        PerCell::Cells(&[]),
        &fuels,
        4,
    )
    .unwrap();
    assert!(empty
        .into_named_vectors()
        .iter()
        .all(|(_, values)| values.is_empty()));
    assert!(SurfaceBatch::run(
        GridShape::new(vec![2]).unwrap(),
        PerCell::Cells(&[]),
        &fuels,
        4
    )
    .is_err());
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
    use behave_fire::{batch::SurfaceBatch, BehaveError, FuelModels, GridShape, PerCell};
    let result = SurfaceBatch::run(
        GridShape::new(vec![2]).unwrap(),
        PerCell::Cells(&[]),
        &FuelModels::default(),
        4,
    );
    assert!(matches!(
        result,
        Err(ParallelError::Work(BehaveError::LengthMismatch {
            expected: 2,
            actual: 0,
            ..
        }))
    ));
}
