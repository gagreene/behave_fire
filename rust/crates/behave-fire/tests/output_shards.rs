//! Final output buffers are written through disjoint, borrowed slices.
use behave_fire::batch::{CrownBatch, SurfaceBatch};
use behave_fire::engine::{Field, OutputRequest};
use behave_fire::output::{fill_chunks, ColumnValues};
use behave_fire::{BehaveError, GridShape, ParallelError};
use std::sync::atomic::{AtomicBool, Ordering};

#[test]
fn empty_work_never_reads_a_prototype_or_calls_a_worker() {
    let mut physical = Vec::<f64>::new();
    let mut codes = Vec::<i32>::new();
    fill_chunks(
        0,
        4,
        &mut [&mut physical],
        &mut [&mut codes],
        &mut [],
        |_, _| {
            panic!("empty output must not execute a prototype");
            #[allow(unreachable_code)]
            Ok::<_, BehaveError>(())
        },
    )
    .unwrap();
}

#[test]
fn invalid_lengths_fail_before_any_work_or_output_write() {
    let called = AtomicBool::new(false);
    let mut data = vec![f64::NAN; 2];
    let error = fill_chunks(3, 4, &mut [&mut data], &mut [], &mut [], |_, _| {
        called.store(true, Ordering::SeqCst);
        Ok::<_, BehaveError>(())
    })
    .unwrap_err();
    assert!(matches!(
        error,
        ParallelError::InvalidInput(BehaveError::LengthMismatch {
            expected: 3,
            actual: 2,
            ..
        })
    ));
    assert!(!called.load(Ordering::SeqCst));
    assert!(data.iter().all(|v| v.is_nan()));
}

#[test]
fn output_shards_write_final_buffers_without_reallocation() {
    for threads in [1, 4, 0] {
        let mut data = vec![f64::NAN; 19];
        let mut code = vec![-1; 19];
        let mut flag = vec![false; 19];
        let pointers = (data.as_ptr(), code.as_ptr(), flag.as_ptr());
        fill_chunks(
            19,
            threads,
            &mut [&mut data],
            &mut [&mut code],
            &mut [&mut flag],
            |range, output| {
                for (local, index) in range.enumerate() {
                    if index == 7 {
                        continue;
                    }
                    output.physical[0][local] = index as f64 + 0.25;
                    output.codes[0][local] = index as i32 * 3;
                    output.flags[0][local] = true;
                }
                Ok::<_, BehaveError>(())
            },
        )
        .unwrap();
        assert_eq!(pointers, (data.as_ptr(), code.as_ptr(), flag.as_ptr()));
        for index in 0..19 {
            if index == 7 {
                assert!(data[index].is_nan());
                assert_eq!(code[index], -1);
                assert!(!flag[index]);
            } else {
                assert_eq!(data[index], index as f64 + 0.25);
                assert_eq!(code[index], index as i32 * 3);
                assert!(flag[index]);
            }
        }
    }
}

#[test]
fn requested_model_columns_keep_typed_codes_and_empty_selection() {
    let shape = GridShape::new(vec![3]).unwrap();
    let request = OutputRequest::new(&[
        Field::FinalSpreadRate,
        Field::FireType,
        Field::FinalSpreadRate,
    ]);
    let mut batch = CrownBatch::allocate(shape.clone(), &request).unwrap();
    assert_eq!(batch.final_spread_rate.len(), 3);
    assert!(batch.final_spread_rate.iter().all(|v| v.is_nan()));
    assert_eq!(batch.fire_type, [-1; 3]);
    assert!(batch.crown_fire_spread_rate.is_empty());
    batch.fire_type[0] = 3;
    let pointer = batch.fire_type.as_ptr();
    let columns = batch.into_named_columns();
    assert_eq!(columns.len(), 2);
    match &columns[1].1 {
        ColumnValues::Int32(values) => {
            assert_eq!(values.as_ptr(), pointer);
            assert_eq!(values.as_slice(), [3, -1, -1]);
        }
        _ => panic!("classification must never widen through float64"),
    }
    let empty = SurfaceBatch::allocate(shape, &OutputRequest::new(&[])).unwrap();
    assert!(empty.into_named_columns().is_empty());
    assert!(SurfaceBatch::allocate(GridShape::new(vec![0]).unwrap(), &request).is_err());
}

#[test]
fn worker_errors_and_panics_are_joined_in_logical_order() {
    let mut data = vec![f64::NAN; 32];
    let error = fill_chunks(32, 4, &mut [&mut data], &mut [], &mut [], |range, _| {
        if range.start == 0 {
            Err("first logical error")
        } else {
            panic!("later worker panic")
        }
    })
    .unwrap_err();
    assert!(matches!(error, ParallelError::Work("first logical error")));
    let error = fill_chunks(32, 1, &mut [&mut data], &mut [], &mut [], |_, _| {
        panic!("serial callback panic");
        #[allow(unreachable_code)]
        Ok::<_, BehaveError>(())
    })
    .unwrap_err();
    assert!(matches!(error, ParallelError::WorkerPanic));
}
