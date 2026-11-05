use behave_fire::{map_chunks, ParallelError};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn empty_work_does_not_call_the_operation() {
    let calls = AtomicUsize::new(0);
    let result = map_chunks(0, usize::MAX, |_| {
        calls.fetch_add(1, Ordering::SeqCst);
        Ok::<_, ()>(())
    })
    .unwrap();
    assert!(result.is_empty());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

#[test]
fn each_cell_is_processed_once_in_logical_order() {
    for threads in [0, 1, 2, 4, 200] {
        let counts: Vec<_> = (0..31).map(|_| AtomicUsize::new(0)).collect();
        let results = map_chunks(31, threads, |range| {
            Ok::<_, ()>(
                range
                    .inspect(|&i| {
                        counts[i].fetch_add(1, Ordering::SeqCst);
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .unwrap();
        assert_eq!(
            results.into_iter().flatten().collect::<Vec<_>>(),
            (0..31).collect::<Vec<_>>()
        );
        assert!(counts.iter().all(|count| count.load(Ordering::SeqCst) == 1));
    }
}

#[test]
fn earliest_logical_work_error_wins() {
    let result = map_chunks(24, 4, |range| {
        for i in range {
            if i == 2 || i == 17 {
                return Err(i);
            }
        }
        Ok(())
    });
    assert!(matches!(result, Err(ParallelError::Work(2))));
}

#[test]
fn worker_panics_do_not_escape_in_serial_or_parallel_modes() {
    for threads in [1, 4] {
        let result = map_chunks(8, threads, |_| -> Result<(), ()> {
            panic!("test worker panic")
        });
        assert!(matches!(result, Err(ParallelError::WorkerPanic)));
    }
}

#[test]
fn scoped_work_can_borrow_owned_native_inputs() {
    let inputs = [1.0_f64, -0.0, f64::INFINITY, f64::NAN, 1e-30];
    let serial = map_chunks(inputs.len(), 1, |r| {
        Ok::<_, ()>(r.map(|i| inputs[i].to_bits()).collect::<Vec<_>>())
    })
    .unwrap();
    let parallel = map_chunks(inputs.len(), 3, |r| {
        Ok::<_, ()>(r.map(|i| inputs[i].to_bits()).collect::<Vec<_>>())
    })
    .unwrap();
    assert_eq!(
        serial.into_iter().flatten().collect::<Vec<_>>(),
        parallel.into_iter().flatten().collect::<Vec<_>>()
    );
}
