use behave_fire::{effective_worker_count, BehaveError, GridOutput, GridShape, PerCell};

#[test]
fn cells_must_match_the_checked_shape() {
    let shape = GridShape::new(vec![2, 3]).unwrap();
    let values = [1, 2, 3];
    let error = PerCell::Cells(&values)
        .validate(&shape, "wind")
        .unwrap_err();
    assert!(
        matches!(error, BehaveError::LengthMismatch { ref field, expected: 6, actual: 3 } if field == "wind")
    );
    assert_eq!(
        PerCell::Cells(&[1, 2, 3, 4, 5, 6]).validate(&shape, "wind"),
        Ok(())
    );
}

#[test]
fn empty_dimensions_make_zero_cells_even_after_large_dimensions() {
    for dimensions in [vec![0], vec![2, 0, 4], vec![usize::MAX, 2, 0]] {
        let shape = GridShape::new(dimensions).unwrap();
        assert_eq!(shape.len(), 0);
        assert!(shape.is_empty());
        assert_eq!(PerCell::<f64>::Cells(&[]).validate(&shape, "wind"), Ok(()));
    }
}

#[test]
fn errors_support_standard_error_consumers() {
    let shape = GridShape::new(vec![2]).unwrap();
    let error = shape.validate_len("moisture", 1).unwrap_err();
    let standard: &dyn std::error::Error = &error;
    assert!(standard.to_string().contains("moisture"));
    assert!(standard.source().is_none());
}

#[test]
fn generic_results_keep_shape_and_checked_owned_cells() {
    let shape = GridShape::new(vec![2]).unwrap();
    let output = GridOutput::new(shape.clone(), vec![true, false]).unwrap();
    assert_eq!(output.shape(), &shape);
    assert_eq!(output.cells(), &[true, false]);
    assert_eq!(output.into_cells(), vec![true, false]);
    assert!(matches!(
        GridOutput::new(shape, vec![true]),
        Err(BehaveError::LengthMismatch {
            expected: 2,
            actual: 1,
            ..
        })
    ));
}

#[test]
fn grid_shapes_reject_nonzero_dimension_product_overflow() {
    let error = GridShape::new(vec![usize::MAX, 2]).unwrap_err();
    assert!(
        matches!(error, BehaveError::ShapeOverflow { ref dimensions } if dimensions == &[usize::MAX, 2])
    );
}

#[test]
fn grid_shapes_retain_dimensions_and_count_logical_cells() {
    let shape = GridShape::new(vec![2, 3, 4]).unwrap();
    assert_eq!(shape.dimensions(), &[2, 3, 4]);
    assert_eq!(shape.len(), 24);
    assert!(!shape.is_empty());
}

#[test]
fn scalar_fields_broadcast_and_slice_fields_preserve_indexing() {
    let shape = GridShape::new(vec![2, 3]).unwrap();
    let scalar = PerCell::Scalar(7);
    assert_eq!(scalar.validate(&shape, "wind"), Ok(()));
    assert_eq!(scalar.get(5), Some(&7));
    let values = [4, 8];
    let cells = PerCell::Cells(&values);
    assert_eq!(cells.get(1), Some(&8));
    assert_eq!(cells.get(2), None);
}

#[test]
fn scalar_shapes_have_one_logical_cell() {
    let shape = GridShape::new(vec![]).unwrap();
    assert!(shape.dimensions().is_empty());
    assert_eq!(shape.len(), 1);
    assert!(!shape.is_empty());
    assert_eq!(shape.validate_len("scalar", 1), Ok(()));
}

#[test]
fn thread_counts_are_capped_by_work_without_rejecting_large_requests() {
    assert_eq!(effective_worker_count(1, 6), 1);
    assert_eq!(effective_worker_count(4, 6), 4);
    assert_eq!(effective_worker_count(usize::MAX, 6), 6);
    assert_eq!(effective_worker_count(0, 0), 0);
    assert_eq!(effective_worker_count(4, 0), 0);
    let automatic = effective_worker_count(0, 6);
    assert!((1..=6).contains(&automatic));
}
