//! Native raster execution must retain shape, ordering, validation, and ownership.
use behave_fire::crown::SurfaceFireInputs;
use behave_fire::engine::{
    CrownCell, CrownPlan, CrownTweaks, PreparedFuels, SurfaceCell, SurfaceOptions, SurfacePlan,
    SurfaceTweaks,
};
use behave_fire::raster::{
    CrownColumns, CrownRaster, CrownTweakColumns, SurfaceColumns, SurfaceRaster,
    SurfaceTweakColumns,
};
use behave_fire::{BehaveError, FuelModels, GridShape, ParallelError, PerCell};

#[test]
fn crown_validity_empty_shapes_and_bad_columns_keep_their_contracts() {
    let shape = GridShape::new(vec![2, 1]).unwrap();
    for values in [&[1.0][..], &[1.0, 2.0, 3.0][..]] {
        assert!(matches!(
            CrownRaster::new(
                shape.clone(),
                CrownColumns {
                    surface_hpua: PerCell::Cells(values),
                    ..CrownColumns::default()
                }
            ),
            Err(BehaveError::LengthMismatch { expected: 2, .. })
        ));
    }
    let validity = [false, true];
    let raster = CrownRaster::new(
        shape,
        CrownColumns {
            canopy_height: PerCell::Scalar(30.0),
            canopy_bulk_density: PerCell::Scalar(0.01),
            surface_hpua: PerCell::Scalar(900.0),
            validity: Some(PerCell::Cells(&validity)),
            ..CrownColumns::default()
        },
    )
    .unwrap();
    let plan = CrownPlan::default();
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let output = raster
        .run(&plan, &fuels, &CrownTweakColumns::default(), 4)
        .unwrap();
    assert_eq!(output.validity.as_deref(), Some(validity.as_slice()));
    assert_eq!(output.fire_type[0], 0);
    for (_, column) in comparison_columns(output.into_named_columns()) {
        assert_eq!(column[0].to_bits(), 0.0_f64.to_bits());
    }
    let empty = CrownRaster::new(GridShape::new(vec![0, 3]).unwrap(), CrownColumns::default())
        .unwrap()
        .run(&plan, &fuels, &CrownTweakColumns::default(), 4)
        .unwrap();
    assert_eq!(empty.shape().dimensions(), [0, 3]);
    assert!(empty.fire_type.is_empty());
    let scalar = CrownRaster::new(GridShape::new(vec![]).unwrap(), CrownColumns::default())
        .unwrap()
        .run(&plan, &fuels, &CrownTweakColumns::default(), 1)
        .unwrap();
    assert_eq!(scalar.shape().dimensions(), []);
    assert_eq!(scalar.fire_type.len(), 1);
}

#[test]
fn raster_parallel_workers_preserve_every_column_and_validity() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let plan = SurfacePlan::default();
    let rates: Vec<_> = (0..1031).map(|index| index as f64 / 7.0).collect();
    let validity: Vec<_> = (0..1031).map(|index| index % 11 != 0).collect();
    let raster = SurfaceRaster::new(
        GridShape::new(vec![1031]).unwrap(),
        SurfaceColumns {
            validity: Some(PerCell::Cells(&validity)),
            ..SurfaceColumns::default()
        },
    )
    .unwrap();
    let tweaks = SurfaceTweakColumns {
        spread_rate: Some(PerCell::Cells(&rates)),
        ..SurfaceTweakColumns::default()
    };
    let serial = raster.run(&plan, &fuels, &tweaks, 1).unwrap();
    let parallel = raster.run(&plan, &fuels, &tweaks, 4).unwrap();
    assert_eq!(serial.validity, parallel.validity);
    for ((left_name, left), (right_name, right)) in comparison_columns(serial.into_named_columns())
        .into_iter()
        .zip(comparison_columns(parallel.into_named_columns()))
    {
        assert_eq!(left_name, right_name);
        assert_eq!(
            left.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            right.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn surface_columns_match_scalar_cells_in_row_major_shape() {
    let models = [1, 10, 101, 0, 124, 4];
    let wind = [0.0, 8.0, 50.0, 100.0, 150.0, 200.0];
    let columns = SurfaceColumns {
        fuel_model_grid: PerCell::Cells(&models),
        wind_speed: PerCell::Cells(&wind),
        slope: PerCell::Scalar(12.0),
        ..SurfaceColumns::default()
    };
    let raster = SurfaceRaster::new(GridShape::new(vec![2, 3]).unwrap(), columns).unwrap();
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let plan = SurfacePlan::new(SurfaceOptions::default());
    let output = raster
        .run(&plan, &fuels, &SurfaceTweakColumns::default(), 1)
        .unwrap();
    assert_eq!(output.shape().dimensions(), [2, 3]);
    assert!(output.validity.is_none());
    for index in 0..models.len() {
        let cell = SurfaceCell {
            wind_speed: wind[index],
            slope: 12.0,
            ..SurfaceCell::new(models[index])
        };
        let expected = plan.run_cell(
            &cell,
            fuels.for_grid(models[index]),
            SurfaceTweaks::default(),
        );
        assert_eq!(
            output.spread_rate[index].to_bits(),
            expected.spread_rate.to_bits()
        );
        assert_eq!(
            output.fireline_intensity[index].to_bits(),
            expected.fireline_intensity.to_bits()
        );
    }
}

#[test]
fn crown_supplied_fields_broadcast_independently_without_fake_surface_records() {
    let supplied_rate = [2.0, 4.0, 8.0, 16.0];
    let columns = CrownColumns {
        surface_ros: PerCell::Cells(&supplied_rate),
        surface_fli: PerCell::Scalar(60.0),
        surface_hpua: PerCell::Scalar(900.0),
        wind_speed: PerCell::Scalar(440.0),
        canopy_base_height: PerCell::Scalar(6.0),
        canopy_height: PerCell::Scalar(30.0),
        canopy_bulk_density: PerCell::Scalar(0.02),
        ..CrownColumns::default()
    };
    let raster = CrownRaster::new(GridShape::new(vec![2, 1, 2]).unwrap(), columns).unwrap();
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let plan = CrownPlan::new(SurfaceOptions::default());
    let output = raster
        .run(&plan, &fuels, &CrownTweakColumns::default(), 1)
        .unwrap();
    assert_eq!(output.shape().dimensions(), [2, 1, 2]);
    for (index, rate) in supplied_rate.into_iter().enumerate() {
        let cell = CrownCell {
            wind_speed: 440.0,
            canopy_base_height: 6.0,
            canopy_height: 30.0,
            canopy_bulk_density: 0.02,
            ..CrownCell::default()
        };
        let mut surface = SurfaceFireInputs::<f64>::new();
        surface.surface_ros = rate;
        surface.surface_fli = 60.0;
        surface.surface_hpua = 900.0;
        let expected = plan.run_cell(&cell, &surface, fuels.for_grid(10), CrownTweaks::default());
        assert_eq!(
            output.final_spread_rate[index].to_bits(),
            expected.final_spread_rate.to_bits()
        );
        assert_eq!(output.fire_type[index], expected.fire_type);
    }
}

#[test]
fn scalar_empty_and_uniform_slice_inputs_keep_checked_shapes() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let plan = SurfacePlan::new(SurfaceOptions::default());
    let scalar = SurfaceRaster::new(GridShape::new(vec![]).unwrap(), SurfaceColumns::default())
        .unwrap()
        .run(&plan, &fuels, &SurfaceTweakColumns::default(), 1)
        .unwrap();
    assert_eq!(scalar.shape().dimensions(), []);
    assert_eq!(scalar.spread_rate.len(), 1);
    let empty = SurfaceRaster::new(
        GridShape::new(vec![0, 7]).unwrap(),
        SurfaceColumns::default(),
    )
    .unwrap()
    .run(&plan, &fuels, &SurfaceTweakColumns::default(), 4)
    .unwrap();
    assert_eq!(empty.shape().dimensions(), [0, 7]);
    assert!(empty.spread_rate.is_empty());
    let identical = [8.0; 6];
    let first = SurfaceRaster::new(
        GridShape::new(vec![6]).unwrap(),
        SurfaceColumns {
            wind_speed: PerCell::Scalar(8.0),
            ..SurfaceColumns::default()
        },
    )
    .unwrap();
    let second = SurfaceRaster::new(
        GridShape::new(vec![6]).unwrap(),
        SurfaceColumns {
            wind_speed: PerCell::Cells(&identical),
            ..SurfaceColumns::default()
        },
    )
    .unwrap();
    assert_eq!(
        first
            .run(&plan, &fuels, &SurfaceTweakColumns::default(), 1)
            .unwrap()
            .spread_rate,
        second
            .run(&plan, &fuels, &SurfaceTweakColumns::default(), 1)
            .unwrap()
            .spread_rate
    );
}

#[test]
fn input_and_override_lengths_fail_before_output_allocation_or_workers() {
    for values in [&[1.0][..], &[1.0, 2.0, 3.0][..]] {
        let result = SurfaceRaster::new(
            GridShape::new(vec![2]).unwrap(),
            SurfaceColumns {
                wind_speed: PerCell::Cells(values),
                ..SurfaceColumns::default()
            },
        );
        assert!(matches!(
            result,
            Err(BehaveError::LengthMismatch { expected: 2, .. })
        ));
    }
    assert!(GridShape::new(vec![usize::MAX, 2]).is_err());
    let huge = SurfaceRaster::new(
        GridShape::new(vec![usize::MAX]).unwrap(),
        SurfaceColumns::default(),
    )
    .unwrap();
    let tweaks = SurfaceTweakColumns {
        spread_rate: Some(PerCell::Cells(&[])),
        ..SurfaceTweakColumns::default()
    };
    let result = huge.run(
        &SurfacePlan::new(SurfaceOptions::default()),
        &PreparedFuels::from_registry(&FuelModels::default()),
        &tweaks,
        4,
    );
    assert!(matches!(
        result,
        Err(ParallelError::Work(BehaveError::LengthMismatch {
            actual: 0,
            ..
        }))
    ));
}

#[test]
fn native_validity_is_not_a_numpy_mask() {
    let validity = [true, false, true];
    let models = [10, 10, 0];
    let columns = SurfaceColumns {
        fuel_model_grid: PerCell::Cells(&models),
        validity: Some(PerCell::Cells(&validity)),
        ..SurfaceColumns::default()
    };
    let raster = SurfaceRaster::new(GridShape::new(vec![3]).unwrap(), columns).unwrap();
    let output = raster
        .run(
            &SurfacePlan::new(SurfaceOptions::default()),
            &PreparedFuels::from_registry(&FuelModels::default()),
            &SurfaceTweakColumns {
                spread_rate: Some(PerCell::Scalar(99.0)),
                ..SurfaceTweakColumns::default()
            },
            1,
        )
        .unwrap();
    assert_eq!(output.validity.as_deref(), Some(validity.as_slice()));
    assert_eq!(output.spread_rate, [99.0, 0.0, 0.0]);
    assert_eq!(output.fire_length_to_width_ratio[1], 0.0);
    assert_eq!(output.fire_length_to_width_ratio[2], 1.0);
}

#[test]
fn irregular_tiles_retain_order_and_reject_invalid_ranges() {
    let wind = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0];
    let raster = SurfaceRaster::new(
        GridShape::new(vec![2, 3]).unwrap(),
        SurfaceColumns {
            wind_speed: PerCell::Cells(&wind),
            ..SurfaceColumns::default()
        },
    )
    .unwrap();
    let plan = SurfacePlan::new(SurfaceOptions::default());
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let tweaks = SurfaceTweakColumns::default();
    let all = raster.run(&plan, &fuels, &tweaks, 1).unwrap();
    let mut combined = Vec::new();
    for range in [0..0, 0..1, 1..4, 4..6, 6..6] {
        let expected = range.len();
        let tile = raster.run_tile(range, &plan, &fuels, &tweaks, 1).unwrap();
        assert_eq!(tile.shape().dimensions(), [expected]);
        combined.extend(tile.spread_rate);
    }
    assert_eq!(combined, all.spread_rate);
    for (start, end) in [(7, 7), (2, 1), (0, 7)] {
        assert!(matches!(
            raster.run_tile(start..end, &plan, &fuels, &tweaks, 1),
            Err(ParallelError::Work(BehaveError::InvalidRange { .. }))
        ));
    }
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
