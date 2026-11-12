//! Actual surviving primitive producers, with isolated allocation observations.
#[path = "../../../rust/crates/behave-py/src/allocation_probe.rs"]
mod allocation_probe;
#[path = "../../../rust/crates/behave-py/src/component_columns.rs"]
mod component_columns;
#[path = "../../../rust/crates/behave-py/src/component_schema.rs"]
mod component_schema;
#[path = "../../../rust/crates/behave-py/src/component_units.rs"]
mod component_units;
#[path = "../../../rust/crates/behave-py/src/prepared/keys.rs"]
mod keys;
#[path = "../../../rust/crates/behave-py/src/model_observer.rs"]
mod model_observer;
#[path = "../../../rust/crates/behave-py/src/operation.rs"]
mod operation;
#[path = "../../../rust/crates/behave-py/src/prepared/primitive.rs"]
mod primitive;
#[path = "../../../rust/crates/behave-py/src/prepared/selectors.rs"]
mod selectors;

mod prepared {
    pub(crate) use super::keys::Field;
    pub(crate) mod primitive {
        pub(crate) use super::super::primitive::*;
    }
    pub(crate) mod selectors {
        pub(crate) use super::super::selectors::*;
    }
}

#[cfg(test)]
mod primitive_component_tests {
    use super::{
        allocation_probe as probe,
        keys::Field,
        primitive::{Arguments, Prepared},
    };
    use behave_fire::GridShape;

    #[test]
    fn typed_components_keep_controls_exact_and_access_without_allocations() {
        let mut a = Arguments::default();
        a.physical
            .insert("spread_rate".into(), vec![1., f64::NAN, 3.]);
        a.codes.insert("fuel_model_grid".into(), vec![124]);
        a.flags.insert("is_crown".into(), vec![true]);
        let p = Prepared::new(&a, &GridShape::new(vec![3]).unwrap()).unwrap();
        p.require(&[Field::SpreadRate, Field::FuelModelGrid, Field::IsCrown])
            .unwrap();
        probe::start();
        for row in 0..3 {
            assert_eq!(p.code(Field::FuelModelGrid, row).unwrap(), 124);
            assert!(p.flag(Field::IsCrown, row).unwrap());
            assert_eq!(
                p.physical(Field::SpreadRate, row).unwrap().is_nan(),
                row == 1
            );
        }
        let counts = probe::stop();
        assert_eq!(counts[0].0, 0);
        assert!(p.physical(Field::FuelModelGrid, 0).is_err());
        assert!(p.code(Field::FuelModelGrid, 3).is_err());
        assert!(p.require(&[Field::FlameLengthFt]).is_err());
    }

    #[test]
    fn typed_component_validation_rejects_wrong_roles_lengths_and_retired_metadata() {
        let shape = GridShape::new(vec![3]).unwrap();
        for (name, values) in [
            ("units", vec![0.]),
            ("_single.spread_rate", vec![1.]),
            ("spread_rate", vec![1., 2.]),
        ] {
            let mut a = Arguments::default();
            a.physical.insert(name.into(), values);
            assert!(Prepared::new(&a, &shape).is_err());
        }
        let mut a = Arguments::default();
        a.codes.insert("fuel_model_grid".into(), vec![i64::MAX]);
        assert!(Prepared::new(&a, &shape).is_err());
        let mut a = Arguments::default();
        a.flags.insert("spread_rate".into(), vec![true]);
        assert!(Prepared::new(&a, &shape).is_err());
        let mut a = Arguments::default();
        a.physical.insert("spread_rate".into(), vec![]);
        let p = Prepared::new(&a, &GridShape::new(vec![0]).unwrap()).unwrap();
        p.require(&[Field::SpreadRate]).unwrap();
        assert!(p.physical(Field::SpreadRate, 0).is_err());
    }
}

#[cfg(test)]
mod component_schema_tests {
    use super::{component_schema::schema, operation::Operation};
    use behave_fire::engine::FieldKind;

    #[test]
    fn every_operation_has_static_unique_typed_columns_without_prototype_evaluation() {
        for &operation in Operation::ALL {
            let fields = schema(operation);
            assert!(!fields.is_empty());
            for (i, field) in fields.iter().enumerate() {
                assert!(!field.name.starts_with('_'));
                assert!(!fields[..i].iter().any(|prior| prior.name == field.name));
            }
        }
        let classify = schema(Operation::CrownClassifyCrownFireType);
        assert_eq!(classify.len(), 4);
        assert_eq!(classify[0].kind, FieldKind::Code);
        assert!(classify[1..].iter().all(|f| f.kind == FieldKind::Flag));
        let crown = schema(Operation::CrownCalculateCrownSurfaceFire);
        assert_eq!(crown.len(), 72);
        assert_eq!(
            crown.iter().filter(|f| f.kind == FieldKind::Flag).count(),
            2
        );
        assert_eq!(crown[0].name, "p_crown.hoc_dead");
        assert_eq!(crown.last().unwrap().name, "crown_length_to_width_ratio");
    }
}

#[cfg(test)]
mod primitive_output_tests {
    use super::{
        allocation_probe as probe, component_columns::Columns, component_schema::schema,
        operation::Operation,
    };
    use behave_fire::output::ColumnValues;

    #[test]
    fn final_typed_component_buffers_use_disjoint_workers_without_cell_allocations() {
        let mut output = Columns::new(
            vec![
                crate::component_schema::Column {
                    name: "code".into(),
                    kind: behave_fire::engine::FieldKind::Code,
                },
                crate::component_schema::Column {
                    name: "first".into(),
                    kind: behave_fire::engine::FieldKind::Physical,
                },
                crate::component_schema::Column {
                    name: "second".into(),
                    kind: behave_fire::engine::FieldKind::Physical,
                },
                crate::component_schema::Column {
                    name: "third".into(),
                    kind: behave_fire::engine::FieldKind::Physical,
                },
            ],
            1031,
        )
        .unwrap();
        probe::start();
        output
            .fill(4, |index, row| {
                row.i32("code", index as i32);
                row.f64("first", index as f64);
                row.f64("second", 2. * index as f64);
                row.f64("third", 3. * index as f64);
                Ok(())
            })
            .unwrap();
        let counts = probe::stop();
        for phase in [probe::KERNEL, probe::ENCODER] {
            assert_eq!((counts[phase].0, counts[phase].1), (0, 0));
            assert_eq!(counts[phase].3, 4);
        }
        assert_eq!(counts[probe::KERNEL].2, 1031);
        assert_eq!(counts[probe::ENCODER].2, 4 * 1031);
        for (index, (_, values)) in output.columns.iter().enumerate() {
            match values {
                ColumnValues::Int32(v) => assert_eq!(v[1030], 1030),
                ColumnValues::Float64(v) => assert_eq!(v[1030], index as f64 * 1030.),
                _ => panic!("unexpected output kind"),
            }
        }
    }

    #[test]
    fn empty_components_never_evaluate_a_prototype_and_fields_start_unavailable() {
        let mut empty = Columns::new(schema(Operation::CrownClassifyCrownFireType), 0).unwrap();
        empty
            .fill(4, |_, _| panic!("empty scientific callback"))
            .unwrap();
        let mut output = Columns::new(schema(Operation::CrownClassifyCrownFireType), 3).unwrap();
        assert!(matches!(&output.columns[0].1, ColumnValues::Int32(v) if v == &[-1, -1, -1]));
        assert!(matches!(&output.columns[1].1, ColumnValues::Bool(v) if v == &[false; 3]));
        assert!(output.fill(1, |_, _| Ok(())).is_err());
    }

    #[test]
    fn compound_crown_scalar_records_follow_the_same_static_typed_layout() {
        use behave_fire::engine::{CrownCell, PreparedFuels, SurfaceOptions, SurfaceTweaks};
        let fuels = PreparedFuels::from_registry(&behave_fire::FuelModels::default());
        let mut output =
            Columns::new(schema(Operation::CrownCalculateCrownSurfaceFire), 1).unwrap();
        output
            .fill(1, |_, row| {
                let result = behave_fire::crown::calculate_crown_surface_fire(
                    &CrownCell::default(),
                    fuels.for_grid(10),
                    &SurfaceOptions::default(),
                    SurfaceTweaks::default(),
                );
                row.with_prefix("p_crown", |row| {
                    for (name, value) in result.p_crown.named_values() {
                        row.f64(name, value);
                    }
                    row.boolean("is_defined", result.p_crown.is_defined);
                    for (name, values) in result.p_crown.named_arrays() {
                        for (index, value) in values.into_iter().enumerate() {
                            row.indexed_f64(name, index, value);
                        }
                    }
                });
                row.with_prefix("ib_crown", |row| {
                    for (name, value) in result.ib_crown.named_values() {
                        row.f64(name, value);
                    }
                    row.boolean("is_defined", result.ib_crown.is_defined);
                });
                row.with_prefix("crown_surface", |row| {
                    for (name, value) in result.crown_surface.named_values() {
                        row.f64(name, value);
                    }
                });
                row.f64("ri_crown", result.ri_crown);
                row.f64("crown_fire_spread_rate", result.crown_fire_spread_rate);
                row.f64(
                    "crown_length_to_width_ratio",
                    result.crown_length_to_width_ratio,
                );
                Ok(())
            })
            .unwrap();
        assert_eq!(output.columns.len(), 72);
        assert!(matches!(&output.columns[4].1, ColumnValues::Bool(v) if v == &[true]));
        let last = &output.columns.last().unwrap().1;
        assert!(matches!(last, ColumnValues::Float64(v) if v[0].is_finite()));
    }
}

#[cfg(test)]
mod unit_producer_tests {
    use super::{
        allocation_probe as probe,
        component_columns::Columns,
        component_schema::schema,
        component_units,
        operation::Operation,
        primitive::{Arguments, Prepared},
    };
    use behave_fire::{output::ColumnValues, GridShape};

    #[test]
    fn varying_integral_units_are_predecoded_before_actual_worker_loops() {
        let cells = 1031;
        let mut input = Arguments::default();
        input.physical.insert("value".into(), vec![2.; cells]);
        input
            .codes
            .insert("units".into(), (0..cells).map(|i| (i % 8) as i64).collect());
        let prepared = Prepared::new(&input, &GridShape::new(vec![cells]).unwrap()).unwrap();
        let operation = Operation::UnitsLengthToBase;
        let mut output = Columns::new(schema(operation), cells).unwrap();
        probe::start();
        component_units::run(operation, &input, &prepared, &mut output, 4)
            .unwrap()
            .unwrap();
        let counts = probe::stop();
        for phase in [probe::KERNEL, probe::ENCODER] {
            assert_eq!((counts[phase].0, counts[phase].1), (0, 0));
            assert_eq!(counts[phase].2, cells);
            assert_eq!(counts[phase].3, 4);
        }
        if let ColumnValues::Float64(values) = &output.columns[0].1 {
            for (index, &actual) in values.iter().enumerate() {
                let expected = behave_fire::units::LengthUnit::try_from((index % 8) as i32)
                    .unwrap()
                    .to_base(2.);
                assert_eq!(actual, expected);
            }
        } else {
            panic!("unit result must be physical f64");
        }
    }
}

#[cfg(test)]
mod configured_scope_tests {
    use super::model_observer::Observer as Probe;
    use behave_fire::engine::{
        run_model_observed, CrownPlan, ModelBatch, ModelRun, OutputRequest, PreparedFuels,
    };
    use behave_fire::raster::{CrownColumns, CrownRaster, CrownTweakColumns};
    use behave_fire::{FuelModels, GridShape, PerCell};
    #[test]
    fn actual_model_kernel_and_encoder_have_zero_absolute_requests() {
        let cells = 10000;
        let fuels = PreparedFuels::from_registry(&FuelModels::default());
        let heat = (0..cells)
            .map(|i| if i % 7 == 0 { f64::NAN } else { 1000. })
            .collect::<Vec<_>>();
        let validity = (0..cells).map(|i| i % 17 != 0).collect::<Vec<_>>();
        let eligible = validity.iter().filter(|&&v| v).count();
        let raster = CrownRaster::new(
            GridShape::new(vec![cells]).unwrap(),
            CrownColumns {
                surface_ros: PerCell::Scalar(2.),
                surface_fli: PerCell::Scalar(200.),
                surface_hpua: PerCell::Cells(&heat),
                canopy_base_height: PerCell::Scalar(4.),
                canopy_height: PerCell::Scalar(30.),
                canopy_bulk_density: PerCell::Scalar(0.01),
                validity: Some(PerCell::Cells(&validity)),
                ..CrownColumns::default()
            },
        )
        .unwrap();
        let run = ModelRun::Crown {
            raster,
            plan: CrownPlan::default(),
            fuels: &fuels,
            tweaks: CrownTweakColumns::default(),
        };
        let request = OutputRequest::crown();
        super::allocation_probe::start();
        let output = run_model_observed(&run, &request, 4, &Probe).unwrap();
        let counts = super::allocation_probe::stop();
        for phase in [
            super::allocation_probe::KERNEL,
            super::allocation_probe::ENCODER,
        ] {
            assert_eq!(counts[phase].0, 0, "absolute request count in scope{phase}");
            assert_eq!(counts[phase].1, 0);
        }
        assert_eq!(counts[super::allocation_probe::KERNEL].2, eligible);
        assert_eq!(counts[super::allocation_probe::ENCODER].2, cells);
        assert_eq!(counts[super::allocation_probe::KERNEL].3, 4);
        if let ModelBatch::Crown(output) = output {
            assert_eq!(output.fire_type[0], -1);
            assert_eq!(output.final_spread_rate.len(), cells);
        } else {
            panic!("wrong result")
        }
    }
}

#[cfg(test)]
mod staged_scope_tests {
    use super::{allocation_probe as probe, model_observer::Observer};
    use behave_fire::engine::{
        CrownPlan, Field, GridSession, ModelRun, OutputRequest, PreparedFuels,
    };
    use behave_fire::raster::{CrownColumns, CrownRaster, CrownTweakColumns};
    use behave_fire::{FuelModels, GridShape, PerCell};

    #[test]
    fn staged_scientific_and_encoder_scopes_allocate_nothing_and_use_actual_workers() {
        let fuels = PreparedFuels::from_registry(&FuelModels::default());
        for workers in [1, 4] {
            let valid = (0..1031).map(|i| i % 17 != 0).collect::<Vec<_>>();
            let run = ModelRun::Crown {
                raster: CrownRaster::new(
                    GridShape::new(vec![1031]).unwrap(),
                    CrownColumns {
                        canopy_base_height: PerCell::Scalar(3.),
                        canopy_height: PerCell::Scalar(30.),
                        canopy_bulk_density: PerCell::Scalar(0.02),
                        surface_ros: PerCell::Scalar(5.),
                        surface_fli: PerCell::Scalar(1000.),
                        surface_hpua: PerCell::Scalar(1000.),
                        wind_speed: PerCell::Scalar(88.),
                        validity: Some(PerCell::Cells(&valid)),
                        ..CrownColumns::default()
                    },
                )
                .unwrap(),
                plan: CrownPlan::default(),
                fuels: &fuels,
                tweaks: CrownTweakColumns::default(),
            };
            let mut s = GridSession::new(&run, OutputRequest::crown(), &[], workers).unwrap();
            s.set_physical(Field::CrownFractionBurned, vec![0.5])
                .unwrap();
            probe::start();
            s.run_remaining_observed(&Observer).unwrap();
            let counts = probe::stop();
            for phase in [probe::KERNEL, probe::ENCODER] {
                assert_eq!((counts[phase].0, counts[phase].1), (0, 0), "{counts:?}");
                assert!(counts[phase].2 > 1031);
                assert!(counts[phase].3 >= workers);
            }
            assert!(s.get(Field::FinalSpreadRate).unwrap().physical().unwrap()[0].is_nan());
            assert!(s.get(Field::FinalSpreadRate).unwrap().physical().unwrap()[1].is_finite());
        }
    }
}
