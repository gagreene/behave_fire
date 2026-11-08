//! Compile the actual boundary data modules without requiring a Python linker.
#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/allocation_probe.rs"]
mod allocation_probe;
#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/fields.rs"]
mod fields;

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/keys.rs"]
mod keys;

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/controls.rs"]
mod controls;

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/units.rs"]
mod units;

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/unit_selectors.rs"]
mod unit_selectors;

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/column_output.rs"]
mod column_output;

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/surface.rs"]
mod surface;

#[cfg(test)]
mod pipeline_tests {
    use super::fields::{Arguments, PreparedArguments};
    use super::surface::PreparedSurface;
    use behave_fire::engine::{PreparedFuels, SurfaceTweaks};
    use behave_fire::{FuelModels, GridShape};
    use std::collections::BTreeMap;

    #[test]
    fn prepared_surface_borrows_owned_columns_and_reuses_numeric_fuels() {
        let mut a = Arguments {
            values: BTreeMap::new(),
            strings: BTreeMap::new(),
            integers: BTreeMap::new(),
        };
        for (key, value) in [
            ("m1h", 0.06),
            ("m10h", 0.07),
            ("m100h", 0.08),
            ("mlh", 0.6),
            ("mlw", 0.9),
            ("wind_direction", 0.0),
            ("aspect", 0.0),
            ("canopy_cover", 0.0),
            ("canopy_height", 0.0),
            ("crown_ratio", 0.0),
            ("wind_speed_units", 0.0),
            ("slope_units", 0.0),
        ] {
            a.values.insert(key.into(), vec![value]);
        }
        a.values.insert("fuel_model_grid".into(), vec![10.0, 124.0]);
        a.values.insert("wind_speed".into(), vec![88.0, 176.0]);
        a.values.insert("slope".into(), vec![0.0, 10.0]);
        for (key, value) in [
            ("wind_orientation_mode", "RelativeToUpslope"),
            ("wind_height_mode", "TwentyFoot"),
            ("waf_method", "CrownRatio"),
        ] {
            a.strings.insert(key.into(), value.into());
        }
        let shape = GridShape::new(vec![2]).unwrap();
        let fields = PreparedArguments::new(&a, &shape).unwrap();
        let prepared = PreparedSurface::new(&fields, shape).unwrap();
        let raster = prepared.raster().unwrap();
        let fuels = PreparedFuels::from_registry(&FuelModels::default());
        super::allocation_probe::start();
        let first = raster
            .evaluate_cell(0, prepared.plan(), &fuels, SurfaceTweaks::default())
            .unwrap();
        let second = raster
            .evaluate_cell(1, prepared.plan(), &fuels, SurfaceTweaks::default())
            .unwrap();
        let counts = super::allocation_probe::stop();
        assert_eq!(
            counts[0].0, 0,
            "prepared execution must not allocate per cell"
        );
        assert!(first.spread_rate.is_finite());
        assert!(second.spread_rate.is_finite());
        assert_ne!(first.spread_rate, second.spread_rate);
        assert!(
            matches!(&prepared.wind, std::borrow::Cow::Borrowed(_)),
            "base-unit arrays should remain borrowed owned snapshots"
        );
    }
}

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/crown.rs"]
mod prepared_crown;

#[cfg(test)]
mod crown_pipeline_tests {
    use super::fields::{Arguments, PreparedArguments};
    use super::prepared_crown::PreparedCrown;
    use behave_fire::engine::{CrownTweaks, PreparedFuels};
    use behave_fire::{FuelModels, GridShape};
    use std::collections::BTreeMap;

    #[test]
    fn prepared_crown_preserves_varying_units_and_independent_surface_columns() {
        let mut a = Arguments {
            values: BTreeMap::new(),
            strings: BTreeMap::new(),
            integers: BTreeMap::new(),
        };
        for (name, values) in [
            ("wind_speed", vec![88.0]),
            ("wind_speed_units", vec![0.0, 1.0]),
            ("slope", vec![0.0]),
            ("slope_units", vec![0.0]),
            ("surface_results.spread_rate", vec![2.0, 3.0]),
            ("surface_results.fireline_intensity", vec![100.0]),
            ("surface_results.heat_per_unit_area", vec![200.0]),
        ] {
            a.values.insert(name.into(), values);
        }
        let shape = GridShape::new(vec![2]).unwrap();
        let fields = PreparedArguments::new(&a, &shape).unwrap();
        let prepared = PreparedCrown::new(&fields, shape).unwrap();
        let raster = prepared.raster().unwrap();
        assert_eq!(raster.input_cell(0).unwrap().wind_speed, 88.0);
        assert_eq!(
            raster.input_cell(1).unwrap().wind_speed,
            behave_fire::units::SpeedUnit::try_from(1)
                .unwrap()
                .to_base(88.0)
        );
        let fuels = PreparedFuels::from_registry(&FuelModels::default());
        super::allocation_probe::start();
        let first = raster
            .evaluate_cell(0, prepared.plan(), &fuels, CrownTweaks::default())
            .unwrap();
        let second = raster
            .evaluate_cell(1, prepared.plan(), &fuels, CrownTweaks::default())
            .unwrap();
        let counts = super::allocation_probe::stop();
        assert_eq!(counts[0].0, 0);
        for (index, actual) in [first, second].into_iter().enumerate() {
            let mut supplied = behave_fire::crown::SurfaceFireInputs::<f64>::new();
            supplied.surface_ros = [2.0, 3.0][index];
            supplied.surface_fli = 100.0;
            supplied.surface_hpua = 200.0;
            let input = behave_fire::engine::CrownCell {
                wind_speed: behave_fire::units::SpeedUnit::try_from(index as i32)
                    .unwrap()
                    .to_base(88.0),
                ..behave_fire::engine::CrownCell::default()
            };
            let expected = behave_fire::engine::CrownPlan::default().run_cell(
                &input,
                &supplied,
                fuels.for_grid(10),
                CrownTweaks::default(),
            );
            assert_eq!(actual.named_values(), expected.named_values());
            assert_eq!(actual.fire_type, expected.fire_type);
        }
    }
}

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/conversions.rs"]
mod conversions;

#[cfg(test)]
mod conversion_pipeline_tests {
    use super::{conversions, fields::Arguments};
    use std::collections::BTreeMap;

    #[test]
    fn fused_conversions_preserve_selector_broadcast_and_derived_crown_ratio() {
        let mut a = Arguments {
            values: BTreeMap::new(),
            strings: BTreeMap::new(),
            integers: BTreeMap::new(),
        };
        for (name, values) in [
            ("wind_speed", vec![2.0, 2.0]),
            ("_convert.wind_speed", vec![0.0, 1.0]),
            ("canopy_height", vec![10.0]),
            ("_convert.canopy_height", vec![4.0]),
            ("canopy_base_height", vec![2.0]),
            ("_convert.canopy_base_height", vec![4.0]),
            ("crown_ratio", vec![1.0]),
        ] {
            a.values.insert(name.into(), values);
        }
        for (name, value) in [
            ("_convert.wind_speed", "Speed"),
            ("_convert.canopy_height", "Length"),
            ("_convert.canopy_base_height", "Length"),
            ("_derive_crown_ratio", "true"),
        ] {
            a.strings.insert(name.into(), value.into());
        }
        conversions::apply(&mut a).unwrap();
        assert_eq!(
            a.values["wind_speed"],
            vec![
                2.0,
                behave_fire::units::SpeedUnit::try_from(1)
                    .unwrap()
                    .to_base(2.0)
            ]
        );
        let height = behave_fire::units::LengthUnit::try_from(4)
            .unwrap()
            .to_base(10.0);
        let base = behave_fire::units::LengthUnit::try_from(4)
            .unwrap()
            .to_base(2.0);
        assert_eq!(
            a.values["crown_ratio"],
            vec![behave_fire::canopy_crown_ratio(height, base)]
        );
        assert_eq!(a.values["canopy_height"], vec![height]);
    }
}

#[cfg(all(test, feature = "allocation-probe"))]
mod direct_output_tests {
    use behave_fire::surface::SurfaceResult;
    use behave_fire::{batch::SurfaceBatch, output::fill_chunks, GridShape};

    #[test]
    fn borrowed_encoder_loop_has_zero_requests() {
        let mut values = vec![f64::NAN; 10000];
        super::allocation_probe::start();
        fill_chunks(
            10000,
            4,
            &mut [&mut values],
            &mut [],
            &mut [],
            |range, shard| {
                let _scope =
                    super::allocation_probe::Scope::enter(super::allocation_probe::ENCODER);
                for (local, index) in range.enumerate() {
                    shard.physical[0][local] = index as f64;
                }
                Ok::<_, ()>(())
            },
        )
        .unwrap();
        let counts = super::allocation_probe::stop();
        assert_eq!(counts[super::allocation_probe::ENCODER].0, 0);
        assert_eq!(counts[super::allocation_probe::ENCODER].1, 0);
        assert_eq!(values[9999], 9999.);
    }

    #[test]
    fn full_batch_request_count_depends_on_fields_and_workers_not_cells() {
        for workers in [1, 4] {
            let mut requests = Vec::new();
            for cells in [4096, 16384] {
                super::allocation_probe::start();
                let result = SurfaceBatch::run_with(
                    GridShape::new(vec![cells]).unwrap(),
                    workers,
                    |index| {
                        let mut result = SurfaceResult::default();
                        result.spread_rate = index as f64;
                        Ok::<_, ()>(result)
                    },
                )
                .unwrap();
                let counts = super::allocation_probe::stop();
                requests.push(counts.iter().map(|scope| scope.0).sum::<usize>());
                assert_eq!(result.spread_rate[cells - 1], (cells - 1) as f64);
                assert_eq!(result.statistics().appended_bytes, 0);
            }
            assert_eq!(
                requests[0], requests[1],
                "batch request counts must not grow per cell, workers={workers}"
            );
        }
    }
}

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/numeric/contract.rs"]
mod contract;
#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/numeric/shape.rs"]
mod numeric_shape;
#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/numeric/snapshot.rs"]
mod snapshot;

#[cfg(test)]
mod numeric_contract_tests {
    use super::{
        contract::InputRole,
        numeric_shape::{plan_groups, OutputDependency},
        snapshot::{ControlSnapshot, ControlValues, PhysicalSnapshot},
    };
    use behave_fire::GridShape;

    #[test]
    fn owned_columns_separate_physics_from_exact_controls() {
        let physical =
            PhysicalSnapshot::new(GridShape::new(vec![2]).unwrap(), vec![-0.0, f64::NAN]).unwrap();
        assert!(physical.values()[0].is_sign_negative());
        assert!(physical.values()[1].is_nan());
        let controls = ControlSnapshot::new(
            InputRole::Identifier,
            GridShape::new(vec![1]).unwrap(),
            ControlValues::Signed(vec![(1_i64 << 53) + 1]),
        )
        .unwrap();
        assert_eq!(controls.signed().unwrap()[0], (1_i64 << 53) + 1);
        assert!(ControlSnapshot::new(
            InputRole::Physical,
            GridShape::new(vec![1]).unwrap(),
            ControlValues::Signed(vec![1])
        )
        .is_err());
        assert!(PhysicalSnapshot::new(GridShape::new(vec![2]).unwrap(), vec![1.0]).is_err());
    }

    #[test]
    fn dependency_groups_keep_independent_and_empty_output_axes() {
        let shapes = vec![
            GridShape::new(vec![2]).unwrap(),
            GridShape::new(vec![3]).unwrap(),
            GridShape::new(vec![]).unwrap(),
        ];
        let groups = plan_groups(
            &shapes,
            &[
                OutputDependency::new(0, &[0, 2]),
                OutputDependency::new(1, &[1, 2]),
            ],
        )
        .unwrap();
        assert_eq!(groups[0].shape.dimensions(), [2]);
        assert_eq!(groups[1].shape.dimensions(), [3]);
        let shapes = vec![
            GridShape::new(vec![]).unwrap(),
            GridShape::new(vec![0]).unwrap(),
        ];
        let groups = plan_groups(
            &shapes,
            &[
                OutputDependency::new(0, &[0]),
                OutputDependency::new(1, &[0, 1]),
            ],
        )
        .unwrap();
        assert_eq!(groups[0].shape.len(), 1);
        assert_eq!(groups[1].shape.len(), 0);
        assert!(plan_groups(&shapes, &[OutputDependency::new(0, &[3])]).is_err());
        assert!(plan_groups(
            &shapes,
            &[
                OutputDependency::new(0, &[0]),
                OutputDependency::new(0, &[1])
            ]
        )
        .is_err());
    }
}

#[cfg(test)]
mod restricted_group_tests {
    use super::{
        numeric_shape::{plan_groups, OutputDependency},
        snapshot::PhysicalSnapshot,
    };
    use behave_fire::GridShape;
    #[test]
    fn group_evaluators_cannot_read_unrelated_inputs_or_fake_empty_rows() {
        let inputs = vec![
            PhysicalSnapshot::new(GridShape::new(vec![2]).unwrap(), vec![1., f64::NAN]).unwrap(),
            PhysicalSnapshot::new(GridShape::new(vec![3]).unwrap(), vec![1.; 3]).unwrap(),
            PhysicalSnapshot::new(GridShape::new(vec![]).unwrap(), vec![2.]).unwrap(),
            PhysicalSnapshot::new(GridShape::new(vec![]).unwrap(), vec![4.]).unwrap(),
        ];
        let shapes = inputs
            .iter()
            .map(|value| value.shape().clone())
            .collect::<Vec<_>>();
        let groups = plan_groups(
            &shapes,
            &[
                OutputDependency::new(0, &[0, 2]),
                OutputDependency::new(1, &[1, 3]),
            ],
        )
        .unwrap();
        assert!(groups[0].read_physical(&inputs, 1).is_err());
        let numerator = groups[0].read_physical(&inputs, 0).unwrap();
        let denominator = groups[0].read_physical(&inputs, 2).unwrap()[0];
        let transition = numerator
            .iter()
            .map(|value| value / denominator)
            .collect::<Vec<_>>();
        assert_eq!(transition[0], 0.5);
        assert!(transition[1].is_nan());
        let numerator = groups[1].read_physical(&inputs, 1).unwrap();
        let denominator = groups[1].read_physical(&inputs, 3).unwrap()[0];
        assert_eq!(
            numerator
                .iter()
                .map(|value| value / denominator)
                .collect::<Vec<_>>(),
            vec![0.25; 3]
        );
    }
}

#[cfg(all(test, feature = "allocation-probe"))]
mod configured_scope_tests {
    use super::allocation_probe::Observer as Probe;
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
