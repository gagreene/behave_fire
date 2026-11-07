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
