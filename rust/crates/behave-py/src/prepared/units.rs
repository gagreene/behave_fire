//! Unit selectors and operations are resolved before direct numeric output loops.
use super::{fields::PreparedArguments, keys::Field, unit_selectors::UnitAccess};
use behave_fire::units::*;
use std::ops::Range;

pub(crate) trait Convert: UnitAccess {
    fn convert(self, value: f64, forward: bool) -> f64;
}

#[derive(Clone, Copy)]
enum Selector<'a, T> {
    Uniform(T),
    Varying(&'a [f64]),
}

pub(crate) struct Conversion<'a, T> {
    values: &'a [f64],
    selector: Selector<'a, T>,
    forward: bool,
}
impl<'a, T: Convert> Conversion<'a, T> {
    fn new(arguments: &PreparedArguments<'a>, forward: bool) -> Result<Self, String> {
        let units = arguments
            .values(Field::Units)
            .ok_or_else(|| "missing argument units".to_owned())?;
        // Preserve first-unit validation before a missing value error.
        let unit = arguments.unit::<T>(0)?;
        let selector = if units.len() == 1 {
            Selector::Uniform(unit)
        } else {
            Selector::Varying(units)
        };
        let values = arguments
            .values(Field::Value)
            .ok_or_else(|| "missing argument value".to_owned())?;
        Ok(Self {
            values,
            selector,
            forward,
        })
    }
    fn run(&self, range: Range<usize>) -> Result<Vec<f64>, String> {
        let mut output = Vec::new();
        output
            .try_reserve_exact(range.len())
            .map_err(|error| format!("MemoryError: {error}"))?;
        match self.selector {
            Selector::Uniform(unit) => {
                for index in range {
                    let value = self
                        .values
                        .get(if self.values.len() == 1 { 0 } else { index })
                        .copied()
                        .ok_or_else(|| "invalid length for value".to_owned())?;
                    output.push(unit.convert(value, self.forward));
                }
            }
            Selector::Varying(units) => {
                for index in range {
                    let code = units
                        .get(index)
                        .copied()
                        .ok_or_else(|| "invalid length for units".to_owned())?;
                    let unit = T::try_from(code as i32).map_err(|error| error.to_string())?;
                    let value = self
                        .values
                        .get(if self.values.len() == 1 { 0 } else { index })
                        .copied()
                        .ok_or_else(|| "invalid length for value".to_owned())?;
                    output.push(unit.convert(value, self.forward));
                }
            }
        }
        Ok(output)
    }
}

macro_rules! units {
    ($($variant:ident: $unit:ty),+ $(,)?) => {
        $(impl Convert for $unit {
            fn convert(self, value: f64, forward: bool) -> f64 {
                if forward { self.to_base(value) } else { self.from_base(value) }
            }
        })+
        pub(crate) enum PreparedUnits<'a> { $($variant(Conversion<'a, $unit>)),+ }
        impl<'a> PreparedUnits<'a> {
            pub(crate) fn new(operation: &str, arguments: &PreparedArguments<'a>) -> Option<Result<Self, String>> {
                // Unit dispatch has always consumed plain f64 buffers; arithmetic metadata does not change unit normalization.
                Some(match operation {
                    $(concat!("units.", stringify!($variant), ".to_base") => Conversion::new(arguments,true).map(Self::$variant),
                      concat!("units.", stringify!($variant), ".from_base") => Conversion::new(arguments,false).map(Self::$variant),)+
                    _ => return None,
                })
            }
            pub(crate) fn run(&self, range: Range<usize>) -> Result<Vec<f64>, String> {
                match self { $(Self::$variant(conversion) => conversion.run(range)),+ }
            }
        }
    };
}
units! {
    Area: AreaUnit, BasalArea: BasalAreaUnit, Length: LengthUnit, Loading: LoadingUnit,
    Pressure: PressureUnit, SurfaceAreaToVolume: SurfaceAreaToVolumeUnit, Speed: SpeedUnit,
    Fraction: FractionUnit, Slope: SlopeUnit, Density: DensityUnit,
    HeatOfCombustion: HeatOfCombustionUnit, HeatSink: HeatSinkUnit,
    HeatPerUnitArea: HeatPerUnitAreaUnit, HeatSourceAndReactionIntensity: HeatSourceAndReactionIntensityUnit,
    FirelineIntensity: FirelineIntensityUnit, Temperature: TemperatureUnit, Time: TimeUnit,
}
#[cfg(all(test, feature = "allocation-probe"))]
mod tests {
    use super::super::fields::{Arguments, PreparedArguments};
    use super::*;
    use behave_fire::GridShape;
    use std::collections::BTreeMap;

    #[test]
    fn every_plain_unit_class_allocates_only_its_output_buffer() {
        for class in [
            "Area",
            "BasalArea",
            "Length",
            "Loading",
            "Pressure",
            "SurfaceAreaToVolume",
            "Speed",
            "Fraction",
            "Slope",
            "Density",
            "HeatOfCombustion",
            "HeatSink",
            "HeatPerUnitArea",
            "HeatSourceAndReactionIntensity",
            "FirelineIntensity",
            "Temperature",
            "Time",
        ] {
            for count in [1000, 10000] {
                let arguments = Arguments {
                    values: BTreeMap::from([
                        ("value".into(), vec![-0.0; count]),
                        ("units".into(), vec![0.0]),
                    ]),
                    strings: BTreeMap::new(),
                    integers: BTreeMap::new(),
                };
                let p = PreparedArguments::new(&arguments, &GridShape::new(vec![count]).unwrap())
                    .unwrap();
                for direction in ["to_base", "from_base"] {
                    let plan = PreparedUnits::new(&format!("units.{class}.{direction}"), &p)
                        .unwrap()
                        .unwrap();
                    crate::allocation_probe::start();
                    let values = plan.run(0..count).unwrap();
                    let counters = crate::allocation_probe::stop();
                    assert_eq!(counters[0].0, 1, "{class}/{direction}/{count}");
                    assert_eq!(values.len(), count);
                    assert!(values.iter().all(|v| v.to_bits() == (-0.0_f64).to_bits()));
                }
            }
        }
    }

    #[test]
    fn varying_units_keep_per_element_conversion_and_error_order() {
        let mut arguments = Arguments {
            values: BTreeMap::from([
                ("value".into(), vec![1.0, 2.0]),
                ("units".into(), vec![0.0, 1.0]),
            ]),
            strings: BTreeMap::new(),
            integers: BTreeMap::new(),
        };
        let p = PreparedArguments::new(&arguments, &GridShape::new(vec![2]).unwrap()).unwrap();
        let plan = PreparedUnits::new("units.Length.to_base", &p)
            .unwrap()
            .unwrap();
        assert_eq!(plan.run(0..2).unwrap(), vec![1.0, 0.16666666666666]);
        arguments.values.remove("value");
        arguments.values.insert("units".into(), vec![-1.0, 0.0]);
        let p = PreparedArguments::new(&arguments, &GridShape::new(vec![2]).unwrap()).unwrap();
        let error = PreparedUnits::new("units.Length.to_base", &p)
            .unwrap()
            .err()
            .unwrap();
        assert_eq!(error, "invalid LengthUnit unit code -1");
    }
}

#[cfg(all(test, feature = "allocation-probe"))]
mod normalization_tests {
    use super::super::fields::{Arguments, PreparedArguments};
    use super::*;
    use behave_fire::GridShape;
    use std::collections::BTreeMap;
    #[test]
    fn unit_dispatch_retains_its_plain_numeric_normalization_with_metadata_present() {
        let arguments = Arguments {
            values: BTreeMap::from([
                ("value".into(), vec![1.0, -0.0]),
                ("units".into(), vec![0.0]),
                ("_single.value".into(), vec![1.0]),
                ("_masked.value".into(), vec![1.0]),
                ("_mask.value".into(), vec![1.0, 0.0]),
            ]),
            strings: BTreeMap::new(),
            integers: BTreeMap::new(),
        };
        let p = PreparedArguments::new(&arguments, &GridShape::new(vec![2]).unwrap()).unwrap();
        let plan = PreparedUnits::new("units.Length.to_base", &p)
            .expect("unit dispatch always converts plain f64 buffers")
            .unwrap();
        let values = plan.run(0..2).unwrap();
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1].to_bits(), (-0.0_f64).to_bits());
    }
}
