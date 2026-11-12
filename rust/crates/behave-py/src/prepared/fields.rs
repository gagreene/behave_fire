//! Owned argument snapshots and prepared native-only field access.
use std::collections::BTreeMap;
#[derive(Clone)]
pub(crate) struct Arguments {
    pub(crate) values: BTreeMap<String, Vec<f64>>,
    pub(crate) strings: BTreeMap<String, String>,
    pub(crate) integers: BTreeMap<String, (String, Vec<u64>)>,
}

impl Arguments {
    pub(crate) fn empty_prototype(&self) -> Self {
        let mut result = self.clone();
        for (name, values) in &mut result.values {
            if values.is_empty() {
                values.push(if name.starts_with("_mask.") { 0.0 } else { 1.0 });
            }
        }
        for (_, bits) in result.integers.values_mut() {
            if bits.is_empty() {
                bits.push(1);
            }
        }
        result
    }
    pub(crate) fn has_crown_python_payloads(&self) -> bool {
        self.values.keys().any(|key| {
            key.starts_with("_object.")
                || (key.starts_with("_weak.")
                    && !matches!(
                        key.as_str(),
                        "_weak.wind_speed_units"
                            | "_weak.slope_units"
                            | "_weak.aspect"
                            | "_weak.moisture_foliar"
                    ))
        })
    }
    pub(crate) fn has_python_payloads(&self) -> bool {
        self.values
            .keys()
            .any(|key| key.starts_with("_object.") || key.starts_with("_weak."))
    }
    pub(crate) fn has_masks(&self) -> bool {
        self.values.keys().any(|key| key.starts_with("_masked."))
    }
    pub(crate) fn has_single(&self) -> bool {
        self.values.keys().any(|key| key.starts_with("_single."))
    }
    pub(crate) fn has_integers(&self) -> bool {
        self.integers.keys().any(|key| {
            let field = key.rsplit('.').next().unwrap_or(key);
            !field.ends_with("units")
                && !field.starts_with('_')
                && !matches!(
                    field,
                    "max_steps"
                        | "charge_type"
                        | "fuel_model_grid"
                        | "equation_number_grid"
                        | "is_crown"
                        | "is_surface_fire"
                        | "is_passive_crown"
                        | "is_active_crown"
                        | "is_defined"
                        | "is_dynamic"
                )
        })
    }
}

use super::keys::Field;
use super::unit_selectors::{Selector, UnitAccess, UnitCache};
use behave_fire::engine::{SurfaceOptions, WafMethod, WindHeight, WindOrientation};
use behave_fire::units::{FractionUnit, PressureUnit, SlopeUnit, SpeedUnit, TemperatureUnit};
use behave_fire::{
    surface::masked::{IntegerKind, IntegerValue, MathProfile},
    GridShape,
};

#[derive(Clone, Copy)]
#[repr(usize)]
enum Meta {
    ObjectKind,
    WeakBool,
    WeakInteger,
    Weak,
    Mask,
    Masked,
    Single,
    Scalar,
    Fill,
    NumpyScalar,
    Object,
    ObjectNumpy,
    IntegerType,
    Empty,
}
impl Meta {
    const COUNT: usize = 14;
    fn from_prefix(prefix: &str) -> Option<Self> {
        match prefix {
            "_object_kind" => Some(Self::ObjectKind),
            "_weak_bool" => Some(Self::WeakBool),
            "_weak_integer" => Some(Self::WeakInteger),
            "_weak" => Some(Self::Weak),
            "_mask" => Some(Self::Mask),
            "_masked" => Some(Self::Masked),
            "_single" => Some(Self::Single),
            "_scalar" => Some(Self::Scalar),
            "_fill" => Some(Self::Fill),
            "_numpy_scalar" => Some(Self::NumpyScalar),
            "_object" => Some(Self::Object),
            "_object_numpy" => Some(Self::ObjectNumpy),
            "_integer_type" => Some(Self::IntegerType),
            "_empty" => Some(Self::Empty),
            _ => None,
        }
    }
}

#[derive(Clone, Copy)]
struct IntegerColumn<'a> {
    name: &'a str,
    kind: Option<IntegerKind>,
    bits: &'a [u64],
}

#[derive(Default)]
struct Descriptor<'a> {
    values: Option<&'a [f64]>,
    text: Option<&'a str>,
    integer: Option<IntegerColumn<'a>>,
    metadata: [Option<&'a [f64]>; Meta::COUNT],
}
impl Descriptor<'_> {
    fn optional(&self, meta: Meta, index: usize) -> Option<f64> {
        self.metadata[meta as usize]
            .and_then(|values| at(values, index))
            .copied()
    }
}

fn at<T>(values: &[T], index: usize) -> Option<&T> {
    values.get(if values.len() == 1 { 0 } else { index })
}

#[derive(Clone, Copy)]
enum Profile<'a> {
    Constant(MathProfile),
    Columns(Option<&'a [f64]>, Option<&'a [f64]>),
}
impl Profile<'_> {
    fn decode(simd: f64, svml: f64) -> MathProfile {
        match (simd != 0.0, svml != 0.0) {
            (true, true) => MathProfile::NumpySimdSvmlAtan2,
            (false, true) => MathProfile::LibmSvmlAtan2,
            (true, false) => MathProfile::NumpySimd,
            (false, false) => MathProfile::Libm,
        }
    }
    fn at(&self, index: usize) -> MathProfile {
        match *self {
            Self::Constant(profile) => profile,
            Self::Columns(simd, svml) => Self::decode(
                simd.and_then(|v| at(v, index)).copied().unwrap_or(0.0),
                svml.and_then(|v| at(v, index)).copied().unwrap_or(0.0),
            ),
        }
    }
}

/// Borrow typed field views only from an already-owned native snapshot.
pub(crate) struct PreparedArguments<'a> {
    slots: [usize; Field::COUNT],
    fields: Vec<Descriptor<'a>>,
    profile: Profile<'a>,
    unit_cache: UnitCache<'a>,
    speed_units: Selector<'a, SpeedUnit>,
    slope_units: Selector<'a, SlopeUnit>,
    temperature_units: Selector<'a, TemperatureUnit>,
    humidity_units: Selector<'a, FractionUnit>,
    pressure_units: Selector<'a, PressureUnit>,
    orientation: Option<WindOrientation>,
    height: Option<WindHeight>,
    waf: Option<WafMethod>,
    masks: bool,
    single: bool,
    integers: bool,
    python_payloads: bool,
    crown_python_payloads: bool,
}
impl<'a> PreparedArguments<'a> {
    pub(crate) fn new(arguments: &'a Arguments, shape: &GridShape) -> Result<Self, String> {
        let count = shape.len();
        let mut result = Self {
            slots: [usize::MAX; Field::COUNT],
            fields: Vec::new(),
            profile: Profile::Constant(MathProfile::Libm),
            unit_cache: UnitCache::new(None),
            speed_units: Selector::new(None),
            slope_units: Selector::new(None),
            temperature_units: Selector::new(None),
            humidity_units: Selector::new(None),
            pressure_units: Selector::new(None),
            orientation: None,
            height: None,
            waf: None,
            masks: arguments.has_masks(),
            single: arguments.has_single(),
            integers: arguments.has_integers(),
            python_payloads: arguments.has_python_payloads(),
            crown_python_payloads: arguments.has_crown_python_payloads(),
        };
        let field_bound = arguments
            .values
            .len()
            .saturating_add(arguments.strings.len())
            .saturating_add(arguments.integers.len())
            .min(Field::COUNT);
        result
            .fields
            .try_reserve_exact(field_bound)
            .map_err(|error| format!("MemoryError: {error}"))?;
        for (name, values) in &arguments.values {
            if values.len() != 1 && values.len() != count {
                return Err(format!("invalid length for {name}"));
            }
            if let Some(field) = Field::from_name(name) {
                result.descriptor_mut(field).values = Some(values);
            } else if let Some((prefix, name)) = name.split_once('.') {
                if let (Some(meta), Some(field)) =
                    (Meta::from_prefix(prefix), Field::from_name(name))
                {
                    result.descriptor_mut(field).metadata[meta as usize] = Some(values);
                }
            }
        }
        for (name, text) in &arguments.strings {
            if let Some(field) = Field::from_name(name) {
                result.descriptor_mut(field).text = Some(text);
            }
        }
        for (name, (kind, bits)) in &arguments.integers {
            if bits.len() != 1 && bits.len() != count {
                return Err(format!("invalid integer length for {name}"));
            }
            if let Some(field) = Field::from_name(name) {
                result.descriptor_mut(field).integer = Some(IntegerColumn {
                    name: kind,
                    kind: IntegerKind::from_name(kind),
                    bits,
                });
            }
        }
        let simd = result
            .descriptor(Field::GlobalSimdExp)
            .and_then(|d| d.values);
        let svml = result
            .descriptor(Field::GlobalSvmlAtan2)
            .and_then(|d| d.values);
        result.profile = if simd.map_or(0, |v| v.len()) <= 1 && svml.map_or(0, |v| v.len()) <= 1 {
            Profile::Constant(Profile::decode(
                simd.and_then(|v| v.first()).copied().unwrap_or(0.0),
                svml.and_then(|v| v.first()).copied().unwrap_or(0.0),
            ))
        } else {
            Profile::Columns(simd, svml)
        };
        result.orientation = result
            .descriptor(Field::WindOrientationMode)
            .and_then(|d| d.text)
            .map(super::controls::orientation);
        result.height = result
            .descriptor(Field::WindHeightMode)
            .and_then(|d| d.text)
            .map(super::controls::height);
        result.waf = result
            .descriptor(Field::WafMethod)
            .and_then(|d| d.text)
            .map(super::controls::waf);
        result.unit_cache = UnitCache::new(result.values(Field::Units));
        result.speed_units = Selector::new(result.values(Field::WindSpeedUnits));
        result.slope_units = Selector::new(result.values(Field::SlopeUnits));
        result.temperature_units = Selector::new(result.values(Field::TempUnits));
        result.humidity_units = Selector::new(result.values(Field::RhUnits));
        result.pressure_units = Selector::new(result.values(Field::OutputUnits));
        Ok(result)
    }
    pub(crate) fn unit<T: UnitAccess>(&self, index: usize) -> Result<T, String> {
        T::cached(&self.unit_cache, index)
    }
    pub(crate) fn speed_units(&self, index: usize) -> Result<SpeedUnit, String> {
        self.speed_units.at(index, Field::WindSpeedUnits)
    }
    pub(crate) fn slope_units(&self, index: usize) -> Result<SlopeUnit, String> {
        self.slope_units.at(index, Field::SlopeUnits)
    }
    pub(crate) fn temperature_units(&self, index: usize) -> Result<TemperatureUnit, String> {
        self.temperature_units.at(index, Field::TempUnits)
    }
    pub(crate) fn humidity_units(&self, index: usize) -> Result<FractionUnit, String> {
        self.humidity_units.at(index, Field::RhUnits)
    }
    pub(crate) fn pressure_units(&self, index: usize) -> Result<PressureUnit, String> {
        self.pressure_units.at(index, Field::OutputUnits)
    }
    pub(crate) fn orientation(&self) -> Result<WindOrientation, String> {
        self.orientation
            .ok_or_else(|| "missing mode wind_orientation_mode".to_owned())
    }
    pub(crate) fn height(&self) -> Result<WindHeight, String> {
        self.height
            .ok_or_else(|| "missing mode wind_height_mode".to_owned())
    }
    pub(crate) fn waf(&self) -> Result<WafMethod, String> {
        self.waf.ok_or_else(|| "missing mode waf_method".to_owned())
    }
    pub(crate) fn surface_options(&self) -> Result<SurfaceOptions, String> {
        Ok(SurfaceOptions {
            wind_orientation: self.orientation()?,
            wind_height: self.height()?,
            waf_method: self.waf()?,
        })
    }
    pub(crate) fn crown_options(&self) -> SurfaceOptions {
        SurfaceOptions {
            wind_orientation: self.orientation.unwrap_or(WindOrientation::Upslope),
            ..SurfaceOptions::default()
        }
    }
    pub(crate) fn values(&self, field: Field) -> Option<&'a [f64]> {
        self.descriptor(field).and_then(|d| d.values)
    }
    fn descriptor(&self, field: Field) -> Option<&Descriptor<'a>> {
        self.fields.get(self.slots[field as usize])
    }
    fn descriptor_mut(&mut self, field: Field) -> &mut Descriptor<'a> {
        let slot = &mut self.slots[field as usize];
        if *slot == usize::MAX {
            *slot = self.fields.len();
            self.fields.push(Descriptor::default());
        }
        &mut self.fields[*slot]
    }
    fn integer_bits(&self, field: Field, index: usize) -> Option<u64> {
        self.descriptor(field)
            .and_then(|d| d.integer)
            .and_then(|column| at(column.bits, index))
            .copied()
    }
    fn integer_value(&self, field: Field, index: usize) -> Result<Option<IntegerValue>, String> {
        if let Some(column) = self.descriptor(field).and_then(|d| d.integer) {
            if let Some(bits) = at(column.bits, index) {
                let kind = column.kind.ok_or_else(|| {
                    format!("TypeError: unsupported integer dtype {}", column.name)
                })?;
                return Ok(Some(IntegerValue::from_bits(kind, *bits)));
            }
        }
        Ok(None)
    }
    pub(crate) fn has_masks(&self) -> bool {
        self.masks
    }
    pub(crate) fn has_single(&self) -> bool {
        self.single
    }
    pub(crate) fn has_integers(&self) -> bool {
        self.integers
    }
    pub(crate) fn has_python_payloads(&self) -> bool {
        self.python_payloads
    }
    pub(crate) fn has_crown_python_payloads(&self) -> bool {
        self.crown_python_payloads
    }
    pub(crate) fn n(&self, field: Field, index: usize) -> Result<f64, String> {
        let values = self
            .descriptor(field)
            .and_then(|d| d.values)
            .ok_or_else(|| format!("missing argument {}", field.name()))?;
        at(values, index)
            .copied()
            .ok_or_else(|| format!("invalid length for {}", field.name()))
    }
    pub(crate) fn optional(&self, field: Field, index: usize) -> Option<f64> {
        self.descriptor(field)
            .and_then(|d| d.values)
            .and_then(|values| at(values, index))
            .copied()
    }
    pub(crate) fn masked(
        &self,
        field: Field,
        index: usize,
    ) -> Result<behave_fire::surface::masked::MaskedValue, String> {
        use behave_fire::surface::masked::{IntegerKind, IntegerValue, MaskedValue, Precision};
        let metadata = self
            .descriptor(field)
            .ok_or_else(|| format!("missing argument {}", field.name()))?;
        let profile = self.profile.at(index);
        let mut integer = self.integer_value(field, index)?;
        if let Some(code) = metadata.optional(Meta::ObjectKind, index) {
            integer = if (0.0..=8.0).contains(&code) {
                let kinds = [
                    IntegerKind::I8,
                    IntegerKind::I16,
                    IntegerKind::I32,
                    IntegerKind::I64,
                    IntegerKind::U8,
                    IntegerKind::U16,
                    IntegerKind::U32,
                    IntegerKind::U64,
                    IntegerKind::Bool,
                ];
                Some(IntegerValue::from_bits(
                    kinds[code as usize],
                    self.integer_bits(field, index).unwrap_or(0),
                ))
            } else {
                None
            };
        }
        if metadata.optional(Meta::WeakBool, index).is_some() {
            return Ok(
                MaskedValue::python_bool(integer.is_some_and(|v| v.raw_bits != 0))
                    .with_math_profile(profile),
            );
        }
        if metadata.optional(Meta::WeakInteger, index).is_some() {
            if let Some(integer) = integer {
                return Ok(if integer.kind.is_signed() {
                    MaskedValue::python_int(integer.as_i128() as i64)
                } else {
                    MaskedValue::python_uint(integer.raw_bits)
                }
                .with_math_profile(profile));
            }
        }
        if metadata.optional(Meta::Weak, index).is_some() {
            return Ok(MaskedValue::scalar(self.n(field, index)?).with_math_profile(profile));
        }
        let value = MaskedValue::new(
            self.n(field, index)?,
            metadata.optional(Meta::Mask, index).unwrap_or(0.0) != 0.0,
            metadata.optional(Meta::Masked, index).is_some(),
            if metadata.optional(Meta::Single, index).is_some()
                || metadata.optional(Meta::ObjectKind, index) == Some(10.0)
            {
                Precision::F32
            } else {
                Precision::F64
            },
        )
        .with_scalar_array(metadata.optional(Meta::Scalar, index).is_some())
        .with_fill_value(metadata.optional(Meta::Fill, index))
        .with_math_profile(profile)
        .with_numpy_scalar(metadata.optional(Meta::NumpyScalar, index).is_some());
        let mut value = integer.map_or(value, |integer| value.with_integer(integer));
        if metadata.optional(Meta::Object, index).is_some() {
            value = value.with_object_payload(
                metadata.optional(Meta::ObjectNumpy, index).unwrap_or(0.0) != 0.0,
            );
        }
        value = value.with_numpy_integer_type(
            metadata
                .optional(Meta::IntegerType, index)
                .filter(|number| (0.0..=10.0).contains(number))
                .map(|number| number as u8),
        );
        Ok(value.with_empty_array(metadata.optional(Meta::Empty, index).is_some()))
    }
}

#[cfg(all(test, feature = "allocation-probe"))]
mod tests {
    use super::*;
    use std::hint::black_box;

    fn arguments() -> Arguments {
        Arguments {
            values: BTreeMap::new(),
            strings: BTreeMap::new(),
            integers: BTreeMap::new(),
        }
    }

    #[test]
    fn absent_optional_fields_do_not_allocate() {
        let a = arguments();
        let p = PreparedArguments::new(&a, &GridShape::new(vec![100]).unwrap()).unwrap();
        crate::allocation_probe::start();
        for index in 0..100 {
            assert!(black_box(p.optional(Field::UserWaf, index)).is_none());
        }
        let counts = crate::allocation_probe::stop();
        assert_eq!(
            counts[0].0, 0,
            "missing optional data must not format discarded errors"
        );
    }

    #[test]
    fn optional_length_mismatch_is_none_without_allocating() {
        let mut a = arguments();
        a.values.insert("value".into(), vec![1.0, 2.0]);
        let p = PreparedArguments::new(&a, &GridShape::new(vec![2]).unwrap()).unwrap();
        crate::allocation_probe::start();
        for index in 2..100 {
            assert!(black_box(p.optional(Field::Value, index)).is_none());
        }
        let counts = crate::allocation_probe::stop();
        assert_eq!(counts[0].0, 0);
        assert_eq!(
            p.n(Field::Value, 2).unwrap_err(),
            "invalid length for value"
        );
    }

    #[test]
    fn required_fields_still_report_missing_and_scalar_values_broadcast() {
        let mut a = arguments();
        let p = PreparedArguments::new(&a, &GridShape::new(vec![1]).unwrap()).unwrap();
        assert_eq!(
            p.n(Field::WindSpeed, 0).unwrap_err(),
            "missing argument wind_speed"
        );
        assert_eq!(p.height().unwrap_err(), "missing mode wind_height_mode");
        a.values.insert("value".into(), vec![-0.0]);
        let p = PreparedArguments::new(&a, &GridShape::new(vec![1]).unwrap()).unwrap();
        assert_eq!(
            p.optional(Field::Value, 999).unwrap().to_bits(),
            (-0.0_f64).to_bits()
        );
    }
}

#[cfg(all(test, feature = "allocation-probe"))]
mod prepared_tests {
    use super::super::keys::Field;
    use super::*;
    use behave_fire::{
        surface::masked::{MathProfile, Precision},
        GridShape,
    };
    use std::hint::black_box;

    fn arguments() -> Arguments {
        Arguments {
            values: BTreeMap::new(),
            strings: BTreeMap::new(),
            integers: BTreeMap::new(),
        }
    }

    #[test]
    fn descriptor_reads_preserve_masks_precision_fill_profile_and_zero_without_allocating() {
        let mut a = arguments();
        for (name, values) in [
            ("value", vec![3.0, -0.0]),
            ("_mask.value", vec![1.0, 0.0]),
            ("_masked.value", vec![0.0]),
            ("_single.value", vec![0.0]),
            ("_scalar.value", vec![0.0]),
            ("_numpy_scalar.value", vec![0.0]),
            ("_fill.value", vec![-17.0]),
            ("_simd_exp", vec![1.0, 0.0]),
            ("_svml_atan2", vec![0.0, 1.0]),
        ] {
            a.values.insert(name.into(), values);
        }
        let prepared = PreparedArguments::new(&a, &GridShape::new(vec![2]).unwrap()).unwrap();
        crate::allocation_probe::start();
        for _ in 0..100 {
            let first = black_box(prepared.masked(Field::Value, 0).unwrap());
            let second = black_box(prepared.masked(Field::Value, 1).unwrap());
            assert_eq!(first.value, 3.0);
            assert!(
                first.mask
                    && first.is_masked_array
                    && first.is_scalar_array
                    && first.is_numpy_scalar
            );
            assert_eq!(first.precision, Precision::F32);
            assert_eq!(first.fill_value, Some(-17.0));
            assert_eq!(first.math_profile, MathProfile::NumpySimd);
            assert_eq!(second.math_profile, MathProfile::LibmSvmlAtan2);
            assert!(!second.mask);
            assert_eq!(second.value.to_bits(), (-0.0_f64).to_bits());
            assert_eq!(prepared.optional(Field::UserWaf, 0), None);
        }
        let counters = crate::allocation_probe::stop();
        assert_eq!(counters[0].0, 0);
    }

    #[test]
    fn descriptor_validation_rejects_wrong_lengths_and_defers_unused_dtype_errors() {
        let mut a = arguments();
        a.values.insert("value".into(), vec![1.0, 2.0]);
        assert!(PreparedArguments::new(&a, &GridShape::new(vec![3]).unwrap()).is_err());
        a.integers
            .insert("wind_speed".into(), ("invalid".into(), vec![1]));
        let p = PreparedArguments::new(&a, &GridShape::new(vec![2]).unwrap()).unwrap();
        assert_eq!(p.masked(Field::Value, 1).unwrap().value, 2.0);
        assert_eq!(
            p.masked(Field::WindSpeed, 0).unwrap_err(),
            "TypeError: unsupported integer dtype invalid"
        );
    }

    #[test]
    fn integer_bits_and_variable_object_kind_are_retained() {
        let mut a = arguments();
        a.values.insert("value".into(), vec![0.0, 1.0]);
        a.integers
            .insert("value".into(), ("uint64".into(), vec![u64::MAX, 1]));
        let p = PreparedArguments::new(&a, &GridShape::new(vec![2]).unwrap()).unwrap();
        assert_eq!(
            p.masked(Field::Value, 0).unwrap().integer.unwrap().raw_bits,
            u64::MAX
        );
        a.values
            .insert("_object_kind.value".into(), vec![8.0, 10.0]);
        let p = PreparedArguments::new(&a, &GridShape::new(vec![2]).unwrap()).unwrap();
        assert!(p.masked(Field::Value, 0).unwrap().integer.is_some());
        assert_eq!(p.masked(Field::Value, 1).unwrap().precision, Precision::F32);
        assert!(p.masked(Field::Value, 1).unwrap().integer.is_none());
    }
}

#[cfg(all(test, feature = "allocation-probe"))]
mod control_tests {
    use super::*;
    use behave_fire::engine::{WafMethod, WindHeight, WindOrientation};
    #[test]
    fn prepared_control_modes_preserve_permissive_parsing_and_crown_defaults() {
        let mut a = Arguments {
            values: BTreeMap::new(),
            strings: BTreeMap::new(),
            integers: BTreeMap::new(),
        };
        let empty = PreparedArguments::new(&a, &GridShape::new(vec![1]).unwrap()).unwrap();
        assert!(empty.surface_options().is_err());
        assert_eq!(
            empty.crown_options().wind_orientation,
            WindOrientation::Upslope
        );
        a.strings
            .insert("wind_orientation_mode".into(), "relative to NORTH".into());
        a.strings
            .insert("wind_height_mode".into(), "ten metre".into());
        a.strings.insert("waf_method".into(), "USER_INPUT".into());
        let p = PreparedArguments::new(&a, &GridShape::new(vec![1]).unwrap()).unwrap();
        let surface = p.surface_options().unwrap();
        assert_eq!(surface.wind_orientation, WindOrientation::North);
        assert_eq!(surface.wind_height, WindHeight::TenMetre);
        assert_eq!(surface.waf_method, WafMethod::UserInput);
        assert_eq!(p.crown_options().wind_height, WindHeight::TwentyFoot);
        assert_eq!(p.crown_options().wind_orientation, WindOrientation::North);
    }
}

#[cfg(all(test, feature = "allocation-probe"))]
mod selector_tests {
    use super::*;
    use behave_fire::units::{LengthUnit, SpeedUnit};
    #[test]
    fn scalar_and_varying_unit_selectors_keep_deferred_errors() {
        let a = Arguments {
            values: BTreeMap::from([
                ("units".into(), vec![0.0, 1.0, -1.0]),
                ("wind_speed_units".into(), vec![0.0]),
            ]),
            strings: BTreeMap::new(),
            integers: BTreeMap::new(),
        };
        let p = PreparedArguments::new(&a, &GridShape::new(vec![3]).unwrap()).unwrap();
        assert_eq!(p.unit::<LengthUnit>(0).unwrap(), LengthUnit::Feet);
        assert_eq!(p.unit::<LengthUnit>(1).unwrap(), LengthUnit::Inches);
        assert_eq!(
            p.unit::<LengthUnit>(2).unwrap_err(),
            "invalid LengthUnit unit code -1"
        );
        assert_eq!(p.speed_units(2).unwrap(), SpeedUnit::FeetPerMinute);
        assert_eq!(
            p.slope_units(0).unwrap_err(),
            "missing argument slope_units"
        );
    }
}

#[cfg(all(test, feature = "allocation-probe"))]
mod tool_selector_tests {
    use super::*;
    #[test]
    fn tool_units_are_cached_without_requiring_unused_controls() {
        let a = Arguments {
            values: BTreeMap::from([
                ("temp_units".into(), vec![0.0]),
                ("rh_units".into(), vec![1.0]),
                ("output_units".into(), vec![1.0]),
            ]),
            strings: BTreeMap::new(),
            integers: BTreeMap::new(),
        };
        let p = PreparedArguments::new(&a, &GridShape::new(vec![1]).unwrap()).unwrap();
        assert_eq!(p.temperature_units(0).unwrap().code(), 0);
        assert_eq!(p.humidity_units(0).unwrap().code(), 1);
        assert_eq!(p.pressure_units(0).unwrap().code(), 1);
        assert!(p.speed_units(0).is_err());
    }
}
