//! Native masked-array semantics for surface stages that operate before `np.asarray` normalization.
//!
//! Arithmetic operators restore the left raw operand at masked positions. NumPy
//! ufuncs instead calculate raw values and propagate masks, with domain fills for
//! square roots. `np.where` discards the mask using the underlying condition/data.
use crate::{
    math::{np_max, np_min, np_remainder},
    units::SlopeUnit,
};
#[path = "masked_records.rs"]
mod records;
pub use records::*;
#[path = "integers.rs"]
mod integers;
pub use integers::{IntegerKind, IntegerValue};
#[path = "complex.rs"]
mod complex;
pub use complex::ComplexValue;
#[path = "objects.rs"]
mod objects;

#[cfg(test)]
mod tests {
    use super::{MaskedValue, MathProfile};
    #[test]
    fn masked_float32_comparisons_promote_python_constants_to_float64() {
        let threshold = MaskedValue::scalar(1e-7);
        assert!(MaskedValue::masked_f32(1e-7, false).greater(threshold));
        assert!(!MaskedValue::plain_f32(1e-7).greater(threshold));
    }
    #[test]
    fn float32_exponential_preserves_numpy_value_after_float64_promotion() {
        let value = MaskedValue::plain_f32(-138.0f32 / 3500.0f32)
            .with_math_profile(MathProfile::NumpySimd)
            .exp()
            .as_array_f64();
        assert_eq!(value.value, (f32::from_bits(1064704585)) as f64);
    }
    #[test]
    fn explicit_scalar_math_profile_matches_numpy_baseline_dispatch() {
        let x = MaskedValue::plain_f32(-138.0f32 / 3500.0f32).with_math_profile(MathProfile::Libm);
        assert_eq!(x.exp().value, f32::from_bits(1064704586) as f64);
        assert_eq!(
            MaskedValue::scalar(1.0).mul(x).math_profile,
            MathProfile::Libm
        );
    }
}

/// Floating precision of a NumPy array, independent of its scalar transport representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// IEEE binary32 arithmetic and output.
    F32,
    /// IEEE binary64 arithmetic and output.
    F64,
}

/// A raw Boolean condition together with the NumPy array rank that broadcasts in selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskedCondition {
    /// Underlying Boolean data, after source comparison semantics.
    pub value: bool,
    /// Whether the entire condition has shape ().
    pub is_scalar_array: bool,
    /// An eagerly evaluated source comparison failed before selection.
    pub error: Option<MaskedError>,
    /// The condition has zero logical elements; its scalar prototype is not a data-loop result.
    pub empty_array: bool,
}
impl MaskedCondition {
    /// Construct a condition from independently evaluated raw data and rank metadata.
    pub fn new(value: bool, is_scalar_array: bool) -> Self {
        Self {
            value,
            is_scalar_array,
            error: None,
            empty_array: false,
        }
    }
    /// Combine conditions using NumPy's Boolean AND broadcasting rank.
    pub fn and(self, other: Self) -> Self {
        Self {
            value: self.value && other.value,
            is_scalar_array: self.is_scalar_array && other.is_scalar_array,
            error: self.error.or(other.error),
            empty_array: self.empty_array || other.empty_array,
        }
    }
    /// Combine conditions using NumPy's Boolean OR broadcasting rank.
    pub fn or(self, other: Self) -> Self {
        Self {
            value: self.value || other.value,
            is_scalar_array: self.is_scalar_array && other.is_scalar_array,
            error: self.error.or(other.error),
            empty_array: self.empty_array || other.empty_array,
        }
    }
    /// Attach zero-work provenance without erasing an earlier scalar exception.
    pub fn with_empty_array(mut self, empty: bool) -> Self {
        self.empty_array = empty;
        if empty {
            self.is_scalar_array = false;
        }
        self
    }
}

/// Float32 transcendental profile selected by the reference NumPy build's CPU dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MathProfile {
    /// NumPy's AVX2/FMA or AVX512 float32 exponential formula, evaluated portably with scalar FMA.
    NumpySimd,
    /// The Rust/platform libc scalar exponential, matching NumPy's non-SIMD fallback.
    Libm,
}
impl MathProfile {
    /// Select the usual NumPy profile for native hardware; bindings should override from actual NumPy capability metadata.
    pub fn native_default() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return Self::NumpySimd;
        }
        Self::Libm
    }
}

/// One logical array element, retaining raw data, element mask, array class, and precision.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct MaskedValue {
    /// Underlying numeric data, including the raw value at a masked position.
    pub value: f64,
    /// Whether this element is masked.
    pub mask: bool,
    /// Whether the source/result is a MaskedArray, including all-false-mask arrays.
    pub is_masked_array: bool,
    /// Precision used for this array's arithmetic and returned NumPy dtype.
    pub precision: Precision,
    /// Whether the full input/result array is zero-dimensional; ordinary per-cell vectors set false.
    pub is_scalar_array: bool,
    /// Whether this value is a NumPy numeric scalar rather than a zero-dimensional ndarray.
    pub is_numpy_scalar: bool,
    /// Explicit NumPy fill value; None retains NumPy's lazy default for the result dtype.
    pub fill_value: Option<f64>,
    /// Float32 math implementation profile, propagated through intermediate expressions.
    pub math_profile: MathProfile,
    /// Exact integer dtype/bits, retained until an operation explicitly promotes to floating point.
    pub integer: Option<IntegerValue>,
    /// Exact NumPy integer type number, distinguishing equal-width C-type aliases.
    pub numpy_integer_type: Option<u8>,
    /// Source-compatible operation failure propagated through eager expressions.
    pub error: Option<MaskedError>,
    /// Generated complex data; Some(0+0j) still requires a complex output dtype.
    pub complex: Option<ComplexValue>,
    /// NumPy object dtype is required for the selected supported numeric payload.
    pub object_result: bool,
    /// A boxed numeric object payload is a NumPy scalar rather than a builtin Python scalar.
    pub object_numpy_scalar: bool,
    /// The original array has zero logical elements; data-loop checks must not inspect its prototype.
    pub empty_array: bool,
    wide_integer: Option<integers::WideInteger>,
    weak: bool,
}

// Named operations make NumPy promotion and mask semantics explicit at each call.
#[allow(clippy::should_implement_trait)]
impl MaskedValue {
    /// Construct an array element from separately marshalled raw data and mask metadata.
    pub fn new(value: f64, mask: bool, is_masked_array: bool, precision: Precision) -> Self {
        Self {
            value: round(value, precision),
            mask,
            is_masked_array,
            precision,
            is_scalar_array: false,
            is_numpy_scalar: false,
            fill_value: None,
            math_profile: MathProfile::native_default(),
            integer: None,
            numpy_integer_type: None,
            error: None,
            wide_integer: None,
            complex: None,
            object_result: false,
            object_numpy_scalar: false,
            empty_array: false,
            weak: false,
        }
    }
    /// Construct a plain float64 array element.
    pub fn plain(value: f64) -> Self {
        Self::new(value, false, false, Precision::F64)
    }
    /// Construct a plain float32 array element.
    pub fn plain_f32(value: f32) -> Self {
        Self::new(value as f64, false, false, Precision::F32)
    }
    /// Construct a float64 MaskedArray element, including a possible false element mask.
    pub fn masked(value: f64, mask: bool) -> Self {
        Self::new(value, mask, true, Precision::F64)
    }
    /// Construct a float32 MaskedArray element, including a possible false element mask.
    pub fn masked_f32(value: f32, mask: bool) -> Self {
        Self::new(value as f64, mask, true, Precision::F32)
    }
    /// Normalize to a plain float64 ndarray, like `np.asarray(value, dtype=float)`.
    pub fn as_array_f64(self) -> Self {
        if self.complex.is_some() {
            return integers::failure(self, MaskedError::UnsupportedComplexOperation);
        }
        let mut source = self;
        source.object_result = false;
        let value = integers::to_float(source, Precision::F64);
        let mut result = Self::plain(value.value)
            .with_scalar_array(self.is_scalar_array)
            .with_math_profile(self.math_profile);
        result = result.with_empty_array(self.empty_array);
        result.error = value.error;
        result
    }
    /// Strip array masks while retaining dtype and rank, like `np.asarray(value)` without a dtype argument.
    pub fn as_array(self) -> Self {
        let mut result = self;
        if let Some(wide) = self.wide_integer {
            if self.integer.is_some_and(|x| x.kind == IntegerKind::Bool) {
                result = result.with_integer(self.integer.unwrap());
            } else if let Some(integer) = wide.fixed() {
                result = result.with_integer(integer);
            } else {
                result.object_result = true;
            }
        }
        result.mask = false;
        result.is_masked_array = false;
        result.is_numpy_scalar = false;
        result.fill_value = None;
        result.weak = false;
        if !result.object_result {
            result.wide_integer = None;
        }
        result
    }
    /// Attach the source array's rank without changing its numeric data or mask.
    pub fn with_scalar_array(mut self, is_scalar_array: bool) -> Self {
        self.is_scalar_array = is_scalar_array;
        self
    }
    /// Attach an owned numeric-object payload kind; no Python object is retained in native code.
    pub fn with_object_payload(mut self, numpy_scalar: bool) -> Self {
        self.object_result = true;
        self.object_numpy_scalar = numpy_scalar;
        self.weak = false;
        self.is_numpy_scalar = false;
        self
    }
    /// Attach zero logical work while preserving dtype-cast and previously computed scalar errors.
    pub fn with_empty_array(mut self, empty: bool) -> Self {
        self.empty_array = empty;
        if empty {
            self.is_scalar_array = false;
            self.is_numpy_scalar = false;
        }
        self
    }
    /// Preserve NumPy scalar dispatch, which differs from zero-dimensional ndarray arithmetic with MaskedArrays.
    pub fn with_numpy_scalar(mut self, is_numpy_scalar: bool) -> Self {
        self.is_numpy_scalar = is_numpy_scalar;
        self.is_scalar_array |= is_numpy_scalar;
        self
    }
    /// Attach an explicitly configured fill value, preserving lazy defaults with None.
    pub fn with_fill_value(mut self, fill_value: Option<f64>) -> Self {
        self.fill_value = fill_value;
        self
    }
    /// Select the reference NumPy build's float32 transcendental profile for this value's expression tree.
    pub fn with_math_profile(mut self, math_profile: MathProfile) -> Self {
        self.math_profile = math_profile;
        self
    }
    /// Add arrays using MaskedArray operator restoration when either source has that subclass.
    pub fn add(self, other: Self) -> Self {
        arithmetic(self, other, Operation::Add)
            .with_empty_array(self.empty_array || other.empty_array)
    }
    /// Subtract arrays using MaskedArray operator restoration.
    pub fn sub(self, other: Self) -> Self {
        arithmetic(self, other, Operation::Sub)
            .with_empty_array(self.empty_array || other.empty_array)
    }
    /// Multiply arrays using MaskedArray operator restoration.
    pub fn mul(self, other: Self) -> Self {
        arithmetic(self, other, Operation::Mul)
            .with_empty_array(self.empty_array || other.empty_array)
    }
    /// Divide arrays, adding MaskedArray domain masks and restoring the raw numerator.
    pub fn div(self, other: Self) -> Self {
        arithmetic(self, other, Operation::Div)
            .with_empty_array(self.empty_array || other.empty_array)
    }
    /// Raise to a power, preserving MaskedArray raw bases and filling nonfinite results with 1e20.
    pub fn pow(self, other: Self) -> Self {
        arithmetic(self, other, Operation::Pow)
            .with_empty_array(self.empty_array || other.empty_array)
    }
    /// Apply NumPy's unary negative ufunc, which computes raw masked data.
    pub fn negative(self) -> Self {
        self.unary(Unary::Negative)
            .with_empty_array(self.empty_array)
    }
    /// Apply NumPy's exponential ufunc while propagating the source element mask.
    pub fn exp(self) -> Self {
        self.unary(Unary::Exp).with_empty_array(self.empty_array)
    }
    /// Apply NumPy's square-root ufunc and its masked/domain-zero fill.
    pub fn sqrt(self) -> Self {
        self.unary(Unary::Sqrt).with_empty_array(self.empty_array)
    }
    /// Apply NumPy's minimum ufunc to raw values, preserving masks and array class.
    pub fn minimum(self, other: Self) -> Self {
        ufunc_binary(self, other, BinaryUfunc::Minimum)
            .with_empty_array(self.empty_array || other.empty_array)
    }
    /// Apply NumPy's maximum ufunc to raw values, preserving masks and array class.
    pub fn maximum(self, other: Self) -> Self {
        ufunc_binary(self, other, BinaryUfunc::Maximum)
            .with_empty_array(self.empty_array || other.empty_array)
    }
    /// Compare raw values after NumPy-style operand promotion; the source mask does not fill the condition.
    pub fn greater(self, other: Self) -> bool {
        if (self.is_scalar_array && self.is_masked_array && self.mask)
            || (other.is_scalar_array && other.is_masked_array && other.mask)
        {
            return false;
        }
        if self.object_result || other.object_result {
            return objects::compare(self, other, 0).unwrap_or(false);
        }
        if self.complex.is_some() || other.complex.is_some() {
            return complex::compare(self, other, 0).unwrap_or(false);
        }
        if integers::is_integer(self) || integers::is_integer(other) {
            return integers::compare(self, other, 0);
        }
        let p = promote(self, other, self.is_masked_array || other.is_masked_array);
        round(self.value, p) > round(other.value, p)
    }
    /// Compare raw values after operand promotion.
    pub fn less(self, other: Self) -> bool {
        if (self.is_scalar_array && self.is_masked_array && self.mask)
            || (other.is_scalar_array && other.is_masked_array && other.mask)
        {
            return false;
        }
        if self.object_result || other.object_result {
            return objects::compare(self, other, 1).unwrap_or(false);
        }
        if self.complex.is_some() || other.complex.is_some() {
            return complex::compare(self, other, 1).unwrap_or(false);
        }
        if integers::is_integer(self) || integers::is_integer(other) {
            return integers::compare(self, other, 1);
        }
        let p = promote(self, other, self.is_masked_array || other.is_masked_array);
        round(self.value, p) < round(other.value, p)
    }
    /// Compare raw values with >=, retaining masked scalar comparison and NaN semantics.
    pub fn greater_equal(self, other: Self) -> bool {
        if (self.is_scalar_array && self.is_masked_array && self.mask)
            || (other.is_scalar_array && other.is_masked_array && other.mask)
        {
            return false;
        }
        if self.object_result || other.object_result {
            return objects::compare(self, other, 2).unwrap_or(false);
        }
        if self.complex.is_some() || other.complex.is_some() {
            return complex::compare(self, other, 2).unwrap_or(false);
        }
        if integers::is_integer(self) || integers::is_integer(other) {
            return integers::compare(self, other, 2);
        }
        let p = promote(self, other, self.is_masked_array || other.is_masked_array);
        round(self.value, p) >= round(other.value, p)
    }
    /// Select using a scalar Boolean condition; use select_with_condition for array conditions.
    pub fn select(condition: bool, yes: Self, no: Self) -> Self {
        Self::select_with_condition(MaskedCondition::new(condition, true), yes, no)
    }
    /// Compare raw values while retaining the broadcast rank of the comparison result.
    pub fn greater_condition(self, other: Self) -> MaskedCondition {
        let mut condition = MaskedCondition::new(
            self.greater(other),
            self.is_scalar_array && other.is_scalar_array,
        );
        condition.error = self.error.or(other.error);
        condition = condition.with_empty_array(self.empty_array || other.empty_array);
        if condition.error.is_none() {
            condition.error = self.comparison_error(other, 0);
        }
        condition
    }
    /// Compare raw values while retaining the broadcast rank of the comparison result.
    pub fn less_condition(self, other: Self) -> MaskedCondition {
        let mut condition = MaskedCondition::new(
            self.less(other),
            self.is_scalar_array && other.is_scalar_array,
        );
        condition.error = self.error.or(other.error);
        condition = condition.with_empty_array(self.empty_array || other.empty_array);
        if condition.error.is_none() {
            condition.error = self.comparison_error(other, 1);
        }
        condition
    }
    /// Compare raw values while retaining the broadcast rank of the comparison result.
    pub fn greater_equal_condition(self, other: Self) -> MaskedCondition {
        let mut condition = MaskedCondition::new(
            self.greater_equal(other),
            self.is_scalar_array && other.is_scalar_array,
        );
        condition.error = self.error.or(other.error);
        condition = condition.with_empty_array(self.empty_array || other.empty_array);
        if condition.error.is_none() {
            condition.error = self.comparison_error(other, 2);
        }
        condition
    }
    /// Retain deferred comparison failures for checked selection rather than fabricating numeric equality.
    fn comparison_error(self, other: Self, operation: u8) -> Option<MaskedError> {
        if self.empty_array || other.empty_array {
            return None;
        }
        if self.object_result || other.object_result {
            return objects::compare(self, other, operation).err();
        }
        if self.complex.is_some() || other.complex.is_some() {
            return complex::compare(self, other, operation).err();
        }
        if integers::is_integer(self) || integers::is_integer(other) {
            return integers::compare_checked(self, other, operation).err();
        }
        None
    }
    /// Select raw branch data as a plain ndarray, broadcasting the condition and both branches.
    pub fn select_with_condition(condition: MaskedCondition, yes: Self, no: Self) -> Self {
        if let Some(error) = condition.error.or(yes.error).or(no.error) {
            return integers::failure(yes, error);
        }
        if complex::object_branch(yes) || complex::object_branch(no) {
            return objects::select(condition, yes, no);
        }
        if yes.complex.is_some() || no.complex.is_some() {
            return complex::select(condition, yes, no)
                .with_empty_array(condition.empty_array || yes.empty_array || no.empty_array);
        }
        if integers::is_integer(yes) || integers::is_integer(no) {
            return integers::select(condition, yes, no)
                .with_empty_array(condition.empty_array || yes.empty_array || no.empty_array);
        }
        let p = promote(yes, no, false);
        let mut result = Self::new(
            if condition.value { yes.value } else { no.value },
            false,
            false,
            p,
        )
        .with_scalar_array(condition.is_scalar_array && yes.is_scalar_array && no.is_scalar_array)
        .with_math_profile(profile(yes, no));
        result =
            result.with_empty_array(condition.empty_array || yes.empty_array || no.empty_array);
        result.error = condition.error.or(yes.error).or(no.error);
        result
    }
    /// Construct a weak Python float scalar for stage constants and reusable expression composition.
    pub fn scalar(value: f64) -> Self {
        let mut result = Self::plain(value).with_scalar_array(true);
        result.weak = true;
        result
    }
    /// Apply a unary ufunc with the array's current precision.
    fn unary(self, operation: Unary) -> Self {
        if self.error.is_some() {
            return self;
        }
        if self.object_result {
            return objects::unary(self, operation);
        }
        if self.complex.is_some() {
            return complex::unary(self, operation);
        }
        if integers::is_integer(self) {
            return integers::unary(self, operation);
        }
        if self.weak && matches!(operation, Unary::Negative) {
            let mut result = self;
            result.value = -result.value;
            return result;
        }
        // NumPy's domained unary ufunc wrapper on 0D MaskedArrays returns a
        // fresh unmasked 0D view of the raw result (including invalid NaNs).
        if self.is_scalar_array && self.is_masked_array && matches!(operation, Unary::Sqrt) {
            let value = match self.precision {
                Precision::F32 => (self.value as f32).sqrt() as f64,
                Precision::F64 => self.value.sqrt(),
            };
            return Self::new(value, false, true, self.precision)
                .with_scalar_array(true)
                .with_math_profile(self.math_profile);
        }
        let domain = self.is_masked_array
            && matches!(operation, Unary::Sqrt)
            && (self.mask || self.value < 0.0);
        let value = if domain {
            0.0
        } else {
            match self.precision {
                Precision::F32 => {
                    ({
                        let x = self.value as f32;
                        match operation {
                            Unary::Negative => -x,
                            Unary::Exp => match self.math_profile {
                                MathProfile::NumpySimd => numpy_simd_exp_f32(x),
                                MathProfile::Libm => x.exp(),
                            },
                            Unary::Sqrt => x.sqrt(),
                            Unary::Degrees => x * (180.0f32 / std::f32::consts::PI),
                        }
                    }) as f64
                }
                Precision::F64 => match operation {
                    Unary::Negative => -self.value,
                    Unary::Exp => self.value.exp(),
                    Unary::Sqrt => self.value.sqrt(),
                    Unary::Degrees => self.value.to_degrees(),
                },
            }
        };
        finish(
            Self::new(
                value,
                self.mask || domain,
                self.is_masked_array,
                self.precision,
            )
            .with_scalar_array(self.is_scalar_array)
            .with_numpy_scalar(self.is_scalar_array && !self.is_masked_array)
            .with_fill_value(self.fill_value)
            .with_math_profile(self.math_profile),
            false,
        )
    }
}

/// Three spread-vector outputs, each retaining its own raw value, mask, class, and dtype.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskedRateVector {
    /// Component along upslope in ft/min.
    pub x_component: MaskedValue,
    /// Component across upslope in ft/min.
    pub y_component: MaskedValue,
    /// Resultant vector magnitude in ft/min.
    pub rate_vector: MaskedValue,
}
impl MaskedRateVector {
    /// Return all vector fields under their exact Python dictionary keys.
    pub fn named_values(&self) -> [(&'static str, MaskedValue); 3] {
        [
            ("x_component", self.x_component),
            ("y_component", self.y_component),
            ("rate_vector", self.rate_vector),
        ]
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}
#[derive(Clone, Copy)]
enum Unary {
    Negative,
    Exp,
    Sqrt,
    Degrees,
}
#[derive(Clone, Copy)]
enum BinaryUfunc {
    Minimum,
    Maximum,
    Atan2,
}

/// A NumPy mask expression that is not accepted by the public Python stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum MaskedError {
    /// Zero-dimensional masked comparisons become float64 MaskedConstants, which cannot be combined with bitwise AND.
    InvalidBooleanAnd,
    /// A weak Python integer does not fit the resolved NumPy integer dtype.
    IntegerOutOfBounds,
    /// NumPy does not define subtraction between Boolean operands.
    BooleanSubtract,
    /// NumPy does not define unary negative for Boolean operands.
    BooleanNegative,
    /// Integer power has a negative integer exponent.
    NegativeIntegerPower,
    /// Ordinary Python scalar division or zero power divides by zero.
    PythonZeroDivision,
    /// The source operation would require an arbitrary Python object integer outside the declared fixed-width boundary.
    UnsupportedIntegerObject,
    /// Ordinary Python real/complex power or integer-to-float conversion overflowed.
    PythonFloatOverflow,
    /// This generated complex operation has no real-numeric NumPy loop.
    UnsupportedComplexOperation,
    /// A masked object scalar arithmetic loop produced a builtin result without ndarray attributes.
    PythonObjectAttribute,
    /// NumPy has no supported numeric payload method/loop for this object operation.
    UnsupportedObjectLoop,
}
impl MaskedError {
    /// Return the Python exception class selected by the thin binding adapter.
    pub fn python_exception(self) -> &'static str {
        match self {
            Self::IntegerOutOfBounds | Self::PythonFloatOverflow => "OverflowError",
            Self::BooleanSubtract
            | Self::BooleanNegative
            | Self::InvalidBooleanAnd
            | Self::UnsupportedIntegerObject
            | Self::UnsupportedComplexOperation
            | Self::UnsupportedObjectLoop => "TypeError",
            Self::NegativeIntegerPower => "ValueError",
            Self::PythonZeroDivision => "ZeroDivisionError",
            Self::PythonObjectAttribute => "AttributeError",
        }
    }
}
/// Validate the inputs used in a source `comparison & comparison` expression.
///
/// Adapters map this error to TypeError, matching NumPy, before calling a stage.
pub fn validate_condition_and(operands: &[MaskedValue]) -> Result<(), MaskedError> {
    if let Some(error) = operands.iter().find_map(|x| x.error) {
        return Err(error);
    }
    if operands
        .iter()
        .any(|x| x.is_scalar_array && x.is_masked_array && x.mask)
    {
        Err(MaskedError::InvalidBooleanAnd)
    } else {
        Ok(())
    }
}
/// Implement NumPy's scalar masked singleton and scalar arithmetic subclass normalization.
fn finish(mut value: MaskedValue, arithmetic: bool) -> MaskedValue {
    if value.error.is_some() {
        return value;
    }
    if value.is_masked_array && value.is_scalar_array {
        if value.mask {
            return MaskedValue::masked(0.0, true)
                .with_scalar_array(true)
                .with_math_profile(value.math_profile);
        }
        if arithmetic {
            value.is_masked_array = false;
            value.fill_value = None;
            value.is_numpy_scalar = true;
        }
    }
    value
}

/// Round one transport value to the selected scalar arithmetic precision.
fn round(value: f64, precision: Precision) -> f64 {
    match precision {
        Precision::F32 => (value as f32) as f64,
        Precision::F64 => value,
    }
}

/// Evaluate NumPy's float32 range-reduced rational exponential using portable scalar FMA.
///
/// NumPy 2.1's SIMD float32 exponential is not always the correctly rounded
/// libc result. Preserve that intermediate value before later f64 promotion.
/// Formula/constants: numpy/_core/src/umath/{loops_exponent_log.dispatch.c.src,npy_simd_data.h}, v2.1.3.
#[allow(clippy::excessive_precision)] // Preserve the published NumPy coefficients.
pub fn numpy_simd_exp_f32(input: f32) -> f32 {
    if input.is_nan() {
        return f32::NAN;
    }
    if input >= 88.72283935546875f32 {
        return f32::INFINITY;
    }
    if input <= -103.97208404541015625f32 {
        return 0.0;
    }
    let quadrant = (input * std::f32::consts::LOG2_E + 12582912.0f32) - 12582912.0f32;
    let x = quadrant.mul_add(-6.93145752e-1f32, input);
    let x = quadrant.mul_add(-1.42860677e-6f32, x);
    let numerator = 5.082762527590694e-4f32
        .mul_add(x, 6.757896990527505e-3f32)
        .mul_add(x, 5.114512081637298e-2f32)
        .mul_add(x, 2.473615434895521e-1f32)
        .mul_add(x, 7.257664613233124e-1f32)
        .mul_add(x, 9.999999999980871e-1f32);
    let denominator = 2.15950937568583e-2f32
        .mul_add(x, -2.742335390411667e-1f32)
        .mul_add(x, 1.0f32);
    let poly = numerator / denominator;
    let exponent = quadrant as i32;
    if exponent <= -125 {
        let scaled = f32::from_bits(poly.to_bits().wrapping_add(((-125i32) << 23) as u32));
        scaled / ((1u32 << (-125 - exponent)) as f32)
    } else {
        f32::from_bits(poly.to_bits().wrapping_add((exponent << 23) as u32))
    }
}
/// Propagate the actual array operand's math profile across weak Python constants.
fn profile(a: MaskedValue, b: MaskedValue) -> MathProfile {
    if a.weak && !b.weak {
        b.math_profile
    } else {
        a.math_profile
    }
}
/// Promote array operands; MaskedArray arithmetic converts Python scalar constants to strong float64 operands.
fn promote(a: MaskedValue, b: MaskedValue, strong_constants: bool) -> Precision {
    if !strong_constants && a.weak && !b.weak {
        b.precision
    } else if !strong_constants && b.weak && !a.weak {
        a.precision
    } else if a.precision == Precision::F64 || b.precision == Precision::F64 {
        Precision::F64
    } else {
        Precision::F32
    }
}
/// Evaluate arithmetic independently of NumPy, including operator-specific mask/domain restoration.
fn arithmetic(a: MaskedValue, b: MaskedValue, operation: Operation) -> MaskedValue {
    if let Some(error) = a.error.or(b.error) {
        return integers::failure(a, error);
    }
    if a.object_result || b.object_result {
        return objects::arithmetic(a, b, operation);
    }
    if a.complex.is_some() || b.complex.is_some() {
        return complex::arithmetic(a, b, operation);
    }
    if a.weak
        && b.weak
        && ((matches!(operation, Operation::Div) && b.value == 0.0)
            || (matches!(operation, Operation::Pow) && a.value == 0.0 && b.value < 0.0))
    {
        return integers::failure(a, MaskedError::PythonZeroDivision);
    }
    if integers::is_integer(a) || integers::is_integer(b) {
        return integers::arithmetic(a, b, operation);
    }
    arithmetic_float(a, b, operation)
}
/// Evaluate the already promoted floating loop without altering its existing numerical operation order.
fn arithmetic_float(a: MaskedValue, b: MaskedValue, operation: Operation) -> MaskedValue {
    if let Some(error) = a.error.or(b.error) {
        return integers::failure(a, error);
    }
    if matches!(operation, Operation::Pow) {
        if let Some(result) = complex::real_power(a, b) {
            return result;
        }
    }
    let ma = a.is_masked_array || b.is_masked_array;
    let ufunc = a.is_numpy_scalar && b.is_masked_array;
    let p = promote(a, b, ma);
    let x = round(a.value, p);
    let y = round(b.value, p);
    let raw = match p {
        Precision::F32 => {
            ({
                let x = x as f32;
                let y = y as f32;
                match operation {
                    Operation::Add => x + y,
                    Operation::Sub => x - y,
                    Operation::Mul => x * y,
                    Operation::Div => x / y,
                    Operation::Pow => x.powf(y),
                }
            }) as f64
        }
        Precision::F64 => match operation {
            Operation::Add => x + y,
            Operation::Sub => x - y,
            Operation::Mul => x * y,
            Operation::Div => x / y,
            Operation::Pow => x.powf(y),
        },
    };
    let mut mask = a.mask || b.mask;
    if a.weak
        && b.weak
        && matches!(operation, Operation::Pow)
        && a.value.is_finite()
        && b.value.is_finite()
        && raw.is_infinite()
    {
        return integers::failure(a, MaskedError::PythonFloatOverflow);
    }
    let mut value = if ma && mask && !ufunc { x } else { raw };
    if ma {
        match operation {
            Operation::Div => {
                let domain = round(x.abs() * f64::MIN_POSITIVE, p) >= y.abs();
                mask |= domain || (!ufunc && !raw.is_finite());
                if ufunc && domain {
                    value = 1.0;
                } else if mask && !ufunc {
                    value = x;
                }
            }
            Operation::Pow if !ufunc && !value.is_finite() => {
                mask = true;
                value = round(a.fill_value.unwrap_or(1e20), p);
            }
            _ => {}
        }
    }
    let fill = if matches!(operation, Operation::Pow) || a.is_masked_array {
        a.fill_value
    } else {
        b.fill_value
    };
    let mut result = MaskedValue::new(value, mask, ma, p)
        .with_scalar_array(a.is_scalar_array && b.is_scalar_array)
        .with_fill_value(fill)
        .with_math_profile(profile(a, b));
    result.weak = a.weak && b.weak;
    result.is_numpy_scalar = result.is_scalar_array && !ma && !result.weak;
    finish(result, !ufunc)
}
/// Evaluate raw ufunc results, with mask propagation rather than operator restoration.
fn ufunc_binary(a: MaskedValue, b: MaskedValue, operation: BinaryUfunc) -> MaskedValue {
    if let Some(error) = a.error.or(b.error) {
        return integers::failure(a, error);
    }
    if a.object_result || b.object_result {
        return objects::ufunc(a, b, operation);
    }
    if a.complex.is_some() || b.complex.is_some() {
        return complex::ufunc(a, b, operation);
    }
    if integers::is_integer(a) || integers::is_integer(b) {
        return integers::ufunc(a, b, operation);
    }
    ufunc_binary_float(a, b, operation)
}
/// Evaluate the already resolved floating ufunc loop.
fn ufunc_binary_float(a: MaskedValue, b: MaskedValue, operation: BinaryUfunc) -> MaskedValue {
    if let Some(error) = a.error.or(b.error) {
        return integers::failure(a, error);
    }
    let p = promote(a, b, false);
    let x = round(a.value, p);
    let y = round(b.value, p);
    let value = match operation {
        BinaryUfunc::Minimum => np_min(x, y),
        BinaryUfunc::Maximum => np_max(x, y),
        BinaryUfunc::Atan2 => match p {
            Precision::F32 => (x as f32).atan2(y as f32) as f64,
            Precision::F64 => x.atan2(y),
        },
    };
    finish(
        MaskedValue::new(
            value,
            a.mask || b.mask,
            a.is_masked_array || b.is_masked_array,
            p,
        )
        .with_scalar_array(a.is_scalar_array && b.is_scalar_array)
        .with_numpy_scalar(
            a.is_scalar_array && b.is_scalar_array && !a.is_masked_array && !b.is_masked_array,
        )
        .with_math_profile(profile(a, b))
        .with_fill_value(if a.is_masked_array {
            a.fill_value
        } else {
            b.fill_value
        }),
        false,
    )
}
/// Construct a weak Python scalar for the translated mathematical expression.
fn c(value: f64) -> MaskedValue {
    MaskedValue::scalar(value)
}

/// Calculate length in feet, stripping rate masks but preserving a masked elapsed-time operand.
pub fn calculate_fire_length(
    forward: MaskedValue,
    backing: MaskedValue,
    elapsed: MaskedValue,
) -> MaskedValue {
    forward.as_array().add(backing.as_array()).mul(elapsed)
}
/// Calculate width in feet with normalized rates/ratio and independently masked elapsed time.
pub fn calculate_fire_width(
    forward: MaskedValue,
    backing: MaskedValue,
    ratio: MaskedValue,
    elapsed: MaskedValue,
) -> MaskedValue {
    let f = forward.as_array_f64();
    let b = backing.as_array_f64();
    let ratio = ratio.as_array_f64();
    let length = f.add(b).mul(elapsed);
    let safe = MaskedValue::select_with_condition(ratio.greater_condition(c(1e-7)), ratio, c(1.0));
    MaskedValue::select_with_condition(ratio.greater_condition(c(1e-7)), length.div(safe), c(0.0))
}
/// Calculate area in square feet, retaining elapsed-time masks through the exact surface/crown expressions.
pub fn calculate_fire_area(
    forward: MaskedValue,
    backing: MaskedValue,
    ratio: MaskedValue,
    elapsed: MaskedValue,
    is_crown: bool,
) -> MaskedValue {
    let f = forward.as_array_f64();
    let b = backing.as_array_f64();
    let ratio = ratio.as_array_f64();
    let safe = MaskedValue::select_with_condition(ratio.greater_condition(c(1e-7)), ratio, c(1.0));
    if is_crown {
        let d = f.mul(elapsed);
        MaskedValue::select_with_condition(
            ratio.greater_condition(c(1e-7)),
            c(std::f64::consts::PI)
                .mul(d.pow(MaskedValue::python_int(2)))
                .div(c(4.0).mul(safe)),
            c(0.0),
        )
    } else {
        let ell_b = f.add(b).div(c(2.0)).mul(elapsed);
        let ell_a = MaskedValue::select_with_condition(
            ratio.greater_condition(c(1e-7)),
            ell_b.div(safe),
            c(0.0),
        );
        c(std::f64::consts::PI).mul(ell_a).mul(ell_b)
    }
}
/// Calculate perimeter in feet using the source ellipse/crown series and elapsed-time mask arithmetic.
pub fn calculate_fire_perimeter(
    forward: MaskedValue,
    backing: MaskedValue,
    ratio: MaskedValue,
    elapsed: MaskedValue,
    is_crown: bool,
) -> MaskedValue {
    let f = forward.as_array_f64();
    let b = backing.as_array_f64();
    let ratio = ratio.as_array_f64();
    let safe = MaskedValue::select_with_condition(ratio.greater_condition(c(1e-7)), ratio, c(1.0));
    if is_crown {
        let d = f.mul(elapsed);
        MaskedValue::select_with_condition(
            ratio.greater_condition(c(1e-7)),
            c(0.5 * std::f64::consts::PI)
                .mul(d)
                .mul(c(1.0).add(c(1.0).div(safe))),
            c(0.0),
        )
    } else {
        let ell_b = f.add(b).div(c(2.0)).mul(elapsed);
        let ell_a = MaskedValue::select_with_condition(
            ratio.greater_condition(c(1e-7)),
            ell_b.div(safe),
            c(0.0),
        );
        let apb = ell_a.add(ell_b);
        let safe2 = MaskedValue::select_with_condition(
            apb.greater_condition(c(1e-7)),
            apb.pow(MaskedValue::python_int(2)),
            c(1.0),
        );
        let h = MaskedValue::select_with_condition(
            apb.greater_condition(c(1e-7)),
            ell_a.sub(ell_b).pow(MaskedValue::python_int(2)).div(safe2),
            c(0.0),
        );
        MaskedValue::select_with_condition(
            apb.greater_condition(c(1e-7)),
            c(std::f64::consts::PI).mul(apb).mul(
                c(1.0)
                    .add(h.div(c(4.0)))
                    .add(h.pow(MaskedValue::python_int(2)).div(c(64.0))),
            ),
            c(0.0),
        )
    }
}

/// Calculate backing spread with NumPy.ma intermediate arithmetic, returning ft/min.
/// Wind coefficient fields with independent object/dtype/empty metadata.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaskedWindCoefficients {
    /// Sigma after the source guard and selection.
    pub safe_sigma: MaskedValue,
    /// Wind coefficient C.
    pub wind_c: MaskedValue,
    /// Wind exponent B.
    pub wind_b: MaskedValue,
    /// Packing exponent E.
    pub wind_e: MaskedValue,
}
impl MaskedWindCoefficients {
    /// Return the exact source dictionary keys.
    pub fn named_values(&self) -> [(&'static str, MaskedValue); 4] {
        [
            ("safe_sigma", self.safe_sigma),
            ("wind_c", self.wind_c),
            ("wind_b", self.wind_b),
            ("wind_e", self.wind_e),
        ]
    }
}
/// Calculate guarded wind coefficients using the same numeric-object and zero-work operations as NumPy.
pub fn calculate_surface_wind_coefficients(sigma: MaskedValue) -> MaskedWindCoefficients {
    let safe_sigma =
        MaskedValue::select_with_condition(sigma.greater_condition(c(1e-7)), sigma, c(1.));
    MaskedWindCoefficients {
        safe_sigma,
        wind_c: c(7.47).mul(c(-0.133).mul(safe_sigma.pow(c(0.55))).exp()),
        wind_b: c(0.02526).mul(safe_sigma.pow(c(0.54))),
        wind_e: c(0.715).mul(c(-0.000359).mul(safe_sigma).exp()),
    }
}
/// Calculate residence time while retaining object-loop boxing and empty-array dtype.
pub fn calculate_residence_time(sigma: MaskedValue) -> MaskedValue {
    let safe = MaskedValue::select_with_condition(sigma.greater_condition(c(1e-7)), sigma, c(1.));
    MaskedValue::select_with_condition(sigma.greater_condition(c(1e-7)), c(384.).div(safe), c(0.))
}
/// Calculate surface flame length after its source np.where normalization, including numeric-object loops.
pub fn calculate_flame_length(intensity: MaskedValue) -> MaskedValue {
    let safe =
        MaskedValue::select_with_condition(intensity.greater_condition(c(0.)), intensity, c(0.));
    MaskedValue::select_with_condition(
        intensity.greater_condition(c(0.)),
        c(0.45).mul(safe.pow(c(0.46))),
        c(0.),
    )
}
/// Calculate backing spread with NumPy.ma intermediate arithmetic, returning ft/min.
pub fn calculate_backing_spread_rate(
    spread_rate: MaskedValue,
    eccentricity: MaskedValue,
) -> MaskedValue {
    let denom = c(1.0).add(eccentricity);
    MaskedValue::select_with_condition(
        denom.greater_condition(c(1e-7)),
        spread_rate.mul(c(1.0).sub(eccentricity)).div(denom),
        c(0.0),
    )
}
/// Calculate eccentricity while preserving masked power, domain-fill, and final `np.where` behavior.
pub fn calculate_eccentricity(ratio: MaskedValue) -> MaskedValue {
    let x = ratio.pow(MaskedValue::python_int(2)).sub(c(1.0));
    MaskedValue::select_with_condition(
        x.greater_condition(c(0.0)),
        x.maximum(c(0.0)).sqrt().div(ratio),
        c(0.0),
    )
}
/// Calculate length-to-width ratio with per-operand masked arithmetic and raw `np.where` selection.
pub fn calculate_fire_length_to_width_ratio(wind: MaskedValue) -> MaskedValue {
    let ratio = c(0.936)
        .mul(c(0.1147).mul(wind).exp())
        .add(c(0.461).mul(c(-0.0692).mul(wind).exp()))
        .sub(c(0.397))
        .minimum(c(8.0));
    MaskedValue::select_with_condition(wind.greater_condition(c(1e-7)), ratio, c(1.0))
}
/// Calculate effective wind in mph, retaining each masked operation before f64 speed normalization.
pub fn calculate_effective_wind_speed(
    spread: MaskedValue,
    r0: MaskedValue,
    b: MaskedValue,
    coefficient: MaskedValue,
    rpr: MaskedValue,
    e: MaskedValue,
) -> MaskedValue {
    let safe_r0 = MaskedValue::select_with_condition(r0.greater_condition(c(1e-7)), r0, c(1.0));
    let phi = MaskedValue::select_with_condition(
        r0.greater_condition(c(1e-7)),
        spread.div(safe_r0).sub(c(1.0)),
        c(0.0),
    );
    let safe_rpr = MaskedValue::select_with_condition(rpr.greater_condition(c(1e-7)), rpr, c(1.0));
    let safe_c = MaskedValue::select_with_condition(
        coefficient.greater_condition(c(1e-7)),
        coefficient,
        c(1.0),
    );
    let safe_b = MaskedValue::select_with_condition(b.greater_condition(c(1e-7)), b, c(1.0));
    let speed = phi.mul(safe_rpr.pow(e)).div(safe_c).pow(c(1.0).div(safe_b));
    let selected = MaskedValue::select_with_condition(
        phi.greater_condition(c(0.0))
            .and(b.greater_condition(c(1e-7)))
            .and(coefficient.greater_condition(c(1e-7)))
            .and(rpr.greater_condition(c(1e-7))),
        speed,
        c(0.0),
    );
    selected.as_array_f64().mul(c(0.01136363636))
}
/// Calculate flanking ft/min with masked addition and division prior to final plain selection.
pub fn calculate_flanking_spread_rate(
    spread: MaskedValue,
    backing: MaskedValue,
    ratio: MaskedValue,
) -> MaskedValue {
    let safe = MaskedValue::select_with_condition(ratio.greater_condition(c(1e-7)), ratio, c(1.0));
    MaskedValue::select_with_condition(
        ratio.greater_condition(c(1e-7)),
        spread.add(backing).div(c(2.0).mul(safe)),
        c(0.0),
    )
}
/// Calculate no-wind ft/min with independent RI, flux, and heat-sink masks.
pub fn calculate_no_wind_no_slope_spread_rate(
    ri: MaskedValue,
    flux: MaskedValue,
    heat_sink: MaskedValue,
) -> MaskedValue {
    let safe =
        MaskedValue::select_with_condition(heat_sink.greater_condition(c(1e-7)), heat_sink, c(1.0));
    MaskedValue::select_with_condition(
        heat_sink.greater_condition(c(1e-7)),
        ri.mul(flux).div(safe),
        c(0.0),
    )
}
/// Calculate wind phi with masked powers/exponents and NumPy's raw conditional data.
pub fn calculate_wind_phi(
    sigma: MaskedValue,
    midflame: MaskedValue,
    b: MaskedValue,
    coefficient: MaskedValue,
    rpr: MaskedValue,
    e: MaskedValue,
) -> MaskedValue {
    let safe = MaskedValue::select_with_condition(rpr.greater_condition(c(1e-7)), rpr, c(1.0));
    let phi = midflame.pow(b).mul(coefficient).mul(safe.pow(e.negative()));
    MaskedValue::select_with_condition(
        sigma
            .greater_condition(c(1e-7))
            .and(midflame.greater_condition(c(1e-7))),
        phi,
        c(0.0),
    )
}
/// Calculate slope phi, stripping slope masks at normalization and retaining reaction masks in the final minimum ufunc.
pub fn calculate_slope_phi(
    slope: MaskedValue,
    units: SlopeUnit,
    packing: MaskedValue,
    ri: MaskedValue,
) -> MaskedValue {
    let slope = slope.as_array_f64();
    if slope.error.is_some() {
        return slope;
    }
    let tan = MaskedValue::plain(units.to_base(slope.value).to_radians().tan())
        .with_scalar_array(slope.is_scalar_array)
        .with_empty_array(slope.empty_array);
    let safe =
        MaskedValue::select_with_condition(packing.greater_condition(c(1e-7)), packing, c(1.0));
    let phi = MaskedValue::select_with_condition(
        tan.greater_condition(c(1e-7))
            .and(packing.greater_condition(c(1e-7))),
        c(5.275)
            .mul(safe.pow(c(-0.3)))
            .mul(tan.pow(MaskedValue::python_int(2))),
        c(0.0),
    );
    phi.minimum(c(0.9).mul(ri))
}
/// Calculate the three vector fields with masked rate operators and normalized direction/aspect arrays.
pub fn calculate_surface_rate_vector(
    r0: MaskedValue,
    wind_phi: MaskedValue,
    slope_phi: MaskedValue,
    direction: MaskedValue,
    orientation: &str,
    aspect: MaskedValue,
) -> MaskedRateVector {
    let direction = direction.as_array_f64();
    let aspect = aspect.as_array_f64();
    if let Some(error) = direction.error.or(aspect.error) {
        let value = integers::failure(direction, error);
        return MaskedRateVector {
            x_component: value,
            y_component: value,
            rate_vector: value,
        };
    }
    let north = super::wind::contains_ascii(orientation, "north");
    let degrees = if north {
        direction.value - aspect.value
    } else {
        direction.value
    };
    let rad = degrees.to_radians();
    let scalar = direction.is_scalar_array && (!north || aspect.is_scalar_array);
    let empty = direction.empty_array || (north && aspect.empty_array);
    let slope = r0.mul(slope_phi);
    let wind = r0.mul(wind_phi);
    let cosine = MaskedValue::plain(rad.cos())
        .with_scalar_array(scalar)
        .with_empty_array(empty);
    let sine = MaskedValue::plain(rad.sin())
        .with_scalar_array(scalar)
        .with_empty_array(empty);
    let x_component = slope.add(wind.mul(cosine));
    let y_component = wind.mul(sine);
    let rate_vector = x_component
        .pow(MaskedValue::python_int(2))
        .add(y_component.pow(MaskedValue::python_int(2)))
        .sqrt();
    MaskedRateVector {
        x_component,
        y_component,
        rate_vector,
    }
}
/// Calculate maximum spread direction, preserving x/y ufunc masks until `np.where` strips them.
pub fn calculate_direction_of_max_spread(
    x: MaskedValue,
    y: MaskedValue,
    rate: MaskedValue,
    orientation: &str,
    aspect: MaskedValue,
) -> MaskedValue {
    let aspect = aspect.as_array_f64();
    if aspect.error.is_some() {
        return aspect;
    }
    let angle = ufunc_binary(y, x, BinaryUfunc::Atan2)
        .unary(Unary::Degrees)
        .with_empty_array(x.empty_array || y.empty_array);
    let mut result =
        MaskedValue::select_with_condition(rate.greater_condition(c(1e-7)), angle, c(0.));
    let snap = MaskedCondition::new(
        result.value.abs() < round(0.5, result.precision),
        result.is_scalar_array,
    )
    .with_empty_array(result.empty_array);
    result = MaskedValue::select_with_condition(snap, c(0.), result);
    result = MaskedValue::select_with_condition(
        result.less_condition(c(-1e-20)),
        result.add(c(360.)),
        result,
    );
    if super::wind::contains_ascii(orientation, "north") {
        result = result.add(aspect).add(c(180.));
        if result.error.is_some() || result.object_result {
            return result;
        }
        let mut value = MaskedValue::plain(np_remainder(result.value, 360.))
            .with_scalar_array(result.is_scalar_array)
            .with_empty_array(result.empty_array)
            .with_math_profile(result.math_profile);
        value.is_numpy_scalar = result.is_scalar_array;
        result = value;
    }
    result
}

/// Run all surface outputs from a masked reaction input and ordinary f64 fuelbed intermediates.
///
/// Each operation retains NumPy.ma behavior until the Python source normalizes
/// or selects its raw data. Final result fields are plain f64 arrays in Python.
pub fn run_surface_fire(
    ri: MaskedValue,
    ib: &super::FuelbedIntermediates,
    input: &super::SurfaceInput,
) -> super::SurfaceResult {
    let plain = MaskedValue::plain;
    let wc = super::calculate_surface_wind_coefficients(ib.sigma);
    let r0 =
        calculate_no_wind_no_slope_spread_rate(ri, plain(ib.propagating_flux), plain(ib.heat_sink));
    let ws = super::calculate_twenty_foot_wind_speed(
        input.wind_speed,
        input.wind_speed_units,
        &input.wind_height_mode,
    );
    let waf = super::calculate_surface_wind_adjustment_factor(
        input.canopy_cover,
        input.canopy_height,
        input.crown_ratio,
        ib.depth,
        &input.waf_method,
        input.user_waf,
    );
    let midflame = super::calculate_midflame_wind_speed(waf, ws);
    let wp = super::calculate_wind_phi(
        ib.sigma,
        midflame,
        wc.wind_b,
        wc.wind_c,
        ib.relative_packing_ratio,
        wc.wind_e,
    );
    let sp = calculate_slope_phi(
        plain(input.slope),
        input.slope_units,
        plain(ib.packing_ratio),
        ri,
    );
    let vector = calculate_surface_rate_vector(
        r0,
        plain(wp),
        sp,
        plain(input.wind_direction),
        &input.wind_orientation_mode,
        plain(input.aspect),
    );
    let spread_rate = super::calculate_forward_spread_rate(r0.value, vector.rate_vector.value);
    let effective_wind_speed = calculate_effective_wind_speed(
        plain(spread_rate),
        r0,
        plain(wc.wind_b),
        plain(wc.wind_c),
        plain(ib.relative_packing_ratio),
        plain(wc.wind_e),
    )
    .value;
    let fire_length_to_width_ratio =
        super::calculate_fire_length_to_width_ratio(effective_wind_speed);
    let eccentricity = super::calculate_eccentricity(fire_length_to_width_ratio);
    let backing_spread_rate = super::calculate_backing_spread_rate(spread_rate, eccentricity);
    let flanking_spread_rate = super::calculate_flanking_spread_rate(
        spread_rate,
        backing_spread_rate,
        fire_length_to_width_ratio,
    );
    let residence_time = super::calculate_residence_time(ib.sigma);
    let heat_per_unit_area = super::calculate_heat_per_unit_area(ri.value, residence_time);
    let fireline_intensity = super::calculate_fireline_intensity(spread_rate, heat_per_unit_area);
    let flame_length = super::calculate_flame_length(fireline_intensity);
    let direction_of_max_spread = calculate_direction_of_max_spread(
        vector.x_component,
        vector.y_component,
        vector.rate_vector,
        &input.wind_orientation_mode,
        plain(input.aspect),
    )
    .value;
    if !ib.is_defined {
        return super::SurfaceResult {
            fire_length_to_width_ratio: 1.0,
            ..super::SurfaceResult::default()
        };
    }
    super::SurfaceResult {
        spread_rate,
        backing_spread_rate,
        flanking_spread_rate,
        flame_length,
        fireline_intensity,
        heat_per_unit_area,
        effective_wind_speed,
        fire_length_to_width_ratio,
        eccentricity,
        direction_of_max_spread,
        residence_time,
        reaction_intensity: ri.value,
        midflame_wind_speed: midflame,
        no_wind_no_slope_spread_rate: r0.value,
    }
}
