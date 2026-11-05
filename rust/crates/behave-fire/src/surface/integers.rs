//! Lossless fixed-width integer metadata and NumPy promotion for the masked scalar engine.
#[path = "integer_magnitude.rs"]
mod magnitude;
use magnitude::Magnitude;

use super::{
    finish, BinaryUfunc, MaskedCondition, MaskedError, MaskedValue, Operation, Precision, Unary,
};

/// Exact NumPy Boolean or fixed-width integer dtype, with stable binding codes.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerKind {
    /// Signed eight-bit integer (binding code 0).
    I8 = 0,
    /// Signed sixteen-bit integer (binding code 1).
    I16 = 1,
    /// Signed thirty-two-bit integer (binding code 2).
    I32 = 2,
    /// Signed sixty-four-bit integer (binding code 3).
    I64 = 3,
    /// Unsigned eight-bit integer (binding code 4).
    U8 = 4,
    /// Unsigned sixteen-bit integer (binding code 5).
    U16 = 5,
    /// Unsigned thirty-two-bit integer (binding code 6).
    U32 = 6,
    /// Unsigned sixty-four-bit integer (binding code 7).
    U64 = 7,
    /// NumPy Boolean (binding code 8).
    Bool = 8,
}
impl IntegerKind {
    /// Return the exact binding metadata code.
    pub fn code(self) -> u8 {
        self as u8
    }
    /// Resolve a declared NumPy dtype name without a runtime NumPy dependency.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "int8" => Some(Self::I8),
            "int16" => Some(Self::I16),
            "int32" => Some(Self::I32),
            "int64" => Some(Self::I64),
            "uint8" => Some(Self::U8),
            "uint16" => Some(Self::U16),
            "uint32" => Some(Self::U32),
            "uint64" => Some(Self::U64),
            "bool" => Some(Self::Bool),
            _ => None,
        }
    }
    /// Return the dtype name used by NumPy and the binding metadata.
    pub fn name(self) -> &'static str {
        match self {
            Self::I8 => "int8",
            Self::I16 => "int16",
            Self::I32 => "int32",
            Self::I64 => "int64",
            Self::U8 => "uint8",
            Self::U16 => "uint16",
            Self::U32 => "uint32",
            Self::U64 => "uint64",
            Self::Bool => "bool",
        }
    }
    /// Return the number of meaningful low bits.
    pub fn bits(self) -> u32 {
        match self {
            Self::Bool => 1,
            Self::I8 | Self::U8 => 8,
            Self::I16 | Self::U16 => 16,
            Self::I32 | Self::U32 => 32,
            Self::I64 | Self::U64 => 64,
        }
    }
    /// Test whether the dtype uses signed two's-complement representation.
    pub fn is_signed(self) -> bool {
        matches!(self, Self::I8 | Self::I16 | Self::I32 | Self::I64)
    }
    /// Return a bit mask for canonical fixed-width output storage.
    pub fn bit_mask(self) -> u64 {
        if self.bits() == 64 {
            u64::MAX
        } else {
            (1u64 << self.bits()) - 1
        }
    }
}

/// One lossless fixed-width value, stored independently of its rounded f64 transport view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerValue {
    /// Original or promoted integer dtype.
    pub kind: IntegerKind,
    /// Meaningful low two's-complement bits; Boolean values are normalized to zero or one.
    pub raw_bits: u64,
}
impl IntegerValue {
    /// Construct a value from exact owned binding bits.
    pub fn from_bits(kind: IntegerKind, raw_bits: u64) -> Self {
        Self {
            kind,
            raw_bits: if kind == IntegerKind::Bool {
                u64::from(raw_bits != 0)
            } else {
                raw_bits & kind.bit_mask()
            },
        }
    }
    /// Construct a signed sixty-four-bit value without float conversion.
    pub fn from_i64(value: i64) -> Self {
        Self::from_bits(IntegerKind::I64, value as u64)
    }
    /// Construct an unsigned sixty-four-bit value without float conversion.
    pub fn from_u64(value: u64) -> Self {
        Self::from_bits(IntegerKind::U64, value)
    }
    /// Return the exact mathematical value in a wider signed representation.
    pub fn as_i128(self) -> i128 {
        if self.kind.is_signed() {
            let shift = 64 - self.kind.bits();
            (((self.raw_bits << shift) as i64) >> shift) as i128
        } else {
            self.raw_bits as i128
        }
    }
    /// Convert directly to f64 at the operation's actual NumPy promotion point.
    pub fn as_f64(self) -> f64 {
        if self.kind.is_signed() {
            self.as_i128() as f64
        } else {
            self.raw_bits as f64
        }
    }
    /// Convert directly to f32, avoiding double rounding through f64.
    pub fn as_f32(self) -> f32 {
        if self.kind.is_signed() {
            self.as_i128() as f32
        } else {
            self.raw_bits as f32
        }
    }
    /// Cast by NumPy's fixed-width low-bit rule, preserving sign extension before narrowing.
    pub fn cast(self, kind: IntegerKind) -> Self {
        Self::from_bits(kind, self.as_i128() as u64)
    }
}

/// Exact intermediate magnitude for ordinary Python integer operations before floating promotion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct WideInteger {
    pub negative: bool,
    magnitude: Magnitude,
    pub exact: bool,
}
impl WideInteger {
    /// Construct an exact wide value from a fixed-width mathematical value.
    pub fn from_integer(value: IntegerValue) -> Self {
        let value = value.as_i128();
        Self {
            negative: value < 0,
            magnitude: Magnitude::from_u128(value.unsigned_abs()),
            exact: true,
        }
    }
    /// Convert directly at the Python integer's float conversion point.
    pub fn as_f64(self) -> f64 {
        if self.negative {
            -self.magnitude.as_f64()
        } else {
            self.magnitude.as_f64()
        }
    }
    /// Infer np.asarray's fixed-width dtype when the value fits the declared real integer boundary.
    pub fn fixed(self) -> Option<IntegerValue> {
        if !self.exact {
            return None;
        }
        if self.negative {
            if self.magnitude <= 1u128 << 63 {
                Some(IntegerValue::from_i64(
                    (-(self.magnitude.low() as i128)) as i64,
                ))
            } else {
                None
            }
        } else if self.magnitude <= i64::MAX as u128 {
            Some(IntegerValue::from_i64(self.magnitude.low() as i64))
        } else if self.magnitude <= u64::MAX as u128 {
            Some(IntegerValue::from_u64(self.magnitude.low()))
        } else {
            None
        }
    }
}

impl MaskedValue {
    /// Attach exact fixed-width input/output data while retaining masks, rank, and math profile.
    pub fn with_integer(mut self, integer: IntegerValue) -> Self {
        let integer = IntegerValue::from_bits(integer.kind, integer.raw_bits);
        if self.integer.map(|value| value.kind) != Some(integer.kind) {
            self.numpy_integer_type = None;
        }
        self.value = integer.as_f64();
        self.integer = Some(integer);
        self.precision = Precision::F64;
        self.wide_integer = None;
        self
    }
    /// Attach an exact NumPy integer type number after fixed-width data normalization.
    pub fn with_numpy_integer_type(mut self, type_number: Option<u8>) -> Self {
        self.numpy_integer_type = type_number.filter(|number| *number <= 10);
        self
    }
    /// Construct an ordinary weak Python signed integer, distinct from NumPy integer scalars.
    pub fn python_int(value: i64) -> Self {
        Self::python_integer(IntegerValue::from_i64(value))
    }
    /// Construct a weak Python nonnegative integer, inferring signed/unsigned ndarray dtype by its value.
    pub fn python_uint(value: u64) -> Self {
        Self::python_integer(if value <= i64::MAX as u64 {
            IntegerValue::from_i64(value as i64)
        } else {
            IntegerValue::from_u64(value)
        })
    }
    /// Construct a weak Python Boolean; ordinary Python Boolean arithmetic differs from NumPy Boolean loops.
    pub fn python_bool(value: bool) -> Self {
        Self::python_integer(IntegerValue::from_bits(IntegerKind::Bool, u64::from(value)))
    }
    /// Initialize weak Python integer metadata without erasing its exact value.
    fn python_integer(integer: IntegerValue) -> Self {
        let mut result = Self::plain(integer.as_f64())
            .with_integer(integer)
            .with_scalar_array(true);
        result.weak = true;
        result.wide_integer = Some(WideInteger::from_integer(integer));
        result
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NumericType {
    Integer(IntegerKind),
    Float(Precision),
}

/// Test whether exact integer or ordinary Python integer metadata is present.
pub(super) fn is_integer(value: MaskedValue) -> bool {
    value.integer.is_some() || value.wide_integer.is_some()
}
/// Select a value's declared dtype before weak-scalar promotion rules apply.
fn scalar_type(value: MaskedValue) -> NumericType {
    if let Some(integer) = value.integer {
        NumericType::Integer(integer.kind)
    } else if value.wide_integer.is_some() {
        NumericType::Integer(IntegerKind::I64)
    } else {
        NumericType::Float(value.precision)
    }
}
/// Promote two fixed integer dtypes using NumPy's signed/unsigned range matrix.
fn integer_type(a: IntegerKind, b: IntegerKind) -> NumericType {
    use IntegerKind::*;
    if a == Bool {
        return NumericType::Integer(b);
    }
    if b == Bool {
        return NumericType::Integer(a);
    }
    if a.is_signed() == b.is_signed() {
        return NumericType::Integer(if a.bits() >= b.bits() { a } else { b });
    }
    let (signed, unsigned) = if a.is_signed() { (a, b) } else { (b, a) };
    if signed.bits() > unsigned.bits() {
        return NumericType::Integer(signed);
    }
    NumericType::Integer(match unsigned.bits() {
        8 => I16,
        16 => I32,
        32 => I64,
        _ => return NumericType::Float(Precision::F64),
    })
}
/// Resolve ordinary dtype promotion without weakening NumPy scalar operands.
fn strong_type(a: NumericType, b: NumericType) -> NumericType {
    match (a, b) {
        (NumericType::Integer(a), NumericType::Integer(b)) => integer_type(a, b),
        (NumericType::Float(a), NumericType::Float(b)) => {
            NumericType::Float(if a == Precision::F64 || b == Precision::F64 {
                Precision::F64
            } else {
                Precision::F32
            })
        }
        (NumericType::Integer(i), NumericType::Float(p))
        | (NumericType::Float(p), NumericType::Integer(i)) => {
            NumericType::Float(if p == Precision::F64 || i.bits() >= 32 {
                Precision::F64
            } else {
                Precision::F32
            })
        }
    }
}
/// Apply NumPy 2 weak Python scalar rules, strengthening constants inside masked arithmetic.
pub(super) fn result_type(a: MaskedValue, b: MaskedValue, strong_constants: bool) -> NumericType {
    let at = scalar_type(a);
    let bt = scalar_type(b);
    if !strong_constants && a.weak != b.weak {
        let (weak, strong) = if a.weak { (at, bt) } else { (bt, at) };
        return match (weak, strong) {
            (_, NumericType::Float(p)) => NumericType::Float(p),
            (NumericType::Float(_), NumericType::Integer(_)) => NumericType::Float(Precision::F64),
            (NumericType::Integer(w), NumericType::Integer(s)) => {
                NumericType::Integer(if s == IntegerKind::Bool && w != IntegerKind::Bool {
                    IntegerKind::I64
                } else {
                    s
                })
            }
        };
    }
    strong_type(at, bt)
}
/// Convert the original integer directly to floating precision at its actual promotion boundary.
pub(super) fn to_float(mut value: MaskedValue, precision: Precision) -> MaskedValue {
    if value
        .wide_integer
        .is_some_and(|wide| !wide.exact || !wide.as_f64().is_finite())
    {
        return failure(value, MaskedError::PythonFloatOverflow);
    }
    value.value = if let Some(integer) = value.integer {
        match precision {
            Precision::F32 => integer.as_f32() as f64,
            Precision::F64 => integer.as_f64(),
        }
    } else if let Some(wide) = value.wide_integer.filter(|x| x.exact) {
        match precision {
            Precision::F32 => {
                let x = wide.magnitude.as_f32();
                (if wide.negative { -x } else { x }) as f64
            }
            Precision::F64 => wide.as_f64(),
        }
    } else {
        match precision {
            Precision::F32 => (value.value as f32) as f64,
            Precision::F64 => value.value,
        }
    };
    value.precision = precision;
    value.integer = None;
    value.numpy_integer_type = None;
    value.wide_integer = None;
    value
}
/// Preserve source exceptions through eager mathematical branches.
pub(super) fn failure(mut value: MaskedValue, error: MaskedError) -> MaskedValue {
    value.error = Some(error);
    value
}
/// Check a weak Python integer against the resolved NumPy integer conversion range.
fn fits(wide: WideInteger, kind: IntegerKind) -> bool {
    if !wide.exact {
        return false;
    }
    if kind == IntegerKind::Bool {
        return !wide.negative && wide.magnitude <= 1;
    }
    if kind.is_signed() {
        let bound = 1u128 << (kind.bits() - 1);
        wide.magnitude < bound || (wide.negative && wide.magnitude == bound)
    } else {
        !wide.negative && wide.magnitude <= kind.bit_mask() as u128
    }
}
/// Cast low bits after an optional weak Python integer bounds check.
fn cast(value: MaskedValue, kind: IntegerKind, checked: bool) -> Result<IntegerValue, MaskedError> {
    let wide = value
        .wide_integer
        .or_else(|| value.integer.map(WideInteger::from_integer))
        .ok_or(MaskedError::UnsupportedIntegerObject)?;
    if checked && !fits(wide, kind) {
        return Err(MaskedError::IntegerOutOfBounds);
    }
    let bits = if wide.negative {
        0u64.wrapping_sub(wide.magnitude.low())
    } else {
        wide.magnitude.low()
    };
    Ok(IntegerValue::from_bits(kind, bits))
}
/// Form an exact ordinary Python integer result, retaining wider magnitudes until float conversion.
fn weak_result(a: MaskedValue, b: MaskedValue, wide: WideInteger) -> MaskedValue {
    let view = if wide.exact {
        wide.as_f64()
    } else if wide.negative {
        -f64::INFINITY
    } else {
        f64::INFINITY
    };
    let mut result = MaskedValue::plain(view)
        .with_scalar_array(true)
        .with_math_profile(super::profile(a, b));
    result.integer = wide.fixed();
    result.wide_integer = Some(wide);
    result.weak = true;
    result
}
/// Evaluate ordinary Python integer operators independently of NumPy's fixed-width loops.
fn weak_arithmetic(a: MaskedValue, b: MaskedValue, operation: Operation) -> MaskedValue {
    let x = a
        .wide_integer
        .or_else(|| a.integer.map(WideInteger::from_integer))
        .unwrap();
    let y = b
        .wide_integer
        .or_else(|| b.integer.map(WideInteger::from_integer))
        .unwrap();
    let small = |negative: bool, magnitude: u128| WideInteger {
        negative,
        magnitude: Magnitude::from_u128(magnitude),
        exact: true,
    };
    if matches!(operation, Operation::Mul)
        && ((x.exact && x.magnitude.is_zero()) || (y.exact && y.magnitude.is_zero()))
    {
        return weak_result(a, b, small(false, 0));
    }
    if matches!(operation, Operation::Pow) && y.exact && y.magnitude.is_zero() {
        return weak_result(a, b, small(false, 1));
    }
    if matches!(operation, Operation::Div) || (matches!(operation, Operation::Pow) && y.negative) {
        if x.exact && x.magnitude.is_zero() && matches!(operation, Operation::Pow)
            || y.exact && y.magnitude.is_zero() && matches!(operation, Operation::Div)
        {
            return failure(a, MaskedError::PythonZeroDivision);
        }
        return super::arithmetic_float(
            to_float(a, Precision::F64),
            to_float(b, Precision::F64),
            operation,
        );
    }
    if matches!(operation, Operation::Pow) && x.exact && (x.magnitude == 0 || x.magnitude == 1) {
        if !y.exact && x.negative {
            return failure(a, MaskedError::UnsupportedIntegerObject);
        }
        return weak_result(
            a,
            b,
            small(
                x.negative && y.magnitude.is_odd(),
                x.magnitude.low() as u128,
            ),
        );
    }
    if !x.exact || !y.exact {
        let negative = match operation {
            Operation::Mul => x.negative ^ y.negative,
            Operation::Pow => {
                if !y.exact && x.negative {
                    return failure(a, MaskedError::UnsupportedIntegerObject);
                }
                x.negative && y.magnitude.is_odd()
            }
            Operation::Add | Operation::Sub => {
                let ys = y.negative ^ matches!(operation, Operation::Sub);
                if y.exact && y.magnitude.is_zero() {
                    return weak_result(a, b, x);
                }
                if x.exact && x.magnitude.is_zero() {
                    let mut value = y;
                    value.negative = ys;
                    return weak_result(a, b, value);
                }
                if x.negative != ys {
                    return failure(a, MaskedError::UnsupportedIntegerObject);
                }
                x.negative
            }
            Operation::Div => unreachable!(),
        };
        return weak_result(
            a,
            b,
            WideInteger {
                negative,
                magnitude: Magnitude::zero(),
                exact: false,
            },
        );
    }
    let (negative, magnitude) = match operation {
        Operation::Add | Operation::Sub => {
            let ys = y.negative ^ matches!(operation, Operation::Sub);
            if x.negative == ys {
                (x.negative, x.magnitude.checked_add(y.magnitude))
            } else if x.magnitude >= y.magnitude {
                (x.negative, Some(x.magnitude - y.magnitude))
            } else {
                (ys, Some(y.magnitude - x.magnitude))
            }
        }
        Operation::Mul => (
            x.negative ^ y.negative,
            x.magnitude.checked_mul(y.magnitude),
        ),
        Operation::Pow => (
            x.negative && y.magnitude.is_odd(),
            if y.magnitude <= u32::MAX as u128 {
                x.magnitude.checked_pow(y.magnitude.low() as u32)
            } else {
                None
            },
        ),
        Operation::Div => unreachable!(),
    };
    let exact = magnitude.is_some();
    let magnitude = magnitude.unwrap_or_else(Magnitude::zero);
    weak_result(
        a,
        b,
        WideInteger {
            negative: negative && (!exact || !magnitude.is_zero()),
            magnitude,
            exact,
        },
    )
}
/// Resolve an exact integer alias after the ordinary numeric promotion has selected its width.
fn result_numpy_integer_type(
    a: MaskedValue,
    b: MaskedValue,
    kind: IntegerKind,
    scalar_arithmetic: bool,
) -> Option<u8> {
    let matching = |value: MaskedValue| {
        if !value.weak && value.integer.is_some_and(|integer| integer.kind == kind) {
            value.numpy_integer_type
        } else {
            None
        }
    };
    let left = matching(a);
    let right = matching(b);
    // NumPy numeric scalar slots preserve the left scalar class when the other
    // scalar fits its numeric kind; ndarray/ufunc promotion instead resolves
    // equal-width aliases by their common dtype descriptor.
    if scalar_arithmetic && a.is_numpy_scalar && (b.is_numpy_scalar || b.weak) && left.is_some() {
        return left;
    }
    if scalar_arithmetic && a.weak && b.is_numpy_scalar && right.is_some() {
        return right;
    }
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(number), None) | (None, Some(number)) => Some(number),
        (None, None) => None,
    }
}
/// Evaluate a fixed-width arithmetic loop before any float conversion.
pub(super) fn arithmetic(a: MaskedValue, b: MaskedValue, operation: Operation) -> MaskedValue {
    let ma = a.is_masked_array || b.is_masked_array;
    if ma
        && ((a.weak && a.wide_integer.is_some() && a.integer.is_none())
            || (b.weak && b.wide_integer.is_some() && b.integer.is_none()))
    {
        let mut x = a;
        let mut y = b;
        if x.weak && x.wide_integer.is_some() && x.integer.is_none() {
            x.object_result = true;
        }
        if y.weak && y.wide_integer.is_some() && y.integer.is_none() {
            y.object_result = true;
        }
        return super::objects::arithmetic(x, y, operation);
    }
    let ufunc = a.is_numpy_scalar && b.is_masked_array;
    if a.weak && b.weak && is_integer(a) && is_integer(b) && !ma {
        return weak_arithmetic(a, b, operation);
    }
    let mut target = result_type(a, b, ma);
    if matches!(operation, Operation::Div) {
        target = NumericType::Float(match target {
            NumericType::Float(p) => p,
            _ => Precision::F64,
        });
    }
    if matches!(operation, Operation::Pow) {
        if target == NumericType::Integer(IntegerKind::Bool) {
            target = NumericType::Integer(IntegerKind::I8);
        }
        if !ma
            && ((a.integer.is_some_and(|x| x.kind == IntegerKind::Bool) && b.weak && is_integer(b))
                || (b.integer.is_some_and(|x| x.kind == IntegerKind::Bool)
                    && a.weak
                    && is_integer(a)))
        {
            target = NumericType::Integer(IntegerKind::I8);
        }
    }
    let kind = match target {
        NumericType::Float(p) => {
            return if ma && matches!(operation, Operation::Div) {
                masked_divide(a, b, p)
            } else {
                super::arithmetic_float(to_float(a, p), to_float(b, p), operation)
            }
        }
        NumericType::Integer(k) => k,
    };
    let x = match cast(a, kind, a.weak && !ma) {
        Ok(x) => x,
        Err(e) => return failure(a, e),
    };
    let y = match cast(b, kind, b.weak && !ma) {
        Ok(x) => x,
        Err(e) => return failure(b, e),
    };
    if matches!(operation, Operation::Sub) && kind == IntegerKind::Bool {
        return failure(a, MaskedError::BooleanSubtract);
    }
    if matches!(operation, Operation::Pow)
        && kind.is_signed()
        && y.as_i128() < 0
        && !a.empty_array
        && !b.empty_array
    {
        return failure(a, MaskedError::NegativeIntegerPower);
    }
    let raw = match operation {
        Operation::Add => {
            if kind == IntegerKind::Bool {
                u64::from(x.raw_bits != 0 || y.raw_bits != 0)
            } else {
                x.raw_bits.wrapping_add(y.raw_bits)
            }
        }
        Operation::Sub => x.raw_bits.wrapping_sub(y.raw_bits),
        Operation::Mul => {
            if kind == IntegerKind::Bool {
                u64::from(x.raw_bits != 0 && y.raw_bits != 0)
            } else {
                x.raw_bits.wrapping_mul(y.raw_bits)
            }
        }
        Operation::Pow => {
            let mut exponent = y.raw_bits;
            let mut base = x.raw_bits;
            let mut result = 1u64;
            while exponent != 0 {
                if exponent & 1 != 0 {
                    result = result.wrapping_mul(base);
                }
                base = base.wrapping_mul(base);
                exponent >>= 1;
            }
            result
        }
        Operation::Div => unreachable!(),
    };
    let mask = a.mask || b.mask;
    let integer = if ma && mask && !ufunc {
        x
    } else {
        IntegerValue::from_bits(kind, raw)
    };
    let fill = if matches!(operation, Operation::Pow) && !ufunc || a.is_masked_array {
        a.fill_value
    } else {
        b.fill_value
    };
    let result = MaskedValue::new(integer.as_f64(), mask, ma, Precision::F64)
        .with_integer(integer)
        .with_numpy_integer_type(result_numpy_integer_type(a, b, kind, true))
        .with_scalar_array(a.is_scalar_array && b.is_scalar_array)
        .with_numpy_scalar(a.is_scalar_array && b.is_scalar_array && !ma)
        .with_fill_value(fill)
        .with_math_profile(super::profile(a, b));
    finish(result, !ufunc)
}
/// Evaluate NumPy's domain check in the original integer dtype, where abs(signed_min) wraps.
fn original_absolute(value: MaskedValue) -> f64 {
    if let Some(integer) = value.integer {
        if integer.kind.is_signed() && integer.as_i128() < 0 {
            IntegerValue::from_bits(integer.kind, 0u64.wrapping_sub(integer.raw_bits)).as_f64()
        } else {
            integer.as_f64()
        }
    } else {
        value.value.abs()
    }
}
/// Preserve original-dtype domain masks before the true-divide floating loop.
fn masked_divide(a: MaskedValue, b: MaskedValue, p: Precision) -> MaskedValue {
    let x = to_float(a, p).value;
    let y = to_float(b, p).value;
    let raw = match p {
        Precision::F32 => ((x as f32) / (y as f32)) as f64,
        Precision::F64 => x / y,
    };
    let domain = original_absolute(a) * f64::MIN_POSITIVE >= original_absolute(b);
    let ufunc = a.is_numpy_scalar && b.is_masked_array;
    let mask = a.mask || b.mask || domain || (!ufunc && !raw.is_finite());
    let value = if ufunc && domain {
        1.0
    } else if mask && !ufunc {
        x
    } else {
        raw
    };
    let result = MaskedValue::new(value, mask, true, p)
        .with_scalar_array(a.is_scalar_array && b.is_scalar_array)
        .with_fill_value(if a.is_masked_array {
            a.fill_value
        } else {
            b.fill_value
        })
        .with_math_profile(super::profile(a, b));
    finish(result, !ufunc)
}
/// Preserve integer values in np.where while applying its unsafe scalar branch cast.
pub(super) fn select(condition: MaskedCondition, yes: MaskedValue, no: MaskedValue) -> MaskedValue {
    let kind = match result_type(yes, no, false) {
        NumericType::Float(p) => {
            return MaskedValue::select_with_condition(condition, to_float(yes, p), to_float(no, p))
        }
        NumericType::Integer(k) => k,
    };
    let selected = if condition.value { yes } else { no };
    let integer = match cast(selected, kind, false) {
        Ok(x) => x,
        Err(e) => return failure(selected, e),
    };
    let mut result = MaskedValue::plain(integer.as_f64())
        .with_integer(integer)
        .with_numpy_integer_type(result_numpy_integer_type(yes, no, kind, false))
        .with_scalar_array(condition.is_scalar_array && yes.is_scalar_array && no.is_scalar_array)
        .with_math_profile(super::profile(yes, no));
    result.error = condition.error.or(yes.error).or(no.error);
    result
}
/// Compare exact integers before promotion, including weak values outside a narrow operand's range.
pub(super) fn compare(a: MaskedValue, b: MaskedValue, operation: u8) -> bool {
    compare_checked(a, b, operation).unwrap_or(false)
}
/// Compare builtin numeric object payloads without rounding an integer before Python rich comparison.
pub(super) fn compare_python_numeric_checked(
    a: MaskedValue,
    b: MaskedValue,
    operation: u8,
) -> Result<bool, MaskedError> {
    let ai = is_integer(a);
    let bi = is_integer(b);
    if ai == bi {
        return compare_checked(a, b, operation);
    }
    let (integer, float) = if ai { (a, b.value) } else { (b, a.value) };
    if float.is_nan() {
        return Ok(false);
    }
    let wide = integer
        .wide_integer
        .or_else(|| integer.integer.map(WideInteger::from_integer))
        .unwrap();
    let order = if float == f64::INFINITY {
        std::cmp::Ordering::Less
    } else if float == f64::NEG_INFINITY {
        std::cmp::Ordering::Greater
    } else {
        let float_negative = float < 0.;
        if wide.negative != float_negative {
            if wide.negative {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        } else {
            let bits = float.abs().to_bits();
            let exponent = ((bits >> 52) & 0x7ff) as i32 - 1023;
            let significand = (bits & ((1u64 << 52) - 1)) | (1u64 << 52);
            let (floor, fractional) = if exponent < 0 {
                (Magnitude::zero(), float != 0.)
            } else if exponent < 52 {
                let shift = (52 - exponent) as u32;
                (
                    Magnitude::from_u128((significand >> shift) as u128),
                    significand & ((1u64 << shift) - 1) != 0,
                )
            } else {
                (
                    Magnitude::from_u128(significand as u128)
                        .checked_mul(
                            Magnitude::from_u128(2)
                                .checked_pow((exponent - 52) as u32)
                                .unwrap(),
                        )
                        .unwrap(),
                    false,
                )
            };
            let mut magnitude_order = if wide.exact {
                wide.magnitude.cmp(&floor)
            } else {
                std::cmp::Ordering::Greater
            };
            if magnitude_order.is_eq() && fractional {
                magnitude_order = std::cmp::Ordering::Less;
            }
            if wide.negative {
                magnitude_order.reverse()
            } else {
                magnitude_order
            }
        }
    };
    let order = if ai { order } else { order.reverse() };
    Ok(match operation {
        0 => order.is_gt(),
        1 => order.is_lt(),
        _ => !order.is_lt(),
    })
}
/// Compare tracked exact/overflow magnitudes without inventing equality for unknown values.
pub(super) fn compare_checked(
    a: MaskedValue,
    b: MaskedValue,
    operation: u8,
) -> Result<bool, MaskedError> {
    let ma = a.is_masked_array || b.is_masked_array;
    let direct = is_integer(a)
        && is_integer(b)
        && (!ma && (a.weak || b.weak) || matches!(result_type(a, b, ma), NumericType::Integer(_)));
    if direct {
        let x = a
            .wide_integer
            .or_else(|| a.integer.map(WideInteger::from_integer))
            .unwrap();
        let y = b
            .wide_integer
            .or_else(|| b.integer.map(WideInteger::from_integer))
            .unwrap();
        let magnitude_order = match (x.exact, y.exact) {
            (true, true) => x.magnitude.cmp(&y.magnitude),
            (false, true) => std::cmp::Ordering::Greater,
            (true, false) => std::cmp::Ordering::Less,
            (false, false) => {
                if x.negative == y.negative {
                    return Err(MaskedError::UnsupportedIntegerObject);
                }
                std::cmp::Ordering::Equal
            }
        };
        let order = if x.negative != y.negative {
            if x.negative {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        } else if x.negative {
            magnitude_order.reverse()
        } else {
            magnitude_order
        };
        return Ok(match operation {
            0 => order.is_gt(),
            1 => order.is_lt(),
            _ => !order.is_lt(),
        });
    }
    let p = match result_type(a, b, ma) {
        NumericType::Float(p) => p,
        _ => Precision::F64,
    };
    let x = to_float(a, p);
    let y = to_float(b, p);
    if let Some(error) = x.error.or(y.error) {
        return Err(error);
    }
    Ok(match operation {
        0 => x.value > y.value,
        1 => x.value < y.value,
        _ => x.value >= y.value,
    })
}
/// Resolve the real unary/binary floating loop, widening would-be float16 loops to float32 by policy.
fn floating_loop(a: MaskedValue, b: Option<MaskedValue>) -> Precision {
    let one = |x: MaskedValue| match scalar_type(x) {
        NumericType::Float(p) => p,
        NumericType::Integer(i) => {
            if i.bits() >= 32 {
                Precision::F64
            } else {
                Precision::F32
            }
        }
    };
    if let Some(b) = b {
        if a.weak != b.weak {
            let (weak, strong) = if a.weak { (a, b) } else { (b, a) };
            if is_integer(weak) {
                if let Some(integer) = strong.integer {
                    if integer.kind == IntegerKind::Bool
                        && weak.integer.is_some_and(|x| x.kind != IntegerKind::Bool)
                    {
                        return Precision::F64;
                    }
                    return one(strong);
                }
            }
        }
        if (a.weak || b.weak)
            && matches!(result_type(a, b, false), NumericType::Float(Precision::F32))
        {
            return Precision::F32;
        }
        if one(a) == Precision::F64 || one(b) == Precision::F64 {
            Precision::F64
        } else {
            Precision::F32
        }
    } else {
        one(a)
    }
}
/// Evaluate integer unary negative exactly, or promote only the requested real ufunc loop.
pub(super) fn unary(value: MaskedValue, operation: Unary) -> MaskedValue {
    if !matches!(operation, Unary::Negative) {
        return to_float(value, floating_loop(value, None)).unary(operation);
    }
    if value.weak {
        let mut wide = value
            .wide_integer
            .or_else(|| value.integer.map(WideInteger::from_integer))
            .unwrap();
        wide.negative = !wide.negative && (!wide.exact || !wide.magnitude.is_zero());
        return weak_result(value, value, wide);
    }
    let integer = value.integer.unwrap();
    if integer.kind == IntegerKind::Bool {
        return failure(value, MaskedError::BooleanNegative);
    }
    let mut result = value.with_integer(IntegerValue::from_bits(
        integer.kind,
        0u64.wrapping_sub(integer.raw_bits),
    ));
    result.is_numpy_scalar = value.is_scalar_array && !value.is_masked_array;
    finish(result, false)
}
/// Evaluate integer min/max or the normalized real atan2 loop without losing integer bits prematurely.
pub(super) fn ufunc(a: MaskedValue, b: MaskedValue, operation: BinaryUfunc) -> MaskedValue {
    if matches!(operation, BinaryUfunc::Atan2) {
        let p = floating_loop(a, Some(b));
        return super::ufunc_binary_float(to_float(a, p), to_float(b, p), operation);
    }
    let kind = match result_type(a, b, false) {
        NumericType::Float(p) => {
            return super::ufunc_binary_float(to_float(a, p), to_float(b, p), operation)
        }
        NumericType::Integer(k) => k,
    };
    let x = match cast(a, kind, a.weak) {
        Ok(x) => x,
        Err(e) => return failure(a, e),
    };
    let y = match cast(b, kind, b.weak) {
        Ok(x) => x,
        Err(e) => return failure(b, e),
    };
    let choose_x = if matches!(operation, BinaryUfunc::Minimum) {
        x.as_i128() < y.as_i128()
    } else {
        x.as_i128() > y.as_i128()
    };
    let integer = if choose_x { x } else { y };
    let ma = a.is_masked_array || b.is_masked_array;
    finish(
        MaskedValue::new(integer.as_f64(), a.mask || b.mask, ma, Precision::F64)
            .with_integer(integer)
            .with_numpy_integer_type(result_numpy_integer_type(a, b, kind, false))
            .with_scalar_array(a.is_scalar_array && b.is_scalar_array)
            .with_numpy_scalar(a.is_scalar_array && b.is_scalar_array && !ma)
            .with_fill_value(if a.is_masked_array {
                a.fill_value
            } else {
                b.fill_value
            })
            .with_math_profile(super::profile(a, b)),
        false,
    )
}

/// Sum a leading particle axis using NumPy's default widened integer accumulator and wrapping order.
pub(super) fn sum(values: [MaskedValue; 5], scalar: bool) -> MaskedValue {
    let input = match values.iter().find_map(|x| x.integer) {
        Some(x) => x.kind,
        None => return failure(values[0], MaskedError::UnsupportedIntegerObject),
    };
    let kind = if input.is_signed() || input == IntegerKind::Bool {
        if usize::BITS == 64 {
            IntegerKind::I64
        } else {
            IntegerKind::I32
        }
    } else {
        if usize::BITS == 64 {
            IntegerKind::U64
        } else {
            IntegerKind::U32
        }
    };
    let ma = values.iter().any(|x| x.is_masked_array);
    let mask = ma && values.iter().all(|x| x.mask);
    let mut bits = 0u64;
    for value in values {
        if let Some(error) = value.error {
            return failure(value, error);
        }
        if !ma || !value.mask {
            let integer = match cast(value, kind, false) {
                Ok(x) => x,
                Err(error) => return failure(value, error),
            };
            bits = bits.wrapping_add(integer.raw_bits);
        }
    }
    let integer = IntegerValue::from_bits(kind, bits);
    finish(
        MaskedValue::new(integer.as_f64(), mask, ma, Precision::F64)
            .with_integer(integer)
            .with_numpy_integer_type(
                values
                    .iter()
                    .find(|value| value.integer.is_some_and(|integer| integer.kind == kind))
                    .and_then(|value| value.numpy_integer_type),
            )
            .with_scalar_array(scalar)
            .with_numpy_scalar(scalar && !ma)
            .with_math_profile(values[0].math_profile),
        true,
    )
}
