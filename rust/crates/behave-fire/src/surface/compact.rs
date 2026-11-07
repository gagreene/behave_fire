//! Small ordinary numeric values; exceptional payloads remain on the checked path.
use super::{IntegerValue, MaskedError, MaskedValue, MathProfile, Precision};
use crate::crown::arithmetic::CrownNumber;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CompactValue {
    value: f64,
    fill: Option<f64>,
    integer: Option<IntegerValue>,
    integer_type: Option<u8>,
    error: Option<MaskedError>,
    precision: Precision,
    profile: MathProfile,
    mask: bool,
    masked: bool,
    scalar: bool,
    numpy_scalar: bool,
    weak: bool,
    unsupported: bool,
}

impl Default for CompactValue {
    fn default() -> Self {
        Self::plain(0.0)
    }
}

#[derive(Clone, Copy)]
enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl CompactValue {
    pub(crate) fn from_checked(value: MaskedValue) -> Option<Self> {
        let result = Self::from_result(value);
        (!result.unsupported && !value.weak && value.error.is_none()).then_some(result)
    }

    fn from_result(value: MaskedValue) -> Self {
        Self {
            value: value.value,
            fill: value.fill_value,
            integer: value.integer,
            integer_type: value.numpy_integer_type,
            error: value.error,
            precision: value.precision,
            profile: value.math_profile,
            mask: value.mask,
            masked: value.is_masked_array,
            scalar: value.is_scalar_array,
            numpy_scalar: value.is_numpy_scalar,
            weak: value.weak,
            unsupported: value.complex.is_some()
                || value.object_result
                || value.object_numpy_scalar
                || value.wide_integer.is_some()
                || value.empty_array,
        }
    }

    pub(crate) fn checked(self) -> MaskedValue {
        MaskedValue {
            value: self.value,
            mask: self.mask,
            is_masked_array: self.masked,
            precision: self.precision,
            is_scalar_array: self.scalar,
            is_numpy_scalar: self.numpy_scalar,
            fill_value: self.fill,
            math_profile: self.profile,
            integer: self.integer,
            numpy_integer_type: self.integer_type,
            error: self.error,
            complex: None,
            object_result: false,
            object_numpy_scalar: false,
            empty_array: false,
            wide_integer: None,
            weak: self.weak,
        }
    }

    pub(crate) fn unsupported(self) -> bool {
        self.unsupported
    }

    fn floating(self, other: Self) -> bool {
        self.precision == Precision::F64
            && other.precision == Precision::F64
            && self.integer.is_none()
            && other.integer.is_none()
            && self.error.is_none()
            && other.error.is_none()
    }

    fn profile(self, other: Self) -> MathProfile {
        if self.weak && !other.weak {
            other.profile
        } else {
            self.profile
        }
    }

    fn binary(self, other: Self, operation: Binary) -> Self {
        if !self.floating(other)
            || self.numpy_scalar
            || (self.scalar && other.scalar)
            || (self.weak && other.weak)
        {
            let (a, b) = (self.checked(), other.checked());
            let value = match operation {
                Binary::Add => a.add(b),
                Binary::Sub => a.sub(b),
                Binary::Mul => a.mul(b),
                Binary::Div => a.div(b),
                Binary::Pow => a.pow(b),
            };
            let mut result = Self::from_result(value);
            result.unsupported |= self.unsupported || other.unsupported;
            return result;
        }
        let (x, y) = (self.value, other.value);
        let raw = match operation {
            Binary::Add => x + y,
            Binary::Sub => x - y,
            Binary::Mul => x * y,
            Binary::Div => x / y,
            Binary::Pow => x.powf(y),
        };
        let masked = self.masked || other.masked;
        let mut mask = self.mask || other.mask;
        let mut value = if masked && mask { x } else { raw };
        if masked {
            match operation {
                Binary::Div => {
                    mask |= x.abs() * f64::MIN_POSITIVE >= y.abs() || !raw.is_finite();
                    if mask {
                        value = x;
                    }
                    if value == 0.0 {
                        value = if mask { 0.0 + x } else { raw + 0.0 * x };
                    }
                }
                Binary::Pow if !value.is_finite() => {
                    mask = true;
                    value = self.fill.unwrap_or(1e20);
                }
                _ => {}
            }
        }
        Self {
            value,
            mask,
            masked,
            fill: if matches!(operation, Binary::Pow) || self.masked {
                self.fill
            } else {
                other.fill
            },
            profile: self.profile(other),
            unsupported: self.unsupported || other.unsupported,
            ..Self::plain(0.0)
        }
    }

    fn extreme(self, other: Self, maximum: bool) -> Self {
        if !self.floating(other) || (self.scalar && other.scalar) {
            let value = if maximum {
                self.checked().maximum(other.checked())
            } else {
                self.checked().minimum(other.checked())
            };
            let mut result = Self::from_result(value);
            result.unsupported |= self.unsupported || other.unsupported;
            return result;
        }
        Self {
            value: if maximum {
                crate::math::np_max(self.value, other.value)
            } else {
                crate::math::np_min(self.value, other.value)
            },
            mask: self.mask || other.mask,
            masked: self.masked || other.masked,
            fill: if self.masked { self.fill } else { other.fill },
            profile: self.profile(other),
            unsupported: self.unsupported || other.unsupported,
            ..Self::plain(0.0)
        }
    }
}

impl CrownNumber for CompactValue {
    fn plain(value: f64) -> Self {
        Self::from_result(MaskedValue::plain(value))
    }
    fn scalar(value: f64) -> Self {
        Self::from_result(MaskedValue::scalar(value))
    }
    fn value(self) -> f64 {
        self.value
    }
    fn add(self, other: Self) -> Self {
        self.binary(other, Binary::Add)
    }
    fn sub(self, other: Self) -> Self {
        self.binary(other, Binary::Sub)
    }
    fn mul(self, other: Self) -> Self {
        self.binary(other, Binary::Mul)
    }
    fn div(self, other: Self) -> Self {
        self.binary(other, Binary::Div)
    }
    fn pow(self, other: Self) -> Self {
        self.binary(other, Binary::Pow)
    }
    fn maximum(self, other: Self) -> Self {
        self.extreme(other, true)
    }
    fn minimum(self, other: Self) -> Self {
        self.extreme(other, false)
    }
    fn greater(self, other: Self) -> bool {
        if !self.floating(other) {
            return self.checked().greater(other.checked());
        }
        !(self.scalar && self.masked && self.mask || other.scalar && other.masked && other.mask)
            && self.value > other.value
    }
    fn less(self, other: Self) -> bool {
        if !self.floating(other) {
            return self.checked().less(other.checked());
        }
        !(self.scalar && self.masked && self.mask || other.scalar && other.masked && other.mask)
            && self.value < other.value
    }
    fn clip_unit(self) -> Self {
        let mut result = self.maximum(Self::scalar(0.0)).minimum(Self::scalar(1.0));
        // np.clip keeps the input on an equal bound, unlike maximum/minimum ufuncs.
        if self.value == 0.0 && result.value == 0.0 {
            result.value = self.value;
        }
        result
    }
    fn select(condition: bool, yes: Self, no: Self) -> Self {
        if !yes.floating(no) {
            let mut result =
                Self::from_result(MaskedValue::select(condition, yes.checked(), no.checked()));
            result.unsupported |= yes.unsupported || no.unsupported;
            return result;
        }
        Self {
            value: if condition { yes.value } else { no.value },
            scalar: yes.scalar && no.scalar,
            profile: yes.profile(no),
            unsupported: yes.unsupported || no.unsupported,
            ..Self::plain(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn ordinary_payload_is_smaller_than_checked_arithmetic() {
        assert!(std::mem::size_of::<super::CompactValue>() <= 80);
        assert!(
            std::mem::size_of::<super::CompactValue>() < std::mem::size_of::<super::MaskedValue>()
        );
    }
}
