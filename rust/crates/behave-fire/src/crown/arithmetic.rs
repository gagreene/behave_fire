//! Numeric operations shared by checked and compact Crown equation evaluation.
use crate::surface::masked::MaskedValue;

/// Arithmetic contract for the Crown equations, including array masks and promotion.
///
/// Implementations must preserve the checked array semantics at every operation.
#[allow(clippy::should_implement_trait)]
pub trait CrownNumber: Copy + Default {
    /// Construct a strong float64 array value.
    fn plain(value: f64) -> Self;
    /// Construct a weak scalar expression constant.
    fn scalar(value: f64) -> Self;
    /// Read the raw numeric value after promotion.
    fn value(self) -> f64;
    /// Add with source promotion and mask restoration.
    fn add(self, other: Self) -> Self;
    /// Subtract with source promotion and mask restoration.
    fn sub(self, other: Self) -> Self;
    /// Multiply with source promotion and mask restoration.
    fn mul(self, other: Self) -> Self;
    /// Divide with source promotion and domain masks.
    fn div(self, other: Self) -> Self;
    /// Raise to a power with source promotion and domain masks.
    fn pow(self, other: Self) -> Self;
    /// Apply the maximum ufunc, retaining raw masked results.
    fn maximum(self, other: Self) -> Self;
    /// Apply the minimum ufunc, retaining raw masked results.
    fn minimum(self, other: Self) -> Self;
    /// Compare using promoted operands and masked scalar rules.
    fn greater(self, other: Self) -> bool;
    /// Compare using promoted operands and masked scalar rules.
    fn less(self, other: Self) -> bool;
    /// Clip the unmasked fraction to [0, 1], retaining zero signs on inclusive bounds.
    fn clip_unit(self) -> Self;
    /// Select promoted raw data while dropping masks.
    fn select(condition: bool, yes: Self, no: Self) -> Self;
}

impl CrownNumber for MaskedValue {
    fn plain(value: f64) -> Self {
        Self::plain(value)
    }
    fn scalar(value: f64) -> Self {
        Self::scalar(value)
    }
    fn value(self) -> f64 {
        self.value
    }
    fn add(self, other: Self) -> Self {
        self.add(other)
    }
    fn sub(self, other: Self) -> Self {
        self.sub(other)
    }
    fn mul(self, other: Self) -> Self {
        self.mul(other)
    }
    fn div(self, other: Self) -> Self {
        self.div(other)
    }
    fn pow(self, other: Self) -> Self {
        self.pow(other)
    }
    fn maximum(self, other: Self) -> Self {
        self.maximum(other)
    }
    fn minimum(self, other: Self) -> Self {
        self.minimum(other)
    }
    fn greater(self, other: Self) -> bool {
        self.greater(other)
    }
    fn less(self, other: Self) -> bool {
        self.less(other)
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
        Self::select(condition, yes, no)
    }
}
