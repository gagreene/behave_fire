//! Exact fixed-capacity weak-integer magnitudes through the complete finite f64 conversion range.
use std::cmp::Ordering;
use std::ops::Sub;

/// A nonnegative 1024-bit value. All storage is inline and Copy; no heap is allocated per cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Magnitude([u64; 16]);
impl Magnitude {
    /// Construct zero without allocation.
    pub fn zero() -> Self {
        Self([0; 16])
    }
    /// Construct the exact magnitude of a fixed-width or u128 intermediate.
    pub fn from_u128(value: u128) -> Self {
        let mut result = Self::zero();
        result.0[0] = value as u64;
        result.0[1] = (value >> 64) as u64;
        result
    }
    /// Return the low word for a validated narrowing or modulo operation.
    pub fn low(self) -> u64 {
        self.0[0]
    }
    /// Test exact zero.
    pub fn is_zero(self) -> bool {
        self.0.iter().all(|x| *x == 0)
    }
    /// Test the low parity bit without floating conversion.
    pub fn is_odd(self) -> bool {
        self.0[0] & 1 != 0
    }
    /// Add two exact magnitudes, reporting capacity overflow without truncation.
    pub fn checked_add(self, other: Self) -> Option<Self> {
        let mut result = Self::zero();
        let mut carry = 0u128;
        for n in 0..16 {
            let sum = self.0[n] as u128 + other.0[n] as u128 + carry;
            result.0[n] = sum as u64;
            carry = sum >> 64;
        }
        if carry != 0 {
            None
        } else {
            Some(result)
        }
    }
    /// Multiply exactly using an inline double-width temporary, rejecting any nonzero high words.
    pub fn checked_mul(self, other: Self) -> Option<Self> {
        let mut product = [0u64; 32];
        let a = self.0.iter().rposition(|x| *x != 0).map_or(0, |n| n + 1);
        let b = other.0.iter().rposition(|x| *x != 0).map_or(0, |n| n + 1);
        for i in 0..a {
            let mut carry = 0u128;
            for j in 0..b {
                let value = self.0[i] as u128 * other.0[j] as u128 + product[i + j] as u128 + carry;
                product[i + j] = value as u64;
                carry = value >> 64;
            }
            product[i + b] = carry as u64;
        }
        if product[16..].iter().any(|x| *x != 0) {
            None
        } else {
            Some(Self(std::array::from_fn(|n| product[n])))
        }
    }
    /// Exponentiate an exact magnitude, stopping once its capacity is exceeded.
    pub fn checked_pow(self, mut exponent: u32) -> Option<Self> {
        let mut base = self;
        let mut result = Self::from_u128(1);
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result.checked_mul(base)?;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.checked_mul(base)?;
            }
        }
        Some(result)
    }
    /// Return the exact binary length.
    fn bits(self) -> u32 {
        self.0
            .iter()
            .rposition(|x| *x != 0)
            .map_or(0, |n| n as u32 * 64 + 64 - self.0[n].leading_zeros())
    }
    /// Extract the low word after a right shift whose retained result fits one word.
    fn shifted(self, shift: u32) -> u64 {
        let word = (shift / 64) as usize;
        let offset = shift % 64;
        let mut result = self.0[word] >> offset;
        if offset != 0 && word + 1 < 16 {
            result |= self.0[word + 1] << (64 - offset);
        }
        result
    }
    /// Test one exact bit.
    fn bit(self, index: u32) -> bool {
        self.0[(index / 64) as usize] & (1u64 << (index % 64)) != 0
    }
    /// Test sticky bits strictly below an index.
    fn below(self, index: u32) -> bool {
        let word = (index / 64) as usize;
        let offset = index % 64;
        self.0[..word].iter().any(|x| *x != 0)
            || (offset != 0 && self.0[word] & ((1u64 << offset) - 1) != 0)
    }
    /// Round once to a binary significand using guard/sticky ties-to-even.
    fn significand(self, width: u32) -> (u64, u32) {
        let bits = self.bits();
        if bits == 0 {
            return (0, 0);
        }
        if bits <= width {
            return (self.0[0] << (width - bits), bits - 1);
        }
        let shift = bits - width;
        let mut mantissa = self.shifted(shift);
        if self.bit(shift - 1) && (self.below(shift - 1) || mantissa & 1 != 0) {
            mantissa += 1;
        }
        if mantissa == 1u64 << width {
            (mantissa >> 1, bits)
        } else {
            (mantissa, bits - 1)
        }
    }
    /// Convert directly to correctly rounded f64 at the explicit promotion boundary.
    pub fn as_f64(self) -> f64 {
        if self.is_zero() {
            return 0.;
        }
        let (mantissa, exponent) = self.significand(53);
        if exponent > 1023 {
            return f64::INFINITY;
        }
        f64::from_bits(((exponent as u64 + 1023) << 52) | (mantissa & ((1u64 << 52) - 1)))
    }
    /// Convert directly to correctly rounded f32, without double rounding through f64.
    pub fn as_f32(self) -> f32 {
        if self.is_zero() {
            return 0.;
        }
        let (mantissa, exponent) = self.significand(24);
        if exponent > 127 {
            return f32::INFINITY;
        }
        f32::from_bits(((exponent + 127) << 23) | ((mantissa as u32) & ((1u32 << 23) - 1)))
    }
}
impl Ord for Magnitude {
    /// Compare exact high words before low words.
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.iter().rev().cmp(other.0.iter().rev())
    }
}
impl PartialOrd for Magnitude {
    /// Return the exact magnitude order.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq<u128> for Magnitude {
    /// Compare a small literal or fixed-width bound exactly.
    fn eq(&self, other: &u128) -> bool {
        *self == Self::from_u128(*other)
    }
}
impl PartialOrd<u128> for Magnitude {
    /// Compare a fixed-width bound without narrowing this value.
    fn partial_cmp(&self, other: &u128) -> Option<Ordering> {
        Some(self.cmp(&Self::from_u128(*other)))
    }
}
impl Sub for Magnitude {
    type Output = Self;
    /// Subtract an already ordered smaller magnitude without borrowing beyond capacity.
    fn sub(self, other: Self) -> Self {
        let mut result = Self::zero();
        let mut borrow = 0u64;
        for n in 0..16 {
            let (a, b1) = self.0[n].overflowing_sub(other.0[n]);
            let (b, b2) = a.overflowing_sub(borrow);
            result.0[n] = b;
            borrow = u64::from(b1 || b2);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::Magnitude;
    #[test]
    fn exact_capacity_and_python_float_rounding() {
        let two = Magnitude::from_u128(2);
        assert_eq!(
            two.checked_pow(1023).unwrap().as_f64().to_bits(),
            2046u64 << 52
        );
        assert!(two.checked_pow(1024).is_none());
        let cases: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/data/weak_scalar.json")).unwrap();
        let case = cases
            .as_array()
            .unwrap()
            .iter()
            .find(|c| {
                c["name"] == "calculate_wind_phi"
                    && c["inputs"]["midflame_wind_speed"]["values"][0].as_f64() == Some(3.)
                    && c["inputs"]["wind_b"]["values"][0].as_f64() == Some(646.)
                    && c["inputs"]["wind_c"]["values"][0].as_f64() == Some(1.)
            })
            .unwrap();
        assert_eq!(
            Magnitude::from_u128(3)
                .checked_pow(646)
                .unwrap()
                .as_f64()
                .to_bits(),
            case["exact_float_bits"][0].as_u64().unwrap()
        );
    }
}
