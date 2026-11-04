//! Scalar operations with NumPy-compatible NaN and signed-zero behavior.

macro_rules! numpy_math {
    ($float:ty) => {
        /// Clip a value, propagating NaN and applying the upper bound last.
        pub fn np_clip(value: $float, lower: $float, upper: $float) -> $float {
            np_min(np_max(value, lower), upper)
        }

        /// Maximum that propagates either NaN and selects the second value on ties.
        pub fn np_max(left: $float, right: $float) -> $float {
            if left.is_nan() || right.is_nan() {
                <$float>::NAN
            } else if left > right {
                left
            } else {
                right
            }
        }

        /// Minimum that propagates either NaN and selects the second value on ties.
        pub fn np_min(left: $float, right: $float) -> $float {
            if left.is_nan() || right.is_nan() {
                <$float>::NAN
            } else if left < right {
                left
            } else {
                right
            }
        }

        /// NumPy/Python remainder, with the divisor's sign rather than the dividend's.
        pub fn np_remainder(dividend: $float, divisor: $float) -> $float {
            let remainder = dividend % divisor;
            if remainder.is_nan() {
                remainder
            } else if remainder == 0.0 {
                (0.0 as $float).copysign(divisor)
            } else if remainder.is_sign_negative() != divisor.is_sign_negative() {
                remainder + divisor
            } else {
                remainder
            }
        }
    };
}

numpy_math!(f64);

/// Single-precision variants for public helpers that retain float32 arithmetic.
pub mod single {
    numpy_math!(f32);
}
