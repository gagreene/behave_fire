//! Live NumPy 2.1.3/2.5.4 masked-division zero-sign regressions.
use behave_fire::surface::masked::{IntegerKind, IntegerValue, MaskedValue, Precision};

#[test]
fn masked_scalar_division_keeps_the_raw_zero_sign() {
    for precision in [Precision::F32, Precision::F64] {
        for numerator in [0.0_f64, -0.0] {
            for denominator in [2_i64, -2] {
                let a = MaskedValue::new(numerator, false, true, precision).with_scalar_array(true);
                let b = MaskedValue::new(denominator as f64, false, false, precision)
                    .with_scalar_array(true);
                for right in [
                    b,
                    b.with_integer(IntegerValue::from_bits(IntegerKind::I8, denominator as u64)),
                ] {
                    let result = a.div(right);
                    assert_eq!(
                        result.value.is_sign_negative(),
                        numerator.is_sign_negative() != (denominator < 0)
                    );
                }
            }
        }
    }
}

#[test]
fn masked_division_restores_zero_through_numpy_addition() {
    for precision in [Precision::F32, Precision::F64] {
        for mask in [false, true] {
            for numerator in [0.0_f64, -0.0] {
                for denominator in [2_i64, -2] {
                    let a = MaskedValue::new(numerator, mask, true, precision);
                    let b = MaskedValue::new(denominator as f64, false, false, precision);
                    // NumPy adds mask * numerator to the cleared result. Only
                    // unmasked -0 / positive retains a negative zero.
                    let negative = !mask && numerator.is_sign_negative() && denominator > 0;
                    for right in [
                        b,
                        b.with_integer(IntegerValue::from_bits(
                            IntegerKind::I8,
                            denominator as u64,
                        )),
                    ] {
                        let result = a.div(right);
                        assert_eq!(result.value, 0.0);
                        assert_eq!(result.mask, mask);
                        assert_eq!(
                            result.value.is_sign_negative(),
                            negative,
                            "{precision:?}, mask={mask}, {numerator:?}/{denominator}"
                        );
                    }
                }
            }
        }
    }
}
