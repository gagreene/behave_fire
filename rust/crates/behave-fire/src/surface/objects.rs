//! Native numeric payload loops for supported object arrays. No Python object is retained or executed.
use super::{
    finish, BinaryUfunc, MaskedCondition, MaskedError, MaskedValue, Operation, Precision, Unary,
};

/// Recover the builtin or NumPy scalar payload that NumPy's object loop passes to numeric operators.
fn unbox(value: MaskedValue) -> MaskedValue {
    let numpy = value.object_result && value.object_numpy_scalar;
    let mut result = value;
    result.object_result = false;
    result.object_numpy_scalar = false;
    result.empty_array = false;
    result.mask = false;
    result.is_masked_array = false;
    result.is_scalar_array = true;
    result.fill_value = None;
    result.is_numpy_scalar = numpy;
    result.weak = !numpy;
    if !numpy {
        if let Some(integer) = result.integer {
            let payload = if integer.kind == super::IntegerKind::Bool {
                MaskedValue::python_bool(integer.raw_bits != 0)
            } else if integer.kind.is_signed() {
                MaskedValue::python_int(integer.as_i128() as i64)
            } else {
                MaskedValue::python_uint(integer.raw_bits)
            };
            result.integer = payload.integer;
            result.wide_integer = payload.wide_integer;
            result.value = payload.value;
            result.precision = Precision::F64;
        } else {
            result.precision = Precision::F64;
        }
    }
    result
}
/// Return a typed object container around one native scalar payload.
fn boxed(
    payload: MaskedValue,
    a: MaskedValue,
    b: MaskedValue,
    mask: bool,
    ma: bool,
    force_scalar_box: bool,
) -> MaskedValue {
    let scalar = a.is_scalar_array && b.is_scalar_array;
    let empty = a.empty_array || b.empty_array;
    if scalar && !force_scalar_box {
        if ma {
            if !payload.is_numpy_scalar {
                return super::integers::failure(payload, MaskedError::PythonObjectAttribute);
            }
            let mut result = payload;
            result.is_masked_array = true;
            result.mask = mask;
            return finish(result, true);
        }
        return payload;
    }
    let mut result = payload;
    result.object_result = true;
    result.object_numpy_scalar = payload.is_numpy_scalar;
    result.is_numpy_scalar = false;
    result.weak = false;
    result.is_scalar_array = scalar;
    result.mask = mask;
    result.is_masked_array = ma;
    result.fill_value = if a.is_masked_array {
        a.fill_value
    } else {
        b.fill_value
    };
    result = result.with_empty_array(empty);
    finish(result, false)
}
/// Construct an empty object result without executing a dummy payload operation.
fn empty(a: MaskedValue, b: MaskedValue, ma: bool) -> MaskedValue {
    boxed(MaskedValue::scalar(0.), a, b, false, ma, true).with_empty_array(true)
}
/// Execute object-loop numeric arithmetic and restore the left boxed data where masked.
pub(super) fn arithmetic(a: MaskedValue, b: MaskedValue, operation: Operation) -> MaskedValue {
    let ma = a.is_masked_array || b.is_masked_array;
    let scalar = a.is_scalar_array && b.is_scalar_array;
    if a.empty_array || b.empty_array {
        if ma && matches!(operation, Operation::Div | Operation::Pow) {
            return super::integers::failure(a, MaskedError::UnsupportedObjectLoop);
        }
        return empty(a, b, ma);
    }
    let x = unbox(a);
    let y = unbox(b);
    let computed = super::arithmetic(x, y, operation);
    if computed.error.is_some() {
        return computed;
    }
    if ma && matches!(operation, Operation::Pow) {
        return super::integers::failure(computed, MaskedError::UnsupportedObjectLoop);
    }
    let mask = a.mask || b.mask;
    if ma && matches!(operation, Operation::Div) {
        if !scalar {
            return super::integers::failure(computed, MaskedError::UnsupportedObjectLoop);
        }
        let nonfinite = !computed.value.is_finite()
            || computed
                .complex
                .is_some_and(|x| !x.real.is_finite() || !x.imag.is_finite());
        if mask || nonfinite {
            return MaskedValue::masked(0., true)
                .with_scalar_array(true)
                .with_math_profile(a.math_profile);
        }
        return computed;
    }
    let ufunc = a.is_numpy_scalar && b.is_masked_array;
    if scalar && ma && !ufunc {
        if !computed.is_numpy_scalar {
            return super::integers::failure(computed, MaskedError::PythonObjectAttribute);
        }
        if mask {
            return MaskedValue::masked(0., true)
                .with_scalar_array(true)
                .with_math_profile(a.math_profile);
        }
        return computed;
    }
    let payload = if ma && mask && !ufunc { x } else { computed };
    boxed(payload, a, b, mask, ma, ufunc && ma)
}
/// Compare the actual payloads; zero work never invokes a placeholder's rich comparison.
pub(super) fn compare(a: MaskedValue, b: MaskedValue, operation: u8) -> Result<bool, MaskedError> {
    if a.empty_array || b.empty_array {
        return Ok(false);
    }
    let x = unbox(a);
    let y = unbox(b);
    if let Some(error) = x.error.or(y.error) {
        return Err(error);
    }
    if x.complex.is_some() || y.complex.is_some() {
        return super::complex::compare(x, y, operation);
    }
    if super::integers::is_integer(x) || super::integers::is_integer(y) {
        if x.weak && y.weak {
            return super::integers::compare_python_numeric_checked(x, y, operation);
        }
        return super::integers::compare_checked(x, y, operation);
    }
    Ok(match operation {
        0 => x.value > y.value,
        1 => x.value < y.value,
        _ => x.value >= y.value,
    })
}
/// Preserve object boxing and selected payload type while erasing masks in np.where.
pub(super) fn select(condition: MaskedCondition, yes: MaskedValue, no: MaskedValue) -> MaskedValue {
    let chosen = if condition.value { yes } else { no };
    let payload = unbox(chosen);
    let mut result = payload;
    result.object_result = true;
    result.object_numpy_scalar = payload.is_numpy_scalar;
    result.is_numpy_scalar = false;
    result.weak = false;
    result.is_masked_array = false;
    result.mask = false;
    result.fill_value = None;
    result.is_scalar_array = condition.is_scalar_array && yes.is_scalar_array && no.is_scalar_array;
    result.error = condition.error.or(yes.error).or(no.error);
    result.with_empty_array(condition.empty_array || yes.empty_array || no.empty_array)
}
/// Numeric-object negation uses scalar operators; unsupported object ufunc methods fail only on nonempty data.
pub(super) fn unary(value: MaskedValue, operation: Unary) -> MaskedValue {
    if value.empty_array {
        return empty(value, value, value.is_masked_array);
    }
    if !matches!(operation, Unary::Negative) {
        return super::integers::failure(value, MaskedError::UnsupportedObjectLoop);
    }
    let payload = unbox(value).negative();
    if payload.error.is_some() {
        return payload;
    }
    boxed(
        payload,
        value,
        value,
        value.mask,
        value.is_masked_array,
        value.is_masked_array,
    )
}
/// Object min/max use Python rich comparison, including its non-propagating first-NaN behavior.
pub(super) fn ufunc(a: MaskedValue, b: MaskedValue, operation: BinaryUfunc) -> MaskedValue {
    if a.empty_array || b.empty_array {
        return empty(a, b, a.is_masked_array || b.is_masked_array);
    }
    if matches!(operation, BinaryUfunc::Atan2) {
        return super::integers::failure(a, MaskedError::PythonObjectAttribute);
    }
    let choose = if matches!(operation, BinaryUfunc::Maximum) {
        compare(a, b, 2)
    } else {
        compare(b, a, 2)
    };
    let payload = match choose {
        Ok(true) => unbox(a),
        Ok(false) => unbox(b),
        Err(error) => return super::integers::failure(a, error),
    };
    boxed(
        payload,
        a,
        b,
        a.mask || b.mask,
        a.is_masked_array || b.is_masked_array,
        a.is_masked_array || b.is_masked_array,
    )
}
/// Reduce a leading object-particle axis using exact payload addition instead of fixed-width accumulation.
pub(super) fn sum(values: [MaskedValue; 5], scalar: bool) -> MaskedValue {
    let ma = values.iter().any(|x| x.is_masked_array);
    let mask = ma && values.iter().all(|x| x.mask);
    for value in values {
        if let Some(error) = value.error {
            return super::integers::failure(value, error);
        }
    }
    // NumPy's object add reduction starts with the first payload. MaskedArray
    // first fills masked positions with builtin integer zero, which can change
    // a NumPy Boolean accumulator into an integer accumulator.
    let filled = |value: MaskedValue| {
        if ma && value.mask {
            MaskedValue::python_int(0)
        } else {
            unbox(value)
        }
    };
    let mut result = filled(values[0]);
    for value in &values[1..] {
        result = result.add(filled(*value));
    }
    if result.error.is_some() {
        return result;
    }
    if scalar {
        if mask {
            return MaskedValue::masked(0., true)
                .with_scalar_array(true)
                .with_math_profile(values[0].math_profile);
        }
        return result;
    }
    let mut output = boxed(
        result,
        values[0].with_scalar_array(false),
        values[0].with_scalar_array(false),
        mask,
        ma,
        false,
    );
    output.fill_value = None;
    output
}

#[cfg(test)]
mod tests {
    use super::super::{IntegerKind, IntegerValue, MathProfile};
    use super::*;
    use serde_json::Value;

    fn node(v: &Value, n: usize) -> MaskedValue {
        if v["dtype"] == "object" {
            let p = &v["object_payloads"][n];
            let mut result = node(p, 0).with_object_payload(p["numpy_scalar"] == true);
            result.mask = v["masks"][n].as_bool().unwrap();
            result.is_masked_array = v["masked_array"] == true;
            return result.with_scalar_array(v["scalar_array"] == true);
        }
        let dtype = v["dtype"].as_str().unwrap();
        let bits = v.get("raw_bits").map(|x| x[n].as_u64().unwrap());
        let value = v["values"][n].as_f64().unwrap();
        let mut result = match v["weak_kind"].as_str() {
            Some("int") if dtype.starts_with("uint") => MaskedValue::python_uint(bits.unwrap()),
            Some("int") => MaskedValue::python_int(bits.unwrap() as i64),
            Some("bool") => MaskedValue::python_bool(bits.unwrap() != 0),
            Some("float") => MaskedValue::scalar(value),
            _ => MaskedValue::new(
                value,
                v["masks"][n].as_bool().unwrap(),
                v["masked_array"] == true,
                if dtype == "float32" {
                    Precision::F32
                } else {
                    Precision::F64
                },
            ),
        };
        if v["weak_kind"].is_null() {
            if let Some(kind) = IntegerKind::from_name(dtype) {
                result = result.with_integer(IntegerValue::from_bits(kind, bits.unwrap()));
            }
        }
        result
            .with_scalar_array(v["scalar_array"] == true)
            .with_numpy_scalar(v["numpy_scalar"] == true)
            .with_math_profile(MathProfile::NumpySimd)
    }

    #[test]
    fn object_reduction_preserves_first_payload_and_masked_zero_types() {
        let cases: Value =
            serde_json::from_str(include_str!("../../tests/data/object_reductions.json")).unwrap();
        for (index, case) in cases.as_array().unwrap().iter().enumerate() {
            let input = &case["inputs"]["a"];
            let expected = &case["output"];
            let actual = sum(
                std::array::from_fn(|n| node(input, n)),
                case["options"]["scalar_cell"] == true,
            )
            .validate_output()
            .unwrap();
            let expected = node(expected, 0);
            assert_eq!(
                actual.object_result, expected.object_result,
                "boxing case {index}"
            );
            assert_eq!(
                actual.object_numpy_scalar, expected.object_numpy_scalar,
                "payload class case {index}"
            );
            assert_eq!(
                actual.is_numpy_scalar, expected.is_numpy_scalar,
                "scalar class case {index}"
            );
            assert_eq!(
                actual.is_scalar_array, expected.is_scalar_array,
                "rank case {index}"
            );
            assert_eq!(
                actual.is_masked_array, expected.is_masked_array,
                "array class case {index}"
            );
            assert_eq!(actual.mask, expected.mask, "mask case {index}");
            assert_eq!(
                actual.integer, expected.integer,
                "exact integer case {index}"
            );
            assert_eq!(
                actual.precision, expected.precision,
                "precision case {index}"
            );
            assert!(
                (actual.value - expected.value).abs() < 1e-6,
                "value case {index}"
            );
        }
    }
}
