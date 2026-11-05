use behave_fire::{
    crown,
    surface::masked::{self, IntegerKind, IntegerValue, MaskedValue, MathProfile, Precision},
};
use serde_json::Value;

#[test]
fn beyond_range_comparisons_are_checked_without_placeholder_equality() {
    let exact = MaskedValue::python_int(2).pow(MaskedValue::python_int(200));
    let positive = exact.greater_condition(MaskedValue::python_int(0));
    assert!(positive.value);
    assert!(positive.error.is_none());
    let huge = MaskedValue::python_int(2).pow(MaskedValue::python_int(2000));
    let positive = huge.greater_condition(MaskedValue::python_int(0));
    assert!(positive.value);
    assert!(positive.error.is_none());
    let other = MaskedValue::python_int(3).pow(MaskedValue::python_int(2000));
    let ambiguous = huge.greater_condition(other);
    assert_eq!(
        ambiguous.error,
        Some(masked::MaskedError::UnsupportedIntegerObject)
    );
    assert_eq!(
        MaskedValue::select_with_condition(
            ambiguous,
            MaskedValue::scalar(1.),
            MaskedValue::scalar(0.)
        )
        .validate_output()
        .err(),
        Some(masked::MaskedError::UnsupportedIntegerObject)
    );
}

#[test]
fn wide_integer_zero_annihilation_precedes_real_conversion() {
    let huge = MaskedValue::python_int(2).pow(MaskedValue::python_int(2000));
    let zero = huge
        .mul(MaskedValue::python_int(0))
        .validate_output()
        .unwrap();
    assert_eq!(zero.integer.unwrap().raw_bits, 0);
    assert_eq!(
        huge.mul(MaskedValue::scalar(0.)).validate_output().err(),
        Some(masked::MaskedError::PythonFloatOverflow)
    );
    let object = MaskedValue::select(false, huge, MaskedValue::scalar(0.));
    assert!(object.object_result);
    assert!(object.validate_output().is_ok());
    assert_eq!(object.as_array_f64().validate_output().unwrap().value, 0.);
}

fn number(v: &Value) -> f64 {
    if let Some(x) = v.as_f64() {
        x
    } else {
        match v.as_str().unwrap() {
            "nan" => f64::NAN,
            "+inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            x => panic!("{x}"),
        }
    }
}
fn arg(v: &Value, n: usize) -> MaskedValue {
    let n = n % v["values"].as_array().unwrap().len();
    let value = number(&v["values"][n]);
    let kind = v["dtype"].as_str().unwrap();
    let bits = v.get("raw_bits").map(|x| x[n].as_u64().unwrap());
    let mut result = match v["weak_kind"].as_str() {
        Some("float") => MaskedValue::scalar(value),
        Some("bool") => MaskedValue::python_bool(bits.unwrap() != 0),
        Some("int") if kind.starts_with("uint") => MaskedValue::python_uint(bits.unwrap()),
        Some("int") => MaskedValue::python_int(bits.unwrap() as i64),
        _ => MaskedValue::new(
            value,
            v["masks"][n].as_bool().unwrap(),
            v["masked_array"].as_bool().unwrap(),
            if kind == "float32" {
                Precision::F32
            } else {
                Precision::F64
            },
        ),
    };
    if v["weak_kind"].is_null() {
        if let Some(bits) = bits {
            result = result.with_integer(IntegerValue::from_bits(
                IntegerKind::from_name(kind).unwrap(),
                bits,
            ));
        }
    }
    result
        .with_scalar_array(v["scalar_array"].as_bool().unwrap())
        .with_numpy_scalar(v["numpy_scalar"].as_bool().unwrap())
        .with_fill_value(v.get("fill_value").filter(|x| !x.is_null()).map(number))
        .with_math_profile(MathProfile::NumpySimd)
}
fn invoke(name: &str, i: &Value, opts: &Value, n: usize) -> Vec<(&'static str, MaskedValue)> {
    let a = |k| arg(&i[k], n);
    let value = match name {
        "pow" => a("a").pow(a("b")),
        "pow_mul" => a("a").pow(a("b")).mul(a("c")),
        "pow_rmul" => a("c").mul(a("a").pow(a("b"))),
        "pow_where" => MaskedValue::select_with_condition(
            masked::MaskedCondition::new(
                a("condition").value != 0.,
                a("condition").is_scalar_array,
            ),
            a("a").pow(a("b")).mul(a("c")),
            MaskedValue::scalar(0.),
        ),
        "calculate_backing_spread_rate" => {
            masked::calculate_backing_spread_rate(a("spread_rate"), a("eccentricity"))
        }
        "calculate_eccentricity" => masked::calculate_eccentricity(a("fire_length_to_width_ratio")),
        "calculate_wind_phi" => masked::calculate_wind_phi(
            a("sigma"),
            a("midflame_wind_speed"),
            a("wind_b"),
            a("wind_c"),
            a("relative_packing_ratio"),
            a("wind_e"),
        ),
        "calculate_crown_fraction_burned" => crown::masked::calculate_crown_fraction_burned(
            a("surface_ros"),
            a("surface_critical_ros"),
            a("crowning_surface_ros"),
        ),
        "calculate_passive_crown_fire_behavior" => {
            return crown::masked::calculate_passive_crown_fire_behavior(
                a("surface_ros"),
                a("surface_hpua"),
                a("crown_ros"),
                a("canopy_hpua"),
                a("cfb"),
            )
            .named_values()
            .to_vec()
        }
        "assign_final_crown_fire_behavior" => {
            return crown::masked::assign_final_crown_fire_behavior(
                opts["is_surface_fire"].as_bool().unwrap(),
                opts["is_passive_crown"].as_bool().unwrap(),
                a("surface_ros"),
                a("surface_hpua"),
                a("surface_fli"),
                a("passive_ros"),
                a("passive_hpua"),
                a("passive_fli"),
                a("passive_fl"),
                a("crown_ros"),
                a("crown_hpua"),
                a("crown_fli"),
                a("crown_fl"),
            )
            .named_values()
            .to_vec()
        }
        _ => panic!("{name}"),
    };
    vec![("", value)]
}
fn near(a: f64, b: f64, p: Precision, context: &str) {
    if a == 0.0 && b == 0.0 {
        assert_eq!(
            a.is_sign_negative(),
            b.is_sign_negative(),
            "zero sign {context}"
        );
    }
    if b.is_nan() {
        assert!(a.is_nan(), "{context}: {a} != NaN");
    } else if b.is_infinite() {
        assert_eq!(a, b, "{context}");
    } else {
        let tol = if p == Precision::F32 { 1e-6 } else { 1e-9 };
        assert!(
            (a - b).abs() <= 1e-12 + tol * b.abs(),
            "{context}: {a:.17e} != {b:.17e}"
        );
    }
}
fn check(actual: MaskedValue, e: &Value, n: usize, context: &str) {
    let n = n % e["values"].as_array().unwrap().len();
    let dtype = e["dtype"].as_str().unwrap();
    assert!(actual.error.is_none(), "{context}: {:?}", actual.error);
    assert_eq!(
        actual.mask,
        e["masks"][n].as_bool().unwrap(),
        "mask {context}"
    );
    assert_eq!(
        actual.is_masked_array,
        e["masked_array"].as_bool().unwrap(),
        "array class {context}"
    );
    assert_eq!(
        actual.is_scalar_array,
        e["scalar_array"].as_bool().unwrap(),
        "rank {context}"
    );
    assert_eq!(
        actual.is_numpy_scalar,
        e["numpy_scalar"].as_bool().unwrap(),
        "numpy scalar {context}"
    );
    if dtype.starts_with("complex") {
        let value = actual
            .complex
            .unwrap_or_else(|| panic!("missing {dtype} metadata: {context}"));
        assert_eq!(
            actual.precision,
            if dtype == "complex64" {
                Precision::F32
            } else {
                Precision::F64
            },
            "complex dtype {context}"
        );
        near(
            value.real,
            number(&e["values"][n]),
            actual.precision,
            context,
        );
        near(
            value.imag,
            number(&e["imaginary"][n]),
            actual.precision,
            context,
        );
    } else if let Some(bits) = e.get("raw_bits") {
        let value = actual
            .integer
            .unwrap_or_else(|| panic!("missing {dtype} metadata: {context}"));
        assert_eq!(value.kind.name(), dtype, "dtype {context}");
        assert_eq!(
            value.raw_bits,
            IntegerValue::from_bits(value.kind, bits[n].as_u64().unwrap()).raw_bits,
            "bits {context}"
        );
    } else {
        assert!(
            actual.complex.is_none(),
            "unexpected complex output: {context}"
        );
        assert!(
            actual.integer.is_none(),
            "unexpected integer output: {context}"
        );
        assert_eq!(
            actual.precision,
            if dtype == "float32" {
                Precision::F32
            } else {
                Precision::F64
            },
            "dtype {context}"
        );
        near(
            actual.value,
            number(&e["values"][n]),
            actual.precision,
            context,
        );
    }
}

#[test]
fn weak_scalar_errors_complex_dtype_and_exact_wide_conversion_match_python() {
    let cases: Value = serde_json::from_str(include_str!("data/weak_scalar.json")).unwrap();
    for (index, c) in cases.as_array().unwrap().iter().enumerate() {
        let name = c["name"].as_str().unwrap();
        let cells = c["inputs"]
            .as_object()
            .unwrap()
            .values()
            .map(|x| x["values"].as_array().unwrap().len())
            .max()
            .unwrap();
        for n in 0..cells {
            let results = invoke(name, &c["inputs"], &c["options"], n);
            let values: Vec<MaskedValue> = results.iter().map(|(_, x)| *x).collect();
            let checked = MaskedValue::validate_outputs(&values);
            if let Some(error) = c
                .get("exception")
                .or_else(|| c.get("native_policy_error"))
                .and_then(Value::as_str)
            {
                assert_eq!(
                    checked.err().map(|e| e.python_exception()),
                    Some(error),
                    "case {index}/{name}"
                );
                continue;
            }
            checked.unwrap_or_else(|e| panic!("unexpected {e:?}: case {index}/{name}"));
            for (key, value) in results {
                let expected = if key.is_empty() {
                    &c["output"]
                } else {
                    &c["output"][key]
                };
                check(
                    value,
                    expected,
                    n,
                    &format!("case {index}/{name}/{key}/{n}"),
                );
                if let Some(bits) = c.get("exact_float_bits") {
                    assert_eq!(
                        value.value.to_bits(),
                        bits[n].as_u64().unwrap(),
                        "exact wide conversion case {index}/{name}"
                    );
                }
            }
        }
    }
}
