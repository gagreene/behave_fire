#[path = "support/engine_fixture.rs"]
mod fixture;
use behave_fire::surface::masked::{
    self, IntegerKind, IntegerValue, MaskedValue, MathProfile, Precision,
};
use serde_json::Value;

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
    let len = v["values"].as_array().unwrap().len();
    let empty = len == 0;
    let n = if empty { 0 } else { n % len };
    if v["dtype"] == "object" {
        let mut value = if empty {
            MaskedValue::scalar(0.)
        } else {
            arg(&v["object_payloads"][n], 0)
        };
        value.mask = if empty {
            false
        } else {
            v["masks"][n].as_bool().unwrap()
        };
        value.is_masked_array = v["masked_array"].as_bool().unwrap();
        return value
            .with_object_payload(!empty && v["object_payloads"][n]["numpy_scalar"] == true)
            .with_scalar_array(v["scalar_array"].as_bool().unwrap())
            .with_empty_array(empty)
            .with_math_profile(MathProfile::NumpySimd);
    }
    let kind = v["dtype"].as_str().unwrap();
    let value = if empty { 0. } else { number(&v["values"][n]) };
    let bits = v
        .get("raw_bits")
        .map(|x| if empty { 0 } else { x[n].as_u64().unwrap() });
    let mut result = match v["weak_kind"].as_str() {
        Some("float") => MaskedValue::scalar(value),
        Some("bool") => MaskedValue::python_bool(bits.unwrap() != 0),
        Some("int") if kind.starts_with("uint") => MaskedValue::python_uint(bits.unwrap()),
        Some("int") => MaskedValue::python_int(bits.unwrap() as i64),
        _ => MaskedValue::new(
            value,
            if empty {
                false
            } else {
                v["masks"][n].as_bool().unwrap()
            },
            v["masked_array"].as_bool().unwrap(),
            if kind == "float32" {
                Precision::F32
            } else {
                Precision::F64
            },
        ),
    };
    if v["weak_kind"].is_null() {
        if let Some(kind) = IntegerKind::from_name(kind) {
            result = result.with_integer(IntegerValue::from_bits(kind, bits.unwrap_or(0)));
        }
    }
    result
        .with_scalar_array(v["scalar_array"].as_bool().unwrap())
        .with_numpy_scalar(v["numpy_scalar"].as_bool().unwrap())
        .with_empty_array(empty)
        .with_math_profile(MathProfile::NumpySimd)
}
fn invoke(name: &str, i: &Value, opts: &Value, n: usize) -> Vec<(&'static str, MaskedValue)> {
    let a = |k| arg(&i[k], n);
    let result = match name {
        "add" => a("a").add(a("b")),
        "sub" => a("a").sub(a("b")),
        "mul" => a("a").mul(a("b")),
        "div" => a("a").div(a("b")),
        "pow" => a("a").pow(a("b")),
        "negative" => a("a").negative(),
        "minimum" => a("a").minimum(a("b")),
        "maximum" => a("a").maximum(a("b")),
        "sqrt" => a("a").sqrt(),
        "exp" => a("a").exp(),
        "calculate_fire_length" => {
            masked::calculate_fire_length(a("forward_ros"), a("backing_ros"), a("elapsed_min"))
        }
        "calculate_flanking_spread_rate" => masked::calculate_flanking_spread_rate(
            a("spread_rate"),
            a("backing_spread_rate"),
            a("fire_length_to_width_ratio"),
        ),
        "calculate_no_wind_no_slope_spread_rate" => masked::calculate_no_wind_no_slope_spread_rate(
            a("ri"),
            a("propagating_flux"),
            a("heat_sink"),
        ),
        "calculate_backing_spread_rate" => {
            masked::calculate_backing_spread_rate(a("spread_rate"), a("eccentricity"))
        }
        "calculate_eccentricity" => masked::calculate_eccentricity(a("fire_length_to_width_ratio")),
        "calculate_fire_length_to_width_ratio" => {
            masked::calculate_fire_length_to_width_ratio(a("effective_wind_speed"))
        }
        "calculate_flame_length" => masked::calculate_flame_length(a("fireline_intensity")),
        "calculate_residence_time" => masked::calculate_residence_time(a("sigma")),
        "calculate_surface_wind_coefficients" => {
            return masked::calculate_surface_wind_coefficients(a("sigma"))
                .named_values()
                .to_vec()
        }
        "calculate_direction_of_max_spread" => masked::calculate_direction_of_max_spread(
            a("x_component"),
            a("y_component"),
            a("rate_vector"),
            fixture::orientation(opts["wind_orientation_mode"].as_str().unwrap()),
            a("aspect"),
        ),
        "calculate_wind_phi" => masked::calculate_wind_phi(
            a("sigma"),
            a("midflame_wind_speed"),
            a("wind_b"),
            a("wind_c"),
            a("relative_packing_ratio"),
            a("wind_e"),
        ),
        "calculate_effective_wind_speed" => masked::calculate_effective_wind_speed(
            a("spread_rate"),
            a("no_wind_no_slope_spread_rate"),
            a("wind_b"),
            a("wind_c"),
            a("relative_packing_ratio"),
            a("wind_e"),
        ),
        "calculate_slope_phi" => masked::calculate_slope_phi(
            a("slope"),
            behave_fire::units::SlopeUnit::try_from(0).unwrap(),
            a("packing_ratio"),
            a("reaction_intensity"),
        ),
        "calculate_surface_rate_vector" => {
            let v = masked::calculate_surface_rate_vector(
                a("no_wind_no_slope_spread_rate"),
                a("wind_phi"),
                a("slope_phi"),
                a("wind_direction"),
                fixture::orientation(opts["wind_orientation_mode"].as_str().unwrap()),
                a("aspect"),
            );
            return vec![
                ("x_component", v.x_component),
                ("y_component", v.y_component),
                ("rate_vector", v.rate_vector),
            ];
        }
        _ => panic!("{name}"),
    };
    vec![("", result)]
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
fn payload(actual: MaskedValue, e: &Value, n: usize, context: &str) {
    let n = n % e["values"].as_array().unwrap().len();
    let dtype = e["dtype"].as_str().unwrap();
    if let Some(bits) = e.get("raw_bits") {
        let x = actual
            .integer
            .unwrap_or_else(|| panic!("missing integer {context}"));
        assert_eq!(x.kind.name(), dtype, "integer dtype {context}");
        assert_eq!(
            x.raw_bits,
            IntegerValue::from_bits(x.kind, bits[n].as_u64().unwrap()).raw_bits,
            "integer bits {context}"
        );
    } else if dtype.starts_with("complex") {
        assert_eq!(
            actual.precision,
            if dtype == "complex64" {
                Precision::F32
            } else {
                Precision::F64
            },
            "complex precision {context}"
        );
        let x = actual.complex.unwrap();
        near(x.real, number(&e["values"][n]), actual.precision, context);
        near(
            x.imag,
            number(&e["imaginary"][n]),
            actual.precision,
            context,
        );
    } else {
        assert!(actual.integer.is_none(), "unexpected integer {context}");
        assert!(actual.complex.is_none(), "unexpected complex {context}");
        assert_eq!(
            actual.precision,
            if dtype == "float32" {
                Precision::F32
            } else {
                Precision::F64
            },
            "precision {context}"
        );
        near(
            actual.value,
            number(&e["values"][n]),
            actual.precision,
            context,
        );
    }
}
fn check(actual: MaskedValue, e: &Value, n: usize, context: &str) {
    let empty = e["values"].as_array().unwrap().is_empty();
    assert_eq!(actual.empty_array, empty, "empty metadata {context}");
    assert_eq!(
        actual.is_scalar_array,
        e["scalar_array"].as_bool().unwrap(),
        "rank {context}"
    );
    assert_eq!(
        actual.is_masked_array,
        e["masked_array"].as_bool().unwrap(),
        "mask class {context}"
    );
    if e["dtype"] == "object" {
        assert!(actual.object_result, "missing object boxing {context}");
        if !empty {
            let p = &e["object_payloads"][n % e["values"].as_array().unwrap().len()];
            assert_eq!(
                actual.object_numpy_scalar,
                p["numpy_scalar"].as_bool().unwrap(),
                "boxed scalar kind {context}"
            );
            payload(actual, p, 0, context);
        }
    } else {
        assert!(!actual.object_result, "unexpected object boxing {context}");
        if !empty {
            payload(actual, e, n, context);
            assert_eq!(
                actual.is_numpy_scalar,
                e["numpy_scalar"].as_bool().unwrap(),
                "numpy scalar {context}"
            );
        }
    }
    if !empty {
        assert_eq!(
            actual.mask,
            e["masks"][n % e["values"].as_array().unwrap().len()]
                .as_bool()
                .unwrap(),
            "mask value {context}"
        );
    }
}

#[test]
fn numeric_objects_and_zero_work_follow_original_scalar_loops() {
    let cases: Value = serde_json::from_str(include_str!("data/objects.json")).unwrap();
    for (index, c) in cases.as_array().unwrap().iter().enumerate() {
        let name = c["name"].as_str().unwrap();
        let cells = c["inputs"]
            .as_object()
            .unwrap()
            .values()
            .map(|x| x["values"].as_array().unwrap().len())
            .max()
            .unwrap()
            .max(1);
        let mut saw_expected_error = false;
        for n in 0..cells {
            let results = invoke(name, &c["inputs"], &c["options"], n);
            let values: Vec<MaskedValue> = results.iter().map(|(_, x)| *x).collect();
            let valid = MaskedValue::validate_outputs(&values);
            if let Some(expected) = c
                .get("exception")
                .or_else(|| c.get("native_policy_error"))
                .and_then(Value::as_str)
            {
                if valid
                    .err()
                    .is_some_and(|e| e.python_exception() == expected)
                {
                    saw_expected_error = true;
                }
                continue;
            }
            valid.unwrap_or_else(|e| panic!("unexpected {e:?}: object case {index}/{name}/{n}"));
            for (key, value) in results {
                check(
                    value,
                    if key.is_empty() {
                        &c["output"]
                    } else {
                        &c["output"][key]
                    },
                    n,
                    &format!("object case {index}/{name}/{key}/{n}"),
                );
            }
        }
        if let Some(expected) = c
            .get("exception")
            .or_else(|| c.get("native_policy_error"))
            .and_then(Value::as_str)
        {
            assert!(
                saw_expected_error,
                "missing {expected}: object case {index}/{name}"
            );
        }
    }
}
