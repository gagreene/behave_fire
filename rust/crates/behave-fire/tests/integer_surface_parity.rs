use behave_fire::surface::masked::{
    self, IntegerKind, IntegerValue, MaskedValue, MathProfile, Precision,
};
use serde_json::Value;

fn number(v: &Value) -> f64 {
    if let Some(tag) = v.get("float") {
        return number(tag);
    }
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
fn argument(v: &Value, n: usize) -> MaskedValue {
    let n = n % v["values"].as_array().unwrap().len();
    let value = number(&v["values"][n]);
    let kind = v["dtype"].as_str().unwrap();
    let bits = v.get("raw_bits").map(|b| b[n].as_u64().unwrap());
    let mut result = match v["weak_kind"].as_str() {
        Some("bool") => MaskedValue::python_bool(bits.unwrap() != 0),
        Some("int") if kind.starts_with("uint") => MaskedValue::python_uint(bits.unwrap()),
        Some("int") => MaskedValue::python_int(bits.unwrap() as i64),
        Some("float") => MaskedValue::scalar(value),
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
fn check(actual: MaskedValue, expected: &Value, n: usize, context: &str) {
    assert!(
        actual.error.is_none(),
        "unexpected {:?}: {context}",
        actual.error
    );
    let n = n % expected["values"].as_array().unwrap().len();
    assert_eq!(
        actual.mask,
        expected["masks"][n].as_bool().unwrap(),
        "mask {context}"
    );
    assert_eq!(
        actual.is_masked_array,
        expected["masked_array"].as_bool().unwrap(),
        "array class {context}"
    );
    assert_eq!(
        actual.is_scalar_array,
        expected["scalar_array"].as_bool().unwrap(),
        "rank {context}"
    );
    assert_eq!(
        actual.is_numpy_scalar,
        expected["numpy_scalar"].as_bool().unwrap(),
        "numpy scalar {context}"
    );
    if let Some(bits) = expected.get("raw_bits") {
        let integer = actual
            .integer
            .unwrap_or_else(|| panic!("missing integer output {} {context}", expected["dtype"]));
        assert_eq!(
            integer.kind.name(),
            expected["dtype"].as_str().unwrap(),
            "integer dtype {context}"
        );
        let expected = IntegerValue::from_bits(integer.kind, bits[n].as_u64().unwrap());
        assert_eq!(
            integer.raw_bits, expected.raw_bits,
            "raw integer bits {context}"
        );
    } else {
        assert!(
            actual.integer.is_none(),
            "unexpected integer output {context}"
        );
        assert_eq!(
            actual.precision,
            if expected["dtype"] == "float32" {
                Precision::F32
            } else {
                Precision::F64
            },
            "float dtype {context}"
        );
        let b = number(&expected["values"][n]);
        let a = actual.value;
        if b.is_nan() {
            assert!(a.is_nan(), "{context}: {a} != NaN");
        } else if b.is_infinite() {
            assert_eq!(a, b, "{context}");
        } else {
            let tol = if actual.precision == Precision::F32 {
                1e-6
            } else {
                1e-9
            };
            assert!(
                (a - b).abs() <= 1e-12 + tol * b.abs(),
                "{context}: {a:.17e} != {b:.17e}"
            );
        }
    }
}
fn invoke(name: &str, i: &Value, options: &Value, n: usize) -> Vec<(&'static str, MaskedValue)> {
    let a = |k| argument(&i[k], n);
    let value = match name {
        "add" => a("a").add(a("b")),
        "sub" => a("a").sub(a("b")),
        "mul" => a("a").mul(a("b")),
        "div" => a("a").div(a("b")),
        "pow" => a("a").pow(a("b")),
        "negative" => a("a").negative(),
        "calculate_eccentricity" => masked::calculate_eccentricity(a("fire_length_to_width_ratio")),
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
        "calculate_wind_phi" => masked::calculate_wind_phi(
            a("sigma"),
            a("midflame_wind_speed"),
            a("wind_b"),
            a("wind_c"),
            a("relative_packing_ratio"),
            a("wind_e"),
        ),
        "calculate_direction_of_max_spread" => masked::calculate_direction_of_max_spread(
            a("x_component"),
            a("y_component"),
            a("rate_vector"),
            options["wind_orientation_mode"].as_str().unwrap(),
            a("aspect"),
        ),
        "calculate_surface_rate_vector" => {
            return masked::calculate_surface_rate_vector(
                a("no_wind_no_slope_spread_rate"),
                a("wind_phi"),
                a("slope_phi"),
                a("wind_direction"),
                options["wind_orientation_mode"].as_str().unwrap(),
                a("aspect"),
            )
            .named_values()
            .to_vec()
        }
        _ => panic!("unhandled operation {name}"),
    };
    vec![("", value)]
}
fn replay(cases: &Value) {
    for (index, c) in cases.as_array().unwrap().iter().enumerate() {
        let name = c["name"].as_str().unwrap();
        let i = &c["inputs"];
        let cells = i
            .as_object()
            .unwrap()
            .values()
            .map(|x| x["values"].as_array().unwrap().len())
            .max()
            .unwrap();
        if let Some(error) = c["exception"].as_str() {
            let found = (0..cells)
                .flat_map(|n| invoke(name, i, &c["options"], n))
                .filter_map(|(_, x)| x.error)
                .find(|x| x.python_exception() == error);
            assert!(
                found.is_some(),
                "missing {error}: case {index}/{name} inputs {i}"
            );
            continue;
        }
        for n in 0..cells {
            for (key, value) in invoke(name, i, &c["options"], n) {
                check(
                    value,
                    if key.is_empty() {
                        &c["output"]
                    } else {
                        &c["output"][key]
                    },
                    n,
                    &format!("case {index}/{name}/{n}/{key}"),
                );
            }
        }
    }
}

#[test]
fn fixed_width_primitives_match_independent_numpy_oracle() {
    let cases: Value = serde_json::from_str(include_str!("data/integer_primitives.json")).unwrap();
    replay(&cases);
}

#[test]
fn integer_surface_stages_wrap_before_float_promotion() {
    let cases: Value = serde_json::from_str(include_str!("data/integer_surface.json")).unwrap();
    replay(&cases);
}

fn intermediates(i: &Value, n: usize) -> masked::MaskedFuelbedIntermediates {
    let mut result = masked::MaskedFuelbedIntermediates::new();
    result.sigma = argument(&i["sigma"], n);
    result.bulk_density = argument(&i["bulk_density"], n);
    result.packing_ratio = argument(&i["packing_ratio"], n);
    result.relative_packing_ratio = argument(&i["relative_packing_ratio"], n);
    result.heat_sink = argument(&i["heat_sink"], n);
    result.propagating_flux = argument(&i["propagating_flux"], n);
    result.w_heat_dead = argument(&i["w_heat_dead"], n);
    result.w_heat_live = argument(&i["w_heat_live"], n);
    result.w_silica_dead = argument(&i["w_silica_dead"], n);
    result.w_silica_live = argument(&i["w_silica_live"], n);
    result.w_moist_dead = argument(&i["w_moist_dead"], n);
    result.w_moist_live = argument(&i["w_moist_live"], n);
    result.wfl_dead = argument(&i["wfl_dead"], n);
    result.wfl_live = argument(&i["wfl_live"], n);
    result.moe_dead = argument(&i["moe_dead"], n);
    result.moe_live = argument(&i["moe_live"], n);
    result.frac_dead = argument(&i["frac_dead"], n);
    result.frac_live = argument(&i["frac_live"], n);
    result.depth = argument(&i["depth"], n);
    result.is_defined = argument(&i["is_defined"], n);
    result
}

#[test]
fn integer_particle_reductions_and_wind_prefixes_match_python() {
    use behave_fire::surface::{masked::MaskedParticles, SurfaceInput};
    let cases: Value = serde_json::from_str(include_str!("data/integer_dictionary.json")).unwrap();
    for (index, c) in cases.as_array().unwrap().iter().enumerate() {
        let name = c["name"].as_str().unwrap();
        let i = &c["inputs"];
        if name == "calculate_wind_phi" {
            replay(&Value::Array(vec![c.clone()]));
            continue;
        }
        let result: Result<Vec<(&str, MaskedValue)>, masked::MaskedError> = if name == "fuelbed" {
            let p0 = &i["p"];
            let mut p = MaskedParticles::new();
            p.scalar_cell = c["scalar_cell"].as_bool().unwrap();
            p.load_dead = std::array::from_fn(|n| argument(&p0["load_dead"], n));
            p.load_live = std::array::from_fn(|n| argument(&p0["load_live"], n));
            p.savr_dead = std::array::from_fn(|n| argument(&p0["savr_dead"], n));
            p.savr_live = std::array::from_fn(|n| argument(&p0["savr_live"], n));
            p.moisture_dead = std::array::from_fn(|n| argument(&p0["moisture_dead"], n));
            p.moisture_live = std::array::from_fn(|n| argument(&p0["moisture_live"], n));
            p.hoc_dead = argument(&p0["hoc_dead"], 0);
            p.hoc_live = argument(&p0["hoc_live"], 0);
            p.depth = argument(&p0["depth"], 0);
            p.moe_dead = argument(&p0["moe_dead"], 0);
            p.is_defined = argument(&p0["is_defined"], 0);
            masked::try_calculate_fuelbed_intermediates(&p).map(|x| x.named_values().to_vec())
        } else {
            let ib = intermediates(&i["ib"], 0);
            let mut input = SurfaceInput::new(124);
            input.wind_speed = 5.;
            input.wind_direction = 45.;
            input.wind_orientation_mode = "RelativeToNorth".into();
            input.slope = 30.;
            input.aspect = 95.;
            masked::try_run_surface_fire_with_intermediates(argument(&i["ri"], 0), &ib, &input).map(
                |x| {
                    x.named_values()
                        .into_iter()
                        .map(|(k, v)| (k, MaskedValue::plain(v)))
                        .collect()
                },
            )
        };
        if let Some(expected) = c["exception"].as_str() {
            assert_eq!(
                result.err().map(|x| x.python_exception()),
                Some(expected),
                "dictionary {index}/{name}"
            );
            continue;
        }
        let result =
            result.unwrap_or_else(|e| panic!("unexpected {e:?}: dictionary {index}/{name}"));
        for (key, value) in result {
            check(
                value,
                &c["output"][key],
                0,
                &format!("dictionary {index}/{name}/{key}"),
            );
        }
    }
}

#[test]
fn integer_direction_uses_targeted_float32_normalization_for_half_loops() {
    let cases: Value = serde_json::from_str(include_str!("data/integer_direction.json")).unwrap();
    let mut widened = 0;
    for c in cases.as_array().unwrap() {
        if c["normalize_half_loop"] == true {
            assert_eq!(c["original_loop_dtype"], "float16");
            if c["options"]["wind_orientation_mode"] == "RelativeToUpslope" {
                assert_eq!(c["original_diagnostic"]["dtype"], "float16");
                assert_eq!(c["output"]["dtype"], "float32");
            } else {
                assert_eq!(c["output"]["dtype"], "float64");
            }
            widened += 1;
        }
    }
    assert!(widened > 10);
    let evidence: Value =
        serde_json::from_str(include_str!("data/integer_direction_platform.json")).unwrap();
    replay(&direction_cases_for_profile(
        &cases,
        &evidence,
        direction_math_fingerprint(),
    ));
}

fn direction_cases_for_profile(cases: &Value, evidence: &Value, fingerprint: [u32; 2]) -> Value {
    let profiles: Vec<_> = evidence["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|profile| profile["atan2_f32_bits"] == serde_json::json!(fingerprint))
        .collect();
    assert_eq!(
        profiles.len(),
        1,
        "unreviewed float32 atan2 profile: {fingerprint:08x?}"
    );
    let mut selected = cases.clone();
    for replacement in profiles[0]["overrides"].as_array().unwrap() {
        let index = replacement["case_index"].as_u64().unwrap() as usize;
        let original = &cases[index];
        let identity = serde_json::json!({
            "name": original["name"],
            "inputs": original["inputs"],
            "options": original["options"],
            "normalize_half_loop": original["normalize_half_loop"],
            "original_loop_dtype": original["original_loop_dtype"],
        });
        assert_eq!(
            identity.to_string(),
            replacement["identity"].to_string(),
            "direction input identity drift at {index}"
        );
        assert_eq!(
            original["output"].to_string(),
            replacement["frozen_output"].to_string(),
            "frozen direction output drift at {index}"
        );
        let mut frozen_metadata = original["output"].clone();
        let mut captured_metadata = replacement["output"].clone();
        frozen_metadata.as_object_mut().unwrap().remove("values");
        captured_metadata.as_object_mut().unwrap().remove("values");
        assert_eq!(
            frozen_metadata, captured_metadata,
            "direction structural oracle drift at {index}"
        );
        assert_eq!(
            original["output"]["values"].as_array().unwrap().len(),
            replacement["output"]["values"].as_array().unwrap().len()
        );
        selected[index]["output"] = replacement["output"].clone();
    }
    selected
}

fn direction_math_fingerprint() -> [u32; 2] {
    let negative_one = std::hint::black_box(-1.0f32);
    [
        std::hint::black_box(65534.0f32)
            .atan2(negative_one)
            .to_bits(),
        std::hint::black_box(0.0f32).atan2(negative_one).to_bits(),
    ]
}

#[test]
fn direction_oracle_rejects_changed_inputs() {
    let mut cases: Value =
        serde_json::from_str(include_str!("data/integer_direction.json")).unwrap();
    let evidence: Value =
        serde_json::from_str(include_str!("data/integer_direction_platform.json")).unwrap();
    let profile = evidence["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|profile| !profile["overrides"].as_array().unwrap().is_empty())
        .unwrap();
    let fingerprint: [u32; 2] = serde_json::from_value(profile["atan2_f32_bits"].clone()).unwrap();
    let index = profile["overrides"][0]["case_index"].as_u64().unwrap() as usize;
    cases[index]["inputs"]["aspect"]["values"][0] = serde_json::json!(96.0);
    assert!(std::panic::catch_unwind(|| direction_cases_for_profile(
        &cases,
        &evidence,
        fingerprint
    ))
    .is_err());
}

#[test]
fn direction_oracle_rejects_changed_zero_sign() {
    let mut cases: Value =
        serde_json::from_str(include_str!("data/integer_direction.json")).unwrap();
    let evidence: Value =
        serde_json::from_str(include_str!("data/integer_direction_platform.json")).unwrap();
    let profile = evidence["profiles"]
        .as_array()
        .unwrap()
        .iter()
        .find(|profile| !profile["overrides"].as_array().unwrap().is_empty())
        .unwrap();
    let fingerprint: [u32; 2] = serde_json::from_value(profile["atan2_f32_bits"].clone()).unwrap();
    let index = profile["overrides"][0]["case_index"].as_u64().unwrap() as usize;
    assert_eq!(
        number(&cases[index]["inputs"]["x_component"]["values"][3]),
        0.0
    );
    cases[index]["inputs"]["x_component"]["values"][3] = serde_json::json!(-0.0);
    assert!(std::panic::catch_unwind(|| direction_cases_for_profile(
        &cases,
        &evidence,
        fingerprint
    ))
    .is_err());
}

#[test]
fn direction_oracle_rejects_structural_changes() {
    let cases: Value = serde_json::from_str(include_str!("data/integer_direction.json")).unwrap();
    let mut evidence: Value =
        serde_json::from_str(include_str!("data/integer_direction_platform.json")).unwrap();
    let profile = evidence["profiles"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|profile| !profile["overrides"].as_array().unwrap().is_empty())
        .unwrap();
    let fingerprint: [u32; 2] = serde_json::from_value(profile["atan2_f32_bits"].clone()).unwrap();
    profile["overrides"][0]["output"]["dtype"] = serde_json::json!("float32");
    assert!(std::panic::catch_unwind(|| direction_cases_for_profile(
        &cases,
        &evidence,
        fingerprint
    ))
    .is_err());
}

#[test]
fn direction_oracle_rejects_unknown_math_profiles() {
    let cases: Value = serde_json::from_str(include_str!("data/integer_direction.json")).unwrap();
    let evidence: Value =
        serde_json::from_str(include_str!("data/integer_direction_platform.json")).unwrap();
    assert!(
        std::panic::catch_unwind(|| direction_cases_for_profile(&cases, &evidence, [0, 0]))
            .is_err()
    );
}
