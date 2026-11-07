use behave_fire::{
    crown::masked,
    surface::{
        masked::{MaskedFuelbedIntermediates, MaskedValue, MathProfile, Precision},
        FuelbedIntermediates,
    },
};
use serde_json::Value;
use std::sync::OnceLock;

fn argument(v: &Value, n: usize) -> MaskedValue {
    MaskedValue::new(
        number(&v["values"][n]),
        v["masks"][n].as_bool().unwrap(),
        v["masked_array"].as_bool().unwrap(),
        if v["dtype"] == "float32" {
            Precision::F32
        } else {
            Precision::F64
        },
    )
    .with_scalar_array(v["scalar_array"].as_bool().unwrap())
    .with_fill_value(v.get("fill_value").filter(|v| !v.is_null()).map(number))
    .with_numpy_scalar(v["numpy_scalar"].as_bool().unwrap_or(false))
    .with_math_profile(match v["math_profile"].as_str() {
        None | Some("numpy_simd" | "NumpySimd") => MathProfile::NumpySimd,
        Some("libm" | "Libm") => MathProfile::Libm,
        Some(other) => panic!("unknown fixture math profile {other}"),
    })
}

#[test]
fn decoder_uses_recorded_transcendental_profile() {
    let mut input = serde_json::json!({"dtype":"float32","values":[(-138.0f32/3500.0f32) as f64],"masks":[false],"masked_array":false,"scalar_array":false,"fill_value":null});
    let simd = argument(&input, 0);
    assert_eq!(simd.math_profile, MathProfile::NumpySimd);
    assert_eq!(
        simd.exp().as_array_f64().value,
        f32::from_bits(1064704585) as f64
    );
    input["math_profile"] = Value::String("libm".into());
    let scalar = argument(&input, 0);
    assert_eq!(scalar.math_profile, MathProfile::Libm);
    assert_eq!(
        scalar.exp().as_array_f64().value,
        f32::from_bits(1064704586) as f64
    );
    assert_ne!(scalar.exp().value, simd.exp().value);
}

#[test]
fn fill_assertions_compare_lazy_defaults_and_nondefault_values() {
    let default = MaskedValue::masked(0., false).with_math_profile(MathProfile::NumpySimd);
    let mut expected = serde_json::json!({"dtype":"float64","values":[0.],"masks":[false],"masked_array":true,"scalar_array":false,"fill_value":null});
    check(default, &expected, 0, "lazy default");
    expected["fill_value"] = serde_json::json!(-12345.);
    assert!(
        std::panic::catch_unwind(|| check(default, &expected, 0, "missing nondefault fill"))
            .is_err()
    );
    expected["fill_value"] = Value::Null;
    assert!(std::panic::catch_unwind(|| check(
        default.with_fill_value(Some(-12345.)),
        &expected,
        0,
        "unexpected nondefault fill"
    ))
    .is_err());
    expected["fill_value"] = serde_json::json!(1e20);
    check(
        default,
        &expected,
        0,
        "explicit default equals lazy default",
    );
    expected["fill_value"] = serde_json::json!({"float":"nan"});
    check(
        default.with_fill_value(Some(f64::NAN)),
        &expected,
        0,
        "tagged NaN fill",
    );
    assert!(std::panic::catch_unwind(|| check(
        default.with_fill_value(Some(f64::INFINITY)),
        &expected,
        0,
        "infinity is not NaN fill"
    ))
    .is_err());
    expected["fill_value"] = Value::String("+inf".into());
    check(
        default.with_fill_value(Some(f64::INFINITY)),
        &expected,
        0,
        "positive infinity fill",
    );
    assert!(std::panic::catch_unwind(|| check(
        default.with_fill_value(Some(f64::NEG_INFINITY)),
        &expected,
        0,
        "infinity sign fill"
    ))
    .is_err());
    let single = MaskedValue::masked_f32(0., false).with_math_profile(MathProfile::NumpySimd);
    let single_expected = serde_json::json!({"dtype":"float32","values":[0.],"masks":[false],"masked_array":true,"scalar_array":false,"fill_value":null});
    check(single, &single_expected, 0, "float32 lazy default");
    check(
        single.with_fill_value(Some((1e20_f32) as f64)),
        &single_expected,
        0,
        "float32 explicit default",
    );
    assert!(std::panic::catch_unwind(|| check(
        single.with_fill_value(Some(1e20)),
        &single_expected,
        0,
        "float64 default is not float32 default"
    ))
    .is_err());
}
fn check(a: MaskedValue, b: &Value, n: usize, ctx: &str) {
    assert_eq!(a.mask, b["masks"][n].as_bool().unwrap(), "mask {ctx}");
    assert_eq!(
        a.is_masked_array,
        b["masked_array"].as_bool().unwrap(),
        "array class {ctx}"
    );
    assert_eq!(
        a.precision,
        if b["dtype"] == "float32" {
            Precision::F32
        } else {
            Precision::F64
        },
        "dtype {ctx}"
    );
    assert_eq!(
        a.is_scalar_array,
        b["scalar_array"].as_bool().unwrap(),
        "scalar rank {ctx}"
    );
    let expected = number(&b["values"][n]);
    if a.value == 0.0 && expected == 0.0 {
        assert_eq!(
            a.value.is_sign_negative(),
            expected.is_sign_negative(),
            "zero sign {ctx}"
        );
    }
    assert!(
        if expected.is_nan() {
            a.value.is_nan()
        } else if expected.is_infinite() {
            a.value == expected
        } else {
            (a.value - expected).abs()
                <= 1e-12
                    + if a.precision == Precision::F32 {
                        1e-6
                    } else {
                        1e-9
                    } * expected.abs()
        },
        "raw {ctx}: {:?} != {:?}",
        a.value,
        expected
    );
    if b["masked_array"] == true {
        let default = match b["dtype"].as_str().unwrap() {
            "float32" => (1e20_f32) as f64,
            "bool" => 1.,
            _ => 1e20,
        };
        let wanted = b
            .get("fill_value")
            .filter(|v| !v.is_null())
            .map(number)
            .unwrap_or(default);
        let actual = a.fill_value.unwrap_or(default);
        if wanted.is_nan() {
            assert!(actual.is_nan(), "fill {ctx}: {actual:?} != NaN");
        } else {
            assert_eq!(actual.to_bits(), wanted.to_bits(), "fill {ctx}");
        }
    } else {
        assert!(
            a.fill_value.is_none(),
            "plain result retains fill metadata {ctx}"
        );
    }
}
fn fixture() -> &'static Value {
    static VALUE: OnceLock<Value> = OnceLock::new();
    VALUE.get_or_init(|| {
        serde_json::from_str(include_str!("data/port_crown_crown_masked.json")).unwrap()
    })
}
fn ib(i: &Value) -> FuelbedIntermediates {
    let mut v = FuelbedIntermediates::default();
    macro_rules! fields {($($k:ident),+)=>{$(v.$k=number(&i[stringify!($k)]);)+}}
    fields!(
        sigma,
        bulk_density,
        packing_ratio,
        relative_packing_ratio,
        heat_sink,
        propagating_flux,
        w_heat_dead,
        w_heat_live,
        w_silica_dead,
        w_silica_live,
        w_moist_dead,
        w_moist_live,
        wfl_dead,
        wfl_live,
        moe_dead,
        moe_live,
        frac_dead,
        frac_live,
        depth
    );
    v.is_defined = i["is_defined"].as_bool().unwrap();
    v
}
fn masked_ib(i: &Value, n: usize) -> MaskedFuelbedIntermediates {
    let mut v = MaskedFuelbedIntermediates::new();
    macro_rules! fields {($($k:ident),+)=>{$(v.$k=argument(&i[stringify!($k)],n);)+}}
    fields!(
        sigma,
        bulk_density,
        packing_ratio,
        relative_packing_ratio,
        heat_sink,
        propagating_flux,
        w_heat_dead,
        w_heat_live,
        w_silica_dead,
        w_silica_live,
        w_moist_dead,
        w_moist_live,
        wfl_dead,
        wfl_live,
        moe_dead,
        moe_live,
        frac_dead,
        frac_live,
        depth,
        is_defined
    );
    v
}
fn number(v: &Value) -> f64 {
    if let Some(tagged) = v.get("float") {
        return number(tagged);
    }
    if let Some(n) = v.as_f64() {
        return n;
    }
    match v.as_str().unwrap() {
        "nan" | "NaN" => f64::NAN,
        "+inf" | "inf" | "Infinity" => f64::INFINITY,
        "-inf" | "-Infinity" => f64::NEG_INFINITY,
        s => panic!("{s}"),
    }
}
fn replay(name: &str) {
    let mut count = 0;
    for (index, c) in fixture()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(_, c)| c["name"] == name)
    {
        let i = &c["inputs"];
        let out = &c["output"];
        let cells = i.as_object().unwrap().values().next().unwrap()["values"]
            .as_array()
            .unwrap()
            .len();
        for n in 0..cells {
            let a = |k| argument(&i[k], n);
            let mut values = Vec::new();
            let scalar = match name {
                "classify_crown_fire_type" => {
                    let result =
                        masked::classify_crown_fire_type(a("transition_ratio"), a("active_ratio"));
                    assert_eq!(
                        result.fire_type,
                        number(&out["fire_type"]["values"][n]) as i32,
                        "classification case {index}/{n}"
                    );
                    for (k, v) in [
                        ("is_surface_fire", result.is_surface_fire),
                        ("is_passive_crown", result.is_passive_crown),
                        ("is_active_crown", result.is_active_crown),
                    ] {
                        assert_eq!(v, number(&out[k]["values"][n]) != 0.);
                    }
                    None
                }
                "assign_final_crown_fire_behavior" => {
                    values = masked::assign_final_crown_fire_behavior(
                        c["options"]["is_surface_fire"].as_bool().unwrap(),
                        c["options"]["is_passive_crown"].as_bool().unwrap(),
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
                    .to_vec();
                    None
                }
                "calculate_critical_crown_fire_spread_rate" => Some(
                    masked::calculate_critical_crown_fire_spread_rate(a("canopy_bulk_density")),
                ),
                "calculate_critical_surface_fireline_intensity" => {
                    Some(masked::calculate_critical_surface_fireline_intensity(
                        a("canopy_base_height"),
                        a("moisture_foliar"),
                    ))
                }
                "calculate_crown_fraction_burned" => Some(masked::calculate_crown_fraction_burned(
                    a("surface_ros"),
                    a("surface_critical_ros"),
                    a("crowning_surface_ros"),
                )),
                "calculate_surface_fire_critical_spread_rate" => {
                    Some(masked::calculate_surface_fire_critical_spread_rate(
                        a("critical_surface_fli"),
                        a("surface_hpua"),
                    ))
                }
                "calculate_crowning_surface_fire_spread_rate" => {
                    Some(if c["options"].get("masked_ib").is_some() {
                        masked::try_calculate_crowning_surface_fire_spread_rate_with_intermediates(
                            a("ri_crown"),
                            &masked_ib(&c["options"]["masked_ib"], n),
                            a("critical_crown_ros"),
                            a("canopy_height"),
                            false,
                        )
                        .unwrap()
                    } else {
                        masked::calculate_crowning_surface_fire_spread_rate_with_shape(
                            a("ri_crown"),
                            &ib(&c["options"]["ib_crown"]),
                            a("critical_crown_ros"),
                            a("canopy_height"),
                            c["scalar"].as_bool().unwrap(),
                        )
                    })
                }
                "calculate_crown_fire_ratios" => {
                    values = masked::calculate_crown_fire_ratios(
                        a("surface_fli"),
                        a("crown_ros"),
                        a("critical_surface_fli"),
                        a("critical_crown_ros"),
                    )
                    .named_values()
                    .to_vec();
                    None
                }
                "calculate_crown_heat_and_intensity" => {
                    values = masked::calculate_crown_heat_and_intensity(
                        a("surface_hpua"),
                        a("crown_ros"),
                        a("canopy_base_height"),
                        a("canopy_height"),
                        a("canopy_bulk_density"),
                    )
                    .named_values()
                    .to_vec();
                    None
                }
                "calculate_passive_crown_fire_behavior" => {
                    values = masked::calculate_passive_crown_fire_behavior(
                        a("surface_ros"),
                        a("surface_hpua"),
                        a("crown_ros"),
                        a("canopy_hpua"),
                        a("cfb"),
                    )
                    .named_values()
                    .to_vec();
                    None
                }
                _ => panic!("{name}"),
            };
            let ctx = format!("{name}/{index}/{n}/{}/{}", c["profile"], c["precision"]);
            if let Some(value) = scalar {
                check(value, out, n, &ctx)
            }
            for (k, v) in values {
                check(v, &out[k], n, &format!("{ctx}/{k}"))
            }
            count += 1;
        }
    }
    assert!(count > 0);
    println!("{name}: {count} masked cases");
}
macro_rules! test {($($name:ident),+)=>{$(#[test] fn $name(){replay(stringify!($name))})+}}
test!(
    assign_final_crown_fire_behavior,
    classify_crown_fire_type,
    calculate_critical_crown_fire_spread_rate,
    calculate_critical_surface_fireline_intensity,
    calculate_crown_fire_ratios,
    calculate_crown_fraction_burned,
    calculate_crown_heat_and_intensity,
    calculate_passive_crown_fire_behavior,
    calculate_surface_fire_critical_spread_rate,
    calculate_crowning_surface_fire_spread_rate
);
