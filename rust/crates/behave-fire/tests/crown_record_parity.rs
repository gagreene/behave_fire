use behave_fire::{
    crown::{self, CrownInput, SurfaceFireInputs},
    fuel_models::FuelModels,
    surface::masked::{MaskedValue, MathProfile, Precision},
    units::{SlopeUnit, SpeedUnit},
};
use serde_json::Value;
use std::sync::OnceLock;

fn argument(value: &Value, n: usize) -> MaskedValue {
    let data = value["values"].as_array().unwrap();
    let index = n % data.len();
    MaskedValue::new(
        number(&data[index]),
        value["masks"][index].as_bool().unwrap(),
        value["masked_array"].as_bool().unwrap(),
        if value["dtype"] == "float32" {
            Precision::F32
        } else {
            Precision::F64
        },
    )
    .with_scalar_array(value["scalar_array"].as_bool().unwrap())
    .with_numpy_scalar(value["numpy_scalar"].as_bool().unwrap())
    .with_fill_value(value.get("fill_value").filter(|x| !x.is_null()).map(number))
    .with_math_profile(MathProfile::NumpySimd)
}
fn check(actual: MaskedValue, expected: &Value, n: usize, context: &str) {
    assert!(
        actual.error.is_none(),
        "unexpected deferred error {context}: {:?}",
        actual.error
    );
    let index = n % expected["values"].as_array().unwrap().len();
    assert_eq!(
        actual.mask,
        expected["masks"][index].as_bool().unwrap(),
        "mask {context}"
    );
    assert_eq!(
        actual.is_masked_array,
        expected["masked_array"].as_bool().unwrap(),
        "array class {context}"
    );
    assert_eq!(
        actual.precision,
        if expected["dtype"] == "float32" {
            Precision::F32
        } else {
            Precision::F64
        },
        "dtype {context}"
    );
    assert_eq!(
        actual.is_scalar_array,
        expected["scalar_array"].as_bool().unwrap(),
        "rank {context}"
    );
    assert_eq!(
        actual.is_numpy_scalar,
        expected["numpy_scalar"].as_bool().unwrap(),
        "NumPy scalar {context}"
    );
    let target = number(&expected["values"][index]);
    if actual.value == 0.0 && target == 0.0 {
        assert_eq!(
            actual.value.is_sign_negative(),
            target.is_sign_negative(),
            "zero sign {context}"
        );
    }
    assert!(
        if target.is_nan() {
            actual.value.is_nan()
        } else if target.is_infinite() {
            actual.value == target
        } else {
            (actual.value - target).abs()
                <= 1e-12
                    + if actual.precision == Precision::F32 {
                        1e-6
                    } else {
                        1e-9
                    } * target.abs()
        },
        "value {context}: {:?} != {target:?}",
        actual.value
    );
    if actual.is_masked_array {
        let default = if actual.precision == Precision::F32 {
            (1e20_f32) as f64
        } else {
            1e20
        };
        let actual_fill = actual.fill_value.unwrap_or(default);
        let expected_fill = expected
            .get("fill_value")
            .filter(|x| !x.is_null())
            .map(number)
            .unwrap_or(default);
        assert!(
            if expected_fill.is_nan() {
                actual_fill.is_nan()
            } else {
                actual_fill.to_bits() == expected_fill.to_bits()
            },
            "fill {context}: {actual_fill:?} != {expected_fill:?}"
        );
    } else {
        assert!(actual.fill_value.is_none(), "plain fill {context}");
    }
}
fn fixture() -> &'static Value {
    static DATA: OnceLock<Value> = OnceLock::new();
    DATA.get_or_init(|| serde_json::from_str(include_str!("data/crown_record.json")).unwrap())
}
fn input(values: &Value, n: usize) -> CrownInput {
    let mut result = CrownInput::default();
    let grid = &values["fuel_model_grid"];
    result.fuel_model_grid = if grid.is_number() {
        number(grid) as i32
    } else {
        number(&grid["values"][n]) as i32
    };
    macro_rules! fields{($($field:ident),+)=>{$(result.$field=scalar(&values[stringify!($field)],n);)+}}
    fields!(
        m1h,
        m10h,
        m100h,
        mlh,
        mlw,
        wind_speed,
        wind_direction,
        slope,
        aspect,
        canopy_base_height,
        canopy_height,
        canopy_bulk_density,
        moisture_foliar
    );
    result.wind_speed_units =
        SpeedUnit::try_from(values["wind_speed_units"].as_i64().unwrap() as i32).unwrap();
    result.slope_units =
        SlopeUnit::try_from(values["slope_units"].as_i64().unwrap() as i32).unwrap();
    result.wind_orientation_mode = values["wind_orientation_mode"].as_str().unwrap().into();
    result
}
fn number(value: &Value) -> f64 {
    if let Some(tag) = value.get("float") {
        return number(tag);
    }
    if let Some(x) = value.as_f64() {
        return x;
    }
    match value.as_str().unwrap() {
        "nan" => f64::NAN,
        "+inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        x => panic!("{x}"),
    }
}
fn scalar(value: &Value, n: usize) -> f64 {
    if let Some(values) = value["values"].as_array() {
        number(&values[n % values.len()])
    } else {
        number(value)
    }
}
fn replay(prefix: &str) {
    let mut classes = std::collections::BTreeSet::new();
    let mut count = 0;
    for case in fixture()["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["id"].as_str().unwrap().starts_with(prefix))
    {
        let label = case["id"].as_str().unwrap();
        let output = &case["output"];
        let cells = output["fire_type"]["values"].as_array().unwrap().len();
        assert_eq!(output["fire_type"]["dtype"], "int32");
        assert_eq!(output["fire_type"]["masked_array"], false);
        assert_eq!(output["fire_type"]["scalar_array"], false);
        assert_eq!(output["fire_type"]["numpy_scalar"], false);
        assert!(output["fire_type"]["masks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|v| v == false));
        let mut fuels = FuelModels::default();
        if case["table"] == "custom10" {
            fuels.get_mut(10).unwrap().dead_1h = 0.4;
            fuels.get_mut(10).unwrap().heat_of_combustion_dead = 9000.;
        }
        for n in 0..cells {
            let mut surface = SurfaceFireInputs::new();
            surface.surface_ros = argument(&case["surface"]["spread_rate"], n);
            surface.surface_fli = argument(&case["surface"]["fireline_intensity"], n);
            surface.surface_hpua = argument(&case["surface"]["heat_per_unit_area"], n);
            let result =
                crown::run_with_surface_inputs(&surface, &input(&case["settings"], n), &fuels)
                    .unwrap();
            classes.insert(result.fire_type);
            assert_eq!(
                result.fire_type,
                number(&output["fire_type"]["values"][n]) as i32,
                "{label}/{n}/fire_type"
            );
            assert_eq!(output.as_object().unwrap().len(), 22);
            for (key, value) in result.named_values() {
                check(value, &output[key], n, &format!("{label}/{n}/{key}"));
            }
            count += 1;
        }
    }
    assert!(count > 0);
    if prefix.is_empty() {
        assert_eq!(classes, std::collections::BTreeSet::from([0, 1, 2, 3]));
    }
    println!("{prefix}: {count} full Crown supplied-record cells");
}
#[test]
fn supplied_float32_records_keep_final_flame_rounding() {
    replay("binding-regression/False")
}
#[test]
fn supplied_masked_records_keep_intermediate_data_and_classification() {
    replay("binding-regression/True")
}
#[test]
fn supplied_surface_records_all_classes_dtypes_masks_and_ranks_match_python() {
    replay("")
}
