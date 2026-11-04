//! Exact NumPy integer aliases use scalar slots and dtype promotion independently.

use behave_fire::surface::masked::{IntegerKind, IntegerValue, MaskedValue};

fn integer(kind: IntegerKind, number: u8, value: u64) -> MaskedValue {
    MaskedValue::plain(0.0)
        .with_integer(IntegerValue::from_bits(kind, value))
        .with_numpy_integer_type(Some(number))
        .with_scalar_array(true)
        .with_numpy_scalar(true)
}

#[test]
fn array_promotion_and_ufuncs_resolve_common_alias() {
    for (kind, lower, upper) in [
        (IntegerKind::I32, 5, 7),
        (IntegerKind::U32, 6, 8),
        (IntegerKind::I64, 7, 9),
        (IntegerKind::U64, 8, 10),
    ] {
        let a = integer(kind, lower, 2);
        let b = integer(kind, upper, 3);
        assert_eq!(a.minimum(b).numpy_integer_type, Some(upper));
        assert_eq!(a.maximum(b).numpy_integer_type, Some(upper));
        assert_eq!(
            a.as_array().add(b.as_array()).numpy_integer_type,
            Some(upper)
        );
        assert_eq!(
            MaskedValue::select(true, a, b).numpy_integer_type,
            Some(upper)
        );
    }
}

#[test]
fn floating_promotion_discards_integer_alias() {
    let a = integer(IntegerKind::I64, 9, 2);
    assert_eq!(a.div(MaskedValue::python_int(2)).numpy_integer_type, None);
    assert_eq!(a.as_array_f64().numpy_integer_type, None);
    assert_eq!(a.add(MaskedValue::scalar(0.5)).numpy_integer_type, None);
}

#[test]
fn independent_linux_and_windows_scalar_oracles() {
    use serde_json::Value;
    for fixture in [
        include_str!("data/integer_alias_linux.json"),
        include_str!("data/integer_alias_windows.json"),
    ] {
        let document: Value = serde_json::from_str(fixture).unwrap();
        let windows = document["platform"] == "Windows";
        let input = |name: &str| match name {
            "builtin" => MaskedValue::python_int(2),
            "bool" => MaskedValue::python_bool(true),
            _ => {
                let kind = if name.contains("unsigned") {
                    IntegerKind::U64
                } else if name == "int32" {
                    IntegerKind::I32
                } else {
                    IntegerKind::I64
                };
                integer(
                    kind,
                    document["types"][name]["num"].as_u64().unwrap() as u8,
                    2,
                )
            }
        };
        for (pair, operations) in document["pairs"].as_object().unwrap() {
            let (left, right) = pair.split_once('/').unwrap();
            let a = input(left);
            let b = input(right);
            for (operation, expected) in operations.as_object().unwrap() {
                let actual = match operation.as_str() {
                    "add" => a.add(b),
                    "multiply" => a.mul(b),
                    "power" => a.pow(b),
                    "minimum" => a.minimum(b),
                    "ordinary_select_left" => MaskedValue::select(true, a, b),
                    "object_select_left" => MaskedValue::select(
                        true,
                        a.with_object_payload(!matches!(left, "builtin" | "bool")),
                        b.with_object_payload(!matches!(right, "builtin" | "bool")),
                    ),
                    _ => panic!("unknown oracle operation"),
                };
                let label = format!("{} {pair} {operation}", document["platform"]);
                assert!(actual.error.is_none(), "{label}: {:?}", actual.error);
                let text = expected["value"].as_str().unwrap();
                let value = match text {
                    "True" => 1.0,
                    "False" => 0.0,
                    _ => text.parse::<f64>().unwrap(),
                };
                assert_eq!(actual.value, value, "{label}");
                if matches!(expected["type"].as_str().unwrap(), "int" | "bool") {
                    assert!(!actual.object_numpy_scalar, "{label}");
                    continue;
                }
                let type_number = if let Some(value) = actual.integer {
                    actual.numpy_integer_type.unwrap_or(match value.kind {
                        IntegerKind::Bool => 0,
                        IntegerKind::I8 => 1,
                        IntegerKind::U8 => 2,
                        IntegerKind::I16 => 3,
                        IntegerKind::U16 => 4,
                        IntegerKind::I32 => {
                            if windows {
                                7
                            } else {
                                5
                            }
                        }
                        IntegerKind::U32 => {
                            if windows {
                                8
                            } else {
                                6
                            }
                        }
                        IntegerKind::I64 => {
                            if windows {
                                9
                            } else {
                                7
                            }
                        }
                        IntegerKind::U64 => {
                            if windows {
                                10
                            } else {
                                8
                            }
                        }
                    })
                } else {
                    12
                };
                assert_eq!(
                    type_number as u64,
                    expected["num"].as_u64().unwrap(),
                    "{label}"
                );
            }
        }
    }
}

#[test]
fn object_selection_retains_selected_payload_alias() {
    let a = integer(IntegerKind::I64, 7, 2).with_object_payload(true);
    let b = integer(IntegerKind::I64, 9, 3).with_object_payload(true);
    assert_eq!(MaskedValue::select(true, a, b).numpy_integer_type, Some(7));
    assert_eq!(MaskedValue::select(false, a, b).numpy_integer_type, Some(9));
    assert_eq!(a.minimum(b).numpy_integer_type, Some(7));
    assert_eq!(a.maximum(b).numpy_integer_type, Some(9));
}

#[test]
fn scalar_arithmetic_preserves_left_equal_width_alias() {
    for (kind, lower, upper) in [
        (IntegerKind::I32, 5, 7),
        (IntegerKind::U32, 6, 8),
        (IntegerKind::I64, 7, 9),
        (IntegerKind::U64, 8, 10),
    ] {
        let a = integer(kind, lower, 2);
        let b = integer(kind, upper, 3);
        assert_eq!(a.add(b).numpy_integer_type, Some(lower));
        assert_eq!(b.add(a).numpy_integer_type, Some(upper));
        assert_eq!(a.mul(b).numpy_integer_type, Some(lower));
        assert_eq!(b.pow(a).numpy_integer_type, Some(upper));
        assert_eq!(a.negative().numpy_integer_type, Some(lower));
    }
}

#[test]
fn weak_builtin_operands_keep_the_numpy_scalar_alias() {
    let a = integer(IntegerKind::I64, 9, 2);
    let weak = MaskedValue::python_int(3);
    assert_eq!(a.add(weak).numpy_integer_type, Some(9));
    assert_eq!(weak.add(a).numpy_integer_type, Some(9));
    assert_eq!(weak.mul(a).numpy_integer_type, Some(9));
}
