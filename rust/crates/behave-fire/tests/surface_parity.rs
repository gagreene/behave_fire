// Fixtures intentionally exercise public default construction and field mutation.
#![allow(clippy::field_reassign_with_default)]

#[path = "support/engine_fixture.rs"]
mod fixture;
use behave_fire::engine::{PreparedFuel, SurfaceCell, SurfacePlan, SurfaceTweaks};
use behave_fire::surface;

#[test]
fn backing_rate_preserves_ellipse_ratio() {
    assert_eq!(surface::calculate_backing_spread_rate(12.0, 0.5), 4.0);
}

use behave_fire::{
    fuel_models::FuelModels,
    surface::*,
    units::{SlopeUnit, SpeedUnit},
};
use serde_json::Value;

#[test]
fn raw_fuel_rows_preserve_undefined_properties_and_dynamic_transfer() {
    let mut fuel = behave_fire::fuel_models::FuelModel::default();
    fuel.dead_1h = 0.125;
    fuel.dead_10h = 0.0625;
    fuel.dead_100h = 0.03125;
    fuel.live_herb = 0.2;
    fuel.live_woody = 0.15;
    fuel.savr_1h = -7.0;
    fuel.savr_live_herb = 777.0;
    fuel.savr_live_woody = 900.0;
    fuel.heat_of_combustion_dead = 8100.0;
    fuel.heat_of_combustion_live = 7900.0;
    fuel.fuel_bed_depth = -0.5;
    fuel.moisture_of_extinction_dead = 0.23;
    fuel.is_defined = false;
    fuel.is_dynamic = true;
    let p = build_particle_arrays_from_prepared(
        &PreparedFuel::from_model(&fuel),
        -0.1,
        0.07,
        0.08,
        0.60,
        0.90,
    );
    assert!(!p.is_defined);
    assert_eq!(p.load_dead[..3], [0.125, 0.0625, 0.03125]);
    check(p.load_dead[3], 0.1334, "raw partial herb transfer", 1e-12);
    check(p.load_live[0], 0.0666, "raw remaining herb", 1e-12);
    assert_eq!(p.load_live[1], 0.15);
    assert_eq!(p.savr_dead, [-7.0, 109.0, 30.0, 777.0, 0.0]);
    assert_eq!(p.savr_live, [777.0, 900.0, 0.0, 0.0, 0.0]);
    assert_eq!(p.moisture_dead, [0.01, 0.07, 0.08, 0.01, 0.0]);
    assert_eq!(p.moisture_live, [0.60, 0.90, 0.0, 0.0, 0.0]);
    assert_eq!(p.hoc_dead, 8100.0);
    assert_eq!(p.hoc_live, 7900.0);
    assert_eq!(p.depth, -0.5);
    assert_eq!(p.moe_dead, 0.23);
    let full = build_particle_arrays_from_prepared(
        &PreparedFuel::from_model(&fuel),
        0.06,
        0.07,
        0.08,
        0.20,
        0.90,
    );
    assert_eq!(full.load_dead[3], 0.2);
    assert_eq!(full.load_live[0], 0.0);
    fuel.is_dynamic = false;
    let static_row = build_particle_arrays_from_prepared(
        &PreparedFuel::from_model(&fuel),
        0.06,
        0.07,
        0.08,
        0.20,
        0.90,
    );
    assert_eq!(static_row.load_dead[3], 0.0);
    assert_eq!(static_row.load_live[0], 0.2);
}

#[test]
fn raw_default_rows_match_grid_particle_path() {
    let fuels = FuelModels::default();
    for model in -1..=261 {
        for mlh in [0.20, 0.30, 0.60, 1.20, 1.200001] {
            assert_eq!(
                build_particle_arrays(&fuels, model, 0.06, 0.07, 0.08, mlh, 0.90),
                build_particle_arrays_from_prepared(
                    &PreparedFuel::from_model(fuels.for_grid(model)),
                    0.06,
                    0.07,
                    0.08,
                    mlh,
                    0.90
                ),
                "model {model}, moisture {mlh}"
            );
        }
    }
}

#[test]
fn oracle_json_preserves_nextafter_one_boundaries() {
    let next: f64 = serde_json::from_str("1.0000000000000002").unwrap();
    let previous: f64 = serde_json::from_str("0.9999999999999999").unwrap();
    assert_eq!(next.to_bits(), 0x3ff0000000000001);
    assert_eq!(previous.to_bits(), 0x3fefffffffffffff);
}

#[test]
fn composed_pipeline_all_grid_models_and_custom_science_match_python() {
    use behave_fire::fuel_models::{CustomFuelModel, FuelModel};
    let fixture: Value =
        serde_json::from_str(include_str!("data/port_surface_composed.json")).unwrap();
    let mut fuels = FuelModels::default();
    for (number, record) in fixture["custom"].as_object().unwrap() {
        let mut model = FuelModel::default();
        model.code = record["code"].as_str().unwrap().into();
        model.name = record["name"].as_str().unwrap().into();
        model.fuel_bed_depth = number_value(&record["fuel_bed_depth"]);
        model.moisture_of_extinction_dead = number_value(&record["moisture_of_extinction_dead"]);
        model.heat_of_combustion_dead = number_value(&record["heat_of_combustion_dead"]);
        model.heat_of_combustion_live = number_value(&record["heat_of_combustion_live"]);
        model.dead_1h = number_value(&record["dead_1h"]);
        model.dead_10h = number_value(&record["dead_10h"]);
        model.dead_100h = number_value(&record["dead_100h"]);
        model.live_herb = number_value(&record["live_herb"]);
        model.live_woody = number_value(&record["live_woody"]);
        model.savr_1h = number_value(&record["savr_1h"]);
        model.savr_live_herb = number_value(&record["savr_live_herb"]);
        model.savr_live_woody = number_value(&record["savr_live_woody"]);
        model.is_dynamic = true;
        model.is_defined = true;
        assert!(fuels.set_custom_fuel_model(
            number.parse().unwrap(),
            CustomFuelModel {
                model,
                ..CustomFuelModel::default()
            }
        ));
    }
    for c in fixture["cases"].as_array().unwrap() {
        let mut x = input(&c["settings"], 0);
        x.fuel_model_grid = c["model"].as_i64().unwrap() as i32;
        x.m1h = number_value(&c["moisture"][0]);
        x.m10h = number_value(&c["moisture"][1]);
        x.m100h = number_value(&c["moisture"][2]);
        x.mlh = number_value(&c["moisture"][3]);
        x.mlw = number_value(&c["moisture"][4]);
        let result = SurfacePlan::new(fixture::surface_options(&c["settings"])).run_cell(
            &x,
            &PreparedFuel::from_model(fuels.for_grid(x.fuel_model_grid)),
            SurfaceTweaks::default(),
        );
        let ctx = format!("model {} {:?}", x.fuel_model_grid, c["moisture"]);
        assert_eq!(c["output"].as_object().unwrap().len(), 14);
        for (k, v) in result.named_values() {
            check(
                v,
                number_value(&c["output"][k]),
                &format!("{ctx}/{k}"),
                1e-9,
            );
        }
        let p = build_particle_arrays(
            &fuels,
            x.fuel_model_grid,
            x.m1h,
            x.m10h,
            x.m100h,
            x.mlh,
            x.mlw,
        );
        assert_eq!(
            p.is_defined,
            c["particles"]["is_defined"].as_bool().unwrap()
        );
        for (k, a) in p.named_arrays() {
            for n in 0..5 {
                check(
                    a[n],
                    number_value(&c["particles"][k][n]),
                    &format!("{ctx}/particles/{k}/{n}"),
                    1e-9,
                );
            }
        }
        let ib = calculate_fuelbed_intermediates(&p);
        assert_eq!(
            ib.is_defined,
            c["intermediates"]["is_defined"].as_bool().unwrap()
        );
        for (k, v) in ib.named_values() {
            check(
                v,
                number_value(&c["intermediates"][k]),
                &format!("{ctx}/intermediates/{k}"),
                1e-9,
            );
        }
    }
}
fn number_value(v: &Value) -> f64 {
    number(v)
}

fn same_direction_json(left: &Value, right: &Value) -> bool {
    // Numeric Value equality merges positive and negative zero; canonical bytes do not.
    serde_json::to_vec(left).unwrap() == serde_json::to_vec(right).unwrap()
}

fn surface_direction_cases_for_profile(
    cases: &Value,
    evidence: &Value,
    fingerprint: [u32; 2],
) -> Value {
    let profiles: Value =
        serde_json::from_str(include_str!("data/integer_direction_platform.json")).unwrap();
    assert_eq!(
        profiles["profiles"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|profile| profile["atan2_f32_bits"] == serde_json::json!(fingerprint))
            .count(),
        1,
        "unreviewed float32 atan2 profile: {fingerprint:08x?}"
    );
    assert_eq!(evidence["schema"], 1);
    assert_eq!(evidence["fixture"], "port_surface_single.json");
    assert_eq!(evidence["case_index"], 11);
    assert_eq!(evidence["cell_index"], 8);
    assert_eq!(
        evidence["atan2_f32_bits"],
        serde_json::json!([1070141531_u32, 1078530010_u32])
    );
    let original = &cases[11];
    assert_eq!(original["name"], "calculate_direction_of_max_spread");
    assert!(
        same_direction_json(original, &evidence["original_case"]),
        "surface direction operation/input/options/frozen-output identity drift"
    );
    let captured = &evidence["output"];
    assert_eq!(captured.as_object().unwrap().len(), 7);
    assert_eq!(captured["dtype"], original["output"]["dtype"]);
    let values = original["output"]["values"].as_array().unwrap();
    let observed = captured["values"].as_array().unwrap();
    assert_eq!(
        captured["values"][8],
        serde_json::json!(71.80140686035156_f64),
        "independently captured macOS value drift"
    );
    assert_eq!(observed.len(), values.len());
    assert_eq!(captured["shape"], serde_json::json!([values.len()]));
    assert_eq!(captured["masked_array"], false);
    assert_eq!(captured["scalar_array"], false);
    assert_eq!(captured["numpy_scalar"], false);
    assert_eq!(
        captured["masks"],
        serde_json::json!(vec![false; values.len()])
    );
    let changed: Vec<_> = values
        .iter()
        .zip(observed)
        .enumerate()
        .filter_map(|(index, (frozen, actual))| {
            (!same_direction_json(frozen, actual)).then_some(index)
        })
        .collect();
    assert_eq!(
        changed,
        vec![8],
        "capture changes outside the reviewed cell"
    );
    let mut selected = cases.clone();
    if evidence["atan2_f32_bits"] == serde_json::json!(fingerprint) {
        selected[11]["output"]["values"][8] = observed[8].clone();
    }
    selected
}

fn surface_direction_math_fingerprint() -> [u32; 2] {
    // Keep the two black-box libc sentinels identical to integer_surface_parity.
    let negative_one = std::hint::black_box(-1.0f32);
    [
        std::hint::black_box(65534.0f32)
            .atan2(negative_one)
            .to_bits(),
        std::hint::black_box(0.0f32).atan2(negative_one).to_bits(),
    ]
}

#[test]
fn surface_direction_oracle_changes_only_the_captured_cell() {
    let cases: Value = serde_json::from_str(include_str!("data/port_surface_single.json")).unwrap();
    let evidence: Value =
        serde_json::from_str(include_str!("data/surface_direction_platform.json")).unwrap();
    let fingerprint: [u32; 2] = serde_json::from_value(evidence["atan2_f32_bits"].clone()).unwrap();
    let actual = surface_direction_cases_for_profile(&cases, &evidence, fingerprint);
    let mut expected = cases.clone();
    expected[11]["output"]["values"][8] = evidence["output"]["values"][8].clone();
    assert!(
        same_direction_json(&actual, &expected),
        "only the captured cell may change"
    );
    check(
        number(&evidence["output"]["values"][8]),
        value(&actual[11]["output"], 8),
        "independent macOS direction capture",
        1e-9,
    );
    for profile in [[1070141531, 1078530011], [1070141530, 1078530011]] {
        assert!(
            same_direction_json(
                &surface_direction_cases_for_profile(&cases, &evidence, profile),
                &cases
            ),
            "other known profiles retain the frozen fixture"
        );
    }
}

#[test]
fn surface_direction_oracle_rejects_case_identity_drift() {
    let cases: Value = serde_json::from_str(include_str!("data/port_surface_single.json")).unwrap();
    let evidence: Value =
        serde_json::from_str(include_str!("data/surface_direction_platform.json")).unwrap();
    for (pointer, replacement) in [
        ("/11/name", serde_json::json!("another_operation")),
        ("/11/inputs/aspect", serde_json::json!(-849.0)),
        (
            "/11/inputs/wind_orientation_mode",
            serde_json::json!("Upslope"),
        ),
        ("/11/inputs/x_component/values/2", serde_json::json!(0.0)),
        ("/11/inputs/x_component/dtype", serde_json::json!("float64")),
        ("/11/output/values/8", serde_json::json!(71.0)),
        ("/11/output/dtype", serde_json::json!("float32")),
    ] {
        let mut changed = cases.clone();
        *changed.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            std::panic::catch_unwind(|| {
                surface_direction_cases_for_profile(&changed, &evidence, [1070141531, 1078530010])
            })
            .is_err(),
            "identity drift accepted at {pointer}"
        );
    }
}

#[test]
fn surface_direction_oracle_rejects_expanded_or_structural_overrides() {
    let cases: Value = serde_json::from_str(include_str!("data/port_surface_single.json")).unwrap();
    let evidence: Value =
        serde_json::from_str(include_str!("data/surface_direction_platform.json")).unwrap();
    for (pointer, replacement) in [
        ("/case_index", serde_json::json!(12)),
        ("/cell_index", serde_json::json!(7)),
        ("/output/values/7", serde_json::json!(126.0)),
        ("/output/values/8", serde_json::json!(70.0)),
        ("/output/dtype", serde_json::json!("float32")),
        ("/output/shape", serde_json::json!([2, 7])),
        ("/output/masks/0", serde_json::json!(true)),
        ("/atan2_f32_bits/0", serde_json::json!(0)),
    ] {
        let mut changed = evidence.clone();
        *changed.pointer_mut(pointer).unwrap() = replacement;
        assert!(
            std::panic::catch_unwind(|| {
                surface_direction_cases_for_profile(&cases, &changed, [1070141531, 1078530010])
            })
            .is_err(),
            "override expansion accepted at {pointer}"
        );
    }
}

#[test]
fn surface_direction_oracle_rejects_unknown_profiles() {
    let cases: Value = serde_json::from_str(include_str!("data/port_surface_single.json")).unwrap();
    let evidence: Value =
        serde_json::from_str(include_str!("data/surface_direction_platform.json")).unwrap();
    assert!(std::panic::catch_unwind(|| {
        surface_direction_cases_for_profile(&cases, &evidence, [0, 0])
    })
    .is_err());
}

#[test]
fn float32_thresholds_and_nonfinite_stages_match_numpy() {
    let cases: Value = serde_json::from_str(include_str!("data/port_surface_single.json")).unwrap();
    let evidence: Value =
        serde_json::from_str(include_str!("data/surface_direction_platform.json")).unwrap();
    let cases = surface_direction_cases_for_profile(
        &cases,
        &evidence,
        surface_direction_math_fingerprint(),
    );
    for c in cases.as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let i = &c["inputs"];
        let out = &c["output"];
        let cells = if let Some(v) = out["values"].as_array() {
            v.len()
        } else {
            out.as_object().unwrap().values().next().unwrap()["values"]
                .as_array()
                .unwrap()
                .len()
        };
        for n in 0..cells {
            let a = |k| f(i, k, n);
            let scalar = match name {
                "calculate_backing_spread_rate" => Some(single::calculate_backing_spread_rate(
                    a("spread_rate") as f32,
                    a("eccentricity") as f32,
                ) as f64),
                "calculate_eccentricity" => Some(single::calculate_eccentricity(a(
                    "fire_length_to_width_ratio",
                )
                    as f32) as f64),
                "calculate_effective_wind_speed" => Some(single::calculate_effective_wind_speed(
                    a("spread_rate") as f32,
                    a("no_wind_no_slope_spread_rate") as f32,
                    a("wind_b") as f32,
                    a("wind_c") as f32,
                    a("relative_packing_ratio") as f32,
                    a("wind_e") as f32,
                ) as f64),
                "calculate_fire_length" => Some(single::calculate_fire_length(
                    a("forward_ros") as f32,
                    a("backing_ros") as f32,
                    a("elapsed_min") as f32,
                ) as f64),
                "calculate_fire_length_to_width_ratio" => Some(
                    single::calculate_fire_length_to_width_ratio(a("effective_wind_speed") as f32)
                        as f64,
                ),
                "calculate_flame_length" => {
                    Some(single::calculate_flame_length(a("fireline_intensity") as f32) as f64)
                }
                "calculate_flanking_spread_rate" => Some(single::calculate_flanking_spread_rate(
                    a("spread_rate") as f32,
                    a("backing_spread_rate") as f32,
                    a("fire_length_to_width_ratio") as f32,
                ) as f64),
                "calculate_no_wind_no_slope_spread_rate" => {
                    Some(single::calculate_no_wind_no_slope_spread_rate(
                        a("ri") as f32,
                        a("propagating_flux") as f32,
                        a("heat_sink") as f32,
                    ) as f64)
                }
                "calculate_residence_time" => {
                    Some(single::calculate_residence_time(a("sigma") as f32) as f64)
                }
                "calculate_wind_phi" => Some(single::calculate_wind_phi(
                    a("sigma") as f32,
                    a("midflame_wind_speed") as f32,
                    a("wind_b") as f32,
                    a("wind_c") as f32,
                    a("relative_packing_ratio") as f32,
                    a("wind_e") as f32,
                ) as f64),
                "calculate_slope_phi" => Some(single::calculate_slope_phi(
                    a("slope"),
                    SlopeUnit::try_from(i["slope_units"].as_i64().unwrap() as i32).unwrap(),
                    a("packing_ratio") as f32,
                    a("reaction_intensity") as f32,
                )),
                "calculate_direction_of_max_spread" => {
                    Some(single::calculate_direction_of_max_spread(
                        a("x_component") as f32,
                        a("y_component") as f32,
                        a("rate_vector") as f32,
                        fixture::orientation(string(
                            i,
                            "wind_orientation_mode",
                            "RelativeToUpslope",
                        )),
                        a("aspect"),
                    ))
                }
                "calculate_reaction_intensity" => {
                    Some(single::calculate_reaction_intensity(&ib(&i["ib"], n)) as f64)
                }
                _ => None,
            };
            let ctx = format!("single/{name}/{n}");
            if let Some(x) = scalar {
                check(
                    x,
                    value(out, n),
                    &ctx,
                    if out["dtype"] == "float32" {
                        1e-6
                    } else {
                        1e-9
                    },
                );
            } else {
                let values = match name {
                    "calculate_fuelbed_intermediates" => {
                        single::calculate_fuelbed_intermediates(&particles(&i["p"], n))
                            .named_values()
                            .to_vec()
                    }
                    "calculate_surface_wind_coefficients" => {
                        single::calculate_surface_wind_coefficients(a("sigma") as f32)
                            .named_values()
                            .to_vec()
                    }
                    "calculate_surface_rate_vector" => single::calculate_surface_rate_vector(
                        a("no_wind_no_slope_spread_rate") as f32,
                        a("wind_phi") as f32,
                        a("slope_phi") as f32,
                        a("wind_direction"),
                        fixture::orientation(string(
                            i,
                            "wind_orientation_mode",
                            "RelativeToUpslope",
                        )),
                        a("aspect"),
                    )
                    .named_values()
                    .to_vec(),
                    _ => panic!("{name}"),
                };
                for (k, x) in values {
                    check(
                        x,
                        f(out, k, n),
                        &format!("{ctx}/{k}"),
                        if out[k]["dtype"] == "float32" {
                            1e-6
                        } else {
                            1e-9
                        },
                    );
                }
            }
        }
    }
}

fn number(v: &Value) -> f64 {
    if let Some(x) = v.get("float") {
        return number(x);
    }
    if let Some(x) = v.as_f64() {
        x
    } else if let Some(x) = v.as_bool() {
        if x {
            1.0
        } else {
            0.0
        }
    } else {
        match v
            .as_str()
            .unwrap_or_else(|| panic!("expected numeric value: {v}"))
        {
            "NaN" | "nan" => f64::NAN,
            "Infinity" | "inf" | "+inf" => f64::INFINITY,
            "-Infinity" | "-inf" => f64::NEG_INFINITY,
            x => panic!("{x}"),
        }
    }
}
fn value(v: &Value, n: usize) -> f64 {
    if v.is_number() || v.is_string() {
        number(v)
    } else {
        let a = v["values"].as_array().unwrap();
        number(&a[n % a.len()])
    }
}
fn f(i: &Value, k: &str, n: usize) -> f64 {
    value(&i[k], n)
}
fn string<'a>(i: &'a Value, k: &str, default: &'a str) -> &'a str {
    i[k].as_str().unwrap_or(default)
}
fn boolval(i: &Value, k: &str, n: usize) -> bool {
    i[k].as_bool().unwrap_or_else(|| {
        let a = i[k]["values"].as_array().unwrap();
        a[n % a.len()]
            .as_bool()
            .unwrap_or_else(|| number(&a[n % a.len()]) != 0.0)
    })
}
fn all_single(v: &Value) -> bool {
    if let Some(d) = v.get("dtype").and_then(Value::as_str) {
        d == "float32" || d == "bool"
    } else {
        v.as_object().map_or(true, |o| o.values().all(all_single))
    }
}
fn check(a: f64, b: f64, ctx: &str, tol: f64) {
    if a == 0.0 && b == 0.0 {
        assert_eq!(
            a.is_sign_negative(),
            b.is_sign_negative(),
            "zero sign {ctx}"
        );
    }
    if b.is_nan() {
        assert!(a.is_nan(), "{ctx}: {a} != NaN")
    } else if b.is_infinite() {
        assert_eq!(a, b, "{ctx}")
    } else {
        assert!(
            (a - b).abs() <= 1e-12 + tol * b.abs(),
            "{ctx}: {a:.17e} != {b:.17e}"
        );
    }
}
fn ib(i: &Value, n: usize) -> FuelbedIntermediates {
    let mut x = FuelbedIntermediates::new();
    x.sigma = f(i, "sigma", n);
    x.bulk_density = f(i, "bulk_density", n);
    x.packing_ratio = f(i, "packing_ratio", n);
    x.relative_packing_ratio = f(i, "relative_packing_ratio", n);
    x.heat_sink = f(i, "heat_sink", n);
    x.propagating_flux = f(i, "propagating_flux", n);
    x.w_heat_dead = f(i, "w_heat_dead", n);
    x.w_heat_live = f(i, "w_heat_live", n);
    x.w_silica_dead = f(i, "w_silica_dead", n);
    x.w_silica_live = f(i, "w_silica_live", n);
    x.w_moist_dead = f(i, "w_moist_dead", n);
    x.w_moist_live = f(i, "w_moist_live", n);
    x.wfl_dead = f(i, "wfl_dead", n);
    x.wfl_live = f(i, "wfl_live", n);
    x.moe_dead = f(i, "moe_dead", n);
    x.moe_live = f(i, "moe_live", n);
    x.frac_dead = f(i, "frac_dead", n);
    x.frac_live = f(i, "frac_live", n);
    x.depth = f(i, "depth", n);
    x.is_defined = boolval(i, "is_defined", n);
    x
}
fn particles(i: &Value, n: usize) -> Particles {
    let array = |k: &str| {
        let a = i[k]["values"].as_array().unwrap();
        let cells = a.len() / 5;
        std::array::from_fn(|p| number(&a[p * cells + n % cells]))
    };
    let mut x = Particles::new();
    x.load_dead = array("load_dead");
    x.load_live = array("load_live");
    x.savr_dead = array("savr_dead");
    x.savr_live = array("savr_live");
    x.moisture_dead = array("moisture_dead");
    x.moisture_live = array("moisture_live");
    x.hoc_dead = f(i, "hoc_dead", n);
    x.hoc_live = f(i, "hoc_live", n);
    x.depth = f(i, "depth", n);
    x.moe_dead = f(i, "moe_dead", n);
    x.is_defined = boolval(i, "is_defined", n);
    x
}
fn input(i: &Value, n: usize) -> SurfaceCell {
    fixture::surface_cell(i, n)
}

#[test]
fn every_public_stage_matches_explicit_python_oracle() {
    let fixture: Value =
        serde_json::from_str(include_str!("data/port_surface_surface.json")).unwrap();
    let fuels = FuelModels::default();
    let mut compared = 0;
    let mut stages = std::collections::BTreeSet::new();
    for c in fixture["cases"].as_array().unwrap() {
        if c.get("exception").is_some()
            || c.get("output").is_none()
            || c["policy"].as_str() == Some("diagnostic")
            || c["id"].as_str().unwrap().contains("/masked")
        {
            continue;
        }
        let name = c["entry_point"]
            .as_str()
            .unwrap()
            .rsplit('.')
            .next()
            .unwrap();
        let i = &c["inputs"];
        let out = &c["output"];
        stages.insert(name);
        // Scalar science is checked here; array rank, layout, masks, and errors belong to bindings.
        let n = if let Some(a) = out["values"].as_array() {
            a.len()
        } else {
            out.as_object()
                .unwrap()
                .values()
                .filter_map(|v| v["values"].as_array().map(Vec::len))
                .min()
                .unwrap()
        };
        for cell in 0..n {
            let ctx = format!("{}/{cell}", c["id"]);
            let a = |k| f(i, k, cell);
            let s = |k, d| string(i, k, d);
            let scalar = match name {
                "calculate_slope_phi"
                    if i["packing_ratio"]["dtype"] == "float32" && all_single(i) =>
                {
                    Some(single::calculate_slope_phi(
                        a("slope"),
                        SlopeUnit::try_from(i["slope_units"].as_i64().unwrap() as i32).unwrap(),
                        a("packing_ratio") as f32,
                        a("reaction_intensity") as f32,
                    ))
                }
                "calculate_flanking_spread_rate"
                    if i["spread_rate"]["dtype"] == "bool"
                        && i["backing_spread_rate"]["dtype"] == "bool" =>
                {
                    Some(calculate_flanking_spread_rate_bool(
                        boolval(i, "spread_rate", cell),
                        boolval(i, "backing_spread_rate", cell),
                        a("fire_length_to_width_ratio"),
                    ))
                }
                "calculate_fire_length"
                    if i["forward_ros"]["dtype"] == "bool"
                        && i["backing_ros"]["dtype"] == "bool" =>
                {
                    Some(calculate_fire_length_bool(
                        boolval(i, "forward_ros", cell),
                        boolval(i, "backing_ros", cell),
                        a("elapsed_min"),
                    ))
                }
                "calculate_backing_spread_rate"
                    if i["spread_rate"]["dtype"] == "float32" && all_single(i) =>
                {
                    Some(single::calculate_backing_spread_rate(
                        a("spread_rate") as f32,
                        a("eccentricity") as f32,
                    ) as f64)
                }
                "calculate_eccentricity"
                    if i["fire_length_to_width_ratio"]["dtype"] == "float32" && all_single(i) =>
                {
                    Some(
                        single::calculate_eccentricity(a("fire_length_to_width_ratio") as f32)
                            as f64,
                    )
                }
                "calculate_effective_wind_speed"
                    if i["spread_rate"]["dtype"] == "float32" && all_single(i) =>
                {
                    Some(single::calculate_effective_wind_speed(
                        a("spread_rate") as f32,
                        a("no_wind_no_slope_spread_rate") as f32,
                        a("wind_b") as f32,
                        a("wind_c") as f32,
                        a("relative_packing_ratio") as f32,
                        a("wind_e") as f32,
                    ) as f64)
                }
                "calculate_fire_length"
                    if i["forward_ros"]["dtype"] == "float32" && all_single(i) =>
                {
                    Some(single::calculate_fire_length(
                        a("forward_ros") as f32,
                        a("backing_ros") as f32,
                        a("elapsed_min") as f32,
                    ) as f64)
                }
                "calculate_fire_length_to_width_ratio"
                    if i["effective_wind_speed"]["dtype"] == "float32" && all_single(i) =>
                {
                    Some(single::calculate_fire_length_to_width_ratio(
                        a("effective_wind_speed") as f32
                    ) as f64)
                }
                "calculate_flame_length"
                    if i["fireline_intensity"]["dtype"] == "float32" && all_single(i) =>
                {
                    Some(single::calculate_flame_length(a("fireline_intensity") as f32) as f64)
                }
                "calculate_flanking_spread_rate"
                    if i["spread_rate"]["dtype"] == "float32" && all_single(i) =>
                {
                    Some(single::calculate_flanking_spread_rate(
                        a("spread_rate") as f32,
                        a("backing_spread_rate") as f32,
                        a("fire_length_to_width_ratio") as f32,
                    ) as f64)
                }
                "calculate_no_wind_no_slope_spread_rate"
                    if i["ri"]["dtype"] == "float32" && all_single(i) =>
                {
                    Some(single::calculate_no_wind_no_slope_spread_rate(
                        a("ri") as f32,
                        a("propagating_flux") as f32,
                        a("heat_sink") as f32,
                    ) as f64)
                }
                "calculate_residence_time" if i["sigma"]["dtype"] == "float32" && all_single(i) => {
                    Some(single::calculate_residence_time(a("sigma") as f32) as f64)
                }
                "calculate_wind_phi" if i["sigma"]["dtype"] == "float32" && all_single(i) => {
                    Some(single::calculate_wind_phi(
                        a("sigma") as f32,
                        a("midflame_wind_speed") as f32,
                        a("wind_b") as f32,
                        a("wind_c") as f32,
                        a("relative_packing_ratio") as f32,
                        a("wind_e") as f32,
                    ) as f64)
                }
                "calculate_reaction_intensity"
                    if i["ib"]["sigma"]["dtype"] == "float32" && all_single(i) =>
                {
                    Some(single::calculate_reaction_intensity(&ib(&i["ib"], cell)) as f64)
                }
                "calculate_direction_of_max_spread"
                    if i["x_component"]["dtype"] == "float32" && all_single(i) =>
                {
                    Some(single::calculate_direction_of_max_spread(
                        a("x_component") as f32,
                        a("y_component") as f32,
                        a("rate_vector") as f32,
                        fixture::orientation(s("wind_orientation_mode", "RelativeToUpslope")),
                        i.get("aspect").map_or(0., |v| value(v, cell)),
                    ))
                }
                "calculate_backing_spread_rate" => Some(calculate_backing_spread_rate(
                    a("spread_rate"),
                    a("eccentricity"),
                )),
                "calculate_direction_of_max_spread" => Some(calculate_direction_of_max_spread(
                    a("x_component"),
                    a("y_component"),
                    a("rate_vector"),
                    fixture::orientation(s("wind_orientation_mode", "RelativeToUpslope")),
                    i.get("aspect").map_or(0., |v| value(v, cell)),
                )),
                "calculate_eccentricity" => {
                    Some(calculate_eccentricity(a("fire_length_to_width_ratio")))
                }
                "calculate_effective_wind_speed" => Some(calculate_effective_wind_speed(
                    a("spread_rate"),
                    a("no_wind_no_slope_spread_rate"),
                    a("wind_b"),
                    a("wind_c"),
                    a("relative_packing_ratio"),
                    a("wind_e"),
                )),
                "calculate_fire_area" => Some(calculate_fire_area(
                    a("forward_ros"),
                    a("backing_ros"),
                    a("lwr"),
                    a("elapsed_min"),
                    i["is_crown"].as_bool().unwrap_or(false),
                )),
                "calculate_fire_length" => Some(calculate_fire_length(
                    a("forward_ros"),
                    a("backing_ros"),
                    a("elapsed_min"),
                )),
                "calculate_fire_length_to_width_ratio" => Some(
                    calculate_fire_length_to_width_ratio(a("effective_wind_speed")),
                ),
                "calculate_fire_perimeter" => Some(calculate_fire_perimeter(
                    a("forward_ros"),
                    a("backing_ros"),
                    a("lwr"),
                    a("elapsed_min"),
                    i["is_crown"].as_bool().unwrap_or(false),
                )),
                "calculate_fire_width" => Some(calculate_fire_width(
                    a("forward_ros"),
                    a("backing_ros"),
                    a("lwr"),
                    a("elapsed_min"),
                )),
                "calculate_fireline_intensity" => Some(calculate_fireline_intensity(
                    a("spread_rate"),
                    a("heat_per_unit_area"),
                )),
                "calculate_flame_length" => Some(calculate_flame_length(a("fireline_intensity"))),
                "calculate_flanking_spread_rate" => Some(calculate_flanking_spread_rate(
                    a("spread_rate"),
                    a("backing_spread_rate"),
                    a("fire_length_to_width_ratio"),
                )),
                "calculate_forward_spread_rate" => Some(calculate_forward_spread_rate(
                    a("no_wind_no_slope_spread_rate"),
                    a("rate_vector"),
                )),
                "calculate_heat_per_unit_area" => Some(calculate_heat_per_unit_area(
                    a("reaction_intensity"),
                    a("residence_time"),
                )),
                "calculate_midflame_wind_speed" => Some(calculate_midflame_wind_speed(
                    a("waf"),
                    a("twenty_foot_wind_speed"),
                )),
                "calculate_no_wind_no_slope_spread_rate" => {
                    Some(calculate_no_wind_no_slope_spread_rate(
                        a("ri"),
                        a("propagating_flux"),
                        a("heat_sink"),
                    ))
                }
                "calculate_reaction_intensity" => {
                    Some(calculate_reaction_intensity(&ib(&i["ib"], cell)))
                }
                "calculate_reaction_intensity_output" => {
                    Some(calculate_reaction_intensity_output(a("reaction_intensity")))
                }
                "calculate_residence_time" => Some(calculate_residence_time(a("sigma"))),
                "calculate_slope_phi" => Some(calculate_slope_phi(
                    a("slope"),
                    SlopeUnit::try_from(i["slope_units"].as_i64().unwrap() as i32).unwrap(),
                    a("packing_ratio"),
                    a("reaction_intensity"),
                )),
                "calculate_surface_wind_adjustment_factor" => {
                    Some(calculate_surface_wind_adjustment_factor(
                        a("canopy_cover"),
                        a("canopy_height"),
                        a("crown_ratio"),
                        a("depth"),
                        fixture::waf(s("waf_method", "UseCrownRatio")),
                        i.get("user_waf")
                            .filter(|v| !v.is_null())
                            .map(|v| value(v, cell)),
                    ))
                }
                "calculate_twenty_foot_wind_speed" => Some(calculate_twenty_foot_wind_speed(
                    a("wind_speed"),
                    SpeedUnit::try_from(i["wind_speed_units"].as_i64().unwrap() as i32).unwrap(),
                    fixture::height(s("wind_height_mode", "TwentyFoot")),
                )),
                "calculate_wind_adjustment_factor" => Some(calculate_wind_adjustment_factor(
                    a("canopy_cover"),
                    a("canopy_height"),
                    a("crown_ratio"),
                    a("depth"),
                )),
                "calculate_wind_phi" => Some(calculate_wind_phi(
                    a("sigma"),
                    a("midflame_wind_speed"),
                    a("wind_b"),
                    a("wind_c"),
                    a("relative_packing_ratio"),
                    a("wind_e"),
                )),
                _ => None,
            };
            if let Some(v) = scalar {
                let tol = if out["dtype"].as_str() == Some("float32") {
                    1e-6
                } else {
                    1e-9
                };
                check(v, value(out, cell), &ctx, tol);
            } else {
                let fields: Vec<(&str, f64)> = match name {
                    "calculate_fuelbed_intermediates" => {
                        calculate_fuelbed_intermediates(&particles(&i["p"], cell))
                            .named_values()
                            .to_vec()
                    }
                    "calculate_surface_wind_coefficients" => {
                        if i["sigma"]["dtype"] == "float32" && all_single(i) {
                            single::calculate_surface_wind_coefficients(a("sigma") as f32)
                                .named_values()
                                .to_vec()
                        } else {
                            calculate_surface_wind_coefficients(a("sigma"))
                                .named_values()
                                .to_vec()
                        }
                    }
                    "calculate_surface_rate_vector"
                        if i["no_wind_no_slope_spread_rate"]["dtype"] == "float32"
                            && all_single(i) =>
                    {
                        single::calculate_surface_rate_vector(
                            a("no_wind_no_slope_spread_rate") as f32,
                            a("wind_phi") as f32,
                            a("slope_phi") as f32,
                            a("wind_direction"),
                            fixture::orientation(s("wind_orientation_mode", "RelativeToUpslope")),
                            i.get("aspect").map_or(0., |v| value(v, cell)),
                        )
                        .named_values()
                        .to_vec()
                    }
                    "calculate_surface_rate_vector" => calculate_surface_rate_vector(
                        a("no_wind_no_slope_spread_rate"),
                        a("wind_phi"),
                        a("slope_phi"),
                        a("wind_direction"),
                        fixture::orientation(s("wind_orientation_mode", "RelativeToUpslope")),
                        i.get("aspect").map_or(0., |v| value(v, cell)),
                    )
                    .named_values()
                    .to_vec(),
                    "run_surface_fire" => {
                        if i["ri"]["dtype"] == "float32" {
                            fixture::surface_with_reaction(
                                a("ri"),
                                &ib(&i["ib"], cell),
                                &input(i, cell),
                                fixture::surface_options(i),
                                true,
                            )
                            .named_values()
                            .to_vec()
                        } else {
                            fixture::surface_with_reaction(
                                a("ri"),
                                &ib(&i["ib"], cell),
                                &input(i, cell),
                                fixture::surface_options(i),
                                false,
                            )
                            .named_values()
                            .to_vec()
                        }
                    }
                    "build_particle_arrays" => {
                        let p = build_particle_arrays(
                            &fuels,
                            a("fuel_model_grid") as i32,
                            a("m1h"),
                            a("m10h"),
                            a("m100h"),
                            a("mlh"),
                            a("mlw"),
                        );
                        for (k, arr) in p.named_arrays() {
                            let vals = out[k]["values"].as_array().unwrap();
                            let cells = vals.len() / 5;
                            for part in 0..5 {
                                check(
                                    arr[part],
                                    number(&vals[part * cells + cell % cells]),
                                    &format!("{ctx}/{k}/{part}"),
                                    1e-9,
                                );
                            }
                        }
                        assert_eq!(p.is_defined, boolval(out, "is_defined", cell));
                        p.named_values().to_vec()
                    }
                    _ => panic!("uncovered stage {name}"),
                };
                for (k, v) in fields {
                    let tol = if out[k]["dtype"] == "float32" {
                        1e-6
                    } else {
                        1e-9
                    };
                    check(v, f(out, k, cell), &format!("{ctx}/{k}"), tol);
                }
            }
            compared += 1;
        }
    }
    assert!(compared > 700, "only {compared} scalar comparisons");
    assert_eq!(stages.len(), 29);
    println!(
        "{compared} scalar oracle cells across {} public surface stages",
        stages.len()
    );
}
