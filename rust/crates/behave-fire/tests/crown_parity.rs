// Fixtures intentionally exercise public default construction and field mutation.
#![allow(clippy::field_reassign_with_default)]

use behave_fire::crown;
use behave_fire::{
    crown::*,
    fuel_models::FuelModels,
    surface::{FuelbedIntermediates, SurfaceResult},
    units::{SlopeUnit, SpeedUnit},
};
use serde_json::Value;

#[test]
fn classification_equal_ratios_are_active() {
    assert_eq!(crown::classify_crown_fire_type(1.0, 1.0).fire_type, 3);
    let below = f64::from_bits(1.0_f64.to_bits() - 1);
    let above = f64::from_bits(1.0_f64.to_bits() + 1);
    for (transition, expected) in [(below, [0, 2, 2]), (1., [1, 3, 3]), (above, [1, 3, 3])] {
        for (active, wanted) in [below, 1., above].into_iter().zip(expected) {
            assert_eq!(
                classify_crown_fire_type(transition, active).fire_type,
                wanted
            );
        }
    }
}

#[test]
fn python_slope_passthrough_and_changed_or_undefined_model_ten_match() {
    let cases: Value =
        serde_json::from_str(include_str!("data/port_crown_crown_coupling.json")).unwrap();
    let mut rates = Vec::new();
    for c in cases.as_array().unwrap() {
        let mut fuels = FuelModels::default();
        let changes = &c["model10_changes"];
        if let Some(v) = changes.get("dead_1h") {
            fuels.get_mut(10).unwrap().dead_1h = number(v);
        }
        if let Some(v) = changes.get("is_defined") {
            fuels.get_mut(10).unwrap().is_defined = v.as_bool().unwrap();
        }
        let x = input(&c["inputs"], 0);
        let result = run(&x, &fuels).unwrap();
        rates.push(result.crown_fire_spread_rate);
        assert_eq!(result.fire_type, number(&c["output"]["fire_type"]) as i32);
        for (k, v) in result.named_values() {
            check(
                v,
                &c["output"][k],
                0,
                &format!("slope {} changes {changes}/{k}", x.slope),
            );
        }
    }
    assert_ne!(
        rates[0], rates[1],
        "Python passes nonzero slope into the first model-10 run"
    );
    assert_ne!(
        rates[1], rates[2],
        "Supplied model-10 loading must affect crown spread"
    );
    assert_eq!(
        rates[3], 0.0,
        "Undefined model-10 surface outputs retain their reference mask"
    );
}

fn number(v: &Value) -> f64 {
    if let Some(x) = v.get("float") {
        return number(x);
    }
    if let Some(x) = v.as_f64() {
        return x;
    }
    if let Some(x) = v.as_bool() {
        return if x { 1.0 } else { 0.0 };
    }
    match v.as_str().unwrap_or_else(|| panic!("numeric value {v}")) {
        "NaN" | "nan" => f64::NAN,
        "Infinity" | "inf" | "+inf" => f64::INFINITY,
        "-Infinity" | "-inf" => f64::NEG_INFINITY,
        x => panic!("number {x}"),
    }
}
fn value(v: &Value, n: usize) -> f64 {
    if let Some(a) = v["values"].as_array() {
        number(&a[n % a.len()])
    } else {
        number(v)
    }
}
fn f(i: &Value, k: &str, n: usize) -> f64 {
    value(&i[k], n)
}
fn optional(i: &Value, k: &str, n: usize, default: f64) -> f64 {
    if i.get(k).is_some() {
        f(i, k, n)
    } else {
        default
    }
}
fn input(i: &Value, n: usize) -> CrownInput {
    let mut x = CrownInput::default();
    let model = optional(i, "fuel_model_grid", n, 10.0);
    x.fuel_model_grid = if model.is_finite() {
        model as i32
    } else {
        i32::MIN
    };
    x.m1h = optional(i, "m1h", n, x.m1h);
    x.m10h = optional(i, "m10h", n, x.m10h);
    x.m100h = optional(i, "m100h", n, x.m100h);
    x.mlh = optional(i, "mlh", n, x.mlh);
    x.mlw = optional(i, "mlw", n, x.mlw);
    x.wind_speed = optional(i, "wind_speed", n, 0.0);
    x.wind_direction = optional(i, "wind_direction", n, 0.0);
    x.wind_speed_units =
        SpeedUnit::try_from(i["wind_speed_units"].as_i64().unwrap_or(5) as i32).unwrap();
    x.wind_orientation_mode = i["wind_orientation_mode"]
        .as_str()
        .unwrap_or("RelativeToUpslope")
        .into();
    x.slope = optional(i, "slope", n, optional(i, "slope_deg", n, 0.0));
    x.slope_units = SlopeUnit::try_from(i["slope_units"].as_i64().unwrap_or(0) as i32).unwrap();
    x.aspect = optional(i, "aspect", n, 0.0);
    x.canopy_base_height = optional(i, "canopy_base_height", n, 0.0);
    x.canopy_height = optional(i, "canopy_height", n, 0.0);
    x.canopy_bulk_density = optional(i, "canopy_bulk_density", n, 0.0);
    x.moisture_foliar = optional(i, "moisture_foliar", n, 100.0);
    if i.get("surface_results").is_some() {
        x.surface_results = surface(&i["surface_results"], n)
    }
    x
}
fn surface(i: &Value, n: usize) -> SurfaceResult {
    let mut x = SurfaceResult::default();
    x.spread_rate = f(i, "spread_rate", n);
    x.fireline_intensity = f(i, "fireline_intensity", n);
    x.heat_per_unit_area = f(i, "heat_per_unit_area", n);
    x
}
fn ib(i: &Value, n: usize) -> FuelbedIntermediates {
    let mut x = FuelbedIntermediates::default();
    macro_rules! fields{($($key:ident),+)=>{$(x.$key=f(i,stringify!($key),n);)+}}
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
    x.is_defined = value(&i["is_defined"], n) != 0.0;
    x
}
fn cells(v: &Value) -> usize {
    if let Some(a) = v["values"].as_array() {
        return a.len();
    }
    if v.get("crown_surface").is_some() {
        return cells(&v["crown_fire_spread_rate"]);
    }
    if v.is_array() {
        return 0;
    }
    v.as_object()
        .map(|o| o.values().map(cells).max().unwrap_or(1))
        .unwrap_or(1)
}

#[test]
fn composed_default_custom_fuels_and_canopy_units_match_python_facade() {
    use behave_fire::{
        fuel_models::{CustomFuelModel, FuelModel},
        surface::{self, SurfaceInput},
        units::{DensityUnit, LengthUnit},
    };
    let fixture: Value =
        serde_json::from_str(include_str!("data/port_crown_crown_composed.json")).unwrap();
    let mut fuels = FuelModels::default();
    let custom = &fixture["custom"];
    let mut model = FuelModel::default();
    model.code = custom["code"].as_str().unwrap().into();
    model.name = custom["name"].as_str().unwrap().into();
    macro_rules! fields{($($k:ident),+)=>{$(model.$k=number(&custom[stringify!($k)]);)+}}
    fields!(
        fuel_bed_depth,
        moisture_of_extinction_dead,
        heat_of_combustion_dead,
        heat_of_combustion_live,
        dead_1h,
        dead_10h,
        dead_100h,
        live_herb,
        live_woody,
        savr_1h,
        savr_live_herb,
        savr_live_woody
    );
    model.is_dynamic = true;
    model.is_defined = true;
    assert!(fuels.set_custom_fuel_model(
        250,
        CustomFuelModel {
            model,
            ..CustomFuelModel::default()
        }
    ));
    let mut classes = std::collections::BTreeSet::new();
    for c in fixture["cases"].as_array().unwrap() {
        let i = &c["inputs"];
        let canopy = &c["canopy"];
        let mut x = input(i, 0);
        x.canopy_base_height =
            LengthUnit::try_from(canopy["canopy_base_height_units"].as_i64().unwrap() as i32)
                .unwrap()
                .to_base(number(&canopy["canopy_base_height"]));
        x.canopy_height =
            LengthUnit::try_from(canopy["canopy_height_units"].as_i64().unwrap() as i32)
                .unwrap()
                .to_base(x.canopy_height);
        x.canopy_bulk_density =
            DensityUnit::try_from(canopy["canopy_bulk_density_units"].as_i64().unwrap() as i32)
                .unwrap()
                .to_base(number(&canopy["canopy_bulk_density"]));
        x.moisture_foliar = number(&canopy["moisture_foliar"]);
        let mut s = SurfaceInput::new(x.fuel_model_grid);
        s.m1h = x.m1h;
        s.m10h = x.m10h;
        s.m100h = x.m100h;
        s.mlh = x.mlh;
        s.mlw = x.mlw;
        s.wind_speed = x.wind_speed;
        s.wind_speed_units = x.wind_speed_units;
        s.wind_direction = x.wind_direction;
        s.wind_orientation_mode = x.wind_orientation_mode.clone();
        s.slope = x.slope;
        s.slope_units = x.slope_units;
        s.aspect = x.aspect;
        s.canopy_height = x.canopy_height;
        x.surface_results = surface::run(&s, &fuels).unwrap();
        let ctx = format!("model {} slope {}", x.fuel_model_grid, x.slope);
        for (k, v) in x.surface_results.named_values() {
            check(v, &c["surface"][k], 0, &format!("{ctx}/surface/{k}"))
        }
        let result = run(&x, &fuels).unwrap();
        classes.insert(result.fire_type);
        assert_eq!(
            result.fire_type,
            number(&c["output"]["fire_type"]) as i32,
            "{ctx}"
        );
        assert_eq!(
            result.named_values().len() + 1,
            c["output"].as_object().unwrap().len()
        );
        for (k, v) in result.named_values() {
            check(v, &c["output"][k], 0, &format!("{ctx}/{k}"))
        }
        assert_eq!(
            run_crown_fire(&x.surface_results, &x, &fuels).unwrap(),
            result
        );
    }
    assert_eq!(classes, std::collections::BTreeSet::from([0, 1, 2, 3]));
}

#[test]
fn supplied_surface_and_live_model_ten_table_are_used() {
    let mut fuels = FuelModels::default();
    let mut s = SurfaceResult::default();
    s.spread_rate = 4.0;
    s.heat_per_unit_area = 900.;
    s.fireline_intensity = 60.;
    let mut x = CrownInput::new(s);
    x.canopy_base_height = 6.;
    x.canopy_height = 30.;
    x.canopy_bulk_density = 0.02;
    x.wind_speed = 5.;
    let original = run(&x, &fuels).unwrap();
    fuels.get_mut(10).unwrap().dead_1h *= 2.0;
    let changed = run(&x, &fuels).unwrap();
    assert_ne!(
        original.crown_fire_spread_rate,
        changed.crown_fire_spread_rate
    );
    let mut other = s;
    other.fireline_intensity = 0.0;
    let overridden = run_crown_fire(&other, &x, &fuels).unwrap();
    assert_eq!(overridden.crown_fire_transition_ratio, 0.0);
    assert_eq!(overridden.final_spread_rate, other.spread_rate);
    assert_eq!(x.moisture_foliar, 100.0);
}
fn check(actual: f64, expected: &Value, n: usize, ctx: &str) {
    let target = value(expected, n);
    let tolerance = if expected["dtype"] == "float32" {
        1e-6
    } else {
        1e-9
    };
    assert!(
        if target.is_nan() {
            actual.is_nan()
        } else if target.is_infinite() {
            actual == target
        } else {
            (actual - target).abs() <= 1e-12 + tolerance * target.abs()
        },
        "{ctx}/{n}: actual {actual:?}, expected {target:?}"
    );
}
fn truth(v: &Value, n: usize) -> bool {
    if let Some(a) = v["values"].as_array() {
        return truth(&a[n % a.len()], 0);
    }
    if let Some(s) = v.as_str() {
        return !s.is_empty();
    }
    if v.is_null() {
        return false;
    }
    number(v) != 0.0
}

#[test]
fn every_source_stage_matches_independent_python_oracle() {
    let fixture: Value = serde_json::from_str(include_str!("data/port_crown_crown.json")).unwrap();
    let fuels = FuelModels::default();
    let mut count = 0;
    for c in fixture["cases"].as_array().unwrap() {
        let i = &c["inputs"];
        let out = &c["output"];
        let id = c["id"].as_str().unwrap();
        let name = c["entry_point"]
            .as_str()
            .unwrap()
            .rsplit('.')
            .next()
            .unwrap();
        for n in 0..cells(out) {
            count += 1;
            let a = |k| f(i, k, n);
            let mut vals: Vec<(&str, f64)> = Vec::new();
            let single = out["dtype"] == "float32"
                || out
                    .as_object()
                    .unwrap()
                    .values()
                    .any(|v| v["dtype"] == "float32");
            macro_rules! scalar {($fun:ident,$($k:literal),+)=>{{if single{Some(single::$fun($(a($k) as f32),+) as f64)}else{Some(crown::$fun($(a($k)),+))}}};}
            macro_rules! record {($fun:ident,$($k:literal),+)=>{{if single{single::$fun($(a($k) as f32),+).named_values().into_iter().map(|(k,v)|(k,v as f64)).collect()}else{crown::$fun($(a($k)),+).named_values()}}};}
            let scalar = match name {
                "calculate_critical_crown_fire_spread_rate" => scalar!(
                    calculate_critical_crown_fire_spread_rate,
                    "canopy_bulk_density"
                ),
                "calculate_critical_surface_fireline_intensity" => {
                    if !single && i["canopy_base_height"]["dtype"] == "float32" {
                        Some(calculate_critical_surface_fireline_intensity_single_height(
                            a("canopy_base_height") as f32,
                            a("moisture_foliar"),
                        ))
                    } else {
                        scalar!(
                            calculate_critical_surface_fireline_intensity,
                            "canopy_base_height",
                            "moisture_foliar"
                        )
                    }
                }
                "calculate_crown_fraction_burned" => scalar!(
                    calculate_crown_fraction_burned,
                    "surface_ros",
                    "surface_critical_ros",
                    "crowning_surface_ros"
                ),
                "calculate_surface_fire_critical_spread_rate" => {
                    if !single && i["critical_surface_fli"]["dtype"] == "float32" {
                        Some(
                            calculate_surface_fire_critical_spread_rate_single_intensity(
                                a("critical_surface_fli") as f32,
                                a("surface_hpua"),
                            ),
                        )
                    } else {
                        scalar!(
                            calculate_surface_fire_critical_spread_rate,
                            "critical_surface_fli",
                            "surface_hpua"
                        )
                    }
                }
                "calculate_crowning_surface_fire_spread_rate" => {
                    Some(if i["critical_crown_ros"]["dtype"] == "float32" {
                        calculate_crowning_surface_fire_spread_rate_single_critical_rate(
                            a("ri_crown"),
                            &ib(&i["ib_crown"], n),
                            a("critical_crown_ros") as f32,
                            a("canopy_height"),
                        )
                    } else {
                        calculate_crowning_surface_fire_spread_rate(
                            a("ri_crown"),
                            &ib(&i["ib_crown"], n),
                            a("critical_crown_ros"),
                            a("canopy_height"),
                        )
                    })
                }
                "calculate_crown_fire_ratios" => {
                    vals = record!(
                        calculate_crown_fire_ratios,
                        "surface_fli",
                        "crown_ros",
                        "critical_surface_fli",
                        "critical_crown_ros"
                    );
                    None
                }
                "calculate_crown_heat_and_intensity" => {
                    vals = record!(
                        calculate_crown_heat_and_intensity,
                        "surface_hpua",
                        "crown_ros",
                        "canopy_base_height",
                        "canopy_height",
                        "canopy_bulk_density"
                    );
                    None
                }
                "calculate_passive_crown_fire_behavior" => {
                    vals = record!(
                        calculate_passive_crown_fire_behavior,
                        "surface_ros",
                        "surface_hpua",
                        "crown_ros",
                        "canopy_hpua",
                        "cfb"
                    );
                    None
                }
                "assign_final_crown_fire_behavior" => {
                    let args = [
                        "surface_ros",
                        "surface_hpua",
                        "surface_fli",
                        "passive_ros",
                        "passive_hpua",
                        "passive_fli",
                        "passive_fl",
                        "crown_ros",
                        "crown_hpua",
                        "crown_fli",
                        "crown_fl",
                    ]
                    .map(a);
                    let b = truth(&i["is_surface_fire"], n);
                    let p = truth(&i["is_passive_crown"], n);
                    vals = if single {
                        let q = args.map(|v| v as f32);
                        single::assign_final_crown_fire_behavior(
                            b, p, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10],
                        )
                        .named_values()
                        .into_iter()
                        .map(|(k, v)| (k, v as f64))
                        .collect()
                    } else if i["surface_fli"]["dtype"] == "float32" {
                        assign_final_crown_fire_behavior_single_surface(
                            b,
                            p,
                            args[0] as f32,
                            args[1] as f32,
                            args[2] as f32,
                            args[3],
                            args[4],
                            args[5],
                            args[6],
                            args[7],
                            args[8],
                            args[9],
                            args[10],
                        )
                        .named_values()
                    } else {
                        assign_final_crown_fire_behavior(
                            b, p, args[0], args[1], args[2], args[3], args[4], args[5], args[6],
                            args[7], args[8], args[9], args[10],
                        )
                        .named_values()
                    };
                    None
                }
                "classify_crown_fire_type" => {
                    let r = classify_crown_fire_type(a("transition_ratio"), a("active_ratio"));
                    assert_eq!(r.fire_type, f(out, "fire_type", n) as i32, "{id}");
                    for (k, b) in [
                        ("is_surface_fire", r.is_surface_fire),
                        ("is_passive_crown", r.is_passive_crown),
                        ("is_active_crown", r.is_active_crown),
                    ] {
                        assert_eq!(b, f(out, k, n) != 0.0, "{id}/{k}")
                    }
                    None
                }
                "coerce_crown_fire_inputs" => {
                    let r = coerce_crown_fire_inputs(&input(i, n));
                    assert_eq!(r.fm_grid, f(out, "fm_grid", n) as i32);
                    vals = vec![
                        ("slope_deg", r.slope_deg),
                        ("aspect", r.aspect),
                        ("wind_dir", r.wind_dir),
                        ("cbh", r.cbh),
                        ("ch", r.ch),
                        ("cbd", r.cbd),
                        ("mf_pct", r.mf_pct),
                        ("wind_speed", r.wind_speed),
                    ];
                    None
                }
                "get_surface_fire_inputs" => {
                    vals =
                        get_surface_fire_inputs(&surface(&i["surface_results"], n)).named_values();
                    None
                }
                "run_crown_fire" => {
                    let x = input(i, n);
                    let r = run(&x, &fuels).unwrap();
                    assert_eq!(r.fire_type, f(out, "fire_type", n) as i32, "{id}");
                    vals = r.named_values().to_vec();
                    None
                }
                "calculate_crown_surface_fire" => {
                    let r = calculate_crown_surface_fire(&input(i, n), &fuels).unwrap();
                    for (k, v) in r.crown_surface.named_values() {
                        check(
                            v,
                            &out["crown_surface"][k],
                            n,
                            &format!("{id}/crown_surface/{k}"),
                        );
                    }
                    for (k, v) in r.ib_crown.named_values() {
                        check(v, &out["ib_crown"][k], n, &format!("{id}/ib/{k}"));
                    }
                    for (k, v) in r.p_crown.named_values() {
                        check(v, &out["p_crown"][k], n, &format!("{id}/particles/{k}"));
                    }
                    let len = out["p_crown"]["load_dead"]["values"]
                        .as_array()
                        .unwrap()
                        .len()
                        / 5;
                    for (k, vs) in r.p_crown.named_arrays() {
                        for (j, v) in vs.into_iter().enumerate() {
                            check(
                                v,
                                &out["p_crown"][k],
                                j * len + n % len,
                                &format!("{id}/particles/{k}"),
                            );
                        }
                    }
                    assert_eq!(
                        r.p_crown.is_defined,
                        f(&out["p_crown"], "is_defined", n) != 0.0
                    );
                    vals = vec![
                        ("ri_crown", r.ri_crown),
                        ("crown_fire_spread_rate", r.crown_fire_spread_rate),
                        ("crown_length_to_width_ratio", r.crown_length_to_width_ratio),
                    ];
                    None
                }
                _ => panic!("missing {name}"),
            };
            if let Some(v) = scalar {
                check(v, out, n, id)
            }
            for (k, v) in vals {
                check(v, &out[k], n, &format!("{id}/{k}"))
            }
        }
    }
    println!("Checked {count} scalar cells across all 14 Python Crown entry points");
}
