use behave_fire::{
    surface::masked::{self, MaskedValue, MathProfile, Precision},
    units::SlopeUnit,
};
use serde_json::Value;

#[test]
fn mixed_rank_conditions_and_nan_fill_match_python() {
    let fixture:Value=serde_json::from_str(r###"[{"name":"calculate_direction_of_max_spread","inputs":{"x_component":{"values":[3.0],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"y_component":{"values":[4.0],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"rate_vector":{"values":[2.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float32","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"aspect":{"values":[95.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{"wind_orientation_mode":"RelativeToNorth"},"output":{"values":[328.13010025024414,275.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"<class 'numpy.float32'>"},{"name":"calculate_direction_of_max_spread","inputs":{"x_component":{"values":[3.0],"masks":[true],"masked_array":true,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":{"float":"nan"}},"y_component":{"values":[4.0],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"rate_vector":{"values":[2.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float32","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"aspect":{"values":[95.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{"wind_orientation_mode":"RelativeToNorth"},"output":{"values":[275.0,275.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"<class 'numpy.float32'>"},{"name":"calculate_wind_phi","inputs":{"sigma":{"values":[1500.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float32","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"midflame_wind_speed":{"values":[100.0],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_b":{"values":[1.2000000476837158],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_c":{"values":[0.4000000059604645],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"relative_packing_ratio":{"values":[0.30000001192092896],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_e":{"values":[0.6000000238418579],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{},"output":{"values":[206.91281127929688,0.0],"masks":[false,false],"masked_array":false,"dtype":"float32","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"<class 'numpy.float32'>"},{"name":"calculate_wind_phi","inputs":{"sigma":{"values":[1500.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float32","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"midflame_wind_speed":{"values":[100.0],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_b":{"values":[1.2000000476837158],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_c":{"values":[0.4000000059604645],"masks":[false],"masked_array":true,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":{"float":"nan"}},"relative_packing_ratio":{"values":[0.30000001192092896],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_e":{"values":[0.6000000238418579],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{},"output":{"values":[206.91281127929688,0.0],"masks":[false,false],"masked_array":false,"dtype":"float32","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"<class 'numpy.float32'>"},{"name":"calculate_wind_phi","inputs":{"sigma":{"values":[1500.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float32","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"midflame_wind_speed":{"values":[100.0],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_b":{"values":[1.2000000476837158],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_c":{"values":[0.4000000059604645],"masks":[true],"masked_array":true,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":"nan"},"relative_packing_ratio":{"values":[0.30000001192092896],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_e":{"values":[0.6000000238418579],"masks":[false],"masked_array":false,"dtype":"float32","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{},"output":{"values":[0.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"<class 'numpy.float32'>"},{"name":"calculate_direction_of_max_spread","inputs":{"x_component":{"values":[3.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"y_component":{"values":[4.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"rate_vector":{"values":[2.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"aspect":{"values":[95.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{"wind_orientation_mode":"RelativeToNorth"},"output":{"values":[328.13010235415595,275.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"<class 'numpy.float64'>"},{"name":"calculate_direction_of_max_spread","inputs":{"x_component":{"values":[3.0],"masks":[true],"masked_array":true,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":"nan"},"y_component":{"values":[4.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"rate_vector":{"values":[2.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"aspect":{"values":[95.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{"wind_orientation_mode":"RelativeToNorth"},"output":{"values":[275.0,275.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"<class 'numpy.float64'>"},{"name":"calculate_wind_phi","inputs":{"sigma":{"values":[1500.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"midflame_wind_speed":{"values":[100.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_b":{"values":[1.2],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_c":{"values":[0.4],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"relative_packing_ratio":{"values":[0.3],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_e":{"values":[0.6],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{},"output":{"values":[206.9127431887146,0.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"<class 'numpy.float64'>"},{"name":"calculate_wind_phi","inputs":{"sigma":{"values":[1500.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"midflame_wind_speed":{"values":[100.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_b":{"values":[1.2],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_c":{"values":[0.4],"masks":[false],"masked_array":true,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":"nan"},"relative_packing_ratio":{"values":[0.3],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_e":{"values":[0.6],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{},"output":{"values":[206.9127431887146,0.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"<class 'numpy.float64'>"},{"name":"calculate_wind_phi","inputs":{"sigma":{"values":[1500.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"midflame_wind_speed":{"values":[100.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_b":{"values":[1.2],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_c":{"values":[0.4],"masks":[true],"masked_array":true,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":{"float":"nan"}},"relative_packing_ratio":{"values":[0.3],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"wind_e":{"values":[0.6],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{},"output":{"values":[0.0,0.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"<class 'numpy.float64'>"},{"name":"calculate_fire_area","inputs":{"forward_ros":{"values":[12.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"backing_ros":{"values":[2.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"elapsed_min":{"values":[10.0,20.0],"masks":[false,true],"masked_array":true,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":"nan"},"lwr":{"values":[2.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{"is_crown":false},"output":{"values":[7696.902001294993,439.822971502571],"masks":[false,true],"masked_array":true,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":"nan"},"profile":"mixedrank","precision":"float64"},{"name":"calculate_fire_area","inputs":{"forward_ros":{"values":[12.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"backing_ros":{"values":[2.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"elapsed_min":{"values":[10.0,20.0],"masks":[false,true],"masked_array":true,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":{"float":"+inf"}},"lwr":{"values":[2.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{"is_crown":false},"output":{"values":[7696.902001294993,439.822971502571],"masks":[false,true],"masked_array":true,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":{"float":"+inf"}},"profile":"mixedrank","precision":"float64"},{"name":"calculate_fire_perimeter","inputs":{"forward_ros":{"values":[12.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"backing_ros":{"values":[2.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"elapsed_min":{"values":[10.0,20.0],"masks":[false,true],"masked_array":true,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":"nan"},"lwr":{"values":[2.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{"is_crown":false},"output":{"values":[339.0938389955441,3.141592653589793],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"float64"},{"name":"calculate_fire_width","inputs":{"forward_ros":{"values":[12.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"backing_ros":{"values":[2.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"elapsed_min":{"values":[10.0,20.0],"masks":[false,true],"masked_array":true,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":{"float":"-inf"}},"lwr":{"values":[2.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null}},"options":{},"output":{"values":[70.0,280.0],"masks":[false,false],"masked_array":false,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"profile":"mixedrank","precision":"float64"},{"name":"calculate_fire_length","inputs":{"forward_ros":{"values":[12.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"backing_ros":{"values":[2.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"elapsed_min":{"values":[10.0,20.0],"masks":[false,true],"masked_array":true,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":"nan"}},"options":{},"output":{"values":[140.0,280.0],"masks":[false,true],"masked_array":true,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":"nan"},"profile":"mixedrank","precision":"float64"},{"name":"calculate_fire_length","inputs":{"forward_ros":{"values":[12.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"backing_ros":{"values":[2.0],"masks":[false],"masked_array":false,"dtype":"float64","scalar_array":true,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":null},"elapsed_min":{"values":[10.0,20.0],"masks":[false,true],"masked_array":true,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":{"float":"-inf"}}},"options":{},"output":{"values":[140.0,280.0],"masks":[false,true],"masked_array":true,"dtype":"float64","scalar_array":false,"numpy_scalar":false,"math_profile":"numpy-simd","fill_value":{"float":"-inf"}},"profile":"mixedrank","precision":"float64"}]"###).unwrap();
    replay(&fixture, 32);
}

#[test]
fn dictionary_selection_broadcasts_condition_rank() {
    let sigma = MaskedValue::plain(1631.0).with_scalar_array(true);
    for (rpr, expected) in [(1.1, 1631.0), (0.0, 1.0)] {
        let ratio = MaskedValue::plain(rpr);
        let valid = sigma
            .greater_condition(MaskedValue::scalar(1e-7))
            .and(ratio.greater_condition(MaskedValue::scalar(0.0)));
        let safe = MaskedValue::select_with_condition(valid, sigma, MaskedValue::scalar(1.0));
        assert!(!safe.is_scalar_array);
        assert_eq!(safe.value, expected);
        assert!(!safe.is_masked_array);
        assert!(safe.fill_value.is_none());
    }
}

#[test]
fn mixed_rank_particle_reduction_uses_lazy_default_fill() {
    use behave_fire::surface::masked::{try_calculate_fuelbed_intermediates, MaskedParticles};
    let mut p = MaskedParticles::new();
    p.load_dead = std::array::from_fn(|n| {
        MaskedValue::masked(if n == 0 { 1.0 } else { 0.0 }, false).with_fill_value(Some(f64::NAN))
    });
    p.savr_dead = std::array::from_fn(|n| MaskedValue::plain(if n == 0 { 1.0 } else { 0.0 }));
    p.moisture_dead = [MaskedValue::plain(0.1); 5];
    p.hoc_dead = MaskedValue::masked(10.0, false)
        .with_scalar_array(true)
        .with_fill_value(Some(f64::NAN));
    p.hoc_live = MaskedValue::plain(0.0).with_scalar_array(true);
    p.depth = MaskedValue::plain(1.0).with_scalar_array(true);
    p.moe_dead = MaskedValue::plain(0.3).with_scalar_array(true);
    p.is_defined = MaskedValue::plain(1.0).with_scalar_array(true);
    let result = try_calculate_fuelbed_intermediates(&p).unwrap();
    assert_eq!(result.w_heat_dead.value, 10.0);
    assert!(result.w_heat_dead.is_masked_array);
    assert!(!result.w_heat_dead.is_scalar_array);
    assert!(result.w_heat_dead.fill_value.is_none());
    assert!(result.is_defined.is_scalar_array);
}

#[test]
fn masked_dictionary_fields_particles_reductions_and_runs_match_python() {
    use behave_fire::surface::{
        masked::{MaskedFuelbedIntermediates, MaskedParticles},
        SurfaceInput,
    };
    let cases: Value =
        serde_json::from_str(include_str!("data/port_surface_masked_dict.json")).unwrap();
    for (index, c) in cases.as_array().unwrap().iter().enumerate() {
        let i = &c["inputs"];
        let stage = c["stage"].as_str().unwrap();
        let ctx = format!("dictionary/{index}/{stage}/{}", c["label"]);
        let scalar_cell = c["scalar_cell"].as_bool().unwrap();
        let output = if stage == "fuelbed" {
            let mut p = MaskedParticles::new();
            p.scalar_cell = scalar_cell;
            p.load_dead = std::array::from_fn(|n| argument(&i["load_dead"], n));
            p.load_live = std::array::from_fn(|n| argument(&i["load_live"], n));
            p.savr_dead = std::array::from_fn(|n| argument(&i["savr_dead"], n));
            p.savr_live = std::array::from_fn(|n| argument(&i["savr_live"], n));
            p.moisture_dead = std::array::from_fn(|n| argument(&i["moisture_dead"], n));
            p.moisture_live = std::array::from_fn(|n| argument(&i["moisture_live"], n));
            p.hoc_dead = argument(&i["hoc_dead"], 0);
            p.hoc_live = argument(&i["hoc_live"], 0);
            p.depth = argument(&i["depth"], 0);
            p.moe_dead = argument(&i["moe_dead"], 0);
            p.is_defined = argument(&i["is_defined"], 0);
            match masked::try_calculate_fuelbed_intermediates(&p) {
                Ok(x) => Some(x.named_values().to_vec()),
                Err(_) => None,
            }
        } else {
            let mut ib = MaskedFuelbedIntermediates::new();
            ib.sigma = argument(&i["sigma"], 0);
            ib.bulk_density = argument(&i["bulk_density"], 0);
            ib.packing_ratio = argument(&i["packing_ratio"], 0);
            ib.relative_packing_ratio = argument(&i["relative_packing_ratio"], 0);
            ib.heat_sink = argument(&i["heat_sink"], 0);
            ib.propagating_flux = argument(&i["propagating_flux"], 0);
            ib.w_heat_dead = argument(&i["w_heat_dead"], 0);
            ib.w_heat_live = argument(&i["w_heat_live"], 0);
            ib.w_silica_dead = argument(&i["w_silica_dead"], 0);
            ib.w_silica_live = argument(&i["w_silica_live"], 0);
            ib.w_moist_dead = argument(&i["w_moist_dead"], 0);
            ib.w_moist_live = argument(&i["w_moist_live"], 0);
            ib.wfl_dead = argument(&i["wfl_dead"], 0);
            ib.wfl_live = argument(&i["wfl_live"], 0);
            ib.moe_dead = argument(&i["moe_dead"], 0);
            ib.moe_live = argument(&i["moe_live"], 0);
            ib.frac_dead = argument(&i["frac_dead"], 0);
            ib.frac_live = argument(&i["frac_live"], 0);
            ib.depth = argument(&i["depth"], 0);
            ib.is_defined = argument(&i["is_defined"], 0);
            if stage == "reaction" {
                match masked::try_calculate_reaction_intensity(&ib) {
                    Ok(x) => Some(vec![("", x)]),
                    Err(_) => None,
                }
            } else {
                let mut input = SurfaceInput::new(c["model"].as_i64().unwrap() as i32);
                input.wind_speed = 5.;
                input.wind_direction = 45.;
                input.wind_orientation_mode = "RelativeToNorth".into();
                input.slope = 30.;
                input.aspect = 95.;
                input.canopy_cover = 0.5;
                input.canopy_height = 30.;
                input.crown_ratio = 0.5;
                let ri = MaskedValue::new(500., false, false, ib.sigma.precision)
                    .with_scalar_array(scalar_cell);
                match masked::try_run_surface_fire_with_intermediates(ri, &ib, &input) {
                    Ok(x) => Some(
                        x.named_values()
                            .into_iter()
                            .map(|(k, v)| (k, MaskedValue::plain(v).with_scalar_array(scalar_cell)))
                            .collect(),
                    ),
                    Err(_) => None,
                }
            }
        };
        if c.get("exception").is_some() {
            assert!(output.is_none(), "missing source TypeError: {ctx}");
            continue;
        }
        let values = output.unwrap_or_else(|| panic!("unexpected TypeError: {ctx}"));
        for (key, result) in values {
            if key == "is_defined" {
                assert_metadata(result, &c["output"][key], &format!("{ctx}/{key}"));
                assert_eq!(
                    result.is_masked_array,
                    c["output"][key]["masked_array"].as_bool().unwrap(),
                    "array kind {ctx}/{key}"
                );
                assert_eq!(
                    result.value,
                    number(&c["output"][key]["values"][0]),
                    "{ctx}/{key}"
                );
                assert_eq!(
                    result.mask,
                    c["output"][key]["masks"][0].as_bool().unwrap(),
                    "{ctx}/{key}"
                );
            } else {
                check(
                    result,
                    if key.is_empty() {
                        &c["output"]
                    } else {
                        &c["output"][key]
                    },
                    0,
                    &format!("{ctx}/{key}"),
                );
            }
        }
    }
}

#[test]
fn composed_surface_runs_preserve_reaction_mask_intermediates() {
    use behave_fire::{
        fuel_models::FuelModels,
        surface::{self, SurfaceInput},
        units::SpeedUnit,
    };
    let cases: Value =
        serde_json::from_str(include_str!("data/port_surface_masked_run.json")).unwrap();
    let fuels = FuelModels::default();
    for (index, c) in cases.as_array().unwrap().iter().enumerate() {
        let model = c["fuel"].as_i64().unwrap() as i32;
        let p = surface::build_particle_arrays(&fuels, model, 0.06, 0.07, 0.08, 0.60, 0.90);
        let ib = surface::calculate_fuelbed_intermediates(&p);
        let mut input = SurfaceInput::new(model);
        let opts = &c["options"];
        input.wind_speed = 5.;
        input.wind_speed_units = SpeedUnit::MilesPerHour;
        input.wind_direction = 45.;
        input.wind_orientation_mode = opts["wind_orientation_mode"].as_str().unwrap().into();
        input.slope = 30.;
        input.slope_units = SlopeUnit::Degrees;
        input.aspect = 95.;
        input.canopy_cover = 0.5;
        input.canopy_height = 30.;
        input.crown_ratio = 0.5;
        let result = masked::run_surface_fire(argument(&c["ri"], 0), &ib, &input);
        for (key, value) in result.named_values() {
            check(
                MaskedValue::plain(value),
                &c["output"][key],
                0,
                &format!("masked run {index}/{model}/{key}"),
            );
        }
    }
}

fn number(v: &Value) -> f64 {
    if let Some(tag) = v.get("float") {
        return number(tag);
    }
    if let Some(boolean) = v.as_bool() {
        return f64::from(u8::from(boolean));
    }
    v.as_f64().unwrap_or_else(|| match v.as_str().unwrap() {
        "nan" => f64::NAN,
        "+inf" => f64::INFINITY,
        "-inf" => f64::NEG_INFINITY,
        x => panic!("{x}"),
    })
}
fn argument(v: &Value, n: usize) -> MaskedValue {
    let n = n % v["values"].as_array().unwrap().len();
    let p = if v["dtype"] == "float32" {
        Precision::F32
    } else {
        Precision::F64
    };
    MaskedValue::new(
        number(&v["values"][n]),
        v["masks"][n].as_bool().unwrap(),
        v["masked_array"].as_bool().unwrap(),
        p,
    )
    .with_scalar_array(v["scalar_array"].as_bool().unwrap_or(false))
    .with_fill_value(v.get("fill_value").filter(|v| !v.is_null()).map(number))
    .with_numpy_scalar(v["numpy_scalar"].as_bool().unwrap_or(false))
    .with_math_profile(if v["math_profile"] == "libm" {
        MathProfile::Libm
    } else {
        MathProfile::NumpySimd
    })
}
fn check(actual: MaskedValue, expected: &Value, n: usize, context: &str) {
    let n = n % expected["values"].as_array().unwrap().len();
    assert_metadata(actual, expected, context);
    assert_eq!(
        actual.mask,
        expected["masks"][n].as_bool().unwrap(),
        "mask {context}"
    );
    assert_eq!(
        actual.is_masked_array,
        expected["masked_array"].as_bool().unwrap(),
        "array kind {context}"
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

fn assert_metadata(actual: MaskedValue, expected: &Value, context: &str) {
    assert_eq!(
        actual.is_scalar_array,
        expected["scalar_array"].as_bool().unwrap_or(false),
        "rank {context}"
    );
    if expected["masked_array"] == true {
        let default = if expected["dtype"] == "bool" {
            1.0
        } else if expected["dtype"] == "float32" {
            (1e20f32) as f64
        } else {
            1e20
        };
        let expected_fill = expected
            .get("fill_value")
            .filter(|x| !x.is_null())
            .map(number)
            .unwrap_or(default);
        let actual_fill = actual.fill_value.unwrap_or(default);
        if expected_fill.is_nan() {
            assert!(actual_fill.is_nan(), "fill {context}: {actual_fill} != NaN");
        } else {
            assert_eq!(actual_fill, expected_fill, "fill {context}");
        }
    } else {
        assert!(
            actual.fill_value.is_none(),
            "plain result retains fill metadata {context}"
        );
    }
}

#[test]
fn every_input_mask_combination_matches_numpy_ma_values_masks_and_dtypes() {
    let fixture: Value =
        serde_json::from_str(include_str!("data/port_surface_masked.json")).unwrap();
    replay(&fixture, 22536);
}

#[test]
fn scalar_masked_constants_and_custom_fill_values_match_numpy_ma() {
    let fixture: Value =
        serde_json::from_str(include_str!("data/port_surface_masked_metadata.json")).unwrap();
    replay(&fixture, 2496);
}

#[test]
fn elapsed_time_masks_preserve_geometry_intermediates() {
    let fixture: Value =
        serde_json::from_str(include_str!("data/port_surface_masked_geometry.json")).unwrap();
    replay(&fixture, 3456);
}

fn replay(fixture: &Value, expected_count: usize) {
    let mut count = 0;
    for (index, c) in fixture.as_array().unwrap().iter().enumerate() {
        let i = &c["inputs"];
        let options = &c["options"];
        let name = c["name"].as_str().unwrap();
        let n = if let Some(values) = c["output"]["values"].as_array() {
            values.len()
        } else {
            i.as_object()
                .unwrap()
                .values()
                .filter_map(|x| x["values"].as_array().map(Vec::len))
                .max()
                .unwrap()
        };
        for cell in 0..n {
            let a = |k| argument(&i[k], cell);
            let orientation = options["wind_orientation_mode"]
                .as_str()
                .unwrap_or("RelativeToUpslope");
            let condition_operands = match name {
                "calculate_effective_wind_speed" => {
                    vec![a("wind_b"), a("wind_c"), a("relative_packing_ratio")]
                }
                "calculate_wind_phi" => vec![a("sigma"), a("midflame_wind_speed")],
                "calculate_slope_phi" => vec![a("packing_ratio")],
                _ => vec![],
            };
            let valid = masked::validate_condition_and(&condition_operands);
            if c.get("exception").is_some() {
                assert!(
                    valid.is_err(),
                    "expected rejected boolean expression: {index}/{name}"
                );
                count += 1;
                continue;
            }
            assert!(
                valid.is_ok(),
                "unexpected rejected boolean expression: {index}/{name}"
            );
            let scalar = match name {
                "calculate_fire_length" => Some(masked::calculate_fire_length(
                    a("forward_ros"),
                    a("backing_ros"),
                    a("elapsed_min"),
                )),
                "calculate_fire_width" => Some(masked::calculate_fire_width(
                    a("forward_ros"),
                    a("backing_ros"),
                    a("lwr"),
                    a("elapsed_min"),
                )),
                "calculate_fire_area" => Some(masked::calculate_fire_area(
                    a("forward_ros"),
                    a("backing_ros"),
                    a("lwr"),
                    a("elapsed_min"),
                    options["is_crown"].as_bool().unwrap(),
                )),
                "calculate_fire_perimeter" => Some(masked::calculate_fire_perimeter(
                    a("forward_ros"),
                    a("backing_ros"),
                    a("lwr"),
                    a("elapsed_min"),
                    options["is_crown"].as_bool().unwrap(),
                )),
                "calculate_backing_spread_rate" => Some(masked::calculate_backing_spread_rate(
                    a("spread_rate"),
                    a("eccentricity"),
                )),
                "calculate_eccentricity" => Some(masked::calculate_eccentricity(a(
                    "fire_length_to_width_ratio",
                ))),
                "calculate_fire_length_to_width_ratio" => Some(
                    masked::calculate_fire_length_to_width_ratio(a("effective_wind_speed")),
                ),
                "calculate_effective_wind_speed" => Some(masked::calculate_effective_wind_speed(
                    a("spread_rate"),
                    a("no_wind_no_slope_spread_rate"),
                    a("wind_b"),
                    a("wind_c"),
                    a("relative_packing_ratio"),
                    a("wind_e"),
                )),
                "calculate_flanking_spread_rate" => Some(masked::calculate_flanking_spread_rate(
                    a("spread_rate"),
                    a("backing_spread_rate"),
                    a("fire_length_to_width_ratio"),
                )),
                "calculate_no_wind_no_slope_spread_rate" => {
                    Some(masked::calculate_no_wind_no_slope_spread_rate(
                        a("ri"),
                        a("propagating_flux"),
                        a("heat_sink"),
                    ))
                }
                "calculate_wind_phi" => Some(masked::calculate_wind_phi(
                    a("sigma"),
                    a("midflame_wind_speed"),
                    a("wind_b"),
                    a("wind_c"),
                    a("relative_packing_ratio"),
                    a("wind_e"),
                )),
                "calculate_slope_phi" => Some(masked::calculate_slope_phi(
                    a("slope"),
                    SlopeUnit::try_from(options["slope_units"].as_i64().unwrap() as i32).unwrap(),
                    a("packing_ratio"),
                    a("reaction_intensity"),
                )),
                "calculate_direction_of_max_spread" => {
                    Some(masked::calculate_direction_of_max_spread(
                        a("x_component"),
                        a("y_component"),
                        a("rate_vector"),
                        orientation,
                        a("aspect"),
                    ))
                }
                "calculate_surface_rate_vector" => None,
                _ => panic!("{name}"),
            };
            let context = format!(
                "case {index}/{name}/{cell}/{}/{}",
                c["profile"], c["precision"]
            );
            if let Some(result) = scalar {
                check(result, &c["output"], cell, &context);
            } else {
                let v = masked::calculate_surface_rate_vector(
                    a("no_wind_no_slope_spread_rate"),
                    a("wind_phi"),
                    a("slope_phi"),
                    a("wind_direction"),
                    orientation,
                    a("aspect"),
                );
                for (key, value) in v.named_values() {
                    check(value, &c["output"][key], cell, &format!("{context}/{key}"));
                }
            }
            count += 1;
        }
    }
    assert_eq!(count, expected_count);
}
