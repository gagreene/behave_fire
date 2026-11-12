//! New missing-data rules are independent of historical MaskedArray arithmetic.
use behave_fire::batch::CrownBatch;
use behave_fire::engine::{
    run_model, CrownPlan, ModelBatch, ModelRun, OutputRequest, PreparedFuels,
};
use behave_fire::raster::{CrownColumns, CrownRaster, CrownTweakColumns};
use behave_fire::{FuelModels, GridShape, PerCell};

fn columns(wind: f64, intensity: f64, heat: f64, density: f64) -> CrownColumns<'static> {
    CrownColumns {
        wind_speed: PerCell::Scalar(wind),
        wind_direction: PerCell::Scalar(35.),
        slope: PerCell::Scalar(12.),
        canopy_base_height: PerCell::Scalar(4.),
        canopy_height: PerCell::Scalar(30.),
        canopy_bulk_density: PerCell::Scalar(density),
        surface_ros: PerCell::Scalar(2.),
        surface_fli: PerCell::Scalar(intensity),
        surface_hpua: PerCell::Scalar(heat),
        ..CrownColumns::default()
    }
}
fn run(inputs: CrownColumns<'_>) -> CrownBatch {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let raster = CrownRaster::new(GridShape::new(vec![1]).unwrap(), inputs).unwrap();
    match run_model(
        &ModelRun::Crown {
            raster,
            plan: CrownPlan::default(),
            fuels: &fuels,
            tweaks: CrownTweakColumns::default(),
        },
        &OutputRequest::crown(),
        1,
    )
    .unwrap()
    {
        ModelBatch::Crown(result) => result,
        _ => panic!("wrong model result"),
    }
}

#[test]
fn explicit_invalid_cells_keep_all_missing_payloads_and_codes() {
    let mut input = columns(8., 32.729299106026254, 1000., 0.01);
    input.validity = Some(PerCell::Scalar(false));
    let output = run(input);
    assert_eq!(output.fire_type, [-1]);
    for (_, column) in output.into_named_columns() {
        if let behave_fire::output::ColumnValues::Float64(values) = column {
            assert!(values[0].is_nan());
        }
    }
}

#[test]
fn missing_density_is_not_a_valid_zero_critical_rate() {
    let output = run(columns(8., 32.729299106026254, 1000., f64::NAN));
    assert!(output.crown_critical_fire_spread_rate[0].is_nan());
    assert!(output.crown_fire_active_ratio[0].is_nan());
    assert!(output.crown_critical_surface_fire_line_intensity[0].is_finite());
    assert!(output.crown_fire_transition_ratio[0].is_finite());
    assert!(output.crown_fire_spread_rate[0].is_finite());
    assert_eq!(output.fire_type, [-1]);
    assert!(output.final_spread_rate[0].is_nan());
}

#[test]
fn missing_heat_preserves_independent_outputs_and_selected_final_fields() {
    for (kind, wind, intensity) in [
        (0, 0., 32.729299106026254),
        (1, 0., 130.91719642410501),
        (2, 5000., 32.729299106026254),
        (3, 5000., 130.91719642410501),
    ] {
        let finite = run(columns(wind, intensity, 1000., 0.01));
        assert_eq!(finite.fire_type[0], kind);
        let output = run(columns(wind, intensity, f64::NAN, 0.01));
        assert_eq!(output.fire_type[0], kind);
        assert_eq!(output.crown_fire_spread_rate, finite.crown_fire_spread_rate);
        assert_eq!(
            output.canopy_heat_per_unit_area,
            finite.canopy_heat_per_unit_area
        );
        assert_eq!(
            output.crown_fire_transition_ratio,
            finite.crown_fire_transition_ratio
        );
        assert_eq!(
            output.crowning_surface_fire_spread_rate,
            finite.crowning_surface_fire_spread_rate
        );
        assert!(output.crown_fire_heat_per_unit_area[0].is_nan());
        assert!(output.crown_fire_line_intensity[0].is_nan());
        assert!(output.crown_flame_length[0].is_nan());
        assert!(output.surface_fire_critical_spread_rate[0].is_nan());
        assert!(output.crown_fraction_burned[0].is_nan());
        assert!(output.passive_crown_fire_spread_rate[0].is_nan());
        assert!(output.passive_crown_fire_heat_per_unit_area[0].is_nan());
        assert!(output.passive_crown_fire_line_intensity[0].is_nan());
        assert!(output.passive_crown_fire_flame_length[0].is_nan());
        assert!(output.final_heat_per_unit_area[0].is_nan());
        if kind == 0 || kind == 2 {
            assert_eq!(output.final_spread_rate, vec![2.]);
            assert_eq!(output.final_fireline_intensity, vec![intensity]);
            assert_eq!(output.final_flame_length, finite.final_flame_length);
        } else {
            assert!(output.final_fireline_intensity[0].is_nan());
            assert!(output.final_flame_length[0].is_nan());
            if kind == 1 {
                assert!(output.final_spread_rate[0].is_nan());
            } else {
                assert_eq!(output.final_spread_rate, finite.crown_fire_spread_rate);
            }
        }
    }
}

#[test]
fn missing_intensity_cannot_become_a_surface_classification() {
    let output = run(columns(8., f64::NAN, 1000., 0.01));
    assert!(output.crown_fire_transition_ratio[0].is_nan());
    assert!(output.crown_fire_active_ratio[0].is_finite());
    assert_eq!(output.fire_type, [-1]);
    assert!(output.final_spread_rate[0].is_nan());
    assert!(output.final_heat_per_unit_area[0].is_nan());
    assert!(output.final_fireline_intensity[0].is_nan());
    assert!(output.final_flame_length[0].is_nan());
}

#[test]
fn unselected_missing_branch_does_not_poison_surface_assignment() {
    let output = run(columns(f64::NAN, 32.729299106026254, 1000., 0.));
    assert!(output.crown_fire_spread_rate[0].is_nan());
    assert_eq!(output.crown_critical_fire_spread_rate, vec![0.]);
    assert_eq!(output.crown_fire_active_ratio, vec![0.]);
    assert_eq!(output.fire_type, vec![0]);
    assert_eq!(output.final_spread_rate, vec![2.]);
    assert_eq!(output.final_heat_per_unit_area, vec![1000.]);
    assert_eq!(output.final_fireline_intensity, vec![32.729299106026254]);
    assert!(output.final_flame_length[0].is_finite());
}

#[test]
fn valid_zero_retains_its_finite_threshold_rule() {
    let output = run(columns(8., 32.729299106026254, 1000., 0.));
    assert_eq!(output.crown_critical_fire_spread_rate, vec![0.]);
    assert_eq!(output.crown_fire_active_ratio, vec![0.]);
    assert_eq!(output.fire_type, vec![0]);
    assert!(output.final_spread_rate[0].is_finite());
}

#[test]
fn supplied_fm10_stage_overrides_bypass_unavailable_upstream_heat_sink() {
    use behave_fire::raster::SurfaceTweakColumns;
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let mut inputs = columns(f64::NAN, 32.729299106026254, 1000., 0.01);
    inputs.m1h = PerCell::Scalar(f64::NAN);
    let raster = CrownRaster::new(GridShape::new(vec![1]).unwrap(), inputs).unwrap();
    let tweaks = CrownTweakColumns {
        fm10: SurfaceTweakColumns {
            reaction_intensity: Some(PerCell::Scalar(913.)),
            no_wind_no_slope_spread_rate: Some(PerCell::Scalar(3.1)),
            midflame_wind_speed: Some(PerCell::Scalar(61.)),
            wind_phi: Some(PerCell::Scalar(2.7)),
            slope_phi: Some(PerCell::Scalar(0.9)),
            ..SurfaceTweakColumns::default()
        },
        ..CrownTweakColumns::default()
    };
    let output = match run_model(
        &ModelRun::Crown {
            raster,
            plan: CrownPlan::default(),
            fuels: &fuels,
            tweaks,
        },
        &OutputRequest::crown(),
        1,
    )
    .unwrap()
    {
        ModelBatch::Crown(output) => output,
        _ => panic!("wrong result"),
    };
    assert!(
        output.crown_fire_spread_rate[0].is_finite(),
        "override closes the spread dependency chain"
    );
    assert!(output.crown_length_to_width_ratio[0].is_finite());
    assert!(
        output.crowning_surface_fire_spread_rate[0].is_nan(),
        "backsolve still needs unavailable fuelbed sink"
    );
}

#[test]
fn widened_f32_threshold_uses_actual_f64_value_instead_of_f32_comparison() {
    let widened = (1e-7_f32) as f64;
    let output = run(columns(8., 32.729299106026254, widened, 0.01));
    assert!(output.surface_fire_critical_spread_rate[0] > 0.);
    let at = run(columns(8., 32.729299106026254, 1e-7_f64, 0.01));
    assert_eq!(at.surface_fire_critical_spread_rate, vec![0.]);
}

#[test]
fn transition_uses_adjacent_f64_values_and_accepts_positive_infinity() {
    let critical = run(columns(0., 0., 1000., 0.01)).crown_critical_surface_fire_line_intensity[0];
    for (intensity, code) in [
        (f64::from_bits(critical.to_bits() - 1), 0),
        (critical, 1),
        (f64::from_bits(critical.to_bits() + 1), 1),
        (f64::INFINITY, 1),
    ] {
        assert_eq!(run(columns(0., intensity, 1000., 0.01)).fire_type, [code]);
    }
    for zero in [0., -0.] {
        let output = run(columns(0., zero, 1000., 0.01));
        assert_eq!(output.fire_type, [0]);
        assert_eq!(output.final_fireline_intensity[0].to_bits(), zero.to_bits());
    }
}
