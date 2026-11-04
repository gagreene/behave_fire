//! Exported Python oracle parity, reference 7933789.
#![allow(unused_imports)]
use behave_fire::units::*;
use behave_fire::{
    contain, fine_dead_fuel_moisture, fire_size, ignite, mortality, safety, slope, spot,
    vapor_pressure_deficit,
};
fn close(actual: f64, expected: f64) {
    assert!(
        (actual.is_nan() && expected.is_nan())
            || actual == expected
            || (actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= (1e-12 + 1e-9 * expected.abs())),
        "actual={actual:?}, expected={expected:?}"
    );
}
// run_contain_sim_array/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_324() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_325() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_326() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_327() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_328() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_329() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_330() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_331() {
    let actual = contain::run_contain_sim_array(1.0, 1.0, 1.0, 1.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.0);
    close(actual.final_area as f64, 0.0007853981633974483);
    close(actual.final_perimeter as f64, 0.6283185307179586);
    close(actual.status as f64, 8.0);
}

// run_contain_sim_array/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_332() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_333() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_334() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_335() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_336() {}

// run_contain_sim_array/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_337() {
    let actual = contain::run_contain_sim_array(0.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.01);
    close(actual.final_area as f64, 0.0);
    close(actual.final_perimeter as f64, 0.0);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_338() {
    let actual = contain::run_contain_sim_array(-1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.01);
    close(actual.final_area as f64, 9.015130433855008e-10);
    close(actual.final_perimeter as f64, -0.0007139983303613166);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_339() {
    let actual = contain::run_contain_sim_array(f64::NAN, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.0);
    close(actual.final_area as f64, f64::NAN);
    close(actual.final_perimeter as f64, f64::NAN);
    close(actual.status as f64, 8.0);
}

// run_contain_sim_array/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_340() {
    let actual = contain::run_contain_sim_array(f64::INFINITY, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.0);
    close(actual.final_area as f64, f64::NAN);
    close(actual.final_perimeter as f64, f64::NAN);
    close(actual.status as f64, 8.0);
}

// run_contain_sim_array/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_341() {
    let actual = contain::run_contain_sim_array(f64::NEG_INFINITY, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.01);
    close(actual.final_area as f64, f64::INFINITY);
    close(actual.final_perimeter as f64, f64::NEG_INFINITY);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_342() {
    let actual =
        contain::run_contain_sim_array(9.999999999999998e-08, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.01);
    close(actual.final_area as f64, 9.015130433855002e-24);
    close(actual.final_perimeter as f64, 7.139983303613164e-11);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_343() {
    let actual = contain::run_contain_sim_array(1e-07, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.01);
    close(actual.final_area as f64, 9.015130433855007e-24);
    close(actual.final_perimeter as f64, 7.139983303613166e-11);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_344() {
    let actual =
        contain::run_contain_sim_array(1.0000000000000001e-07, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.01);
    close(actual.final_area as f64, 9.015130433855011e-24);
    close(actual.final_perimeter as f64, 7.139983303613167e-11);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_345() {
    let actual = contain::run_contain_sim_array(300.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.0);
    close(actual.final_area as f64, 0.00039269908169872427);
    close(actual.final_perimeter as f64, 0.4712388980384691);
    close(actual.status as f64, 8.0);
}

// run_contain_sim_array/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_346() {
    let actual = contain::run_contain_sim_array(300.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.0);
    close(actual.final_area as f64, 0.00039269908169872427);
    close(actual.final_perimeter as f64, 0.4712388980384691);
    close(actual.status as f64, 8.0);
}

// run_contain_sim_array/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_347() {
    let actual = contain::run_contain_sim_array(0.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.01);
    close(actual.final_area as f64, 0.0);
    close(actual.final_perimeter as f64, 0.0);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_348() {
    let actual = contain::run_contain_sim_array(0.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.01);
    close(actual.final_area as f64, 0.0);
    close(actual.final_perimeter as f64, 0.0);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_349() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
    let actual = contain::run_contain_sim_array(2.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.33);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_350() {
    let actual = contain::run_contain_sim_array(0.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.01);
    close(actual.final_area as f64, 0.0);
    close(actual.final_perimeter as f64, 0.0);
    close(actual.status as f64, 3.0);
    let actual = contain::run_contain_sim_array(2.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.33);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_351() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// run_contain_sim_array/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_352() {
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
    let actual = contain::run_contain_sim_array(1.0, 2.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 3.926990816987241e-06);
    close(actual.final_perimeter as f64, 0.0471238898038469);
    close(actual.status as f64, 3.0);
}

// containment/mixed-terminal
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_353() {
    let actual = contain::run_contain_sim_array(1.0, 1.0, 1.0, 1000.0, 0.0, 10, 0.01);
    close(actual.contained_time as f64, 0.66);
    close(actual.final_area as f64, 7.853981633974482e-06);
    close(actual.final_perimeter as f64, 0.06283185307179587);
    close(actual.status as f64, 3.0);
    let actual = contain::run_contain_sim_array(1.0, 1.0, 1e-08, 0.0, 1000.0, 10, 0.01);
    close(actual.contained_time as f64, 0.0);
    close(actual.final_area as f64, 7.853981633974482e-06);
    close(actual.final_perimeter as f64, 0.06283185307179587);
    close(actual.status as f64, 4.0);
    let actual = contain::run_contain_sim_array(1.0, 1.0, 1.0, 0.0, 1000.0, 10, 0.01);
    close(actual.contained_time as f64, 0.0);
    close(actual.final_area as f64, 0.0007853981633974483);
    close(actual.final_perimeter as f64, 0.6283185307179586);
    close(actual.status as f64, 8.0);
}

// containment/no-steps
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_354() {
    let actual = contain::run_contain_sim_array(1.0, 1.0, 1.0, 1000.0, 0.0, 0, 0.01);
    close(actual.contained_time as f64, 0.0);
    close(actual.final_area as f64, 0.0);
    close(actual.final_perimeter as f64, 0.0);
    close(actual.status as f64, 8.0);
    let actual = contain::run_contain_sim_array(1.0, 1.0, 1e-08, 0.0, 1000.0, 0, 0.01);
    close(actual.contained_time as f64, 0.0);
    close(actual.final_area as f64, 0.0);
    close(actual.final_perimeter as f64, 0.0);
    close(actual.status as f64, 8.0);
    let actual = contain::run_contain_sim_array(1.0, 1.0, 1.0, 0.0, 1000.0, 0, 0.01);
    close(actual.contained_time as f64, 0.0);
    close(actual.final_area as f64, 0.0);
    close(actual.final_perimeter as f64, 0.0);
    close(actual.status as f64, 8.0);
}

#[test]
fn default_controls_and_terminal_freezing() {
    let input = contain::ContainInput {
        report_spread_rate: 10.0,
        lw_ratio: 2.0,
        report_size: 1.0,
        production_rate: 1000.0,
        attack_time: 0.0,
        ..Default::default()
    };
    let result = contain::run(input);
    assert_eq!(
        result,
        contain::run_contain_sim_array(10.0, 2.0, 1.0, 1000.0, 0.0, 500, 0.01)
    );
    assert_eq!(result.status, contain::STATUS_CONTAINED);
    assert_eq!(
        result,
        contain::run(contain::ContainInput {
            max_steps: 100000,
            ..input
        })
    );
    assert_eq!(
        result.named_values().map(|entry| entry.0),
        ["status", "contained_time", "final_perimeter", "final_area"]
    );
    assert_eq!(
        contain::run(contain::ContainInput {
            max_steps: 0,
            ..input
        })
        .status,
        contain::STATUS_TIME_LIMIT
    );
    assert_eq!(
        contain::run(contain::ContainInput {
            max_steps: -10,
            ..input
        })
        .final_area,
        0.0
    );
}

#[test]
#[should_panic]
fn comparator_rejects_near_zero_error_above_absolute_tolerance() {
    close(5e-10, 0.0);
}
