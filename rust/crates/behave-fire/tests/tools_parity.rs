//! Exported Python oracle parity, reference 7933789.
#![allow(unused_imports)]
// These exported state traces retain the reference's uniform casts, literal
// expectations, None-return capture, and ordered post-construction mutations.
#![allow(
    clippy::bool_assert_comparison,
    clippy::field_reassign_with_default,
    clippy::let_unit_value,
    clippy::unnecessary_cast
)]
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
// calculate_vpd/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_266() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
}

// calculate_vpd/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_267() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.300048828125,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.3118279956154);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.8250698644665);
}

// calculate_vpd/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_268() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.300048828125,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.3118279956154);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.8250698644665);
}

// calculate_vpd/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_269() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.30000001192092896,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1411110471452);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.995786812937);
}

// calculate_vpd/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_270() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
}

// calculate_vpd/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_271() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.0,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 0.0);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 3497.136897860082);
}

// calculate_vpd/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_272() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.0,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 0.0);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 3497.136897860082);
}

// calculate_vpd/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_273() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        1.0,
        TemperatureUnit::try_from(0).unwrap(),
        1.0,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 158.17123036179956);
    close(actual.saturated_vp as f64, 158.17123036179956);
    close(actual.vpd as f64, 0.0);
}

// calculate_vpd/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_274() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
}

// calculate_vpd/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_275() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
}

// calculate_vpd/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_276() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
}

// calculate_vpd/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_277() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
    let actual = vapor_pressure_deficit::calculate_vpd(
        80.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 1049.1410693580247);
    close(actual.saturated_vp as f64, 3497.136897860082);
    close(actual.vpd as f64, 2447.9958285020575);
}

// calculate_vpd/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_278() {}

// calculate_vpd/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_279() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        0.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 45.2671134869953);
    close(actual.saturated_vp as f64, 150.89037828998434);
    close(actual.vpd as f64, 105.62326480298903);
}

// calculate_vpd/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_280() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        -1.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 43.1730772883861);
    close(actual.saturated_vp as f64, 143.91025762795365);
    close(actual.vpd as f64, 100.73718033956757);
}

// calculate_vpd/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_281() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        f64::NAN,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, f64::NAN);
    close(actual.saturated_vp as f64, f64::NAN);
    close(actual.vpd as f64, f64::NAN);
}

// calculate_vpd/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_282() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        f64::INFINITY,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, f64::NAN);
    close(actual.saturated_vp as f64, f64::NAN);
    close(actual.vpd as f64, f64::NAN);
}

// calculate_vpd/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_283() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        f64::NEG_INFINITY,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, f64::NAN);
    close(actual.saturated_vp as f64, f64::NAN);
    close(actual.vpd as f64, f64::NAN);
}

// calculate_vpd/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_284() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        9.999999999999998e-08,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 45.26711370085453);
    close(actual.saturated_vp as f64, 150.89037900284845);
    close(actual.vpd as f64, 105.62326530199391);
}

// calculate_vpd/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_285() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        1e-07,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 45.26711370085453);
    close(actual.saturated_vp as f64, 150.89037900284845);
    close(actual.vpd as f64, 105.62326530199391);
}

// calculate_vpd/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_286() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        1.0000000000000001e-07,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 45.26711370085453);
    close(actual.saturated_vp as f64, 150.89037900284845);
    close(actual.vpd as f64, 105.62326530199391);
}

// calculate_vpd/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_287() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        300.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.300048828125,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 142802.89599793428);
    close(actual.saturated_vp as f64, 475932.1904048322);
    close(actual.vpd as f64, 333129.2944068979);
}

// calculate_vpd/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_288() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        300.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.300048828125,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 142802.89599793428);
    close(actual.saturated_vp as f64, 475932.1904048322);
    close(actual.vpd as f64, 333129.2944068979);
}

// calculate_vpd/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_289() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        0.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.300048828125,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 45.27448118124774);
    close(actual.saturated_vp as f64, 150.89037828998434);
    close(actual.vpd as f64, 105.6158971087366);
}

// calculate_vpd/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_290() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        0.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.300048828125,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 45.27448118124774);
    close(actual.saturated_vp as f64, 150.89037828998434);
    close(actual.vpd as f64, 105.6158971087366);
}

// calculate_vpd/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_291() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        1.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 47.451369108539865);
    close(actual.saturated_vp as f64, 158.17123036179956);
    close(actual.vpd as f64, 110.7198612532597);
    let actual = vapor_pressure_deficit::calculate_vpd(
        2.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 49.729217602900725);
    close(actual.saturated_vp as f64, 165.76405867633576);
    close(actual.vpd as f64, 116.03484107343505);
}

// calculate_vpd/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_292() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        0.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 45.2671134869953);
    close(actual.saturated_vp as f64, 150.89037828998434);
    close(actual.vpd as f64, 105.62326480298903);
    let actual = vapor_pressure_deficit::calculate_vpd(
        2.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 49.729217602900725);
    close(actual.saturated_vp as f64, 165.76405867633576);
    close(actual.vpd as f64, 116.03484107343505);
}

// calculate_vpd/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_293() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        1.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 47.451369108539865);
    close(actual.saturated_vp as f64, 158.17123036179956);
    close(actual.vpd as f64, 110.7198612532597);
}

// calculate_vpd/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_294() {
    let actual = vapor_pressure_deficit::calculate_vpd(
        1.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 47.451369108539865);
    close(actual.saturated_vp as f64, 158.17123036179956);
    close(actual.vpd as f64, 110.7198612532597);
    let actual = vapor_pressure_deficit::calculate_vpd(
        1.0,
        TemperatureUnit::try_from(0).unwrap(),
        0.3,
        FractionUnit::try_from(0).unwrap(),
        PressureUnit::try_from(0).unwrap(),
    );
    close(actual.actual_vp as f64, 47.451369108539865);
    close(actual.saturated_vp as f64, 158.17123036179956);
    close(actual.vpd as f64, 110.7198612532597);
}

// calculate_fine_dead_fuel_moisture/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_295() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_296() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_297() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_298() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_299() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_300() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_301() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_302() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_303() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_304() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_305() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_306() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_307() {}

// calculate_fine_dead_fuel_moisture/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_308() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_309() {
    let actual =
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(-1, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, -0.02);
}

// calculate_fine_dead_fuel_moisture/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_310() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(
        -2147483648,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    close(actual as f64, -0.02);
}

// calculate_fine_dead_fuel_moisture/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_311() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(
        -2147483648,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    close(actual as f64, -0.02);
}

// calculate_fine_dead_fuel_moisture/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_312() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(
        -2147483648,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    );
    close(actual as f64, -0.02);
}

// calculate_fine_dead_fuel_moisture/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_313() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_314() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_315() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_316() {
    let actual =
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(300, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, -0.02);
}

// calculate_fine_dead_fuel_moisture/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_317() {
    let actual =
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(300, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, -0.02);
}

// calculate_fine_dead_fuel_moisture/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_318() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_319() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_320() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_321() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_322() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// calculate_fine_dead_fuel_moisture/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_323() {
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
    let actual = fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 0, 0, 0, 0, 0, 0, 0);
    close(actual as f64, 0.03);
}

// FireSize.calculate_backing_spread_rate/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_355() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.calculate_backing_spread_rate().unwrap();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_backing_spread_rate/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_356() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.calculate_backing_spread_rate().unwrap();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_crown_fire_length_to_width_ratio/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_357() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.calculate_crown_fire_length_to_width_ratio();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(helper.fire_length_to_width_ratio_ as f64, 1.625);
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_crown_fire_length_to_width_ratio/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_358() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.calculate_crown_fire_length_to_width_ratio();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(helper.fire_length_to_width_ratio_ as f64, 1.625);
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_elliptical_dimensions/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_359() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.calculate_elliptical_dimensions().unwrap();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_elliptical_dimensions/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_360() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.calculate_elliptical_dimensions().unwrap();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_fire_basic_dimensions/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_361() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper
        .calculate_fire_basic_dimensions(
            false,
            5.0,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::try_from(0).unwrap(),
        )
        .unwrap();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_fire_basic_dimensions/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_362() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper
        .calculate_fire_basic_dimensions(
            false,
            5.0,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::try_from(0).unwrap(),
        )
        .unwrap();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_fire_eccentricity/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_363() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.calculate_fire_eccentricity();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_fire_eccentricity/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_364() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.calculate_fire_eccentricity();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_flanking_spread_rate/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_365() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.calculate_flanking_spread_rate().unwrap();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_flanking_spread_rate/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_366() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.calculate_flanking_spread_rate().unwrap();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_surface_fire_length_to_width_ratio/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_367() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper
        .calculate_surface_fire_length_to_width_ratio()
        .unwrap();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.calculate_surface_fire_length_to_width_ratio/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_368() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper
        .calculate_surface_fire_length_to_width_ratio()
        .unwrap();
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_backing_spread_rate/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_369() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_backing_spread_rate(SpeedUnit::try_from(0).unwrap());
    close(actual as f64, 1.2518732040192728);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_backing_spread_rate/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_370() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_backing_spread_rate(SpeedUnit::try_from(0).unwrap());
    close(actual as f64, 1.2518732040192728);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_eccentricity/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_371() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_eccentricity();
    close(actual as f64, 0.7774818145707344);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_eccentricity/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_372() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_eccentricity();
    close(actual as f64, 0.7774818145707344);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_elliptical_a/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_373() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_elliptical_a(
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 35.38182024739927);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_elliptical_a/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_374() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_elliptical_a(
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 35.38182024739927);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_elliptical_b/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_375() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_elliptical_b(
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 56.259366020096365);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_elliptical_b/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_376() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_elliptical_b(
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 56.259366020096365);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_elliptical_c/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_377() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_elliptical_c(
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 43.740633979903635);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_elliptical_c/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_378() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_elliptical_c(
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 43.740633979903635);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_fire_area/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_379() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_fire_area(
        false,
        AreaUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 6253.5248264527745);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_fire_area/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_380() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_fire_area(
        false,
        AreaUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 6253.5248264527745);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_fire_length/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_381() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_fire_length(
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 112.51873204019273);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_fire_length/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_382() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_fire_length(
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 112.51873204019273);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_fire_length_to_width_ratio/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_383() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_fire_length_to_width_ratio();
    close(actual as f64, 1.5900642088709862);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_fire_length_to_width_ratio/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_384() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_fire_length_to_width_ratio();
    close(actual as f64, 1.5900642088709862);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_fire_perimeter/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_385() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper
        .get_fire_perimeter(
            false,
            LengthUnit::try_from(0).unwrap(),
            10.0,
            TimeUnit::try_from(0).unwrap(),
        )
        .unwrap();
    close(actual as f64, 291.6469753429357);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_fire_perimeter/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_386() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper
        .get_fire_perimeter(
            false,
            LengthUnit::try_from(0).unwrap(),
            10.0,
            TimeUnit::try_from(0).unwrap(),
        )
        .unwrap();
    close(actual as f64, 291.6469753429357);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_flanking_spread_rate/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_387() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_flanking_spread_rate(SpeedUnit::try_from(0).unwrap());
    close(actual as f64, 3.538182024739927);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_flanking_spread_rate/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_388() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_flanking_spread_rate(SpeedUnit::try_from(0).unwrap());
    close(actual as f64, 3.538182024739927);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_heading_to_backing_ratio/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_389() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_heading_to_backing_ratio();
    close(actual as f64, 7.988029432928135);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_heading_to_backing_ratio/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_390() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_heading_to_backing_ratio();
    close(actual as f64, 7.988029432928135);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_max_fire_width/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_391() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_max_fire_width(
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 70.76364049479854);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// FireSize.get_max_fire_width/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_392() {
    let mut helper = fire_size::FireSize::default();
    helper.backing_spread_rate_ = 1.2518732040192728;
    helper.eccentricity_ = 0.7774818145707344;
    helper.effective_wind_speed_ = 5.0;
    helper.elapsed_time_ = 0.0;
    helper.elliptical_a_ = 3.538182024739927;
    helper.elliptical_b_ = 5.625936602009636;
    helper.elliptical_c_ = 4.374063397990364;
    helper.fire_length_to_width_ratio_ = 1.5900642088709862;
    helper.flanking_spread_rate_ = 3.538182024739927;
    helper.forward_spread_rate_ = 10.0;
    helper.heading_to_backing_ratio_ = 7.988029432928135;
    let actual = helper.get_max_fire_width(
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
    close(actual as f64, 70.76364049479854);
    close(helper.backing_spread_rate_ as f64, 1.2518732040192728);
    close(helper.eccentricity_ as f64, 0.7774818145707344);
    close(helper.effective_wind_speed_ as f64, 5.0);
    close(helper.elapsed_time_ as f64, 0.0);
    close(helper.elliptical_a_ as f64, 3.538182024739927);
    close(helper.elliptical_b_ as f64, 5.625936602009636);
    close(helper.elliptical_c_ as f64, 4.374063397990364);
    close(
        helper.fire_length_to_width_ratio_ as f64,
        1.5900642088709862,
    );
    close(helper.flanking_spread_rate_ as f64, 3.538182024739927);
    close(helper.forward_spread_rate_ as f64, 10.0);
    close(helper.heading_to_backing_ratio_ as f64, 7.988029432928135);
}

// SlopeTool.calculate_horizontal_distance/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_393() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper
        .calculate_horizontal_distance(
            10.0,
            LengthUnit::try_from(0).unwrap(),
            30.0,
            SlopeUnit::try_from(0).unwrap(),
        )
        .unwrap();
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.calculate_horizontal_distance/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_394() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper
        .calculate_horizontal_distance(
            10.0,
            LengthUnit::try_from(0).unwrap(),
            30.0,
            SlopeUnit::try_from(0).unwrap(),
        )
        .unwrap();
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.calculate_slope_from_map_measurements/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_395() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.calculate_slope_from_map_measurements(
        24000.0,
        1.0,
        LengthUnit::try_from(0).unwrap(),
        20.0,
        3.0,
        LengthUnit::try_from(0).unwrap(),
    );
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.calculate_slope_from_map_measurements/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_396() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.calculate_slope_from_map_measurements(
        24000.0,
        1.0,
        LengthUnit::try_from(0).unwrap(),
        20.0,
        3.0,
        LengthUnit::try_from(0).unwrap(),
    );
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_centimeters_per_kilometer_at_index/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_397() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_centimeters_per_kilometer_at_index(0);
    close(actual as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_centimeters_per_kilometer_at_index/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_398() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_centimeters_per_kilometer_at_index(-1);
    close(actual as f64, -1.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_horizontal_distance_at_index/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_399() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_horizontal_distance_at_index(0, LengthUnit::try_from(0).unwrap());
    close(actual as f64, 8.660254037844041);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_horizontal_distance_at_index/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_400() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_horizontal_distance_at_index(-1, LengthUnit::try_from(0).unwrap());
    close(actual as f64, -1.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_inches_per_mile_at_index/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_401() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_inches_per_mile_at_index(0);
    close(actual as f64, 32.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_inches_per_mile_at_index/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_402() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_inches_per_mile_at_index(-1);
    close(actual as f64, -1.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_kilometers_per_centimeter_at_index/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_403() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_kilometers_per_centimeter_at_index(0);
    close(actual as f64, 0.0198);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_kilometers_per_centimeter_at_index/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_404() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_kilometers_per_centimeter_at_index(-1);
    close(actual as f64, -1.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_miles_per_inch_at_index/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_405() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_miles_per_inch_at_index(0);
    close(actual as f64, 0.03125);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_miles_per_inch_at_index/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_406() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_miles_per_inch_at_index(-1);
    close(actual as f64, -1.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_number_of_horizontal_distances/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_407() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_number_of_horizontal_distances();
    close(actual as f64, 7.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_number_of_horizontal_distances/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_408() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_number_of_horizontal_distances();
    close(actual as f64, 7.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_number_of_representative_fractions/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_409() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_number_of_representative_fractions();
    close(actual as f64, 18.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_number_of_representative_fractions/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_410() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_number_of_representative_fractions();
    close(actual as f64, 18.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_representative_fraction_at_index/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_411() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_representative_fraction_at_index(0);
    close(actual as f64, 1980.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_representative_fraction_at_index/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_412() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_representative_fraction_at_index(-1);
    close(actual as f64, -1.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_slope_elevation_change_from_map_measurements/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_413() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual =
        helper.get_slope_elevation_change_from_map_measurements(LengthUnit::try_from(0).unwrap());
    close(actual as f64, 60.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_slope_elevation_change_from_map_measurements/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_414() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual =
        helper.get_slope_elevation_change_from_map_measurements(LengthUnit::try_from(0).unwrap());
    close(actual as f64, 60.0);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_slope_from_map_measurements/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_415() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_slope_from_map_measurements(SlopeUnit::try_from(0).unwrap());
    close(actual as f64, 0.1432391503683123);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_slope_from_map_measurements/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_416() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_slope_from_map_measurements(SlopeUnit::try_from(0).unwrap());
    close(actual as f64, 0.1432391503683123);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_slope_from_map_measurements_in_degrees/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_417() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_slope_from_map_measurements_in_degrees();
    close(actual as f64, 0.1432391503683123);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_slope_from_map_measurements_in_degrees/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_418() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_slope_from_map_measurements_in_degrees();
    close(actual as f64, 0.1432391503683123);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_slope_from_map_measurements_in_percent/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_419() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_slope_from_map_measurements_in_percent();
    close(actual as f64, 0.25000000000001005);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_slope_from_map_measurements_in_percent/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_420() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper.get_slope_from_map_measurements_in_percent();
    close(actual as f64, 0.25000000000001005);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_slope_horizontal_distance_from_map_measurements/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_421() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper
        .get_slope_horizontal_distance_from_map_measurements(LengthUnit::try_from(0).unwrap());
    close(actual as f64, 23999.99999999904);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}

// SlopeTool.get_slope_horizontal_distance_from_map_measurements/invalid-index
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_422() {
    let mut helper = slope::SlopeTool::default();
    helper.centimeters_per_kilometer_ = [
        50.5051, 25.2525, 12.6263, 10.0, 6.3131, 4.7348, 4.1667, 3.1566, 2.0, 1.6, 1.5783, 1.0,
        0.7891, 0.4, 0.3946, 0.1973, 0.1, 0.0986,
    ];
    helper.horizontal_distances_ = [
        8.660254037844041,
        8.756408079383258,
        9.013878188659614,
        9.35414346693448,
        9.682458365518155,
        9.915912340642066,
        9.9999999999996,
    ];
    helper.inches_per_mile_ = [
        32.0, 16.0, 8.0, 6.336, 4.0, 3.0, 2.64, 2.0, 1.2672, 1.0138, 1.0, 0.6336, 0.5, 0.2534,
        0.25, 0.125, 0.0634, 0.0625,
    ];
    helper.kilometers_per_centimeter_ = [
        0.0198, 0.0396, 0.0792, 0.1, 0.1584, 0.2112, 0.24, 0.3168, 0.5, 0.625, 0.6336, 1.0, 1.2672,
        2.5, 2.5344, 5.0688, 10.0, 10.1376,
    ];
    helper.max_slope_in_degrees_ = 30.0;
    helper.max_slope_in_percent_ = 57.735026918962575;
    helper.miles_per_inch_ = [
        0.03125, 0.0625, 0.125, 0.15783, 0.25, 0.3333, 0.37879, 0.5, 0.78914, 0.98643, 1.0,
        1.57828, 2.0, 3.94571, 4.0, 8.0, 15.78283, 16.0,
    ];
    helper.representative_fractions_ = [
        1980, 3960, 7920, 10000, 15840, 21120, 24000, 31680, 50000, 62500, 63360, 100000, 126720,
        250000, 253440, 506880, 1000000, 1013760,
    ];
    helper.slope_elevation_change_ = 60.0;
    helper.slope_from_map_measurements_ = 0.1432391503683123;
    helper.slope_horizontal_distance_ = 23999.99999999904;
    let actual = helper
        .get_slope_horizontal_distance_from_map_measurements(LengthUnit::try_from(0).unwrap());
    close(actual as f64, 23999.99999999904);
    close(helper.centimeters_per_kilometer_[0] as f64, 50.5051);
    close(helper.centimeters_per_kilometer_[1] as f64, 25.2525);
    close(helper.centimeters_per_kilometer_[2] as f64, 12.6263);
    close(helper.centimeters_per_kilometer_[3] as f64, 10.0);
    close(helper.centimeters_per_kilometer_[4] as f64, 6.3131);
    close(helper.centimeters_per_kilometer_[5] as f64, 4.7348);
    close(helper.centimeters_per_kilometer_[6] as f64, 4.1667);
    close(helper.centimeters_per_kilometer_[7] as f64, 3.1566);
    close(helper.centimeters_per_kilometer_[8] as f64, 2.0);
    close(helper.centimeters_per_kilometer_[9] as f64, 1.6);
    close(helper.centimeters_per_kilometer_[10] as f64, 1.5783);
    close(helper.centimeters_per_kilometer_[11] as f64, 1.0);
    close(helper.centimeters_per_kilometer_[12] as f64, 0.7891);
    close(helper.centimeters_per_kilometer_[13] as f64, 0.4);
    close(helper.centimeters_per_kilometer_[14] as f64, 0.3946);
    close(helper.centimeters_per_kilometer_[15] as f64, 0.1973);
    close(helper.centimeters_per_kilometer_[16] as f64, 0.1);
    close(helper.centimeters_per_kilometer_[17] as f64, 0.0986);
    close(helper.horizontal_distances_[0] as f64, 8.660254037844041);
    close(helper.horizontal_distances_[1] as f64, 8.756408079383258);
    close(helper.horizontal_distances_[2] as f64, 9.013878188659614);
    close(helper.horizontal_distances_[3] as f64, 9.35414346693448);
    close(helper.horizontal_distances_[4] as f64, 9.682458365518155);
    close(helper.horizontal_distances_[5] as f64, 9.915912340642066);
    close(helper.horizontal_distances_[6] as f64, 9.9999999999996);
    close(helper.inches_per_mile_[0] as f64, 32.0);
    close(helper.inches_per_mile_[1] as f64, 16.0);
    close(helper.inches_per_mile_[2] as f64, 8.0);
    close(helper.inches_per_mile_[3] as f64, 6.336);
    close(helper.inches_per_mile_[4] as f64, 4.0);
    close(helper.inches_per_mile_[5] as f64, 3.0);
    close(helper.inches_per_mile_[6] as f64, 2.64);
    close(helper.inches_per_mile_[7] as f64, 2.0);
    close(helper.inches_per_mile_[8] as f64, 1.2672);
    close(helper.inches_per_mile_[9] as f64, 1.0138);
    close(helper.inches_per_mile_[10] as f64, 1.0);
    close(helper.inches_per_mile_[11] as f64, 0.6336);
    close(helper.inches_per_mile_[12] as f64, 0.5);
    close(helper.inches_per_mile_[13] as f64, 0.2534);
    close(helper.inches_per_mile_[14] as f64, 0.25);
    close(helper.inches_per_mile_[15] as f64, 0.125);
    close(helper.inches_per_mile_[16] as f64, 0.0634);
    close(helper.inches_per_mile_[17] as f64, 0.0625);
    close(helper.kilometers_per_centimeter_[0] as f64, 0.0198);
    close(helper.kilometers_per_centimeter_[1] as f64, 0.0396);
    close(helper.kilometers_per_centimeter_[2] as f64, 0.0792);
    close(helper.kilometers_per_centimeter_[3] as f64, 0.1);
    close(helper.kilometers_per_centimeter_[4] as f64, 0.1584);
    close(helper.kilometers_per_centimeter_[5] as f64, 0.2112);
    close(helper.kilometers_per_centimeter_[6] as f64, 0.24);
    close(helper.kilometers_per_centimeter_[7] as f64, 0.3168);
    close(helper.kilometers_per_centimeter_[8] as f64, 0.5);
    close(helper.kilometers_per_centimeter_[9] as f64, 0.625);
    close(helper.kilometers_per_centimeter_[10] as f64, 0.6336);
    close(helper.kilometers_per_centimeter_[11] as f64, 1.0);
    close(helper.kilometers_per_centimeter_[12] as f64, 1.2672);
    close(helper.kilometers_per_centimeter_[13] as f64, 2.5);
    close(helper.kilometers_per_centimeter_[14] as f64, 2.5344);
    close(helper.kilometers_per_centimeter_[15] as f64, 5.0688);
    close(helper.kilometers_per_centimeter_[16] as f64, 10.0);
    close(helper.kilometers_per_centimeter_[17] as f64, 10.1376);
    close(helper.max_slope_in_degrees_ as f64, 30.0);
    close(helper.max_slope_in_percent_ as f64, 57.735026918962575);
    close(helper.miles_per_inch_[0] as f64, 0.03125);
    close(helper.miles_per_inch_[1] as f64, 0.0625);
    close(helper.miles_per_inch_[2] as f64, 0.125);
    close(helper.miles_per_inch_[3] as f64, 0.15783);
    close(helper.miles_per_inch_[4] as f64, 0.25);
    close(helper.miles_per_inch_[5] as f64, 0.3333);
    close(helper.miles_per_inch_[6] as f64, 0.37879);
    close(helper.miles_per_inch_[7] as f64, 0.5);
    close(helper.miles_per_inch_[8] as f64, 0.78914);
    close(helper.miles_per_inch_[9] as f64, 0.98643);
    close(helper.miles_per_inch_[10] as f64, 1.0);
    close(helper.miles_per_inch_[11] as f64, 1.57828);
    close(helper.miles_per_inch_[12] as f64, 2.0);
    close(helper.miles_per_inch_[13] as f64, 3.94571);
    close(helper.miles_per_inch_[14] as f64, 4.0);
    close(helper.miles_per_inch_[15] as f64, 8.0);
    close(helper.miles_per_inch_[16] as f64, 15.78283);
    close(helper.miles_per_inch_[17] as f64, 16.0);
    close(helper.representative_fractions_[0] as f64, 1980.0);
    close(helper.representative_fractions_[1] as f64, 3960.0);
    close(helper.representative_fractions_[2] as f64, 7920.0);
    close(helper.representative_fractions_[3] as f64, 10000.0);
    close(helper.representative_fractions_[4] as f64, 15840.0);
    close(helper.representative_fractions_[5] as f64, 21120.0);
    close(helper.representative_fractions_[6] as f64, 24000.0);
    close(helper.representative_fractions_[7] as f64, 31680.0);
    close(helper.representative_fractions_[8] as f64, 50000.0);
    close(helper.representative_fractions_[9] as f64, 62500.0);
    close(helper.representative_fractions_[10] as f64, 63360.0);
    close(helper.representative_fractions_[11] as f64, 100000.0);
    close(helper.representative_fractions_[12] as f64, 126720.0);
    close(helper.representative_fractions_[13] as f64, 250000.0);
    close(helper.representative_fractions_[14] as f64, 253440.0);
    close(helper.representative_fractions_[15] as f64, 506880.0);
    close(helper.representative_fractions_[16] as f64, 1000000.0);
    close(helper.representative_fractions_[17] as f64, 1013760.0);
    close(helper.slope_elevation_change_ as f64, 60.0);
    close(
        helper.slope_from_map_measurements_ as f64,
        0.1432391503683123,
    );
    close(helper.slope_horizontal_distance_ as f64, 23999.99999999904);
}
#[test]
fn all_moisture_reference_and_correction_cells() {
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 1, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 2, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 3, 0, 0, 0, 0, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 4, 0, 0, 0, 0, 0, 0),
        0.06,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 5, 0, 0, 0, 0, 0, 0),
        0.07,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 6, 0, 0, 0, 0, 0, 0),
        0.07,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 7, 0, 0, 0, 0, 0, 0),
        0.08,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 8, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 9, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 10, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 11, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 12, 0, 0, 0, 0, 0, 0),
        0.11,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 13, 0, 0, 0, 0, 0, 0),
        0.11,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 14, 0, 0, 0, 0, 0, 0),
        0.12000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 15, 0, 0, 0, 0, 0, 0),
        0.13,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 16, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 17, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 18, 0, 0, 0, 0, 0, 0),
        0.15,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 19, 0, 0, 0, 0, 0, 0),
        0.15,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 20, 0, 0, 0, 0, 0, 0),
        0.16,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 0, 0, 0, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 1, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 2, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 3, 0, 0, 0, 0, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 4, 0, 0, 0, 0, 0, 0),
        0.06,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 5, 0, 0, 0, 0, 0, 0),
        0.07,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 6, 0, 0, 0, 0, 0, 0),
        0.07,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 7, 0, 0, 0, 0, 0, 0),
        0.08,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 8, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 9, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 10, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 11, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 12, 0, 0, 0, 0, 0, 0),
        0.11,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 13, 0, 0, 0, 0, 0, 0),
        0.11,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 14, 0, 0, 0, 0, 0, 0),
        0.12000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 15, 0, 0, 0, 0, 0, 0),
        0.12000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 16, 0, 0, 0, 0, 0, 0),
        0.13,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 17, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 18, 0, 0, 0, 0, 0, 0),
        0.15,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 19, 0, 0, 0, 0, 0, 0),
        0.15,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(1, 20, 0, 0, 0, 0, 0, 0),
        0.15,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 0, 0, 0, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 1, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 2, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 3, 0, 0, 0, 0, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 4, 0, 0, 0, 0, 0, 0),
        0.06,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 5, 0, 0, 0, 0, 0, 0),
        0.07,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 6, 0, 0, 0, 0, 0, 0),
        0.07,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 7, 0, 0, 0, 0, 0, 0),
        0.08,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 8, 0, 0, 0, 0, 0, 0),
        0.08,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 9, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 10, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 11, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 12, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 13, 0, 0, 0, 0, 0, 0),
        0.11,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 14, 0, 0, 0, 0, 0, 0),
        0.11,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 15, 0, 0, 0, 0, 0, 0),
        0.12000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 16, 0, 0, 0, 0, 0, 0),
        0.13,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 17, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 18, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 19, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(2, 20, 0, 0, 0, 0, 0, 0),
        0.15,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 0, 0, 0, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 1, 0, 0, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 2, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 3, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 4, 0, 0, 0, 0, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 5, 0, 0, 0, 0, 0, 0),
        0.06,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 6, 0, 0, 0, 0, 0, 0),
        0.07,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 7, 0, 0, 0, 0, 0, 0),
        0.07,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 8, 0, 0, 0, 0, 0, 0),
        0.08,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 9, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 10, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 11, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 12, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 13, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 14, 0, 0, 0, 0, 0, 0),
        0.11,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 15, 0, 0, 0, 0, 0, 0),
        0.12000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 16, 0, 0, 0, 0, 0, 0),
        0.12000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 17, 0, 0, 0, 0, 0, 0),
        0.13,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 18, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 19, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(3, 20, 0, 0, 0, 0, 0, 0),
        0.15,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 0, 0, 0, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 1, 0, 0, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 2, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 3, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 4, 0, 0, 0, 0, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 5, 0, 0, 0, 0, 0, 0),
        0.06,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 6, 0, 0, 0, 0, 0, 0),
        0.06,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 7, 0, 0, 0, 0, 0, 0),
        0.07,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 8, 0, 0, 0, 0, 0, 0),
        0.08,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 9, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 10, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 11, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 12, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 13, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 14, 0, 0, 0, 0, 0, 0),
        0.11,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 15, 0, 0, 0, 0, 0, 0),
        0.12000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 16, 0, 0, 0, 0, 0, 0),
        0.12000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 17, 0, 0, 0, 0, 0, 0),
        0.13,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 18, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 19, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(4, 20, 0, 0, 0, 0, 0, 0),
        0.15,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 0, 0, 0, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 1, 0, 0, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 2, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 3, 0, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 4, 0, 0, 0, 0, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 5, 0, 0, 0, 0, 0, 0),
        0.06,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 6, 0, 0, 0, 0, 0, 0),
        0.06,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 7, 0, 0, 0, 0, 0, 0),
        0.07,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 8, 0, 0, 0, 0, 0, 0),
        0.08,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 9, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 10, 0, 0, 0, 0, 0, 0),
        0.09000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 11, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 12, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 13, 0, 0, 0, 0, 0, 0),
        0.1,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 14, 0, 0, 0, 0, 0, 0),
        0.11,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 15, 0, 0, 0, 0, 0, 0),
        0.12000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 16, 0, 0, 0, 0, 0, 0),
        0.12000000000000001,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 17, 0, 0, 0, 0, 0, 0),
        0.13,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 18, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 19, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(5, 20, 0, 0, 0, 0, 0, 0),
        0.13999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 1, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 2, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 1, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 2, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 1, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 2, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 1, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 2, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 1, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 2, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 0, 5),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 1, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 0, 2, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 1, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 2, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 1, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 2, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 0, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 1, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 2, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 0, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 1, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 2, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 0, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 1, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 2, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 0, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 1, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 0, 2, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 1, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 2, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 1, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 2, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 0, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 1, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 2, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 0, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 1, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 2, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 0, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 1, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 2, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 0, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 1, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 0, 2, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 0, 0),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 1, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 2, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 0, 1),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 1, 1),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 2, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 0, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 1, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 2, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 0, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 1, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 2, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 0, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 1, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 2, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 0, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 1, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 2, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 1, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 2, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 0, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 1, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 2, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 0, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 1, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 2, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 0, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 1, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 2, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 0, 5),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 1, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 0, 2, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 1, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 2, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 1, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 2, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 0, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 1, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 2, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 0, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 1, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 2, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 0, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 1, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 2, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 0, 5),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 1, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 0, 2, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 0, 0),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 1, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 2, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 1, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 2, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 0, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 1, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 2, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 0, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 1, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 2, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 0, 4),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 1, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 2, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 0, 5),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 1, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 0, 2, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 0, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 1, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 2, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 0, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 1, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 2, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 0, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 1, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 2, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 0, 4),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 1, 4),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 2, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 0, 5),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 1, 5),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 0, 2, 5),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 2, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 1, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 2, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 0, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 1, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 2, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 0, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 1, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 2, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 1, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 2, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 0, 2, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 1, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 2, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 1, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 2, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 0, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 1, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 2, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 0, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 1, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 2, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 1, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 2, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 0, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 1, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 2, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 1, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 2, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 0, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 1, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 2, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 0, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 1, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 2, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 1, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 2, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 0, 2, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 1, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 2, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 0, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 1, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 2, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 0, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 1, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 2, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 1, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 2, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 1, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 0, 2, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 1, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 2, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 1, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 2, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 0, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 1, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 2, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 0, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 1, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 2, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 0, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 1, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 2, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 0, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 1, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 1, 2, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 1, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 2, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 1, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 2, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 0, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 1, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 2, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 0, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 1, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 2, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 1, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 2, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 0, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 1, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 1, 2, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 1, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 2, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 1, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 2, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 0, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 1, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 2, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 0, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 1, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 2, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 0, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 1, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 2, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 0, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 1, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 1, 2, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 1, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 2, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 1, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 2, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 0, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 1, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 2, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 0, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 1, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 2, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 1, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 2, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 0, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 1, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 1, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 1, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 2, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 1, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 2, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 0, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 1, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 2, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 0, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 1, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 2, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 0, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 1, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 2, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 0, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 1, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 1, 2, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 1, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 2, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 1, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 2, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 0, 2),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 1, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 2, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 0, 3),
        0.01,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 1, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 2, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 0, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 1, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 2, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 0, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 1, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 1, 2, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 0, 0),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 1, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 2, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 0, 1),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 1, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 2, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 0, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 1, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 2, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 0, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 1, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 2, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 0, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 1, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 2, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 0, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 1, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 1, 2, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 1, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 2, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 0, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 1, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 2, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 0, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 1, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 2, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 0, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 1, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 2, 4),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 0, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 1, 5),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 1, 2, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 0, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 1, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 2, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 0, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 1, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 2, 2),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 0, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 1, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 2, 3),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 0, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 1, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 2, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 1, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 1, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 2, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 0, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 1, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 2, 2),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 0, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 1, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 2, 3),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 0, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 1, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 2, 4),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 1, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 1, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 2, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 0, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 1, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 2, 2),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 0, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 1, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 2, 3),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 1, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 2, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 1, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 0, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 1, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 2, 1),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 0, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 1, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 2, 2),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 0, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 1, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 2, 3),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 1, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 2, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 1, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 1, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 2, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 0, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 1, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 2, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 0, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 1, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 2, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 1, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 2, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 0, 2, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 0, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 1, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 2, 1),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 0, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 1, 2),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 2, 2),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 0, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 1, 3),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 2, 3),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 0, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 1, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 2, 4),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 0, 0, 2, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 1, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 2, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 0, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 1, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 2, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 0, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 1, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 2, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 1, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 2, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 0, 2, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 0, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 1, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 2, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 0, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 1, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 2, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 0, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 1, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 2, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 0, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 1, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 2, 4),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 1, 0, 2, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 1, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 2, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 0, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 1, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 2, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 0, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 1, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 2, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 1, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 2, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 0, 2, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 0, 1),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 1, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 2, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 0, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 1, 2),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 2, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 0, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 1, 3),
        0.02,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 2, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 0, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 1, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 2, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 2, 0, 2, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 0, 1),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 1, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 2, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 0, 2),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 1, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 2, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 0, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 1, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 2, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 0, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 1, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 2, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 0, 2, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 0, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 1, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 2, 1),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 0, 2),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 1, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 2, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 0, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 1, 3),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 2, 3),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 0, 4),
        0.03,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 1, 4),
        0.04,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 2, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 1, 3, 0, 2, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 0, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 1, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 2, 1),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 0, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 1, 2),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 2, 2),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 0, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 1, 3),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 2, 3),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 0, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 1, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 2, 4),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 0, 1, 2, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 0, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 1, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 2, 1),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 0, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 1, 2),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 2, 2),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 0, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 1, 3),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 2, 3),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 0, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 1, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 2, 4),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 1, 1, 2, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 0, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 1, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 2, 1),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 0, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 1, 2),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 2, 2),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 0, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 1, 3),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 2, 3),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 0, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 1, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 2, 4),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 2, 1, 2, 2, 5),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 0, 0),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 1, 0),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 2, 0),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 0, 1),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 1, 1),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 2, 1),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 0, 2),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 1, 2),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 2, 2),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 0, 3),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 1, 3),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 2, 3),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 0, 4),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 1, 4),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 2, 4),
        0.06999999999999999,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 0, 5),
        0.05,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 1, 5),
        0.060000000000000005,
    );
    close(
        fine_dead_fuel_moisture::calculate_fine_dead_fuel_moisture(0, 0, 0, 3, 1, 2, 2, 5),
        0.06999999999999999,
    );
}
#[test]
fn scalar_geometry_exceptions_and_numpy_division() {
    use fire_size::{FireSize, GeometryError};
    let mut fire = FireSize::default();
    fire.effective_wind_speed_ = 10000.0;
    assert_eq!(
        fire.calculate_surface_fire_length_to_width_ratio(),
        Err(GeometryError::Overflow)
    );
    close(fire.fire_length_to_width_ratio_, 1.0);
    fire.effective_wind_speed_ = f64::INFINITY;
    fire.calculate_surface_fire_length_to_width_ratio().unwrap();
    close(fire.fire_length_to_width_ratio_, 8.0);
    fire.effective_wind_speed_ = f64::NAN;
    fire.calculate_surface_fire_length_to_width_ratio().unwrap();
    close(fire.fire_length_to_width_ratio_, 1.0);
    fire.fire_length_to_width_ratio_ = 0.5;
    assert_eq!(
        fire.calculate_elliptical_dimensions(),
        Err(GeometryError::Domain)
    );
    fire.fire_length_to_width_ratio_ = 1e9;
    assert_eq!(
        fire.calculate_elliptical_dimensions(),
        Err(GeometryError::DivisionByZero)
    );
    fire.fire_length_to_width_ratio_ = 1e200;
    assert_eq!(
        fire.calculate_elliptical_dimensions(),
        Err(GeometryError::Overflow)
    );
    fire.fire_length_to_width_ratio_ = f64::INFINITY;
    fire.calculate_elliptical_dimensions().unwrap();
    assert!(fire.heading_to_backing_ratio_.is_nan());
    fire.fire_length_to_width_ratio_ = 0.0;
    assert_eq!(
        fire.calculate_flanking_spread_rate(),
        Err(GeometryError::DivisionByZero)
    );
    assert_eq!(
        fire.get_fire_perimeter(true, LengthUnit::Feet, 1.0, TimeUnit::Minutes),
        Err(GeometryError::DivisionByZero)
    );
    fire.calculate_fire_basic_dimensions(
        false,
        5.0,
        SpeedUnit::MilesPerHour,
        10.0,
        SpeedUnit::FeetPerMinute,
    )
    .unwrap();
    fire.fire_length_to_width_ratio_ = 0.0;
    fire.calculate_flanking_spread_rate().unwrap();
    assert!(fire.flanking_spread_rate_.is_infinite());
    assert!(fire
        .get_fire_area(true, AreaUnit::SquareFeet, 1.0, TimeUnit::Minutes)
        .is_infinite());
    let mut slope = slope::SlopeTool::default();
    assert_eq!(
        slope.calculate_horizontal_distance(
            10.0,
            LengthUnit::Feet,
            f64::INFINITY,
            SlopeUnit::Degrees
        ),
        Err(GeometryError::Domain)
    );
    assert!(slope.max_slope_in_degrees_.is_infinite());
    assert!(slope.max_slope_in_percent_.is_nan());
    assert_eq!(slope.horizontal_distances_, [0.0; 7]);
    slope
        .calculate_horizontal_distance(10.0, LengthUnit::Feet, f64::NAN, SlopeUnit::Degrees)
        .unwrap();
    assert!(slope
        .horizontal_distances_
        .iter()
        .all(|value| value.is_nan()));
}

#[test]
fn finite_slope_angle_overflow_is_a_domain_error() {
    let mut tool = slope::SlopeTool::default();
    assert_eq!(
        tool.calculate_horizontal_distance(10.0, LengthUnit::Feet, 1e308, SlopeUnit::Degrees),
        Err(fire_size::GeometryError::Domain)
    );
}

#[test]
fn crown_wind_numeric_kinds_preserve_python_state() {
    use fire_size::{FireSize, GeometryError, NumericKind};
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            5.0,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 5.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.441311542550502);
        close(fire.elliptical_b_, 5.592131256644566);
        close(fire.elliptical_c_, 4.407868743355435);
        close(fire.eccentricity_, 0.7882269819968921);
        close(fire.backing_spread_rate_, 1.184262513289132);
        close(fire.flanking_spread_rate_, 3.441311542550502);
        close(fire.fire_length_to_width_ratio_, 1.625);
        close(fire.heading_to_backing_ratio_, 8.444073748671086);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            5.0,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.0568181818);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 8.877080193965405);
        close(fire.elliptical_b_, 8.940127638504643);
        close(fire.elliptical_c_, 1.0598723614953558);
        close(fire.eccentricity_, 0.11855226282570552);
        close(fire.backing_spread_rate_, 7.8802552770092875);
        close(fire.flanking_spread_rate_, 8.877080193965405);
        close(fire.fire_length_to_width_ratio_, 1.007102272725);
        close(fire.heading_to_backing_ratio_, 1.2689944232105128);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            13.3,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 13.3);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 1.9492907903339078);
        close(fire.elliptical_b_, 5.18998672926403);
        close(fire.elliptical_c_, 4.810013270735971);
        close(fire.eccentricity_, 0.9267872003630457);
        close(fire.backing_spread_rate_, 0.37997345852805897);
        close(fire.flanking_spread_rate_, 1.9492907903339078);
        close(fire.fire_length_to_width_ratio_, 2.6625);
        close(fire.heading_to_backing_ratio_, 26.317627654147216);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            13.3,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.151136363588);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 8.235948353347144);
        close(fire.elliptical_b_, 8.391542263950077);
        close(fire.elliptical_c_, 1.608457736049922);
        close(fire.eccentricity_, 0.1916760573273676);
        close(fire.backing_spread_rate_, 6.783084527900155);
        close(fire.flanking_spread_rate_, 8.235948353347144);
        close(fire.fire_length_to_width_ratio_, 1.0188920454485);
        close(fire.heading_to_backing_ratio_, 1.474255548322897);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1000000000.0,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert_eq!(result, Err(GeometryError::DivisionByZero));
        close(fire.effective_wind_speed_, 1000000000.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 1.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 3.999999968e-08);
        close(fire.fire_length_to_width_ratio_, 125000001.0);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1000000000.0,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 11363636.36);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.5199975230485797e-06);
        close(fire.elliptical_b_, 5.00000000000062);
        close(fire.elliptical_c_, 4.999999999999381);
        close(fire.eccentricity_, 0.9999999999997522);
        close(fire.backing_spread_rate_, 1.2390088954818282e-12);
        close(fire.flanking_spread_rate_, 3.5199975230485797e-06);
        close(fire.fire_length_to_width_ratio_, 1420455.545);
        close(fire.heading_to_backing_ratio_, 8069854644439.286);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e+30,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert_eq!(result, Err(GeometryError::DivisionByZero));
        close(fire.effective_wind_speed_, 1e+30);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 1.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 4e-29);
        close(fire.fire_length_to_width_ratio_, 1.25e+29);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e+30,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 1.136363636e+28);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.5200000011263996e-27);
        close(fire.elliptical_b_, 5.0);
        close(fire.elliptical_c_, 5.0);
        close(fire.eccentricity_, 1.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 3.5200000011263996e-27);
        close(fire.fire_length_to_width_ratio_, 1.420454545e+27);
        close(fire.heading_to_backing_ratio_, f64::INFINITY);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e+200,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert_eq!(result, Err(GeometryError::Overflow));
        close(fire.effective_wind_speed_, 1e+200);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, f64::INFINITY);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, 1.25e+199);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e+200,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 1.1363636359999999e+198);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::INFINITY);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, 1.4204545449999999e+197);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::INFINITY,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::NAN);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, f64::INFINITY);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::INFINITY,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::NAN);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, f64::INFINITY);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::NAN,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::NAN,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            5.0,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 5.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.441311542550502);
        close(fire.elliptical_b_, 5.592131256644566);
        close(fire.elliptical_c_, 4.407868743355435);
        close(fire.eccentricity_, 0.7882269819968921);
        close(fire.backing_spread_rate_, 1.184262513289132);
        close(fire.flanking_spread_rate_, 3.441311542550502);
        close(fire.fire_length_to_width_ratio_, 1.625);
        close(fire.heading_to_backing_ratio_, 8.444073748671086);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            5.0,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.0568181818);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 8.877080193965405);
        close(fire.elliptical_b_, 8.940127638504643);
        close(fire.elliptical_c_, 1.0598723614953558);
        close(fire.eccentricity_, 0.11855226282570552);
        close(fire.backing_spread_rate_, 7.8802552770092875);
        close(fire.flanking_spread_rate_, 8.877080193965405);
        close(fire.fire_length_to_width_ratio_, 1.007102272725);
        close(fire.heading_to_backing_ratio_, 1.2689944232105128);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            13.3,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 13.3);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 1.9492907903339078);
        close(fire.elliptical_b_, 5.18998672926403);
        close(fire.elliptical_c_, 4.810013270735971);
        close(fire.eccentricity_, 0.9267872003630457);
        close(fire.backing_spread_rate_, 0.37997345852805897);
        close(fire.flanking_spread_rate_, 1.9492907903339078);
        close(fire.fire_length_to_width_ratio_, 2.6625);
        close(fire.heading_to_backing_ratio_, 26.317627654147216);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            13.3,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.151136363588);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 8.235948353347144);
        close(fire.elliptical_b_, 8.391542263950077);
        close(fire.elliptical_c_, 1.608457736049922);
        close(fire.eccentricity_, 0.1916760573273676);
        close(fire.backing_spread_rate_, 6.783084527900155);
        close(fire.flanking_spread_rate_, 8.235948353347144);
        close(fire.fire_length_to_width_ratio_, 1.0188920454485);
        close(fire.heading_to_backing_ratio_, 1.474255548322897);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1000000000.0,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 1000000000.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.999999968e-08);
        close(fire.elliptical_b_, 5.0);
        close(fire.elliptical_c_, 5.0);
        close(fire.eccentricity_, 1.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 3.999999968e-08);
        close(fire.fire_length_to_width_ratio_, 125000001.0);
        close(fire.heading_to_backing_ratio_, f64::INFINITY);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1000000000.0,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 11363636.36);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.5199975230485797e-06);
        close(fire.elliptical_b_, 5.00000000000062);
        close(fire.elliptical_c_, 4.999999999999381);
        close(fire.eccentricity_, 0.9999999999997522);
        close(fire.backing_spread_rate_, 1.2390088954818282e-12);
        close(fire.flanking_spread_rate_, 3.5199975230485797e-06);
        close(fire.fire_length_to_width_ratio_, 1420455.545);
        close(fire.heading_to_backing_ratio_, 8069854644439.286);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e+30,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 1e+30);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 4e-29);
        close(fire.elliptical_b_, 5.0);
        close(fire.elliptical_c_, 5.0);
        close(fire.eccentricity_, 1.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 4e-29);
        close(fire.fire_length_to_width_ratio_, 1.25e+29);
        close(fire.heading_to_backing_ratio_, f64::INFINITY);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e+30,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 1.136363636e+28);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.5200000011263996e-27);
        close(fire.elliptical_b_, 5.0);
        close(fire.elliptical_c_, 5.0);
        close(fire.eccentricity_, 1.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 3.5200000011263996e-27);
        close(fire.fire_length_to_width_ratio_, 1.420454545e+27);
        close(fire.heading_to_backing_ratio_, f64::INFINITY);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e+200,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 1e+200);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::INFINITY);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, 1.25e+199);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e+200,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 1.1363636359999999e+198);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::INFINITY);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, 1.4204545449999999e+197);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::INFINITY,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::NAN);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, f64::INFINITY);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::INFINITY,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::NAN);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, f64::INFINITY);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::NAN,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::NAN,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            5.0,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 5.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.4413114540102976);
        close(fire.elliptical_b_, 5.592131112766734);
        close(fire.elliptical_c_, 4.407868887233267);
        close(fire.eccentricity_, 0.7882270216941833);
        close(fire.backing_spread_rate_, 1.1842622255334672);
        close(fire.flanking_spread_rate_, 3.4413114540102976);
        close(fire.fire_length_to_width_ratio_, 1.625);
        close(fire.heading_to_backing_ratio_, 8.444075584411621);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            5.0,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.0568181818);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 8.877080193965405);
        close(fire.elliptical_b_, 8.940127638504643);
        close(fire.elliptical_c_, 1.0598723614953558);
        close(fire.eccentricity_, 0.11855226282570552);
        close(fire.backing_spread_rate_, 7.8802552770092875);
        close(fire.flanking_spread_rate_, 8.877080193965405);
        close(fire.fire_length_to_width_ratio_, 1.007102272725);
        close(fire.heading_to_backing_ratio_, 1.2689944232105128);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            13.300000190734863,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 13.300000190734863);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 1.9492908651680323);
        close(fire.elliptical_b_, 5.189986742611023);
        close(fire.elliptical_c_, 4.810013257388977);
        close(fire.eccentricity_, 0.9267871975898743);
        close(fire.backing_spread_rate_, 0.37997348522204516);
        close(fire.flanking_spread_rate_, 1.9492908651680323);
        close(fire.fire_length_to_width_ratio_, 2.6624999046325684);
        close(fire.heading_to_backing_ratio_, 26.3176212310791);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            13.300000190734863,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.15113636575544162);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 8.23594834192165);
        close(fire.elliptical_b_, 8.391542254540099);
        close(fire.elliptical_c_, 1.6084577454599014);
        close(fire.eccentricity_, 0.1916760586636709);
        close(fire.backing_spread_rate_, 6.783084509080197);
        close(fire.flanking_spread_rate_, 8.23594834192165);
        close(fire.fire_length_to_width_ratio_, 1.0188920457194302);
        close(fire.heading_to_backing_ratio_, 1.4742555524132814);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1000000000.0,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 1000000000.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 4e-08);
        close(fire.elliptical_b_, 5.0);
        close(fire.elliptical_c_, 5.0);
        close(fire.eccentricity_, 1.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 4e-08);
        close(fire.fire_length_to_width_ratio_, 125000000.0);
        close(fire.heading_to_backing_ratio_, f64::INFINITY);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1000000000.0,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 11363636.36);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.5199975230485797e-06);
        close(fire.elliptical_b_, 5.00000000000062);
        close(fire.elliptical_c_, 4.999999999999381);
        close(fire.eccentricity_, 0.9999999999997522);
        close(fire.backing_spread_rate_, 1.2390088954818282e-12);
        close(fire.flanking_spread_rate_, 3.5199975230485797e-06);
        close(fire.fire_length_to_width_ratio_, 1420455.545);
        close(fire.heading_to_backing_ratio_, 8069854644439.286);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1.0000000150474662e+30,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 1.0000000150474662e+30);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::INFINITY);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, 1.2500000188093328e+29);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1.0000000150474662e+30,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 1.1363636530993935e+28);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.5199999481593194e-27);
        close(fire.elliptical_b_, 5.0);
        close(fire.elliptical_c_, 5.0);
        close(fire.eccentricity_, 1.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 3.5199999481593194e-27);
        close(fire.fire_length_to_width_ratio_, 1.420454566374242e+27);
        close(fire.heading_to_backing_ratio_, f64::INFINITY);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::INFINITY,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::NAN);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, f64::INFINITY);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::INFINITY,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::NAN);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, f64::INFINITY);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::INFINITY,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::NAN);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, f64::INFINITY);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::INFINITY,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::NAN);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, f64::INFINITY);
        close(fire.heading_to_backing_ratio_, f64::NAN);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::NAN,
            SpeedUnit::try_from(5).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
    }
    {
        let mut fire = FireSize::default();
        fire.elliptical_a_ = 7.0;
        fire.elliptical_b_ = 8.0;
        fire.elliptical_c_ = 9.0;
        fire.heading_to_backing_ratio_ = 10.0;
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::NAN,
            SpeedUnit::try_from(0).unwrap(),
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
    }
}
#[test]
#[should_panic]
fn comparator_rejects_near_zero_error_above_absolute_tolerance() {
    close(5e-10, 0.0);
}

#[test]
fn surface_numeric_kinds_and_crown_getters_match_python() {
    use fire_size::{FireSize, GeometryError, NumericKind};
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            5.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 5.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.538182024739927);
        close(fire.elliptical_b_, 5.625936602009636);
        close(fire.elliptical_c_, 4.374063397990364);
        close(fire.eccentricity_, 0.7774818145707344);
        close(fire.backing_spread_rate_, 1.2518732040192728);
        close(fire.flanking_spread_rate_, 3.538182024739927);
        close(fire.fire_length_to_width_ratio_, 1.5900642088709862);
        close(fire.heading_to_backing_ratio_, 7.988029432928135);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            18072.68674844852,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            495.7998580829907,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            5.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 5.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.441311542550502);
        close(fire.elliptical_b_, 5.592131256644566);
        close(fire.elliptical_c_, 4.407868743355435);
        close(fire.eccentricity_, 0.7882269819968921);
        close(fire.backing_spread_rate_, 1.184262513289132);
        close(fire.flanking_spread_rate_, 3.441311542550502);
        close(fire.fire_length_to_width_ratio_, 1.625);
        close(fire.heading_to_backing_ratio_, 8.444073748671086);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            13968.004259806925,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            431.36483743521393,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            13.3,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 13.3);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 1.2413627556321922);
        close(fire.elliptical_b_, 5.077049074553537);
        close(fire.elliptical_c_, 4.922950925446462);
        close(fire.eccentricity_, 0.9696480875319043);
        close(fire.backing_spread_rate_, 0.15409814910707503);
        close(fire.flanking_spread_rate_, 1.2413627556321922);
        close(fire.fire_length_to_width_ratio_, 4.089899629675884);
        close(fire.heading_to_backing_ratio_, 64.89370610838093);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            5722.13089203983,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            369.25372122385505,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            13.3,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 13.3);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 1.9492907903339078);
        close(fire.elliptical_b_, 5.18998672926403);
        close(fire.elliptical_c_, 4.810013270735971);
        close(fire.eccentricity_, 0.9267872003630457);
        close(fire.backing_spread_rate_, 0.37997345852805897);
        close(fire.flanking_spread_rate_, 1.9492907903339078);
        close(fire.fire_length_to_width_ratio_, 2.6625);
        close(fire.heading_to_backing_ratio_, 26.317627654147216);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            8525.073022417371,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            367.3303522894545,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            10000.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert_eq!(result, Err(GeometryError::Overflow));
        close(fire.effective_wind_speed_, 10000.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 0.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            10000.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 10000.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.003996803196420295);
        close(fire.elliptical_b_, 5.000000798721789);
        close(fire.elliptical_c_, 4.99999920127821);
        close(fire.eccentricity_, 0.9999996805113353);
        close(fire.backing_spread_rate_, 1.5974435789134652e-06);
        close(fire.flanking_spread_rate_, 0.003996803196420295);
        close(fire.fire_length_to_width_ratio_, 1251.0);
        close(fire.heading_to_backing_ratio_, 6260002.0012767585);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            18.143890425408678,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            267.248833089549,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            1e+38,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert_eq!(result, Err(GeometryError::Overflow));
        close(fire.effective_wind_speed_, 1e+38);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 0.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e+38,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert_eq!(result, Err(GeometryError::DivisionByZero));
        close(fire.effective_wind_speed_, 1e+38);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 1.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 4e-37);
        close(fire.fire_length_to_width_ratio_, 1.25e+37);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            f64::INFINITY,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.6274606680622823);
        close(fire.elliptical_b_, 5.019685344498258);
        close(fire.elliptical_c_, 4.980314655501742);
        close(fire.eccentricity_, 0.9921567416492215);
        close(fire.backing_spread_rate_, 0.03937068899651632);
        close(fire.flanking_spread_rate_, 0.6274606680622823);
        close(fire.fire_length_to_width_ratio_, 8.0);
        close(fire.heading_to_backing_ratio_, 253.9960629311019);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            2859.63574839736,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            348.9340534455657,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::INFINITY,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::NAN);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, f64::INFINITY);
        close(fire.heading_to_backing_ratio_, f64::NAN);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            0.0,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            267.0353755551324,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            f64::NAN,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            90792.02768874503,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            1068.1415022205297,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::NAN,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            22698.006922186254,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            534.0707511102648,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            5.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 5.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.538182024739927);
        close(fire.elliptical_b_, 5.625936602009636);
        close(fire.elliptical_c_, 4.374063397990364);
        close(fire.eccentricity_, 0.7774818145707344);
        close(fire.backing_spread_rate_, 1.2518732040192728);
        close(fire.flanking_spread_rate_, 3.538182024739927);
        close(fire.fire_length_to_width_ratio_, 1.5900642088709862);
        close(fire.heading_to_backing_ratio_, 7.988029432928135);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            18072.68674844852,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            495.7998580829907,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            5.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 5.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.441311542550502);
        close(fire.elliptical_b_, 5.592131256644566);
        close(fire.elliptical_c_, 4.407868743355435);
        close(fire.eccentricity_, 0.7882269819968921);
        close(fire.backing_spread_rate_, 1.184262513289132);
        close(fire.flanking_spread_rate_, 3.441311542550502);
        close(fire.fire_length_to_width_ratio_, 1.625);
        close(fire.heading_to_backing_ratio_, 8.444073748671086);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            13968.004259806925,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            431.36483743521393,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            13.3,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 13.3);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 1.2413627556321922);
        close(fire.elliptical_b_, 5.077049074553537);
        close(fire.elliptical_c_, 4.922950925446462);
        close(fire.eccentricity_, 0.9696480875319043);
        close(fire.backing_spread_rate_, 0.15409814910707503);
        close(fire.flanking_spread_rate_, 1.2413627556321922);
        close(fire.fire_length_to_width_ratio_, 4.089899629675884);
        close(fire.heading_to_backing_ratio_, 64.89370610838093);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            5722.13089203983,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            369.25372122385505,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            13.3,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 13.3);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 1.9492907903339078);
        close(fire.elliptical_b_, 5.18998672926403);
        close(fire.elliptical_c_, 4.810013270735971);
        close(fire.eccentricity_, 0.9267872003630457);
        close(fire.backing_spread_rate_, 0.37997345852805897);
        close(fire.flanking_spread_rate_, 1.9492907903339078);
        close(fire.fire_length_to_width_ratio_, 2.6625);
        close(fire.heading_to_backing_ratio_, 26.317627654147216);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            8525.073022417371,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            367.3303522894545,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            10000.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert_eq!(result, Err(GeometryError::Overflow));
        close(fire.effective_wind_speed_, 10000.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 0.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            10000.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 10000.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.003996803196420295);
        close(fire.elliptical_b_, 5.000000798721789);
        close(fire.elliptical_c_, 4.99999920127821);
        close(fire.eccentricity_, 0.9999996805113353);
        close(fire.backing_spread_rate_, 1.5974435789134652e-06);
        close(fire.flanking_spread_rate_, 0.003996803196420295);
        close(fire.fire_length_to_width_ratio_, 1251.0);
        close(fire.heading_to_backing_ratio_, 6260002.0012767585);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            18.143890425408678,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            267.248833089549,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            1e+38,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert_eq!(result, Err(GeometryError::Overflow));
        close(fire.effective_wind_speed_, 1e+38);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 0.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e+38,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 1e+38);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 4e-37);
        close(fire.elliptical_b_, 5.0);
        close(fire.elliptical_c_, 5.0);
        close(fire.eccentricity_, 1.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 4e-37);
        close(fire.fire_length_to_width_ratio_, 1.25e+37);
        close(fire.heading_to_backing_ratio_, f64::INFINITY);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            1.8158405537749005e-33,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            267.0353755551324,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            f64::INFINITY,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.6274606680622823);
        close(fire.elliptical_b_, 5.019685344498258);
        close(fire.elliptical_c_, 4.980314655501742);
        close(fire.eccentricity_, 0.9921567416492215);
        close(fire.backing_spread_rate_, 0.03937068899651632);
        close(fire.flanking_spread_rate_, 0.6274606680622823);
        close(fire.fire_length_to_width_ratio_, 8.0);
        close(fire.heading_to_backing_ratio_, 253.9960629311019);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            2859.63574839736,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            348.9340534455657,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::INFINITY,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::NAN);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, f64::INFINITY);
        close(fire.heading_to_backing_ratio_, f64::NAN);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            0.0,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            267.0353755551324,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            f64::NAN,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            90792.02768874503,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            1068.1415022205297,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::NAN,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            22698.006922186254,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            534.0707511102648,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            5.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 5.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.5381821459626868);
        close(fire.elliptical_b_, 5.625936644900456);
        close(fire.elliptical_c_, 4.374063355099543);
        close(fire.eccentricity_, 0.7774818010196304);
        close(fire.backing_spread_rate_, 1.2518732898009126);
        close(fire.flanking_spread_rate_, 3.5381821459626868);
        close(fire.fire_length_to_width_ratio_, 1.5900641665155772);
        close(fire.heading_to_backing_ratio_, 7.9880288855674175);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            18072.68750542441,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            495.7998662533556,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            5.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 5.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 3.4413114540102976);
        close(fire.elliptical_b_, 5.592131112766734);
        close(fire.elliptical_c_, 4.407868887233267);
        close(fire.eccentricity_, 0.7882270216941833);
        close(fire.backing_spread_rate_, 1.1842622255334672);
        close(fire.flanking_spread_rate_, 3.4413114540102976);
        close(fire.fire_length_to_width_ratio_, 1.625);
        close(fire.heading_to_backing_ratio_, 8.444075584411621);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            13968.004259806925,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            431.3648276404147,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            13.300000190734863,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 13.300000190734863);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 1.2413628199217401);
        close(fire.elliptical_b_, 5.077049082534203);
        close(fire.elliptical_c_, 4.922950917465798);
        close(fire.eccentricity_, 0.9696480844357944);
        close(fire.backing_spread_rate_, 0.1540981650684055);
        close(fire.flanking_spread_rate_, 1.2413628199217401);
        close(fire.fire_length_to_width_ratio_, 4.089899424291021);
        close(fire.heading_to_backing_ratio_, 64.89369938675722);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            5722.131197380766,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            369.2537237485578,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            13.300000190734863,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 13.300000190734863);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 1.9492908651680323);
        close(fire.elliptical_b_, 5.189986742611023);
        close(fire.elliptical_c_, 4.810013257388977);
        close(fire.eccentricity_, 0.9267871975898743);
        close(fire.backing_spread_rate_, 0.37997348522204516);
        close(fire.flanking_spread_rate_, 1.9492908651680323);
        close(fire.fire_length_to_width_ratio_, 2.6624999046325684);
        close(fire.heading_to_backing_ratio_, 26.3176212310791);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            8525.073327774873,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            367.33035572683116,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            10000.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert_eq!(result, Err(GeometryError::Overflow));
        close(fire.effective_wind_speed_, 10000.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 0.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            10000.0,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 10000.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0039968031535237);
        close(fire.elliptical_b_, 5.0000007450581485);
        close(fire.elliptical_c_, 4.9999992549418515);
        close(fire.eccentricity_, 0.9999997019767761);
        close(fire.backing_spread_rate_, 1.4901162970204707e-06);
        close(fire.flanking_spread_rate_, 0.0039968031535237);
        close(fire.fire_length_to_width_ratio_, 1251.0);
        close(fire.heading_to_backing_ratio_, 6832126.5);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            18.143890425408678,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            267.24884830632936,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            9.999999680285692e+37,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert_eq!(result, Err(GeometryError::Overflow));
        close(fire.effective_wind_speed_, 9.999999680285692e+37);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 0.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 0.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            9.999999680285692e+37,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 9.999999680285692e+37);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::INFINITY);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, 1.2499999600357116e+37);
        close(fire.heading_to_backing_ratio_, f64::NAN);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            1.8158406118299227e-33,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            267.0353755551324,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            f64::INFINITY,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.6274606680622823);
        close(fire.elliptical_b_, 5.019685344498258);
        close(fire.elliptical_c_, 4.980314655501742);
        close(fire.eccentricity_, 0.9921567416492215);
        close(fire.backing_spread_rate_, 0.03937068899651632);
        close(fire.flanking_spread_rate_, 0.6274606680622823);
        close(fire.fire_length_to_width_ratio_, 8.0);
        close(fire.heading_to_backing_ratio_, 253.9960629311019);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            2859.63574839736,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            348.9340534455657,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::INFINITY,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::INFINITY);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, f64::NAN);
        close(fire.elliptical_b_, f64::NAN);
        close(fire.elliptical_c_, f64::NAN);
        close(fire.eccentricity_, f64::NAN);
        close(fire.backing_spread_rate_, f64::NAN);
        close(fire.flanking_spread_rate_, f64::NAN);
        close(fire.fire_length_to_width_ratio_, f64::INFINITY);
        close(fire.heading_to_backing_ratio_, f64::NAN);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            0.0,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            267.0353755551324,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            f64::NAN,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        close(
            fire.get_fire_area(false, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            90792.02768874503,
        );
        close(
            fire.get_fire_perimeter(false, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            1068.1415022205297,
        );
    }
    {
        let mut fire = FireSize::default();
        let result = fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            f64::NAN,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        );
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, f64::NAN);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        close(
            fire.get_fire_area(true, AreaUnit::SquareFeet, 17.0, TimeUnit::Minutes),
            22698.006922186254,
        );
        close(
            fire.get_fire_perimeter(true, LengthUnit::Feet, 17.0, TimeUnit::Minutes)
                .unwrap(),
            534.0707511102648,
        );
    }
}

#[test]
fn wind_thresholds_match_original_numeric_kind() {
    use fire_size::{FireSize, NumericKind};
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            9.999999999999998e-08,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 9.999999999999998e-08);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            9.999999999999998e-08,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 9.999999999999998e-08);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            1e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            1.0000000000000001e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1.0000000000000001e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 9.998771596707114);
        close(fire.elliptical_b_, 9.998771672155847);
        close(fire.elliptical_c_, 0.001228327844151167);
        close(fire.eccentricity_, 0.00012284787416166578);
        close(fire.backing_spread_rate_, 9.997543344311696);
        close(fire.flanking_spread_rate_, 9.998771596707114);
        close(fire.fire_length_to_width_ratio_, 1.0000000075458002);
        close(fire.heading_to_backing_ratio_, 1.0002457259352322);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1.0000000000000001e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::PythonFloat,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1.0000000000000001e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 9.9984189861677);
        close(fire.elliptical_b_, 9.998419111147935);
        close(fire.elliptical_c_, 0.0015808888520645326);
        close(fire.eccentricity_, 0.0001581138812536959);
        close(fire.backing_spread_rate_, 9.996838222295871);
        close(fire.flanking_spread_rate_, 9.9984189861677);
        close(fire.fire_length_to_width_ratio_, 1.0000000125);
        close(fire.heading_to_backing_ratio_, 1.0003162777704133);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            9.999999999999998e-08,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 9.999999999999998e-08);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat64);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            9.999999999999998e-08,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 9.999999999999998e-08);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat64);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            1e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat64);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat64);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            1.0000000000000001e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1.0000000000000001e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 9.998771596707114);
        close(fire.elliptical_b_, 9.998771672155847);
        close(fire.elliptical_c_, 0.001228327844151167);
        close(fire.eccentricity_, 0.00012284787416166578);
        close(fire.backing_spread_rate_, 9.997543344311696);
        close(fire.flanking_spread_rate_, 9.998771596707114);
        close(fire.fire_length_to_width_ratio_, 1.0000000075458002);
        close(fire.heading_to_backing_ratio_, 1.0002457259352322);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat64);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1.0000000000000001e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat64,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1.0000000000000001e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 9.9984189861677);
        close(fire.elliptical_b_, 9.998419111147935);
        close(fire.elliptical_c_, 0.0015808888520645326);
        close(fire.eccentricity_, 0.0001581138812536959);
        close(fire.backing_spread_rate_, 9.996838222295871);
        close(fire.flanking_spread_rate_, 9.9984189861677);
        close(fire.fire_length_to_width_ratio_, 1.0000000125);
        close(fire.heading_to_backing_ratio_, 1.0003162777704133);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat64);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::NumpyFloat64);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::NumpyFloat64);
        assert_eq!(fire.ratio_is_numpy_, true);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            9.999999406318238e-08,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 9.999999406318238e-08);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat32);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            9.999999406318238e-08,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 9.999999406318238e-08);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat32);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            1.0000000116860974e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1.0000000116860974e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat32);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1.0000000116860974e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1.0000000116860974e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat32);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            false,
            1.000000082740371e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1.000000082740371e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 9.998771596670972);
        close(fire.elliptical_b_, 9.998771672119709);
        close(fire.elliptical_c_, 0.0012283278802929232);
        close(fire.eccentricity_, 0.00012284787777661795);
        close(fire.backing_spread_rate_, 9.997543344239416);
        close(fire.flanking_spread_rate_, 9.998771596670972);
        close(fire.fire_length_to_width_ratio_, 1.0000000075458007);
        close(fire.heading_to_backing_ratio_, 1.000245725942464);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat32);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, false);
    }
    {
        let mut fire = FireSize::default();
        fire.calculate_fire_basic_dimensions_with_wind_kind(
            true,
            1.000000082740371e-07,
            SpeedUnit::MilesPerHour,
            10.0,
            SpeedUnit::FeetPerMinute,
            NumericKind::NumpyFloat32,
        )
        .unwrap();
        close(fire.effective_wind_speed_, 1.000000082740371e-07);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 10.0);
        close(fire.elliptical_b_, 10.0);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 10.0);
        close(fire.flanking_spread_rate_, 10.0);
        close(fire.fire_length_to_width_ratio_, 1.0);
        close(fire.heading_to_backing_ratio_, 1.0);
        assert_eq!(fire.effective_wind_numeric_kind_, NumericKind::NumpyFloat32);
        assert_eq!(fire.ratio_numeric_kind_, NumericKind::NumpyFloat32);
        assert_eq!(fire.eccentricity_numeric_kind_, NumericKind::PythonFloat);
        assert_eq!(fire.ratio_is_numpy_, true);
    }
}
#[test]
fn ellipse_ratio_thresholds_match_original_numeric_kind() {
    use fire_size::{FireSize, GeometryError, NumericKind};
    {
        let mut fire = FireSize::default();
        fire.forward_spread_rate_ = 10.0;
        fire.backing_spread_rate_ = 1.0;
        fire.fire_length_to_width_ratio_ = 9.999999999999998e-08;
        fire.ratio_numeric_kind_ = NumericKind::PythonFloat;
        fire.ratio_is_numpy_ = false;
        let result = fire.calculate_elliptical_dimensions();
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.5);
        close(fire.elliptical_c_, 4.5);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 1.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 9.999999999999998e-08);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        fire.forward_spread_rate_ = 10.0;
        fire.backing_spread_rate_ = 1.0;
        fire.fire_length_to_width_ratio_ = 1e-07;
        fire.ratio_numeric_kind_ = NumericKind::PythonFloat;
        fire.ratio_is_numpy_ = false;
        let result = fire.calculate_elliptical_dimensions();
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.5);
        close(fire.elliptical_c_, 4.5);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 1.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1e-07);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        fire.forward_spread_rate_ = 10.0;
        fire.backing_spread_rate_ = 1.0;
        fire.fire_length_to_width_ratio_ = 1.0000000000000001e-07;
        fire.ratio_numeric_kind_ = NumericKind::PythonFloat;
        fire.ratio_is_numpy_ = false;
        let result = fire.calculate_elliptical_dimensions();
        assert_eq!(result, Err(GeometryError::Domain));
        close(fire.effective_wind_speed_, 0.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.5);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 1.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1.0000000000000001e-07);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        fire.forward_spread_rate_ = 10.0;
        fire.backing_spread_rate_ = 1.0;
        fire.fire_length_to_width_ratio_ = 9.999999999999998e-08;
        fire.ratio_numeric_kind_ = NumericKind::NumpyFloat64;
        fire.ratio_is_numpy_ = true;
        let result = fire.calculate_elliptical_dimensions();
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.5);
        close(fire.elliptical_c_, 4.5);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 1.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 9.999999999999998e-08);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        fire.forward_spread_rate_ = 10.0;
        fire.backing_spread_rate_ = 1.0;
        fire.fire_length_to_width_ratio_ = 1e-07;
        fire.ratio_numeric_kind_ = NumericKind::NumpyFloat64;
        fire.ratio_is_numpy_ = true;
        let result = fire.calculate_elliptical_dimensions();
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.5);
        close(fire.elliptical_c_, 4.5);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 1.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1e-07);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        fire.forward_spread_rate_ = 10.0;
        fire.backing_spread_rate_ = 1.0;
        fire.fire_length_to_width_ratio_ = 1.0000000000000001e-07;
        fire.ratio_numeric_kind_ = NumericKind::NumpyFloat64;
        fire.ratio_is_numpy_ = true;
        let result = fire.calculate_elliptical_dimensions();
        assert_eq!(result, Err(GeometryError::Domain));
        close(fire.effective_wind_speed_, 0.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.5);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 1.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1.0000000000000001e-07);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        fire.forward_spread_rate_ = 10.0;
        fire.backing_spread_rate_ = 1.0;
        fire.fire_length_to_width_ratio_ = 9.999999406318238e-08;
        fire.ratio_numeric_kind_ = NumericKind::NumpyFloat32;
        fire.ratio_is_numpy_ = true;
        let result = fire.calculate_elliptical_dimensions();
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.5);
        close(fire.elliptical_c_, 4.5);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 1.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 9.999999406318238e-08);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        fire.forward_spread_rate_ = 10.0;
        fire.backing_spread_rate_ = 1.0;
        fire.fire_length_to_width_ratio_ = 1.0000000116860974e-07;
        fire.ratio_numeric_kind_ = NumericKind::NumpyFloat32;
        fire.ratio_is_numpy_ = true;
        let result = fire.calculate_elliptical_dimensions();
        assert!(result.is_ok());
        close(fire.effective_wind_speed_, 0.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.5);
        close(fire.elliptical_c_, 4.5);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 1.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1.0000000116860974e-07);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
    {
        let mut fire = FireSize::default();
        fire.forward_spread_rate_ = 10.0;
        fire.backing_spread_rate_ = 1.0;
        fire.fire_length_to_width_ratio_ = 1.000000082740371e-07;
        fire.ratio_numeric_kind_ = NumericKind::NumpyFloat32;
        fire.ratio_is_numpy_ = true;
        let result = fire.calculate_elliptical_dimensions();
        assert_eq!(result, Err(GeometryError::Domain));
        close(fire.effective_wind_speed_, 0.0);
        close(fire.forward_spread_rate_, 10.0);
        close(fire.elapsed_time_, 0.0);
        close(fire.elliptical_a_, 0.0);
        close(fire.elliptical_b_, 5.5);
        close(fire.elliptical_c_, 0.0);
        close(fire.eccentricity_, 0.0);
        close(fire.backing_spread_rate_, 1.0);
        close(fire.flanking_spread_rate_, 0.0);
        close(fire.fire_length_to_width_ratio_, 1.000000082740371e-07);
        close(fire.heading_to_backing_ratio_, 0.0);
    }
}
