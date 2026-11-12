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
    if actual == 0.0 && expected == 0.0 {
        assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
    }
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
    let actual = helper.calculate_backing_spread_rate();
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
    let actual = helper.calculate_backing_spread_rate();
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
    let actual = helper.calculate_elliptical_dimensions();
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
    let actual = helper.calculate_elliptical_dimensions();
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
    let actual = helper.calculate_fire_basic_dimensions(
        false,
        5.0,
        SpeedUnit::try_from(5).unwrap(),
        10.0,
        SpeedUnit::try_from(0).unwrap(),
    );
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
    let actual = helper.calculate_fire_basic_dimensions(
        false,
        5.0,
        SpeedUnit::try_from(5).unwrap(),
        10.0,
        SpeedUnit::try_from(0).unwrap(),
    );
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
    let actual = helper.calculate_flanking_spread_rate();
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
    let actual = helper.calculate_flanking_spread_rate();
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
    let actual = helper.calculate_surface_fire_length_to_width_ratio();
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
    let actual = helper.calculate_surface_fire_length_to_width_ratio();
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
    let actual = helper.get_fire_perimeter(
        false,
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
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
    let actual = helper.get_fire_perimeter(
        false,
        LengthUnit::try_from(0).unwrap(),
        10.0,
        TimeUnit::try_from(0).unwrap(),
    );
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
    let actual = helper.calculate_horizontal_distance(
        10.0,
        LengthUnit::try_from(0).unwrap(),
        30.0,
        SlopeUnit::try_from(0).unwrap(),
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
    let actual = helper.calculate_horizontal_distance(
        10.0,
        LengthUnit::try_from(0).unwrap(),
        30.0,
        SlopeUnit::try_from(0).unwrap(),
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
// The appended Python numeric-kind and scalar-exception emulation tests were
// replaced by f64_helpers.rs and f64_state_domains.rs. Original Python evidence
// remains in tests/data/rust_port; finite oracle traces and all LUT cells above remain.
#[test]
#[should_panic]
fn comparator_rejects_near_zero_error_above_absolute_tolerance() {
    close(5e-10, 0.0);
}
