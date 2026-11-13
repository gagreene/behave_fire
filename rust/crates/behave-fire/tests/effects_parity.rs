//! Exported Python oracle parity, reference 7933789.
#![allow(unused_imports)]
use behave_fire::units::*;
use behave_fire::{
    fine_dead_fuel_moisture, fire_size, ignite, mortality, safety, slope, spot,
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
#[test]
fn comparator_rejects_zero_sign_changes_but_allows_roundoff() {
    assert!(std::panic::catch_unwind(|| close(0.0, -0.0)).is_err());
    close(1.0 + 1e-10, 1.0);
}
// build_mortality_lookup/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_0() {
    let actual = mortality::build_mortality_lookup(None);
    close(actual[0][0], 0.0);
    close(actual[0][1], 0.0);
    close(actual[0][2], 0.0);
    close(actual[1][0], -2.4027);
    close(actual[1][1], 0.0);
    close(actual[1][2], 0.9568);
    close(actual[2][0], -2.2471);
    close(actual[2][1], 0.0);
    close(actual[2][2], 0.8093);
    close(actual[3][0], -3.1789);
    close(actual[3][1], 0.0);
    close(actual[3][2], 1.5416);
    close(actual[4][0], -2.3327);
    close(actual[4][1], 0.0);
    close(actual[4][2], 0.9577);
    close(actual[5][0], -2.3327);
    close(actual[5][1], 0.0);
    close(actual[5][2], 0.9577);
    close(actual[6][0], -1.4667);
    close(actual[6][1], 0.0);
    close(actual[6][2], 0.7401);
    close(actual[7][0], -2.4027);
    close(actual[7][1], 0.0);
    close(actual[7][2], 0.9568);
    close(actual[8][0], -3.3701);
    close(actual[8][1], 0.0);
    close(actual[8][2], 1.4804);
    close(actual[9][0], -3.3701);
    close(actual[9][1], 0.0);
    close(actual[9][2], 1.4804);
    close(actual[10][0], -2.9986);
    close(actual[10][1], 0.0);
    close(actual[10][2], 1.4804);
    close(actual[11][0], -2.9986);
    close(actual[11][1], 0.0);
    close(actual[11][2], 1.4804);
    close(actual[12][0], -2.4027);
    close(actual[12][1], 0.0);
    close(actual[12][2], 0.9568);
    close(actual[13][0], -3.0471);
    close(actual[13][1], 0.0);
    close(actual[13][2], 1.2668);
    close(actual[14][0], -2.3327);
    close(actual[14][1], 0.0);
    close(actual[14][2], 0.9577);
    close(actual[15][0], -2.7735);
    close(actual[15][1], 0.0);
    close(actual[15][2], 1.007);
    close(actual[16][0], -3.3701);
    close(actual[16][1], 0.0);
    close(actual[16][2], 1.4804);
    close(actual[17][0], -2.2471);
    close(actual[17][1], 0.0);
    close(actual[17][2], 0.8093);
    close(actual[18][0], -2.4027);
    close(actual[18][1], 0.0);
    close(actual[18][2], 0.9568);
    close(actual[19][0], -2.2471);
    close(actual[19][1], 0.0);
    close(actual[19][2], 0.8093);
    close(actual[20][0], -3.3701);
    close(actual[20][1], 0.0);
    close(actual[20][2], 1.4804);
    close(actual[21][0], 0.0);
    close(actual[21][1], 0.0);
    close(actual[21][2], 0.0);
    close(actual[22][0], 0.0);
    close(actual[22][1], 0.0);
    close(actual[22][2], 0.0);
    close(actual[23][0], 0.0);
    close(actual[23][1], 0.0);
    close(actual[23][2], 0.0);
    close(actual[24][0], 0.0);
    close(actual[24][1], 0.0);
    close(actual[24][2], 0.0);
    close(actual[25][0], 0.0);
    close(actual[25][1], 0.0);
    close(actual[25][2], 0.0);
    close(actual[26][0], 0.0);
    close(actual[26][1], 0.0);
    close(actual[26][2], 0.0);
    close(actual[27][0], 0.0);
    close(actual[27][1], 0.0);
    close(actual[27][2], 0.0);
    close(actual[28][0], 0.0);
    close(actual[28][1], 0.0);
    close(actual[28][2], 0.0);
    close(actual[29][0], 0.0);
    close(actual[29][1], 0.0);
    close(actual[29][2], 0.0);
    close(actual[30][0], 0.0);
    close(actual[30][1], 0.0);
    close(actual[30][2], 0.0);
    close(actual[31][0], 0.0);
    close(actual[31][1], 0.0);
    close(actual[31][2], 0.0);
    close(actual[32][0], 0.0);
    close(actual[32][1], 0.0);
    close(actual[32][2], 0.0);
    close(actual[33][0], 0.0);
    close(actual[33][1], 0.0);
    close(actual[33][2], 0.0);
    close(actual[34][0], 0.0);
    close(actual[34][1], 0.0);
    close(actual[34][2], 0.0);
    close(actual[35][0], 0.0);
    close(actual[35][1], 0.0);
    close(actual[35][2], 0.0);
    close(actual[36][0], 0.0);
    close(actual[36][1], 0.0);
    close(actual[36][2], 0.0);
    close(actual[37][0], 0.0);
    close(actual[37][1], 0.0);
    close(actual[37][2], 0.0);
    close(actual[38][0], 0.0);
    close(actual[38][1], 0.0);
    close(actual[38][2], 0.0);
    close(actual[39][0], 0.0);
    close(actual[39][1], 0.0);
    close(actual[39][2], 0.0);
    close(actual[40][0], 0.0);
    close(actual[40][1], 0.0);
    close(actual[40][2], 0.0);
    close(actual[41][0], 0.0);
    close(actual[41][1], 0.0);
    close(actual[41][2], 0.0);
    close(actual[42][0], 0.0);
    close(actual[42][1], 0.0);
    close(actual[42][2], 0.0);
    close(actual[43][0], 0.0);
    close(actual[43][1], 0.0);
    close(actual[43][2], 0.0);
    close(actual[44][0], 0.0);
    close(actual[44][1], 0.0);
    close(actual[44][2], 0.0);
    close(actual[45][0], 0.0);
    close(actual[45][1], 0.0);
    close(actual[45][2], 0.0);
    close(actual[46][0], 0.0);
    close(actual[46][1], 0.0);
    close(actual[46][2], 0.0);
    close(actual[47][0], 0.0);
    close(actual[47][1], 0.0);
    close(actual[47][2], 0.0);
    close(actual[48][0], 0.0);
    close(actual[48][1], 0.0);
    close(actual[48][2], 0.0);
    close(actual[49][0], 0.0);
    close(actual[49][1], 0.0);
    close(actual[49][2], 0.0);
    close(actual[50][0], 0.0);
    close(actual[50][1], 0.0);
    close(actual[50][2], 0.0);
    close(actual[51][0], 0.0);
    close(actual[51][1], 0.0);
    close(actual[51][2], 0.0);
    close(actual[52][0], 0.0);
    close(actual[52][1], 0.0);
    close(actual[52][2], 0.0);
    close(actual[53][0], 0.0);
    close(actual[53][1], 0.0);
    close(actual[53][2], 0.0);
    close(actual[54][0], 0.0);
    close(actual[54][1], 0.0);
    close(actual[54][2], 0.0);
    close(actual[55][0], 0.0);
    close(actual[55][1], 0.0);
    close(actual[55][2], 0.0);
    close(actual[56][0], 0.0);
    close(actual[56][1], 0.0);
    close(actual[56][2], 0.0);
    close(actual[57][0], 0.0);
    close(actual[57][1], 0.0);
    close(actual[57][2], 0.0);
    close(actual[58][0], 0.0);
    close(actual[58][1], 0.0);
    close(actual[58][2], 0.0);
    close(actual[59][0], 0.0);
    close(actual[59][1], 0.0);
    close(actual[59][2], 0.0);
    close(actual[60][0], 0.0);
    close(actual[60][1], 0.0);
    close(actual[60][2], 0.0);
    close(actual[61][0], 0.0);
    close(actual[61][1], 0.0);
    close(actual[61][2], 0.0);
    close(actual[62][0], 0.0);
    close(actual[62][1], 0.0);
    close(actual[62][2], 0.0);
    close(actual[63][0], 0.0);
    close(actual[63][1], 0.0);
    close(actual[63][2], 0.0);
    close(actual[64][0], 0.0);
    close(actual[64][1], 0.0);
    close(actual[64][2], 0.0);
    close(actual[65][0], 0.0);
    close(actual[65][1], 0.0);
    close(actual[65][2], 0.0);
    close(actual[66][0], 0.0);
    close(actual[66][1], 0.0);
    close(actual[66][2], 0.0);
    close(actual[67][0], 0.0);
    close(actual[67][1], 0.0);
    close(actual[67][2], 0.0);
    close(actual[68][0], 0.0);
    close(actual[68][1], 0.0);
    close(actual[68][2], 0.0);
    close(actual[69][0], 0.0);
    close(actual[69][1], 0.0);
    close(actual[69][2], 0.0);
    close(actual[70][0], 0.0);
    close(actual[70][1], 0.0);
    close(actual[70][2], 0.0);
    close(actual[71][0], 0.0);
    close(actual[71][1], 0.0);
    close(actual[71][2], 0.0);
    close(actual[72][0], 0.0);
    close(actual[72][1], 0.0);
    close(actual[72][2], 0.0);
    close(actual[73][0], 0.0);
    close(actual[73][1], 0.0);
    close(actual[73][2], 0.0);
    close(actual[74][0], 0.0);
    close(actual[74][1], 0.0);
    close(actual[74][2], 0.0);
    close(actual[75][0], 0.0);
    close(actual[75][1], 0.0);
    close(actual[75][2], 0.0);
    close(actual[76][0], 0.0);
    close(actual[76][1], 0.0);
    close(actual[76][2], 0.0);
    close(actual[77][0], 0.0);
    close(actual[77][1], 0.0);
    close(actual[77][2], 0.0);
    close(actual[78][0], 0.0);
    close(actual[78][1], 0.0);
    close(actual[78][2], 0.0);
    close(actual[79][0], 0.0);
    close(actual[79][1], 0.0);
    close(actual[79][2], 0.0);
    close(actual[80][0], 0.0);
    close(actual[80][1], 0.0);
    close(actual[80][2], 0.0);
    close(actual[81][0], 0.0);
    close(actual[81][1], 0.0);
    close(actual[81][2], 0.0);
    close(actual[82][0], 0.0);
    close(actual[82][1], 0.0);
    close(actual[82][2], 0.0);
    close(actual[83][0], 0.0);
    close(actual[83][1], 0.0);
    close(actual[83][2], 0.0);
    close(actual[84][0], 0.0);
    close(actual[84][1], 0.0);
    close(actual[84][2], 0.0);
    close(actual[85][0], 0.0);
    close(actual[85][1], 0.0);
    close(actual[85][2], 0.0);
    close(actual[86][0], 0.0);
    close(actual[86][1], 0.0);
    close(actual[86][2], 0.0);
    close(actual[87][0], 0.0);
    close(actual[87][1], 0.0);
    close(actual[87][2], 0.0);
    close(actual[88][0], 0.0);
    close(actual[88][1], 0.0);
    close(actual[88][2], 0.0);
    close(actual[89][0], 0.0);
    close(actual[89][1], 0.0);
    close(actual[89][2], 0.0);
    close(actual[90][0], 0.0);
    close(actual[90][1], 0.0);
    close(actual[90][2], 0.0);
    close(actual[91][0], 0.0);
    close(actual[91][1], 0.0);
    close(actual[91][2], 0.0);
    close(actual[92][0], 0.0);
    close(actual[92][1], 0.0);
    close(actual[92][2], 0.0);
    close(actual[93][0], 0.0);
    close(actual[93][1], 0.0);
    close(actual[93][2], 0.0);
    close(actual[94][0], 0.0);
    close(actual[94][1], 0.0);
    close(actual[94][2], 0.0);
    close(actual[95][0], 0.0);
    close(actual[95][1], 0.0);
    close(actual[95][2], 0.0);
    close(actual[96][0], 0.0);
    close(actual[96][1], 0.0);
    close(actual[96][2], 0.0);
    close(actual[97][0], 0.0);
    close(actual[97][1], 0.0);
    close(actual[97][2], 0.0);
    close(actual[98][0], 0.0);
    close(actual[98][1], 0.0);
    close(actual[98][2], 0.0);
    close(actual[99][0], 0.0);
    close(actual[99][1], 0.0);
    close(actual[99][2], 0.0);
    close(actual[100][0], 2.3014);
    close(actual[100][1], -0.3267);
    close(actual[100][2], 1.1137);
    close(actual[101][0], -0.8727);
    close(actual[101][1], -0.1814);
    close(actual[101][2], 4.1947);
    close(actual[102][0], 2.7899);
    close(actual[102][1], -0.5511);
    close(actual[102][2], 1.2888);
    close(actual[103][0], 1.9438);
    close(actual[103][1], -0.4602);
    close(actual[103][2], 1.6352);
    close(actual[104][0], -1.8137);
    close(actual[104][1], -0.0603);
    close(actual[104][2], 0.8666);
    close(actual[105][0], -1.6262);
    close(actual[105][1], -0.0339);
    close(actual[105][2], 0.6901);
    close(actual[106][0], 0.3714);
    close(actual[106][1], -0.1005);
    close(actual[106][2], 1.5577);
    close(actual[107][0], -1.4416);
    close(actual[107][1], -0.1469);
    close(actual[107][2], 1.3159);
    close(actual[108][0], 0.1122);
    close(actual[108][1], -0.1287);
    close(actual[108][2], 1.2612);
    close(actual[109][0], 1.6779);
    close(actual[109][1], -1.0299);
    close(actual[109][2], 10.2855);
    close(actual[110][0], 0.0);
    close(actual[110][1], 0.0);
    close(actual[110][2], 0.0);
    close(actual[111][0], 0.0);
    close(actual[111][1], 0.0);
    close(actual[111][2], 0.0);
    close(actual[112][0], 0.0);
    close(actual[112][1], 0.0);
    close(actual[112][2], 0.0);
    close(actual[113][0], 0.0);
    close(actual[113][1], 0.0);
    close(actual[113][2], 0.0);
    close(actual[114][0], 0.0);
    close(actual[114][1], 0.0);
    close(actual[114][2], 0.0);
    close(actual[115][0], 0.0);
    close(actual[115][1], 0.0);
    close(actual[115][2], 0.0);
    close(actual[116][0], 0.0);
    close(actual[116][1], 0.0);
    close(actual[116][2], 0.0);
    close(actual[117][0], 0.0);
    close(actual[117][1], 0.0);
    close(actual[117][2], 0.0);
    close(actual[118][0], 0.0);
    close(actual[118][1], 0.0);
    close(actual[118][2], 0.0);
    close(actual[119][0], 0.0);
    close(actual[119][1], 0.0);
    close(actual[119][2], 0.0);
    close(actual[120][0], 0.0);
    close(actual[120][1], 0.0);
    close(actual[120][2], 0.0);
}

// calculate_crown_scorch_mortality/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_1() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_2() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_3() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_4() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_5() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_6() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.0,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_7() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.0,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_8() {
    let actual = mortality::calculate_crown_scorch_mortality(
        1.0,
        1.0,
        1.0,
        1.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 1.0);
    close(actual.crown_volume_scorch as f64, 1.0);
    close(actual.probability_mortality as f64, 0.1906333621198654);
}

// calculate_crown_scorch_mortality/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_9() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_10() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_11() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_12() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_13() {}

// calculate_crown_scorch_mortality/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_14() {
    let actual = mortality::calculate_crown_scorch_mortality(
        0.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_15() {
    let actual = mortality::calculate_crown_scorch_mortality(
        -1.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_16() {
    let actual = mortality::calculate_crown_scorch_mortality(
        f64::NAN,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, f64::NAN);
    close(actual.crown_volume_scorch as f64, f64::NAN);
    close(actual.probability_mortality as f64, f64::NAN);
}

// calculate_crown_scorch_mortality/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_17() {
    let actual = mortality::calculate_crown_scorch_mortality(
        f64::INFINITY,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 1.0);
    close(actual.crown_volume_scorch as f64, 1.0);
    close(actual.probability_mortality as f64, 0.1906333621198654);
}

// calculate_crown_scorch_mortality/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_18() {
    let actual = mortality::calculate_crown_scorch_mortality(
        f64::NEG_INFINITY,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_19() {
    let actual = mortality::calculate_crown_scorch_mortality(
        9.999999999999998e-08,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_20() {
    let actual = mortality::calculate_crown_scorch_mortality(
        1e-07,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_21() {
    let actual = mortality::calculate_crown_scorch_mortality(
        1.0000000000000001e-07,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_22() {
    let actual = mortality::calculate_crown_scorch_mortality(
        300.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 1.0);
    close(actual.crown_volume_scorch as f64, 1.0);
    close(actual.probability_mortality as f64, 0.1906333621198654);
}

// calculate_crown_scorch_mortality/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_23() {
    let actual = mortality::calculate_crown_scorch_mortality(
        300.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 1.0);
    close(actual.crown_volume_scorch as f64, 1.0);
    close(actual.probability_mortality as f64, 0.1906333621198654);
}

// calculate_crown_scorch_mortality/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_24() {
    let actual = mortality::calculate_crown_scorch_mortality(
        0.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_25() {
    let actual = mortality::calculate_crown_scorch_mortality(
        0.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_26() {
    let actual = mortality::calculate_crown_scorch_mortality(
        1.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        2.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_27() {
    let actual = mortality::calculate_crown_scorch_mortality(
        0.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        2.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_28() {
    let actual = mortality::calculate_crown_scorch_mortality(
        1.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_crown_scorch_mortality/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_29() {
    let actual = mortality::calculate_crown_scorch_mortality(
        1.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        1.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
}

// calculate_scorch_height/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_30() {
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
}

// calculate_scorch_height/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_31() {
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
}

// calculate_scorch_height/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_32() {
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
}

// calculate_scorch_height/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_33() {
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
}

// calculate_scorch_height/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_34() {
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
}

// calculate_scorch_height/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_35() {
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
}

// calculate_scorch_height/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_36() {
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
}

// calculate_scorch_height/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_37() {
    let actual = mortality::calculate_scorch_height(1.0, 1.0, 1.0);
    close(actual as f64, 0.3204872461492985);
}

// calculate_scorch_height/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_38() {
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
}

// calculate_scorch_height/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_39() {
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
}

// calculate_scorch_height/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_40() {
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
}

// calculate_scorch_height/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_41() {
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
    let actual = mortality::calculate_scorch_height(100.0, 5.0, 80.0);
    close(actual as f64, 15.08106598049722);
}

// calculate_scorch_height/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_42() {}

// calculate_scorch_height/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_43() {
    let actual = mortality::calculate_scorch_height(0.0, 5.0, 80.0);
    close(actual as f64, 0.0);
}

// calculate_scorch_height/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_44() {
    let actual = mortality::calculate_scorch_height(-1.0, 5.0, 80.0);
    close(actual as f64, 0.0);
}

// calculate_scorch_height/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_45() {
    let actual = mortality::calculate_scorch_height(f64::NAN, 5.0, 80.0);
    // A missing intensity is not the finite nonpositive-intensity branch.
    close(actual as f64, f64::NAN);
}

// calculate_scorch_height/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_46() {
    let actual = mortality::calculate_scorch_height(f64::INFINITY, 5.0, 80.0);
    close(actual as f64, f64::NAN);
}

// calculate_scorch_height/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_47() {
    let actual = mortality::calculate_scorch_height(f64::NEG_INFINITY, 5.0, 80.0);
    close(actual as f64, 0.0);
}

// calculate_scorch_height/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_48() {
    let actual = mortality::calculate_scorch_height(9.999999999999998e-08, 5.0, 80.0);
    close(actual as f64, 0.0);
}

// calculate_scorch_height/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_49() {
    let actual = mortality::calculate_scorch_height(1e-07, 5.0, 80.0);
    close(actual as f64, 0.0);
}

// calculate_scorch_height/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_50() {
    let actual = mortality::calculate_scorch_height(1.0000000000000001e-07, 5.0, 80.0);
    close(actual as f64, 6.39831021272567e-10);
}

// calculate_scorch_height/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_51() {
    let actual = mortality::calculate_scorch_height(300.0, 5.0, 80.0);
    close(actual as f64, 39.53397259927919);
}

// calculate_scorch_height/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_52() {
    let actual = mortality::calculate_scorch_height(300.0, 5.0, 80.0);
    close(actual as f64, 39.53397259927919);
}

// calculate_scorch_height/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_53() {
    let actual = mortality::calculate_scorch_height(0.0, 5.0, 80.0);
    close(actual as f64, 0.0);
}

// calculate_scorch_height/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_54() {
    let actual = mortality::calculate_scorch_height(0.0, 5.0, 80.0);
    close(actual as f64, 0.0);
}

// calculate_scorch_height/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_55() {
    let actual = mortality::calculate_scorch_height(1.0, 5.0, 80.0);
    close(actual as f64, 0.09354143466934854);
    let actual = mortality::calculate_scorch_height(2.0, 5.0, 80.0);
    close(actual as f64, 0.20916508940676373);
}

// calculate_scorch_height/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_56() {
    let actual = mortality::calculate_scorch_height(0.0, 5.0, 80.0);
    close(actual as f64, 0.0);
    let actual = mortality::calculate_scorch_height(2.0, 5.0, 80.0);
    close(actual as f64, 0.20916508940676373);
}

// calculate_scorch_height/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_57() {
    let actual = mortality::calculate_scorch_height(1.0, 5.0, 80.0);
    close(actual as f64, 0.09354143466934854);
}

// calculate_scorch_height/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_58() {
    let actual = mortality::calculate_scorch_height(1.0, 5.0, 80.0);
    close(actual as f64, 0.09354143466934854);
    let actual = mortality::calculate_scorch_height(1.0, 5.0, 80.0);
    close(actual as f64, 0.09354143466934854);
}

// mortality/all-equations
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_59() {
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        -1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.0);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        0,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.0);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        2,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.09559990665250463);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        3,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.0399675197298558);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        4,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08845072779027041);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        5,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08845072779027041);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        6,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.1874447159696992);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        7,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        8,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.03324309489031025);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        9,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.03324309489031025);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        10,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.0474891605899967);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        11,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.0474891605899967);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        12,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        13,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.045342840012864656);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        14,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08845072779027041);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        15,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.058773098326393526);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        16,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.03324309489031025);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        17,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.09559990665250463);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        18,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.08296703957465054);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        19,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.09559990665250463);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        20,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.03324309489031025);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        100,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.16534282858797358);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        101,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.045239064819306375);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        102,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.021388109343767128);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        103,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.02715667959942745);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        104,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.07328432937962163);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        105,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.11578143958467713);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        106,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.3026733038097942);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        107,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.03900047834131898);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        108,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.19275613017678195);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        109,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 2.2979156790173467e-05);
    let actual = mortality::calculate_crown_scorch_mortality(
        20.0,
        60.0,
        0.5,
        12.0,
        999,
        &mortality::build_mortality_lookup(None),
    );
    close(actual.crown_length_scorch as f64, 0.0);
    close(actual.crown_volume_scorch as f64, 0.0);
    close(actual.probability_mortality as f64, 0.0);
}

// calculate_spotting_from_burning_pile/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_60() {
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
}

// calculate_spotting_from_burning_pile/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_61() {
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
}

// calculate_spotting_from_burning_pile/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_62() {
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
}

// calculate_spotting_from_burning_pile/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_63() {
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
}

// calculate_spotting_from_burning_pile/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_64() {
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
}

// calculate_spotting_from_burning_pile/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_65() {
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
}

// calculate_spotting_from_burning_pile/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_66() {
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
}

// calculate_spotting_from_burning_pile/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_67() {
    let actual = spot::calculate_spotting_from_burning_pile(1.0, 1.0, 1.0);
    close(actual as f64, 2.995850444732822);
}

// calculate_spotting_from_burning_pile/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_68() {
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
}

// calculate_spotting_from_burning_pile/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_69() {
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
}

// calculate_spotting_from_burning_pile/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_70() {
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
}

// calculate_spotting_from_burning_pile/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_71() {
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
    let actual = spot::calculate_spotting_from_burning_pile(5.0, 5.0, 10.0);
    close(actual as f64, 29.157674975477576);
}

// calculate_spotting_from_burning_pile/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_72() {}

// calculate_spotting_from_burning_pile/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_73() {
    let actual = spot::calculate_spotting_from_burning_pile(0.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_burning_pile/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_74() {
    let actual = spot::calculate_spotting_from_burning_pile(-1.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_burning_pile/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_75() {
    let actual = spot::calculate_spotting_from_burning_pile(f64::NAN, 5.0, 10.0);
    close(actual as f64, f64::NAN); // Unknown required flame condition stays unavailable.
}

// calculate_spotting_from_burning_pile/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_76() {
    let actual = spot::calculate_spotting_from_burning_pile(f64::INFINITY, 5.0, 10.0);
    close(actual as f64, f64::NAN);
}

// calculate_spotting_from_burning_pile/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_77() {
    let actual = spot::calculate_spotting_from_burning_pile(f64::NEG_INFINITY, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_burning_pile/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_78() {
    let actual = spot::calculate_spotting_from_burning_pile(9.999999999999998e-08, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_burning_pile/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_79() {
    let actual = spot::calculate_spotting_from_burning_pile(1e-07, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_burning_pile/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_80() {
    let actual = spot::calculate_spotting_from_burning_pile(1.0000000000000001e-07, 5.0, 10.0);
    close(actual as f64, 19.310390260585514);
}

// calculate_spotting_from_burning_pile/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_81() {
    let actual = spot::calculate_spotting_from_burning_pile(300.0, 5.0, 10.0);
    close(actual as f64, 186.71958901561635);
}

// calculate_spotting_from_burning_pile/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_82() {
    let actual = spot::calculate_spotting_from_burning_pile(300.0, 5.0, 10.0);
    close(actual as f64, 186.71958901561635);
}

// calculate_spotting_from_burning_pile/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_83() {
    let actual = spot::calculate_spotting_from_burning_pile(0.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_burning_pile/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_84() {
    let actual = spot::calculate_spotting_from_burning_pile(0.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_burning_pile/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_85() {
    let actual = spot::calculate_spotting_from_burning_pile(1.0, 5.0, 10.0);
    close(actual as f64, 13.4184338017691);
    let actual = spot::calculate_spotting_from_burning_pile(2.0, 5.0, 10.0);
    close(actual as f64, 18.621328862448703);
}

// calculate_spotting_from_burning_pile/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_86() {
    let actual = spot::calculate_spotting_from_burning_pile(0.0, 5.0, 10.0);
    close(actual as f64, 0.0);
    let actual = spot::calculate_spotting_from_burning_pile(2.0, 5.0, 10.0);
    close(actual as f64, 18.621328862448703);
}

// calculate_spotting_from_burning_pile/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_87() {
    let actual = spot::calculate_spotting_from_burning_pile(1.0, 5.0, 10.0);
    close(actual as f64, 13.4184338017691);
}

// calculate_spotting_from_burning_pile/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_88() {
    let actual = spot::calculate_spotting_from_burning_pile(1.0, 5.0, 10.0);
    close(actual as f64, 13.4184338017691);
    let actual = spot::calculate_spotting_from_burning_pile(1.0, 5.0, 10.0);
    close(actual as f64, 13.4184338017691);
}

// calculate_spotting_from_surface_fire/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_89() {
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
}

// calculate_spotting_from_surface_fire/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_90() {
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
}

// calculate_spotting_from_surface_fire/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_91() {
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
}

// calculate_spotting_from_surface_fire/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_92() {
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
}

// calculate_spotting_from_surface_fire/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_93() {
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
}

// calculate_spotting_from_surface_fire/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_94() {
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
}

// calculate_spotting_from_surface_fire/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_95() {
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
}

// calculate_spotting_from_surface_fire/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_96() {
    let actual = spot::calculate_spotting_from_surface_fire(1.0, 1.0, 1.0);
    close(actual as f64, 64.27460887690135);
}

// calculate_spotting_from_surface_fire/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_97() {
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
}

// calculate_spotting_from_surface_fire/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_98() {
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
}

// calculate_spotting_from_surface_fire/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_99() {
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
}

// calculate_spotting_from_surface_fire/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_100() {
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
    let actual = spot::calculate_spotting_from_surface_fire(5.0, 5.0, 10.0);
    close(actual as f64, 565.5562062193633);
}

// calculate_spotting_from_surface_fire/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_101() {}

// calculate_spotting_from_surface_fire/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_102() {
    let actual = spot::calculate_spotting_from_surface_fire(0.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_surface_fire/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_103() {
    let actual = spot::calculate_spotting_from_surface_fire(-1.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_surface_fire/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_104() {
    let actual = spot::calculate_spotting_from_surface_fire(f64::NAN, 5.0, 10.0);
    close(actual as f64, f64::NAN); // Unknown required flame condition stays unavailable.
}

// calculate_spotting_from_surface_fire/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_105() {
    let actual = spot::calculate_spotting_from_surface_fire(f64::INFINITY, 5.0, 10.0);
    close(actual as f64, f64::NAN);
}

// calculate_spotting_from_surface_fire/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_106() {
    let actual = spot::calculate_spotting_from_surface_fire(f64::NEG_INFINITY, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_surface_fire/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_107() {
    let actual = spot::calculate_spotting_from_surface_fire(9.999999999999998e-08, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_surface_fire/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_108() {
    let actual = spot::calculate_spotting_from_surface_fire(1e-07, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_surface_fire/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_109() {
    let actual = spot::calculate_spotting_from_surface_fire(1.0000000000000001e-07, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_surface_fire/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_110() {
    let actual = spot::calculate_spotting_from_surface_fire(300.0, 5.0, 10.0);
    close(actual as f64, 9907.077182483255);
}

// calculate_spotting_from_surface_fire/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_111() {
    let actual = spot::calculate_spotting_from_surface_fire(300.0, 5.0, 10.0);
    close(actual as f64, 9907.077182483255);
}

// calculate_spotting_from_surface_fire/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_112() {
    let actual = spot::calculate_spotting_from_surface_fire(0.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_surface_fire/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_113() {
    let actual = spot::calculate_spotting_from_surface_fire(0.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_surface_fire/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_114() {
    let actual = spot::calculate_spotting_from_surface_fire(1.0, 5.0, 10.0);
    close(actual as f64, 140.69144948679883);
    let actual = spot::calculate_spotting_from_surface_fire(2.0, 5.0, 10.0);
    close(actual as f64, 261.9787050643918);
}

// calculate_spotting_from_surface_fire/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_115() {
    let actual = spot::calculate_spotting_from_surface_fire(0.0, 5.0, 10.0);
    close(actual as f64, 0.0);
    let actual = spot::calculate_spotting_from_surface_fire(2.0, 5.0, 10.0);
    close(actual as f64, 261.9787050643918);
}

// calculate_spotting_from_surface_fire/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_116() {
    let actual = spot::calculate_spotting_from_surface_fire(1.0, 5.0, 10.0);
    close(actual as f64, 140.69144948679883);
}

// calculate_spotting_from_surface_fire/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_117() {
    let actual = spot::calculate_spotting_from_surface_fire(1.0, 5.0, 10.0);
    close(actual as f64, 140.69144948679883);
    let actual = spot::calculate_spotting_from_surface_fire(1.0, 5.0, 10.0);
    close(actual as f64, 140.69144948679883);
}

// calculate_spotting_from_torching_trees/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_118() {
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
}

// calculate_spotting_from_torching_trees/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_119() {
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
}

// calculate_spotting_from_torching_trees/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_120() {
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
}

// calculate_spotting_from_torching_trees/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_121() {
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
}

// calculate_spotting_from_torching_trees/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_122() {
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
}

// calculate_spotting_from_torching_trees/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_123() {
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
}

// calculate_spotting_from_torching_trees/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_124() {
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
}

// calculate_spotting_from_torching_trees/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_125() {
    let actual = spot::calculate_spotting_from_torching_trees(1.0, 1.0, 1.0, 1.0, 1.0);
    close(actual as f64, 4.664752330796295);
}

// calculate_spotting_from_torching_trees/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_126() {
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
}

// calculate_spotting_from_torching_trees/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_127() {
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
}

// calculate_spotting_from_torching_trees/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_128() {
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
}

// calculate_spotting_from_torching_trees/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_129() {
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
    let actual = spot::calculate_spotting_from_torching_trees(12.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 101.26121957494824);
}

// calculate_spotting_from_torching_trees/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_130() {}

// calculate_spotting_from_torching_trees/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_131() {
    let actual = spot::calculate_spotting_from_torching_trees(0.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_torching_trees/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_132() {
    let actual = spot::calculate_spotting_from_torching_trees(-1.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_torching_trees/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_133() {
    let actual = spot::calculate_spotting_from_torching_trees(f64::NAN, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, f64::NAN); // Unknown required diameter condition stays unavailable.
}

// calculate_spotting_from_torching_trees/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_134() {
    let actual = spot::calculate_spotting_from_torching_trees(f64::INFINITY, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, f64::NAN);
}

// calculate_spotting_from_torching_trees/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_135() {
    let actual =
        spot::calculate_spotting_from_torching_trees(f64::NEG_INFINITY, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_torching_trees/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_136() {
    let actual =
        spot::calculate_spotting_from_torching_trees(9.999999999999998e-08, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_torching_trees/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_137() {
    let actual = spot::calculate_spotting_from_torching_trees(1e-07, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_torching_trees/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_138() {
    let actual =
        spot::calculate_spotting_from_torching_trees(1.0000000000000001e-07, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 68.46722190146808);
}

// calculate_spotting_from_torching_trees/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_139() {
    let actual = spot::calculate_spotting_from_torching_trees(300.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 311.7567315170764);
}

// calculate_spotting_from_torching_trees/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_140() {
    let actual = spot::calculate_spotting_from_torching_trees(300.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 311.7567315170764);
}

// calculate_spotting_from_torching_trees/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_141() {
    let actual = spot::calculate_spotting_from_torching_trees(0.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_torching_trees/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_142() {
    let actual = spot::calculate_spotting_from_torching_trees(0.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 0.0);
}

// calculate_spotting_from_torching_trees/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_143() {
    let actual = spot::calculate_spotting_from_torching_trees(1.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 71.98208280849632);
    let actual = spot::calculate_spotting_from_torching_trees(2.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 75.2884567519687);
}

// calculate_spotting_from_torching_trees/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_144() {
    let actual = spot::calculate_spotting_from_torching_trees(0.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 0.0);
    let actual = spot::calculate_spotting_from_torching_trees(2.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 75.2884567519687);
}

// calculate_spotting_from_torching_trees/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_145() {
    let actual = spot::calculate_spotting_from_torching_trees(1.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 71.98208280849632);
}

// calculate_spotting_from_torching_trees/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_146() {
    let actual = spot::calculate_spotting_from_torching_trees(1.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 71.98208280849632);
    let actual = spot::calculate_spotting_from_torching_trees(1.0, 60.0, 3.0, 5.0, 10.0);
    close(actual as f64, 71.98208280849632);
}

// calculate_firebrand_ignition_probability/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_147() {
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
}

// calculate_firebrand_ignition_probability/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_148() {
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.05999755859375);
    close(actual as f64, 0.5483359961524821);
}

// calculate_firebrand_ignition_probability/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_149() {
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.05999755859375);
    close(actual as f64, 0.5483359961524821);
}

// calculate_firebrand_ignition_probability/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_150() {
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.05999999865889549);
    close(actual as f64, 0.5483170580612344);
}

// calculate_firebrand_ignition_probability/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_151() {
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
}

// calculate_firebrand_ignition_probability/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_152() {
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.0, 0.0);
    close(actual as f64, 1.0);
}

// calculate_firebrand_ignition_probability/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_153() {
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.0, 0.0);
    close(actual as f64, 1.0);
}

// calculate_firebrand_ignition_probability/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_154() {
    let actual = ignite::calculate_firebrand_ignition_probability(1.0, 1.0, 1.0);
    close(actual as f64, 0.0);
}

// calculate_firebrand_ignition_probability/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_155() {
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
}

// calculate_firebrand_ignition_probability/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_156() {
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
}

// calculate_firebrand_ignition_probability/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_157() {
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
}

// calculate_firebrand_ignition_probability/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_158() {
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
    let actual = ignite::calculate_firebrand_ignition_probability(80.0, 0.5, 0.06);
    close(actual as f64, 0.5483170476526804);
}

// calculate_firebrand_ignition_probability/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_159() {}

// calculate_firebrand_ignition_probability/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_160() {
    let actual = ignite::calculate_firebrand_ignition_probability(0.0, 0.5, 0.06);
    close(actual as f64, 0.4023901057394422);
}

// calculate_firebrand_ignition_probability/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_161() {
    let actual = ignite::calculate_firebrand_ignition_probability(-1.0, 0.5, 0.06);
    close(actual as f64, 0.4009020330325474);
}

// calculate_firebrand_ignition_probability/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_162() {
    let actual = ignite::calculate_firebrand_ignition_probability(f64::NAN, 0.5, 0.06);
    close(actual as f64, f64::NAN);
}

// calculate_firebrand_ignition_probability/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_163() {
    let actual = ignite::calculate_firebrand_ignition_probability(f64::INFINITY, 0.5, 0.06);
    close(actual as f64, 1.0);
}

// calculate_firebrand_ignition_probability/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_164() {
    let actual = ignite::calculate_firebrand_ignition_probability(f64::NEG_INFINITY, 0.5, 0.06);
    close(actual as f64, f64::NAN);
}

// calculate_firebrand_ignition_probability/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_165() {
    let actual = ignite::calculate_firebrand_ignition_probability(9.999999999999998e-08, 0.5, 0.06);
    close(actual as f64, 0.40239010588861346);
}

// calculate_firebrand_ignition_probability/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_166() {
    let actual = ignite::calculate_firebrand_ignition_probability(1e-07, 0.5, 0.06);
    close(actual as f64, 0.40239010588861346);
}

// calculate_firebrand_ignition_probability/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_167() {
    let actual =
        ignite::calculate_firebrand_ignition_probability(1.0000000000000001e-07, 0.5, 0.06);
    close(actual as f64, 0.40239010588861346);
}

// calculate_firebrand_ignition_probability/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_168() {
    let actual = ignite::calculate_firebrand_ignition_probability(300.0, 0.5, 0.05999755859375);
    close(actual as f64, 1.0);
}

// calculate_firebrand_ignition_probability/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_169() {
    let actual = ignite::calculate_firebrand_ignition_probability(300.0, 0.5, 0.05999755859375);
    close(actual as f64, 1.0);
}

// calculate_firebrand_ignition_probability/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_170() {
    let actual = ignite::calculate_firebrand_ignition_probability(0.0, 0.5, 0.05999755859375);
    close(actual as f64, 0.4024059737376151);
}

// calculate_firebrand_ignition_probability/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_171() {
    let actual = ignite::calculate_firebrand_ignition_probability(0.0, 0.5, 0.05999755859375);
    close(actual as f64, 0.4024059737376151);
}

// calculate_firebrand_ignition_probability/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_172() {
    let actual = ignite::calculate_firebrand_ignition_probability(1.0, 0.5, 0.06);
    close(actual as f64, 0.4038854738065774);
    let actual = ignite::calculate_firebrand_ignition_probability(2.0, 0.5, 0.06);
    close(actual as f64, 0.40538817177620523);
}

// calculate_firebrand_ignition_probability/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_173() {
    let actual = ignite::calculate_firebrand_ignition_probability(0.0, 0.5, 0.06);
    close(actual as f64, 0.4023901057394422);
    let actual = ignite::calculate_firebrand_ignition_probability(2.0, 0.5, 0.06);
    close(actual as f64, 0.40538817177620523);
}

// calculate_firebrand_ignition_probability/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_174() {
    let actual = ignite::calculate_firebrand_ignition_probability(1.0, 0.5, 0.06);
    close(actual as f64, 0.4038854738065774);
}

// calculate_firebrand_ignition_probability/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_175() {
    let actual = ignite::calculate_firebrand_ignition_probability(1.0, 0.5, 0.06);
    close(actual as f64, 0.4038854738065774);
    let actual = ignite::calculate_firebrand_ignition_probability(1.0, 0.5, 0.06);
    close(actual as f64, 0.4038854738065774);
}

// calculate_fuel_temperature/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_176() {
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
}

// calculate_fuel_temperature/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_177() {
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
}

// calculate_fuel_temperature/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_178() {
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
}

// calculate_fuel_temperature/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_179() {
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
}

// calculate_fuel_temperature/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_180() {
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
}

// calculate_fuel_temperature/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_181() {
    let actual = ignite::calculate_fuel_temperature(80.0, 0.0);
    close(actual as f64, 105.0);
}

// calculate_fuel_temperature/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_182() {
    let actual = ignite::calculate_fuel_temperature(80.0, 0.0);
    close(actual as f64, 105.0);
}

// calculate_fuel_temperature/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_183() {
    let actual = ignite::calculate_fuel_temperature(1.0, 1.0);
    close(actual as f64, 6.0);
}

// calculate_fuel_temperature/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_184() {
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
}

// calculate_fuel_temperature/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_185() {
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
}

// calculate_fuel_temperature/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_186() {
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
}

// calculate_fuel_temperature/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_187() {
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
    let actual = ignite::calculate_fuel_temperature(80.0, 0.5);
    close(actual as f64, 95.0);
}

// calculate_fuel_temperature/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_188() {}

// calculate_fuel_temperature/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_189() {
    let actual = ignite::calculate_fuel_temperature(0.0, 0.5);
    close(actual as f64, 15.0);
}

// calculate_fuel_temperature/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_190() {
    let actual = ignite::calculate_fuel_temperature(-1.0, 0.5);
    close(actual as f64, 14.0);
}

// calculate_fuel_temperature/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_191() {
    let actual = ignite::calculate_fuel_temperature(f64::NAN, 0.5);
    close(actual as f64, f64::NAN);
}

// calculate_fuel_temperature/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_192() {
    let actual = ignite::calculate_fuel_temperature(f64::INFINITY, 0.5);
    close(actual as f64, f64::INFINITY);
}

// calculate_fuel_temperature/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_193() {
    let actual = ignite::calculate_fuel_temperature(f64::NEG_INFINITY, 0.5);
    close(actual as f64, f64::NEG_INFINITY);
}

// calculate_fuel_temperature/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_194() {
    let actual = ignite::calculate_fuel_temperature(9.999999999999998e-08, 0.5);
    close(actual as f64, 15.0000001);
}

// calculate_fuel_temperature/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_195() {
    let actual = ignite::calculate_fuel_temperature(1e-07, 0.5);
    close(actual as f64, 15.0000001);
}

// calculate_fuel_temperature/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_196() {
    let actual = ignite::calculate_fuel_temperature(1.0000000000000001e-07, 0.5);
    close(actual as f64, 15.0000001);
}

// calculate_fuel_temperature/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_197() {
    let actual = ignite::calculate_fuel_temperature(300.0, 0.5);
    close(actual as f64, 315.0);
}

// calculate_fuel_temperature/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_198() {
    let actual = ignite::calculate_fuel_temperature(300.0, 0.5);
    close(actual as f64, 315.0);
}

// calculate_fuel_temperature/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_199() {
    let actual = ignite::calculate_fuel_temperature(0.0, 0.5);
    close(actual as f64, 15.0);
}

// calculate_fuel_temperature/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_200() {
    let actual = ignite::calculate_fuel_temperature(0.0, 0.5);
    close(actual as f64, 15.0);
}

// calculate_fuel_temperature/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_201() {
    let actual = ignite::calculate_fuel_temperature(1.0, 0.5);
    close(actual as f64, 16.0);
    let actual = ignite::calculate_fuel_temperature(2.0, 0.5);
    close(actual as f64, 17.0);
}

// calculate_fuel_temperature/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_202() {
    let actual = ignite::calculate_fuel_temperature(0.0, 0.5);
    close(actual as f64, 15.0);
    let actual = ignite::calculate_fuel_temperature(2.0, 0.5);
    close(actual as f64, 17.0);
}

// calculate_fuel_temperature/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_203() {
    let actual = ignite::calculate_fuel_temperature(1.0, 0.5);
    close(actual as f64, 16.0);
}

// calculate_fuel_temperature/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_204() {
    let actual = ignite::calculate_fuel_temperature(1.0, 0.5);
    close(actual as f64, 16.0);
    let actual = ignite::calculate_fuel_temperature(1.0, 0.5);
    close(actual as f64, 16.0);
}

// calculate_lightning_ignition_probability/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_205() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_206() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08001708984375, 2.0, 0);
    close(actual as f64, 0.1350230891486481);
}

// calculate_lightning_ignition_probability/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_207() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08001708984375, 2.0, 0);
    close(actual as f64, 0.1350230891486481);
}

// calculate_lightning_ignition_probability/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_208() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.07999999821186066, 2.0, 0);
    close(actual as f64, 0.13503555165440753);
}

// calculate_lightning_ignition_probability/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_209() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_210() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.0, 2.0, 0);
    close(actual as f64, 0.20800000000000002);
}

// calculate_lightning_ignition_probability/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_211() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.0, 2.0, 0);
    close(actual as f64, 0.20800000000000002);
}

// calculate_lightning_ignition_probability/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_212() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 1.0, 1.0, 0);
    close(actual as f64, 0.023987625175917005);
}

// calculate_lightning_ignition_probability/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_213() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_214() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_215() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_216() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_217() {}

// calculate_lightning_ignition_probability/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_218() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_219() {
    let actual = ignite::calculate_lightning_ignition_probability(-1, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_220() {
    let actual = ignite::calculate_lightning_ignition_probability(-2147483648, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_221() {
    let actual = ignite::calculate_lightning_ignition_probability(-2147483648, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_222() {
    let actual = ignite::calculate_lightning_ignition_probability(-2147483648, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_223() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_224() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_225() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
}

// calculate_lightning_ignition_probability/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_226() {
    let actual = ignite::calculate_lightning_ignition_probability(300, 0.08001708984375, 2.0, 0);
    close(actual as f64, 0.10394495085914819);
}

// calculate_lightning_ignition_probability/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_227() {
    let actual = ignite::calculate_lightning_ignition_probability(300, 0.08001708984375, 2.0, 0);
    close(actual as f64, 0.10394495085914819);
}

// calculate_lightning_ignition_probability/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_228() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08001708984375, 2.0, 0);
    close(actual as f64, 0.1350230891486481);
}

// calculate_lightning_ignition_probability/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_229() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08001708984375, 2.0, 0);
    close(actual as f64, 0.1350230891486481);
}

// calculate_lightning_ignition_probability/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_230() {
    let actual = ignite::calculate_lightning_ignition_probability(1, 0.08, 2.0, 0);
    close(actual as f64, 0.05562788612121744);
    let actual = ignite::calculate_lightning_ignition_probability(2, 0.08, 2.0, 0);
    close(actual as f64, 0.11500284312574494);
}

// calculate_lightning_ignition_probability/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_231() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
    let actual = ignite::calculate_lightning_ignition_probability(2, 0.08, 2.0, 0);
    close(actual as f64, 0.11500284312574494);
}

// calculate_lightning_ignition_probability/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_232() {
    let actual = ignite::calculate_lightning_ignition_probability(1, 0.08, 2.0, 0);
    close(actual as f64, 0.05562788612121744);
}

// calculate_lightning_ignition_probability/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_233() {
    let actual = ignite::calculate_lightning_ignition_probability(1, 0.08, 2.0, 0);
    close(actual as f64, 0.05562788612121744);
    let actual = ignite::calculate_lightning_ignition_probability(1, 0.08, 2.0, 0);
    close(actual as f64, 0.05562788612121744);
}

// lightning/charge-0
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_234() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 0);
    close(actual as f64, 0.13503555035051068);
    let actual = ignite::calculate_lightning_ignition_probability(1, 0.08, 2.0, 0);
    close(actual as f64, 0.05562788612121744);
    let actual = ignite::calculate_lightning_ignition_probability(2, 0.08, 2.0, 0);
    close(actual as f64, 0.11500284312574494);
    let actual = ignite::calculate_lightning_ignition_probability(3, 0.08, 2.0, 0);
    close(actual as f64, 0.12840000000000001);
    let actual = ignite::calculate_lightning_ignition_probability(4, 0.08, 2.0, 0);
    close(actual as f64, 0.06234787840824612);
    let actual = ignite::calculate_lightning_ignition_probability(5, 0.08, 2.0, 0);
    close(actual as f64, 0.1470879653325673);
    let actual = ignite::calculate_lightning_ignition_probability(6, 0.08, 2.0, 0);
    close(actual as f64, 0.13760000000000003);
    let actual = ignite::calculate_lightning_ignition_probability(7, 0.08, 2.0, 0);
    close(actual as f64, 0.10395560982343166);
}

// lightning/charge-1
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_235() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 1);
    close(actual as f64, 0.41282061583672736);
    let actual = ignite::calculate_lightning_ignition_probability(1, 0.08, 2.0, 1);
    close(actual as f64, 0.16425403302590624);
    let actual = ignite::calculate_lightning_ignition_probability(2, 0.08, 2.0, 1);
    close(actual as f64, 0.478938345257953);
    let actual = ignite::calculate_lightning_ignition_probability(3, 0.08, 2.0, 1);
    close(actual as f64, 0.4945108734305958);
    let actual = ignite::calculate_lightning_ignition_probability(4, 0.08, 2.0, 1);
    close(actual as f64, 0.14191108277255313);
    let actual = ignite::calculate_lightning_ignition_probability(5, 0.08, 2.0, 1);
    close(actual as f64, 0.5325740179957923);
    let actual = ignite::calculate_lightning_ignition_probability(6, 0.08, 2.0, 1);
    close(actual as f64, 0.3740385856878868);
    let actual = ignite::calculate_lightning_ignition_probability(7, 0.08, 2.0, 1);
    close(actual as f64, 0.3650025917993927);
}

// lightning/charge-2
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_236() {
    let actual = ignite::calculate_lightning_ignition_probability(0, 0.08, 2.0, 2);
    close(actual as f64, 0.2119820134901927);
    let actual = ignite::calculate_lightning_ignition_probability(1, 0.08, 2.0, 2);
    close(actual as f64, 0.08571732881381625);
    let actual = ignite::calculate_lightning_ignition_probability(2, 0.08, 2.0, 2);
    close(actual as f64, 0.21581297721636658);
    let actual = ignite::calculate_lightning_ignition_probability(3, 0.08, 2.0, 2);
    close(actual as f64, 0.22981271194027506);
    let actual = ignite::calculate_lightning_ignition_probability(4, 0.08, 2.0, 2);
    close(actual as f64, 0.08438688601715916);
    let actual = ignite::calculate_lightning_ignition_probability(5, 0.08, 2.0, 2);
    close(actual as f64, 0.25386760192028063);
    let actual = ignite::calculate_lightning_ignition_probability(6, 0.08, 2.0, 2);
    close(actual as f64, 0.20309348823554466);
    let actual = ignite::calculate_lightning_ignition_probability(7, 0.08, 2.0, 2);
    close(actual as f64, 0.17626562383077288);
}

// calculate_safety_zone/baseline
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_237() {
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
}

// calculate_safety_zone/dtype-float16
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_238() {
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
}

// calculate_safety_zone/dtype-float16/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_239() {
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
}

// calculate_safety_zone/dtype-float32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_240() {
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
}

// calculate_safety_zone/dtype-float64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_241() {
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
}

// calculate_safety_zone/dtype-int64
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_242() {
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
}

// calculate_safety_zone/dtype-uint32
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_243() {
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
}

// calculate_safety_zone/dtype-bool
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_244() {
    let actual = safety::calculate_safety_zone(1.0, 1.0, 1.0, 1.0, 1.0);
    close(actual.area as f64, 72.31850865448469);
    close(actual.radius as f64, 4.797884560802865);
    close(actual.separation_distance as f64, 4.0);
}

// calculate_safety_zone/dtype-object
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_245() {
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
}

// calculate_safety_zone/scalar
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_246() {
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
}

// calculate_safety_zone/2d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_247() {
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
}

// calculate_safety_zone/3d
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_248() {
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
    let actual = safety::calculate_safety_zone(5.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 3341.9679804783214);
    close(actual.radius as f64, 32.6156626101008);
    close(actual.separation_distance as f64, 20.0);
}

// calculate_safety_zone/empty
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_249() {}

// calculate_safety_zone/zero
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_250() {
    let actual = safety::calculate_safety_zone(0.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 500.0);
    close(actual.radius as f64, 12.6156626101008);
    close(actual.separation_distance as f64, 0.0);
}

// calculate_safety_zone/negative
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_251() {
    let actual = safety::calculate_safety_zone(-1.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 233.19929864895585);
    close(actual.radius as f64, 8.6156626101008);
    close(actual.separation_distance as f64, -4.0);
}

// calculate_safety_zone/nan
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_252() {
    let actual = safety::calculate_safety_zone(f64::NAN, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, f64::NAN);
    close(actual.radius as f64, f64::NAN);
    close(actual.separation_distance as f64, f64::NAN);
}

// calculate_safety_zone/positive-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_253() {
    let actual = safety::calculate_safety_zone(f64::INFINITY, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, f64::INFINITY);
    close(actual.radius as f64, f64::INFINITY);
    close(actual.separation_distance as f64, f64::INFINITY);
}

// calculate_safety_zone/negative-infinity
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_254() {
    let actual = safety::calculate_safety_zone(f64::NEG_INFINITY, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, f64::INFINITY);
    close(actual.radius as f64, f64::NEG_INFINITY);
    close(actual.separation_distance as f64, f64::NEG_INFINITY);
}

// calculate_safety_zone/below-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_255() {
    let actual = safety::calculate_safety_zone(9.999999999999998e-08, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 500.00003170661887);
    close(actual.radius as f64, 12.6156630101008);
    close(actual.separation_distance as f64, 3.9999999999999993e-07);
}

// calculate_safety_zone/at-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_256() {
    let actual = safety::calculate_safety_zone(1e-07, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 500.00003170661887);
    close(actual.radius as f64, 12.6156630101008);
    close(actual.separation_distance as f64, 4e-07);
}

// calculate_safety_zone/above-threshold
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_257() {
    let actual = safety::calculate_safety_zone(1.0000000000000001e-07, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 500.00003170661887);
    close(actual.radius as f64, 12.6156630101008);
    close(actual.separation_distance as f64, 4.0000000000000003e-07);
}

// calculate_safety_zone/float16-overflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_258() {
    let actual = safety::calculate_safety_zone(300.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 4619513.276311846);
    close(actual.radius as f64, 1212.6156626101008);
    close(actual.separation_distance as f64, 1200.0);
}

// calculate_safety_zone/float16-overflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_259() {
    let actual = safety::calculate_safety_zone(300.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 4619513.276311846);
    close(actual.radius as f64, 1212.6156626101008);
    close(actual.separation_distance as f64, 1200.0);
}

// calculate_safety_zone/float16-underflow
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_260() {
    let actual = safety::calculate_safety_zone(0.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 500.0);
    close(actual.radius as f64, 12.6156626101008);
    close(actual.separation_distance as f64, 0.0);
}

// calculate_safety_zone/float16-underflow/normalized
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_261() {
    let actual = safety::calculate_safety_zone(0.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 500.0);
    close(actual.radius as f64, 12.6156626101008);
    close(actual.separation_distance as f64, 0.0);
}

// calculate_safety_zone/masked
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_262() {
    let actual = safety::calculate_safety_zone(1.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 867.3316662659173);
    close(actual.radius as f64, 16.6156626101008);
    close(actual.separation_distance as f64, 4.0);
    let actual = safety::calculate_safety_zone(2.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 1335.1942974467083);
    close(actual.radius as f64, 20.6156626101008);
    close(actual.separation_distance as f64, 8.0);
}

// calculate_safety_zone/strided
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_263() {
    let actual = safety::calculate_safety_zone(0.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 500.0);
    close(actual.radius as f64, 12.6156626101008);
    close(actual.separation_distance as f64, 0.0);
    let actual = safety::calculate_safety_zone(2.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 1335.1942974467083);
    close(actual.radius as f64, 20.6156626101008);
    close(actual.separation_distance as f64, 8.0);
}

// calculate_safety_zone/mixed-dtype
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_264() {
    let actual = safety::calculate_safety_zone(1.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 867.3316662659173);
    close(actual.radius as f64, 16.6156626101008);
    close(actual.separation_distance as f64, 4.0);
}

// calculate_safety_zone/read-only
#[test]
#[allow(deprecated, unused_variables, unused_mut)]
fn oracle_265() {
    let actual = safety::calculate_safety_zone(1.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 867.3316662659173);
    close(actual.radius as f64, 16.6156626101008);
    close(actual.separation_distance as f64, 4.0);
    let actual = safety::calculate_safety_zone(1.0, 4.0, 50.0, 1.0, 300.0);
    close(actual.area as f64, 867.3316662659173);
    close(actual.radius as f64, 16.6156626101008);
    close(actual.separation_distance as f64, 4.0);
}

#[test]
fn mortality_diagnostics_and_dictionary_names() {
    let mut table = behave_fire::species::SpeciesMasterTable::default();
    table.records.truncate(1);
    table.records[0].mortality_equation_number = 999;
    let diagnostics = mortality::missing_coefficient_equations(&table);
    assert_eq!(
        diagnostics,
        vec![(table.records[0].species_code.as_str(), 999)]
    );
    assert_eq!(
        mortality::build_mortality_lookup(Some(&table)),
        mortality::build_mortality_lookup(None)
    );
    let result = mortality::calculate_crown_scorch_mortality(
        30.0,
        50.0,
        0.5,
        10.0,
        1,
        &mortality::build_mortality_lookup(None),
    );
    assert_eq!(
        result.named_values().map(|entry| entry.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let zone = safety::calculate_safety_zone(5.0, 10.0, 30.0, 1.0, 200.0);
    assert_eq!(
        zone.named_values().map(|entry| entry.0),
        ["separation_distance", "radius", "area"]
    );
}

#[test]
#[should_panic]
fn comparator_rejects_near_zero_error_above_absolute_tolerance() {
    close(5e-10, 0.0);
}

#[test]
fn selected_coefficient_rows_match_python_custom_tables() {
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        30.0,
        50.0,
        0.5,
        10.0,
        [-5.0, 0.4, 3.0],
    );
    close(result.crown_length_scorch, 0.2);
    close(result.crown_volume_scorch, 0.10400000000000002);
    close(result.probability_mortality, 0.3344781307879406);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        100.0,
        50.0,
        0.5,
        10.0,
        [-5.0, 0.4, 3.0],
    );
    close(result.crown_length_scorch, 1.0);
    close(result.crown_volume_scorch, 1.0);
    close(result.probability_mortality, 0.8807970779778823);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        -5.0,
        0.0,
        0.5,
        0.0,
        [-5.0, 0.4, 3.0],
    );
    close(result.crown_length_scorch, 0.0);
    close(result.crown_volume_scorch, 0.0);
    close(result.probability_mortality, 0.0066928509242848554);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        1.0,
        1e-08,
        1.0,
        f64::INFINITY,
        [-5.0, 0.4, 3.0],
    );
    close(result.crown_length_scorch, 0.0);
    close(result.crown_volume_scorch, 0.0);
    close(result.probability_mortality, 1.0);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        30.0,
        50.0,
        0.5,
        10.0,
        [0.0, 0.0, 0.0],
    );
    close(result.crown_length_scorch, 0.2);
    close(result.crown_volume_scorch, 0.10400000000000002);
    close(result.probability_mortality, 0.0);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        100.0,
        50.0,
        0.5,
        10.0,
        [0.0, 0.0, 0.0],
    );
    close(result.crown_length_scorch, 1.0);
    close(result.crown_volume_scorch, 1.0);
    close(result.probability_mortality, 0.0);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        -5.0,
        0.0,
        0.5,
        0.0,
        [0.0, 0.0, 0.0],
    );
    close(result.crown_length_scorch, 0.0);
    close(result.crown_volume_scorch, 0.0);
    close(result.probability_mortality, 0.0);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        1.0,
        1e-08,
        1.0,
        f64::INFINITY,
        [0.0, 0.0, 0.0],
    );
    close(result.crown_length_scorch, 0.0);
    close(result.crown_volume_scorch, 0.0);
    close(result.probability_mortality, 0.0);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        30.0,
        50.0,
        0.5,
        10.0,
        [2.0, -0.1, 1.5],
    );
    close(result.crown_length_scorch, 0.2);
    close(result.crown_volume_scorch, 0.10400000000000002);
    close(result.probability_mortality, 0.7606051344021186);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        100.0,
        50.0,
        0.5,
        10.0,
        [2.0, -0.1, 1.5],
    );
    close(result.crown_length_scorch, 1.0);
    close(result.crown_volume_scorch, 1.0);
    close(result.probability_mortality, 0.9241418199787566);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        -5.0,
        0.0,
        0.5,
        0.0,
        [2.0, -0.1, 1.5],
    );
    close(result.crown_length_scorch, 0.0);
    close(result.crown_volume_scorch, 0.0);
    close(result.probability_mortality, 0.8807970779778823);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        1.0,
        1e-08,
        1.0,
        f64::INFINITY,
        [2.0, -0.1, 1.5],
    );
    close(result.crown_length_scorch, 0.0);
    close(result.crown_volume_scorch, 0.0);
    close(result.probability_mortality, 0.0);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        30.0,
        50.0,
        0.5,
        10.0,
        [f64::NAN, 0.0, 1.0],
    );
    close(result.crown_length_scorch, 0.2);
    close(result.crown_volume_scorch, 0.10400000000000002);
    close(result.probability_mortality, f64::NAN);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        100.0,
        50.0,
        0.5,
        10.0,
        [f64::NAN, 0.0, 1.0],
    );
    close(result.crown_length_scorch, 1.0);
    close(result.crown_volume_scorch, 1.0);
    close(result.probability_mortality, f64::NAN);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        -5.0,
        0.0,
        0.5,
        0.0,
        [f64::NAN, 0.0, 1.0],
    );
    close(result.crown_length_scorch, 0.0);
    close(result.crown_volume_scorch, 0.0);
    close(result.probability_mortality, f64::NAN);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
    let result = mortality::calculate_crown_scorch_mortality_with_coefficients(
        1.0,
        1e-08,
        1.0,
        f64::INFINITY,
        [f64::NAN, 0.0, 1.0],
    );
    close(result.crown_length_scorch, 0.0);
    close(result.crown_volume_scorch, 0.0);
    close(result.probability_mortality, f64::NAN);
    assert_eq!(
        result.named_values().map(|pair| pair.0),
        [
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality"
        ]
    );
}
#[test]
fn selected_rows_match_all_lookup_equations_and_clipping() {
    let coefficients = mortality::build_mortality_lookup(None);
    for equation in -2i32..=122 {
        let row = coefficients[equation.clamp(0, 120) as usize];
        for scorch in [-1.0, 30.0, 100.0] {
            let existing = mortality::calculate_crown_scorch_mortality(
                scorch,
                50.0,
                0.5,
                10.0,
                equation,
                &coefficients,
            );
            let selected = mortality::calculate_crown_scorch_mortality_with_coefficients(
                scorch, 50.0, 0.5, 10.0, row,
            );
            assert_eq!(selected.named_values(), existing.named_values());
        }
    }
}
