use behave_fire::math::{np_clip, np_max, np_min, np_remainder};

#[test]
fn nan_propagates_from_either_minmax_operand() {
    assert!(np_min(f64::NAN, 2.0).is_nan());
    assert!(np_min(2.0, f64::NAN).is_nan());
    assert!(np_max(f64::NAN, 2.0).is_nan());
    assert!(np_max(2.0, f64::NAN).is_nan());
}

#[test]
fn equal_zero_selects_second_operand() {
    assert!(np_max(0.0, -0.0).is_sign_negative());
    assert!(!np_max(-0.0, 0.0).is_sign_negative());
    assert!(np_min(0.0, -0.0).is_sign_negative());
    assert!(!np_min(-0.0, 0.0).is_sign_negative());
}

#[test]
fn clipping_preserves_nan_and_reversed_bound_behavior() {
    assert_eq!(np_clip(1.0, 2.0, 3.0), 2.0);
    assert_eq!(np_clip(4.0, 2.0, 3.0), 3.0);
    assert_eq!(np_clip(2.0, 3.0, 1.0), 1.0);
    assert!(np_clip(f64::NAN, 0.0, 1.0).is_nan());
}

#[test]
fn remainder_has_divisor_sign_and_handles_infinities() {
    assert_eq!(np_remainder(-10.0, 360.0), 350.0);
    assert_eq!(np_remainder(10.0, -360.0), -350.0);
    assert!(np_remainder(-360.0, 360.0).is_sign_positive());
    assert!(np_remainder(360.0, -360.0).is_sign_negative());
    assert!(np_remainder(1.0, 0.0).is_nan());
    assert!(np_remainder(f64::INFINITY, 360.0).is_nan());
}
