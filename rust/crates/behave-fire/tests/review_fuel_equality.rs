//! Public fuel equality describes physical values, independently of preparation history.
use behave_fire::engine::{PreparedFuel, PreparedFuels};
use behave_fire::FuelModels;

#[test]
fn all_public_physical_fields_participate_in_equality() {
    let original = PreparedFuel::default();
    for field in 0..14 {
        let mut changed = original;
        match field {
            0 => changed.fuel_bed_depth = 1.,
            1 => changed.moisture_of_extinction_dead = 1.,
            2 => changed.heat_of_combustion_dead = 1.,
            3 => changed.heat_of_combustion_live = 1.,
            4 => changed.dead_1h = 1.,
            5 => changed.dead_10h = 1.,
            6 => changed.dead_100h = 1.,
            7 => changed.live_herb = 1.,
            8 => changed.live_woody = 1.,
            9 => changed.savr_1h = 1.,
            10 => changed.savr_live_herb = 1.,
            11 => changed.savr_live_woody = 1.,
            12 => changed.is_dynamic = true,
            13 => changed.is_defined = true,
            _ => unreachable!(),
        }
        assert_ne!(original, changed, "public field {field}");
    }
}

#[test]
fn ordinary_float_equality_keeps_nan_and_signed_zero_conventions() {
    let original = PreparedFuel::default();
    let mut changed = original;
    changed.fuel_bed_depth = -0.;
    assert_eq!(original, changed);
    changed.fuel_bed_depth = f64::NAN;
    assert_ne!(changed, changed);
}

#[test]
fn preparation_history_does_not_change_public_equality() {
    let original = PreparedFuel::default();
    assert_eq!(original, original.prepared());
    assert_eq!(original.prepared(), original.prepared());
    let registry = FuelModels::default();
    assert_eq!(
        PreparedFuels::from_registry(&registry),
        PreparedFuels::from_registry(&registry)
    );
}
