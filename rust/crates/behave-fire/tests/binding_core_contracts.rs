//! Standalone operations needed by the Python adapter's snapshot and facade routes.
use behave_fire::{canopy_crown_ratio, FuelModel, FuelModels};
use std::collections::BTreeMap;

#[test]
fn canopy_ratio_preserves_clipping_and_nonfinite_propagation() {
    for (height, base, expected) in [
        (80.0, 20.0, 0.75),
        (80.0, 100.0, 0.0),
        (80.0, -20.0, 1.0),
        (0.0, -1.0, 1.0),
        (-2.0, -3.0, 1.0),
        (1.0, f64::INFINITY, 0.0),
        (1.0, f64::NEG_INFINITY, 1.0),
    ] {
        assert_eq!(canopy_crown_ratio(height, base), expected);
    }
    assert!(canopy_crown_ratio(f64::NAN, 20.0).is_nan());
    assert!(canopy_crown_ratio(80.0, f64::NAN).is_nan());
    assert!(canopy_crown_ratio(f64::INFINITY, 20.0).is_nan());
}

#[test]
// FuelModel is intentionally non-exhaustive; snapshots can mutate public fields.
#[allow(clippy::field_reassign_with_default)]
fn imported_records_are_restored_without_defaults_or_unit_conversion() {
    let mut record = FuelModel::default();
    record.code = "SNAP".into();
    record.fuel_bed_depth = 0.35;
    record.heat_of_combustion_dead = 19000.0;
    record.is_dynamic = true;
    record.is_reserved = true;
    record.is_defined = false;
    let records = BTreeMap::from([(0, record.clone()), (300, record.clone())]);
    let imported = FuelModels::from_records(records);
    assert_eq!(imported.records().len(), 2);
    assert_eq!(imported.get(300), Some(&record));
    assert_eq!(imported.get(0), Some(&record));
    assert!(imported.get(1).is_none());
    assert_eq!(imported.for_grid(0), &FuelModel::default());
    assert_eq!(imported.for_grid(300), &FuelModel::default());
}
