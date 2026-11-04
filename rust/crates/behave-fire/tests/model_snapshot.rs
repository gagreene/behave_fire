use behave_fire::{BehaveRun, FuelModels};

#[test]
fn model_owns_its_registry_snapshot() {
    let mut original = FuelModels::default();
    let model = BehaveRun::new(original.clone());
    original.get_mut(1).unwrap().fuel_bed_depth = 91.0;
    assert_eq!(model.fuel_models().get(1).unwrap().fuel_bed_depth, 1.0);
    let cloned = model.clone();
    assert_eq!(cloned.fuel_models().get(1).unwrap().fuel_bed_depth, 1.0);
}
