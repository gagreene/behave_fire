//! Undefined identifiers follow the existing zero-row calculation path.
use behave_fire::{
    engine::{PreparedFuel, PreparedFuels, SurfaceCell, SurfacePlan, SurfaceTweaks},
    fuel_models::{CustomFuelModel, FuelModels},
};

const INVALID_IDS: [i32; 7] = [-100, -1, 0, 257, 261, 999, i32::MAX];

#[test]
fn custom_registration_rejects_outside_range_and_preserves_upper_endpoint() {
    let mut fuels = FuelModels::default();
    for id in INVALID_IDS.into_iter().chain([258, 259, 260]) {
        let before = fuels.clone();
        assert!(!fuels.set_custom_fuel_model(id, CustomFuelModel::default()));
        assert_eq!(fuels, before);
        assert!(!fuels.is_fuel_model_defined(id));
    }
    let custom = CustomFuelModel {
        model: fuels.get(10).unwrap().clone(),
        ..CustomFuelModel::default()
    };
    assert!(fuels.set_custom_fuel_model(256, custom));
    assert!(fuels.is_fuel_model_defined(256));
    let prepared = PreparedFuels::from_registry(&fuels);
    assert_eq!(
        prepared.for_grid(256).fuel_bed_depth,
        fuels.for_grid(256).fuel_bed_depth
    );
    assert!(prepared.for_grid(256).is_defined);
    for id in INVALID_IDS {
        assert_eq!(fuels.for_grid(id), fuels.for_grid(0));
        assert_eq!(prepared.for_grid(id), prepared.for_grid(0));
    }
}

#[test]
fn invalid_surface_ids_match_undefined_zero_outputs() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let plan = SurfacePlan::default();
    let zero = plan.run_cell(
        &SurfaceCell::new(0),
        fuels.for_grid(0),
        SurfaceTweaks::default(),
    );
    for id in INVALID_IDS {
        let actual = plan.run_cell(
            &SurfaceCell::new(id),
            fuels.for_grid(id),
            SurfaceTweaks::default(),
        );
        assert_eq!(actual, zero, "fuel {id}");
        assert_eq!(actual.spread_rate, 0., "fuel {id}/spread_rate");
        assert_eq!(
            actual.fireline_intensity, 0.,
            "fuel {id}/fireline_intensity"
        );
    }
}

#[test]
fn missing_short_lookup_rows_use_zero_instead_of_the_last_real_fuel() {
    let registry = FuelModels::default();
    let row = PreparedFuel::from_model(registry.for_grid(10));
    let fuels = PreparedFuels::from_rows(vec![PreparedFuel::default(), row]).unwrap();
    assert_eq!(fuels.for_grid(1), &row);
    for id in INVALID_IDS.into_iter().chain([2, 10, 256]) {
        assert_eq!(fuels.for_grid(id), &PreparedFuel::default(), "fuel {id}");
    }
}
