//! Injection must happen before dependent outputs, without stale plan state.
#![allow(clippy::field_reassign_with_default)]

use behave_fire::crown::SurfaceFireInputs;
use behave_fire::engine::{
    CrownCell, CrownPlan, CrownTweaks, PreparedFuels, SurfaceCell, SurfaceOptions, SurfacePlan,
    SurfaceTweaks,
};
use behave_fire::FuelModels;

fn inputs() -> (SurfacePlan, SurfaceCell, PreparedFuels) {
    let mut cell = SurfaceCell::new(10);
    cell.wind_speed = 8.0;
    cell.wind_direction = 35.0;
    cell.slope = 12.0;
    (
        SurfacePlan::new(SurfaceOptions::default()),
        cell,
        PreparedFuels::from_registry(&FuelModels::default()),
    )
}

#[test]
fn crown_overrides_precede_classification_and_passive_outputs() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let plan = CrownPlan::new(SurfaceOptions::default());
    let mut cell = CrownCell::default();
    cell.wind_speed = 440.0;
    cell.canopy_base_height = 6.0;
    cell.canopy_height = 30.0;
    cell.canopy_bulk_density = 0.02;
    let mut supplied = SurfaceFireInputs::<f64>::new();
    supplied.surface_ros = 4.0;
    supplied.surface_hpua = 900.0;
    supplied.surface_fli = 60.0;
    let original = plan.run_cell(&cell, &supplied, fuels.for_grid(10), CrownTweaks::default());
    let mut tweaks = CrownTweaks::default();
    tweaks.fm10.spread_rate = Some(3.0);
    tweaks.critical_surface_intensity = Some(1.0);
    tweaks.critical_crown_rate = Some(1e6);
    tweaks.crown_fraction_burned = Some(0.5);
    let result = plan.run_cell(&cell, &supplied, fuels.for_grid(10), tweaks);
    assert_eq!(result.crown_fire_spread_rate, 3.34 * 3.0);
    assert_eq!(result.crown_critical_surface_fire_line_intensity, 1.0);
    assert_eq!(result.crown_critical_fire_spread_rate, 1e6);
    assert_eq!(result.crown_fraction_burned, 0.5);
    assert_eq!(result.fire_type, 1);
    assert_eq!(
        result.final_spread_rate,
        4.0 + 0.5 * (result.crown_fire_spread_rate - 4.0)
    );
    assert_eq!(
        result.final_heat_per_unit_area,
        900.0 + 0.5 * result.canopy_heat_per_unit_area
    );
    assert_eq!(
        original,
        plan.run_cell(&cell, &supplied, fuels.for_grid(10), CrownTweaks::default())
    );
}

#[test]
fn fuelbed_edit_is_used_without_rebuild() {
    let (plan, cell, fuels) = inputs();
    let mut fuelbed = plan.prepare_fuelbed(&cell, fuels.for_grid(10));
    let before = plan.run_from_fuelbed(&cell, &fuelbed, SurfaceTweaks::default());
    fuelbed.propagating_flux *= 2.0;
    let changed = plan.run_from_fuelbed(&cell, &fuelbed, SurfaceTweaks::default());
    assert!(changed.no_wind_no_slope_spread_rate > before.no_wind_no_slope_spread_rate);
    assert_eq!(changed.reaction_intensity, before.reaction_intensity);
}

#[test]
fn later_overrides_take_precedence() {
    let (plan, cell, fuels) = inputs();
    let mut tweaks = SurfaceTweaks::default();
    tweaks.reaction_intensity = Some(100.0);
    tweaks.midflame_wind_speed = Some(1000.0);
    tweaks.wind_phi = Some(0.0);
    tweaks.slope_phi = Some(0.0);
    tweaks.spread_rate = Some(12.0);
    let result = plan.run_cell(&cell, fuels.for_grid(10), tweaks);
    assert_eq!(result.spread_rate, 12.0);
    assert_eq!(result.midflame_wind_speed, 1000.0);
    assert_eq!(
        result.fireline_intensity,
        12.0 * result.heat_per_unit_area / 60.0
    );
}

#[test]
fn plan_reuse_has_no_stale_values() {
    let (plan, cell, fuels) = inputs();
    let original = plan.run_cell(&cell, fuels.for_grid(10), SurfaceTweaks::default());
    let mut tweaks = SurfaceTweaks::default();
    tweaks.spread_rate = Some(99.0);
    assert_eq!(
        plan.run_cell(&cell, fuels.for_grid(10), tweaks).spread_rate,
        99.0
    );
    assert_eq!(
        original,
        plan.run_cell(&cell, fuels.for_grid(10), SurfaceTweaks::default())
    );
}

#[test]
fn reaction_override_changes_heat_and_slope_cap() {
    let (plan, mut cell, fuels) = inputs();
    cell.wind_speed = 0.0;
    cell.slope = 75.0;
    let mut tweaks = SurfaceTweaks::default();
    tweaks.reaction_intensity = Some(1.0);
    let low = plan.run_cell(&cell, fuels.for_grid(10), tweaks);
    tweaks.reaction_intensity = Some(100.0);
    let high = plan.run_cell(&cell, fuels.for_grid(10), tweaks);
    assert!(high.heat_per_unit_area > low.heat_per_unit_area);
    assert!(high.spread_rate > low.spread_rate);
    assert_eq!(low.reaction_intensity, 1.0);
    assert_eq!(high.reaction_intensity, 100.0);
}

#[test]
fn spread_override_recomputes_shape_and_intensity() {
    let (plan, cell, fuels) = inputs();
    let before = plan.run_cell(&cell, fuels.for_grid(10), SurfaceTweaks::default());
    let mut tweaks = SurfaceTweaks::default();
    tweaks.spread_rate = Some(20.0);
    let result = plan.run_cell(&cell, fuels.for_grid(10), tweaks);
    assert_eq!(result.spread_rate, 20.0);
    assert_eq!(result.heat_per_unit_area, before.heat_per_unit_area);
    assert_eq!(
        result.fireline_intensity,
        20.0 * result.heat_per_unit_area / 60.0
    );
    assert_ne!(result.effective_wind_speed, before.effective_wind_speed);
    assert_ne!(result.backing_spread_rate, before.backing_spread_rate);
    assert_ne!(result.flame_length, before.flame_length);
}

#[test]
fn undefined_fuel_mask_is_final_and_special_values_are_not_sanitized() {
    let (plan, mut cell, fuels) = inputs();
    let mut tweaks = SurfaceTweaks::default();
    tweaks.reaction_intensity = Some(f64::NAN);
    assert!(plan
        .run_cell(&cell, fuels.for_grid(10), tweaks)
        .heat_per_unit_area
        .is_nan());
    tweaks.reaction_intensity = None;
    tweaks.spread_rate = Some(-0.0);
    assert_eq!(
        plan.run_cell(&cell, fuels.for_grid(10), tweaks)
            .spread_rate
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    cell.fuel_model_grid = 0;
    tweaks.spread_rate = Some(99.0);
    let masked = plan.run_cell(&cell, fuels.for_grid(0), tweaks);
    assert_eq!(masked.spread_rate, 0.0);
    assert_eq!(masked.fire_length_to_width_ratio, 1.0);
}
