//! Structured native crown inputs and nested outputs.
use crate::{arrays::Arguments, dispatch::Cell, structures};
use behave_fire::units::*;
use behave_fire::{
    crown::*,
    engine::{CrownCell, PreparedFuel},
};
pub fn input(a: &Arguments, i: usize) -> Result<CrownCell, String> {
    let mut r = CrownCell::default();
    r.m1h = a.optional("m1h", i).unwrap_or(r.m1h);
    r.m10h = a.optional("m10h", i).unwrap_or(r.m10h);
    r.m100h = a.optional("m100h", i).unwrap_or(r.m100h);
    r.mlh = a.optional("mlh", i).unwrap_or(r.mlh);
    r.mlw = a.optional("mlw", i).unwrap_or(r.mlw);
    let wind_units = SpeedUnit::try_from(a.optional("wind_speed_units", i).unwrap_or(5.0) as i32)
        .map_err(|e| e.to_string())?;
    r.wind_speed = wind_units.to_base(a.optional("wind_speed", i).unwrap_or(0.0));
    r.wind_direction = a.optional("wind_direction", i).unwrap_or(r.wind_direction);
    let slope_units = SlopeUnit::try_from(a.optional("slope_units", i).unwrap_or(0.0) as i32)
        .map_err(|e| e.to_string())?;
    r.slope = slope_units.to_base(a.optional("slope", i).unwrap_or(0.0));
    r.aspect = a.optional("aspect", i).unwrap_or(r.aspect);
    r.canopy_base_height = a
        .optional("canopy_base_height", i)
        .unwrap_or(r.canopy_base_height);
    r.canopy_height = a.optional("canopy_height", i).unwrap_or(r.canopy_height);
    r.canopy_bulk_density = a
        .optional("canopy_bulk_density", i)
        .unwrap_or(r.canopy_bulk_density);
    r.moisture_foliar = a
        .optional("moisture_foliar", i)
        .unwrap_or(r.moisture_foliar);
    Ok(r)
}
pub fn fuel(a: &Arguments, i: usize) -> Result<PreparedFuel, String> {
    structures::fuel(a, i)
}
pub fn surface_values(a: &Arguments, i: usize) -> SurfaceFireInputs<f64> {
    let mut values = SurfaceFireInputs::<f64>::new();
    values.surface_ros = a.optional("surface_results.spread_rate", i).unwrap_or(0.0);
    values.surface_fli = a
        .optional("surface_results.fireline_intensity", i)
        .unwrap_or(0.0);
    values.surface_hpua = a
        .optional("surface_results.heat_per_unit_area", i)
        .unwrap_or(0.0);
    values
}
pub fn surface_output(r: CrownSurfaceResult) -> Cell {
    let mut out = Cell::new();
    for (k, v) in structures::particles_output(r.p_crown) {
        out.insert(format!("p_crown.{k}"), v);
    }
    for (k, v) in structures::fuelbed_output(r.ib_crown) {
        out.insert(format!("ib_crown.{k}"), v);
    }
    for (k, v) in r.crown_surface.named_values() {
        out.insert(format!("crown_surface.{k}"), v);
    }
    out.insert("ri_crown".into(), r.ri_crown);
    out.insert("crown_fire_spread_rate".into(), r.crown_fire_spread_rate);
    out.insert(
        "crown_length_to_width_ratio".into(),
        r.crown_length_to_width_ratio,
    );
    out
}
