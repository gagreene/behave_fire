//! Structured native crown inputs and nested outputs.
use crate::{
    component_columns::RowWriter,
    prepared::{Field, PreparedArguments},
    structures,
};

use behave_fire::{
    crown::*,
    engine::{CrownCell, PreparedFuel},
};
pub fn input(a: &PreparedArguments<'_>, i: usize) -> Result<CrownCell, String> {
    let mut r = CrownCell::default();
    r.m1h = a.optional(Field::M1h, i).unwrap_or(r.m1h);
    r.m10h = a.optional(Field::M10h, i).unwrap_or(r.m10h);
    r.m100h = a.optional(Field::M100h, i).unwrap_or(r.m100h);
    r.mlh = a.optional(Field::Mlh, i).unwrap_or(r.mlh);
    r.mlw = a.optional(Field::Mlw, i).unwrap_or(r.mlw);
    let wind_units = a.speed_units(i)?;
    r.wind_speed = wind_units.to_base(a.optional(Field::WindSpeed, i).unwrap_or(0.0));
    r.wind_direction = a
        .optional(Field::WindDirection, i)
        .unwrap_or(r.wind_direction);
    let slope_units = a.slope_units(i)?;
    r.slope = slope_units.to_base(a.optional(Field::Slope, i).unwrap_or(0.0));
    r.aspect = a.optional(Field::Aspect, i).unwrap_or(r.aspect);
    r.canopy_base_height = a
        .optional(Field::CanopyBaseHeight, i)
        .unwrap_or(r.canopy_base_height);
    r.canopy_height = a
        .optional(Field::CanopyHeight, i)
        .unwrap_or(r.canopy_height);
    r.canopy_bulk_density = a
        .optional(Field::CanopyBulkDensity, i)
        .unwrap_or(r.canopy_bulk_density);
    r.moisture_foliar = a
        .optional(Field::MoistureFoliar, i)
        .unwrap_or(r.moisture_foliar);
    Ok(r)
}
pub fn fuel(a: &PreparedArguments<'_>, i: usize) -> Result<PreparedFuel, String> {
    structures::fuel(a, i)
}
pub fn surface_output(out: &mut RowWriter<'_, '_>, result: CrownSurfaceResult) {
    out.with_prefix("p_crown", |out| {
        structures::particles_output(out, result.p_crown)
    });
    out.with_prefix("ib_crown", |out| {
        structures::fuelbed_output(out, result.ib_crown)
    });
    out.with_prefix("crown_surface", |out| {
        structures::surface_output(out, result.crown_surface)
    });
    out.f64("ri_crown", result.ri_crown);
    out.f64("crown_fire_spread_rate", result.crown_fire_spread_rate);
    out.f64(
        "crown_length_to_width_ratio",
        result.crown_length_to_width_ratio,
    );
}
