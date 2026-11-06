//! Typed particle and fuelbed dictionaries preserve per-field dtype and masks.
use crate::{arrays::Arguments, dispatch::Cell, masked::output};
use behave_fire::surface::masked::*;
pub fn particles(a: &Arguments, i: usize) -> Result<MaskedParticles, String> {
    let mut r = MaskedParticles::new();
    r.load_dead = [
        a.masked("p.load_dead.0", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.load_dead.1", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.load_dead.2", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.load_dead.3", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.load_dead.4", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.load_live = [
        a.masked("p.load_live.0", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.load_live.1", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.load_live.2", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.load_live.3", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.load_live.4", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.savr_dead = [
        a.masked("p.savr_dead.0", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.savr_dead.1", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.savr_dead.2", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.savr_dead.3", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.savr_dead.4", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.savr_live = [
        a.masked("p.savr_live.0", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.savr_live.1", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.savr_live.2", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.savr_live.3", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.savr_live.4", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.moisture_dead = [
        a.masked("p.moisture_dead.0", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.moisture_dead.1", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.moisture_dead.2", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.moisture_dead.3", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.moisture_dead.4", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.moisture_live = [
        a.masked("p.moisture_live.0", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.moisture_live.1", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.moisture_live.2", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.moisture_live.3", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked("p.moisture_live.4", i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.hoc_dead = if a.optional("p.hoc_dead", i).is_some() {
        a.masked("p.hoc_dead", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.hoc_live = if a.optional("p.hoc_live", i).is_some() {
        a.masked("p.hoc_live", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.depth = if a.optional("p.depth", i).is_some() {
        a.masked("p.depth", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.moe_dead = if a.optional("p.moe_dead", i).is_some() {
        a.masked("p.moe_dead", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.is_defined = if a.optional("p.is_defined", i).is_some() {
        a.masked("p.is_defined", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.scalar_cell = a.optional("_particle_scalar", i).is_some();
    Ok(r)
}
pub fn intermediates(a: &Arguments, i: usize) -> Result<MaskedFuelbedIntermediates, String> {
    let mut r = MaskedFuelbedIntermediates::new();
    r.sigma = if a.optional("ib.sigma", i).is_some() {
        a.masked("ib.sigma", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.bulk_density = if a.optional("ib.bulk_density", i).is_some() {
        a.masked("ib.bulk_density", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.packing_ratio = if a.optional("ib.packing_ratio", i).is_some() {
        a.masked("ib.packing_ratio", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.relative_packing_ratio = if a.optional("ib.relative_packing_ratio", i).is_some() {
        a.masked("ib.relative_packing_ratio", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.heat_sink = if a.optional("ib.heat_sink", i).is_some() {
        a.masked("ib.heat_sink", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.propagating_flux = if a.optional("ib.propagating_flux", i).is_some() {
        a.masked("ib.propagating_flux", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_heat_dead = if a.optional("ib.w_heat_dead", i).is_some() {
        a.masked("ib.w_heat_dead", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_heat_live = if a.optional("ib.w_heat_live", i).is_some() {
        a.masked("ib.w_heat_live", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_silica_dead = if a.optional("ib.w_silica_dead", i).is_some() {
        a.masked("ib.w_silica_dead", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_silica_live = if a.optional("ib.w_silica_live", i).is_some() {
        a.masked("ib.w_silica_live", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_moist_dead = if a.optional("ib.w_moist_dead", i).is_some() {
        a.masked("ib.w_moist_dead", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_moist_live = if a.optional("ib.w_moist_live", i).is_some() {
        a.masked("ib.w_moist_live", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.wfl_dead = if a.optional("ib.wfl_dead", i).is_some() {
        a.masked("ib.wfl_dead", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.wfl_live = if a.optional("ib.wfl_live", i).is_some() {
        a.masked("ib.wfl_live", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.moe_dead = if a.optional("ib.moe_dead", i).is_some() {
        a.masked("ib.moe_dead", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.moe_live = if a.optional("ib.moe_live", i).is_some() {
        a.masked("ib.moe_live", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.frac_dead = if a.optional("ib.frac_dead", i).is_some() {
        a.masked("ib.frac_dead", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.frac_live = if a.optional("ib.frac_live", i).is_some() {
        a.masked("ib.frac_live", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.depth = if a.optional("ib.depth", i).is_some() {
        a.masked("ib.depth", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.is_defined = if a.optional("ib.is_defined", i).is_some() {
        a.masked("ib.is_defined", i)?
    } else {
        MaskedValue::plain(0.0)
    };
    Ok(r)
}
pub fn fields(r: MaskedFuelbedIntermediates) -> Result<Cell, String> {
    crate::masked::outputs(r.named_values())
}
pub fn evaluate(operation: &str, a: &Arguments, i: usize) -> Result<Option<Cell>, String> {
    Ok(Some(match operation {
        "surface.calculate_fuelbed_intermediates" => fields(
            try_calculate_fuelbed_intermediates(&particles(a, i)?)
                .map_err(crate::arrays::masked_error)?,
        )?,
        "surface.calculate_reaction_intensity" => output(
            try_calculate_reaction_intensity(&intermediates(a, i)?)
                .map_err(crate::arrays::masked_error)?,
            "",
        )?,
        "surface.run_surface_fire" => crate::structures::surface_output(
            try_run_surface_fire_with_intermediates_and_shape(
                a.masked("ri", i)?,
                &intermediates(a, i)?,
                &crate::structures::input(a, i)?,
                &crate::controls::surface(a)?,
                &input_shape(a, i),
            )
            .map_err(crate::arrays::masked_error)?,
        ),
        _ => return Ok(None),
    }))
}
fn input_shape(a: &Arguments, i: usize) -> SurfaceInputShape {
    let mut shape = SurfaceInputShape::uniform(false);
    shape.wind_speed = a.optional("_scalar.wind_speed", i).is_some();
    shape.wind_direction = a.optional("_scalar.wind_direction", i).is_some();
    shape.slope = a.optional("_scalar.slope", i).is_some();
    shape.aspect = a.optional("_scalar.aspect", i).is_some();
    shape.canopy_cover = a.optional("_scalar.canopy_cover", i).is_some();
    shape.canopy_height = a.optional("_scalar.canopy_height", i).is_some();
    shape.crown_ratio = a.optional("_scalar.crown_ratio", i).is_some();
    shape.user_waf = a.optional("_scalar.user_waf", i).is_some();
    shape
}
