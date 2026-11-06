//! Structured dictionaries marshalled to typed core records.
use crate::arrays::Arguments;
use crate::dispatch::Cell;
use behave_fire::units::*;
use behave_fire::{
    engine::{PreparedFuel, SurfaceCell},
    surface::*,
};
pub fn p(a: &Arguments, i: usize) -> Result<Particles, String> {
    let mut r = Particles::default();
    r.load_dead = [
        a.optional("p.load_dead.0", i).unwrap_or(0.0),
        a.optional("p.load_dead.1", i).unwrap_or(0.0),
        a.optional("p.load_dead.2", i).unwrap_or(0.0),
        a.optional("p.load_dead.3", i).unwrap_or(0.0),
        a.optional("p.load_dead.4", i).unwrap_or(0.0),
    ];
    r.load_live = [
        a.optional("p.load_live.0", i).unwrap_or(0.0),
        a.optional("p.load_live.1", i).unwrap_or(0.0),
        a.optional("p.load_live.2", i).unwrap_or(0.0),
        a.optional("p.load_live.3", i).unwrap_or(0.0),
        a.optional("p.load_live.4", i).unwrap_or(0.0),
    ];
    r.savr_dead = [
        a.optional("p.savr_dead.0", i).unwrap_or(0.0),
        a.optional("p.savr_dead.1", i).unwrap_or(0.0),
        a.optional("p.savr_dead.2", i).unwrap_or(0.0),
        a.optional("p.savr_dead.3", i).unwrap_or(0.0),
        a.optional("p.savr_dead.4", i).unwrap_or(0.0),
    ];
    r.savr_live = [
        a.optional("p.savr_live.0", i).unwrap_or(0.0),
        a.optional("p.savr_live.1", i).unwrap_or(0.0),
        a.optional("p.savr_live.2", i).unwrap_or(0.0),
        a.optional("p.savr_live.3", i).unwrap_or(0.0),
        a.optional("p.savr_live.4", i).unwrap_or(0.0),
    ];
    r.moisture_dead = [
        a.optional("p.moisture_dead.0", i).unwrap_or(0.0),
        a.optional("p.moisture_dead.1", i).unwrap_or(0.0),
        a.optional("p.moisture_dead.2", i).unwrap_or(0.0),
        a.optional("p.moisture_dead.3", i).unwrap_or(0.0),
        a.optional("p.moisture_dead.4", i).unwrap_or(0.0),
    ];
    r.moisture_live = [
        a.optional("p.moisture_live.0", i).unwrap_or(0.0),
        a.optional("p.moisture_live.1", i).unwrap_or(0.0),
        a.optional("p.moisture_live.2", i).unwrap_or(0.0),
        a.optional("p.moisture_live.3", i).unwrap_or(0.0),
        a.optional("p.moisture_live.4", i).unwrap_or(0.0),
    ];
    r.hoc_dead = a.optional("p.hoc_dead", i).unwrap_or(0.0);
    r.hoc_live = a.optional("p.hoc_live", i).unwrap_or(0.0);
    r.depth = a.optional("p.depth", i).unwrap_or(0.0);
    r.moe_dead = a.optional("p.moe_dead", i).unwrap_or(0.0);
    r.is_defined = a.optional("p.is_defined", i).unwrap_or(0.0) != 0.0;
    Ok(r)
}
pub fn ib(a: &Arguments, i: usize) -> Result<FuelbedIntermediates, String> {
    let mut r = FuelbedIntermediates::default();
    r.sigma = a.optional("ib.sigma", i).unwrap_or(0.0);
    r.bulk_density = a.optional("ib.bulk_density", i).unwrap_or(0.0);
    r.packing_ratio = a.optional("ib.packing_ratio", i).unwrap_or(0.0);
    r.relative_packing_ratio = a.optional("ib.relative_packing_ratio", i).unwrap_or(0.0);
    r.heat_sink = a.optional("ib.heat_sink", i).unwrap_or(0.0);
    r.propagating_flux = a.optional("ib.propagating_flux", i).unwrap_or(0.0);
    r.w_heat_dead = a.optional("ib.w_heat_dead", i).unwrap_or(0.0);
    r.w_heat_live = a.optional("ib.w_heat_live", i).unwrap_or(0.0);
    r.w_silica_dead = a.optional("ib.w_silica_dead", i).unwrap_or(0.0);
    r.w_silica_live = a.optional("ib.w_silica_live", i).unwrap_or(0.0);
    r.w_moist_dead = a.optional("ib.w_moist_dead", i).unwrap_or(0.0);
    r.w_moist_live = a.optional("ib.w_moist_live", i).unwrap_or(0.0);
    r.wfl_dead = a.optional("ib.wfl_dead", i).unwrap_or(0.0);
    r.wfl_live = a.optional("ib.wfl_live", i).unwrap_or(0.0);
    r.moe_dead = a.optional("ib.moe_dead", i).unwrap_or(0.0);
    r.moe_live = a.optional("ib.moe_live", i).unwrap_or(0.0);
    r.frac_dead = a.optional("ib.frac_dead", i).unwrap_or(0.0);
    r.frac_live = a.optional("ib.frac_live", i).unwrap_or(0.0);
    r.depth = a.optional("ib.depth", i).unwrap_or(0.0);
    r.is_defined = a.optional("ib.is_defined", i).unwrap_or(0.0) != 0.0;
    Ok(r)
}
pub fn input(a: &Arguments, i: usize) -> Result<SurfaceCell, String> {
    Ok(SurfaceCell {
        wind_speed: SpeedUnit::try_from(a.n("wind_speed_units", i)? as i32)
            .map_err(|e| e.to_string())?
            .to_base(a.n("wind_speed", i)?),
        wind_direction: a.n("wind_direction", i)?,
        slope: SlopeUnit::try_from(a.n("slope_units", i)? as i32)
            .map_err(|e| e.to_string())?
            .to_base(a.n("slope", i)?),
        aspect: a.n("aspect", i)?,
        canopy_cover: a.n("canopy_cover", i)?,
        canopy_height: a.n("canopy_height", i)?,
        crown_ratio: a.n("crown_ratio", i)?,
        user_waf: a.optional("user_waf", i),
        ..SurfaceCell::default()
    })
}
// Physical fuel fields are copied without carrying registry names into cell loops.
#[allow(clippy::field_reassign_with_default)]
pub fn fuel(a: &Arguments, i: usize) -> Result<PreparedFuel, String> {
    let mut r = PreparedFuel::default();
    r.fuel_bed_depth = a.n("model.fuel_bed_depth", i)?;
    r.moisture_of_extinction_dead = a.n("model.moisture_of_extinction_dead", i)?;
    r.heat_of_combustion_dead = a.n("model.heat_of_combustion_dead", i)?;
    r.heat_of_combustion_live = a.n("model.heat_of_combustion_live", i)?;
    r.dead_1h = a.n("model.dead_1h", i)?;
    r.dead_10h = a.n("model.dead_10h", i)?;
    r.dead_100h = a.n("model.dead_100h", i)?;
    r.live_herb = a.n("model.live_herb", i)?;
    r.live_woody = a.n("model.live_woody", i)?;
    r.savr_1h = a.n("model.savr_1h", i)?;
    r.savr_live_herb = a.n("model.savr_live_herb", i)?;
    r.savr_live_woody = a.n("model.savr_live_woody", i)?;
    r.is_dynamic = a.n("model.is_dynamic", i)? != 0.0;
    r.is_defined = a.n("model.is_defined", i)? != 0.0;
    Ok(r)
}
/// Resume supplied reaction/fuelbed state, retaining the source's f32 slope-cap rounding.
pub fn run_surface(a: &Arguments, i: usize, single: bool) -> Result<SurfaceResult, String> {
    let cell = input(a, i)?;
    let fuelbed = ib(a, i)?;
    let options = crate::controls::surface(a)?;
    let reaction = a.n("ri", i)?;
    let mut tweaks = behave_fire::engine::SurfaceTweaks {
        reaction_intensity: Some(reaction),
        ..behave_fire::engine::SurfaceTweaks::default()
    };
    if single {
        let reaction = reaction as f32;
        tweaks.reaction_intensity = Some(reaction as f64);
        tweaks.slope_phi = Some(behave_fire::math::np_min(
            calculate_slope_phi(
                cell.slope,
                SlopeUnit::Degrees,
                fuelbed.packing_ratio,
                f64::INFINITY,
            ),
            (0.9_f32 * reaction) as f64,
        ));
    }
    Ok(behave_fire::engine::SurfacePlan::new(options).run_from_fuelbed(&cell, &fuelbed, tweaks))
}
pub fn particles_output(p: Particles) -> Cell {
    let mut result: Cell = p
        .named_values()
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
    result.insert("is_defined".into(), f64::from(u8::from(p.is_defined)));
    for (name, values) in p.named_arrays() {
        for (j, value) in values.into_iter().enumerate() {
            result.insert(format!("{name}.{j}"), value);
        }
    }
    result
}
pub fn fuelbed_output(p: FuelbedIntermediates) -> Cell {
    let mut result: Cell = p
        .named_values()
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
    result.insert("is_defined".into(), f64::from(u8::from(p.is_defined)));
    result
}
pub fn surface_output(r: SurfaceResult) -> Cell {
    r.named_values()
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect()
}
