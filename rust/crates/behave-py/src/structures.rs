//! Structured dictionaries marshalled to typed core records.
use crate::component_columns::RowWriter;
use crate::dispatch::named;
use crate::prepared::{Field, PreparedArguments};
use behave_fire::{
    engine::{PreparedFuel, SurfaceCell},
    surface::*,
};
pub fn p(a: &PreparedArguments<'_>, i: usize) -> Result<Particles, String> {
    let mut r = Particles::default();
    r.load_dead = [
        a.optional(Field::PLoadDead0, i).unwrap_or(0.0),
        a.optional(Field::PLoadDead1, i).unwrap_or(0.0),
        a.optional(Field::PLoadDead2, i).unwrap_or(0.0),
        a.optional(Field::PLoadDead3, i).unwrap_or(0.0),
        a.optional(Field::PLoadDead4, i).unwrap_or(0.0),
    ];
    r.load_live = [
        a.optional(Field::PLoadLive0, i).unwrap_or(0.0),
        a.optional(Field::PLoadLive1, i).unwrap_or(0.0),
        a.optional(Field::PLoadLive2, i).unwrap_or(0.0),
        a.optional(Field::PLoadLive3, i).unwrap_or(0.0),
        a.optional(Field::PLoadLive4, i).unwrap_or(0.0),
    ];
    r.savr_dead = [
        a.optional(Field::PSavrDead0, i).unwrap_or(0.0),
        a.optional(Field::PSavrDead1, i).unwrap_or(0.0),
        a.optional(Field::PSavrDead2, i).unwrap_or(0.0),
        a.optional(Field::PSavrDead3, i).unwrap_or(0.0),
        a.optional(Field::PSavrDead4, i).unwrap_or(0.0),
    ];
    r.savr_live = [
        a.optional(Field::PSavrLive0, i).unwrap_or(0.0),
        a.optional(Field::PSavrLive1, i).unwrap_or(0.0),
        a.optional(Field::PSavrLive2, i).unwrap_or(0.0),
        a.optional(Field::PSavrLive3, i).unwrap_or(0.0),
        a.optional(Field::PSavrLive4, i).unwrap_or(0.0),
    ];
    r.moisture_dead = [
        a.optional(Field::PMoistureDead0, i).unwrap_or(0.0),
        a.optional(Field::PMoistureDead1, i).unwrap_or(0.0),
        a.optional(Field::PMoistureDead2, i).unwrap_or(0.0),
        a.optional(Field::PMoistureDead3, i).unwrap_or(0.0),
        a.optional(Field::PMoistureDead4, i).unwrap_or(0.0),
    ];
    r.moisture_live = [
        a.optional(Field::PMoistureLive0, i).unwrap_or(0.0),
        a.optional(Field::PMoistureLive1, i).unwrap_or(0.0),
        a.optional(Field::PMoistureLive2, i).unwrap_or(0.0),
        a.optional(Field::PMoistureLive3, i).unwrap_or(0.0),
        a.optional(Field::PMoistureLive4, i).unwrap_or(0.0),
    ];
    r.hoc_dead = a.optional(Field::PHocDead, i).unwrap_or(0.0);
    r.hoc_live = a.optional(Field::PHocLive, i).unwrap_or(0.0);
    r.depth = a.optional(Field::PDepth, i).unwrap_or(0.0);
    r.moe_dead = a.optional(Field::PMoeDead, i).unwrap_or(0.0);
    r.is_defined = a.flag(Field::PIsDefined, i)?;
    Ok(r)
}
pub fn ib(a: &PreparedArguments<'_>, i: usize) -> Result<FuelbedIntermediates, String> {
    let mut r = FuelbedIntermediates::default();
    r.sigma = a.optional(Field::IbSigma, i).unwrap_or(0.0);
    r.bulk_density = a.optional(Field::IbBulkDensity, i).unwrap_or(0.0);
    r.packing_ratio = a.optional(Field::IbPackingRatio, i).unwrap_or(0.0);
    r.relative_packing_ratio = a.optional(Field::IbRelativePackingRatio, i).unwrap_or(0.0);
    r.heat_sink = a.optional(Field::IbHeatSink, i).unwrap_or(0.0);
    r.propagating_flux = a.optional(Field::IbPropagatingFlux, i).unwrap_or(0.0);
    r.w_heat_dead = a.optional(Field::IbWHeatDead, i).unwrap_or(0.0);
    r.w_heat_live = a.optional(Field::IbWHeatLive, i).unwrap_or(0.0);
    r.w_silica_dead = a.optional(Field::IbWSilicaDead, i).unwrap_or(0.0);
    r.w_silica_live = a.optional(Field::IbWSilicaLive, i).unwrap_or(0.0);
    r.w_moist_dead = a.optional(Field::IbWMoistDead, i).unwrap_or(0.0);
    r.w_moist_live = a.optional(Field::IbWMoistLive, i).unwrap_or(0.0);
    r.wfl_dead = a.optional(Field::IbWflDead, i).unwrap_or(0.0);
    r.wfl_live = a.optional(Field::IbWflLive, i).unwrap_or(0.0);
    r.moe_dead = a.optional(Field::IbMoeDead, i).unwrap_or(0.0);
    r.moe_live = a.optional(Field::IbMoeLive, i).unwrap_or(0.0);
    r.frac_dead = a.optional(Field::IbFracDead, i).unwrap_or(0.0);
    r.frac_live = a.optional(Field::IbFracLive, i).unwrap_or(0.0);
    r.depth = a.optional(Field::IbDepth, i).unwrap_or(0.0);
    r.is_defined = if a.has(Field::IbIsDefined) {
        a.flag(Field::IbIsDefined, i)?
    } else {
        false
    };
    Ok(r)
}
pub fn input(a: &PreparedArguments<'_>, i: usize) -> Result<SurfaceCell, String> {
    Ok(SurfaceCell {
        wind_speed: a.speed_units(i)?.to_base(a.n(Field::WindSpeed, i)?),
        wind_direction: a.n(Field::WindDirection, i)?,
        slope: a.slope_units(i)?.to_base(a.n(Field::Slope, i)?),
        aspect: a.optional(Field::Aspect, i).unwrap_or(f64::NAN),
        canopy_cover: a.optional(Field::CanopyCover, i).unwrap_or(f64::NAN),
        canopy_height: a.optional(Field::CanopyHeight, i).unwrap_or(f64::NAN),
        crown_ratio: a.optional(Field::CrownRatio, i).unwrap_or(f64::NAN),
        user_waf: a.optional(Field::UserWaf, i),
        ..SurfaceCell::default()
    })
}
// Physical fuel fields are copied without carrying registry names into cell loops.
#[allow(clippy::field_reassign_with_default)]
pub fn fuel(a: &PreparedArguments<'_>, i: usize) -> Result<PreparedFuel, String> {
    let mut r = PreparedFuel::default();
    r.fuel_bed_depth = a.n(Field::ModelFuelBedDepth, i)?;
    r.moisture_of_extinction_dead = a.n(Field::ModelMoistureOfExtinctionDead, i)?;
    r.heat_of_combustion_dead = a.n(Field::ModelHeatOfCombustionDead, i)?;
    r.heat_of_combustion_live = a.n(Field::ModelHeatOfCombustionLive, i)?;
    r.dead_1h = a.n(Field::ModelDead1h, i)?;
    r.dead_10h = a.n(Field::ModelDead10h, i)?;
    r.dead_100h = a.n(Field::ModelDead100h, i)?;
    r.live_herb = a.n(Field::ModelLiveHerb, i)?;
    r.live_woody = a.n(Field::ModelLiveWoody, i)?;
    r.savr_1h = a.n(Field::ModelSavr1h, i)?;
    r.savr_live_herb = a.n(Field::ModelSavrLiveHerb, i)?;
    r.savr_live_woody = a.n(Field::ModelSavrLiveWoody, i)?;
    r.is_dynamic = a.flag(Field::ModelIsDynamic, i)?;
    r.is_defined = a.flag(Field::ModelIsDefined, i)?;
    Ok(r)
}
/// Resume supplied reaction/fuelbed state, using the shared float64 stage equations.
pub fn run_surface(a: &PreparedArguments<'_>, i: usize) -> Result<SurfaceResult, String> {
    let cell = input(a, i)?;
    let fuelbed = ib(a, i)?;
    let options = crate::prepared::controls::surface(a)?;
    let reaction = a.n(Field::Ri, i)?;
    let tweaks = behave_fire::engine::SurfaceTweaks {
        reaction_intensity: Some(reaction),
        ..behave_fire::engine::SurfaceTweaks::default()
    };

    Ok(behave_fire::engine::SurfacePlan::new(options).run_from_fuelbed(&cell, &fuelbed, tweaks))
}
pub fn particles_output(out: &mut RowWriter<'_, '_>, p: Particles) {
    named(out, p.named_values());
    out.boolean("is_defined", p.is_defined);
    for (name, values) in p.named_arrays() {
        for (index, value) in values.into_iter().enumerate() {
            out.indexed_f64(name, index, value);
        }
    }
}
pub fn fuelbed_output(out: &mut RowWriter<'_, '_>, p: FuelbedIntermediates) {
    named(out, p.named_values());
    out.boolean("is_defined", p.is_defined);
}
pub fn surface_output(out: &mut RowWriter<'_, '_>, result: SurfaceResult) {
    named(out, result.named_values());
}
