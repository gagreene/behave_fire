//! Typed particle and fuelbed dictionaries preserve per-field dtype and masks.
use crate::{
    dispatch::Cell,
    masked::output,
    prepared::{Field, PreparedArguments},
};
use behave_fire::surface::masked::*;
pub fn particles(a: &PreparedArguments<'_>, i: usize) -> Result<MaskedParticles, String> {
    let mut r = MaskedParticles::new();
    r.load_dead = [
        a.masked(Field::PLoadDead0, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PLoadDead1, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PLoadDead2, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PLoadDead3, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PLoadDead4, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.load_live = [
        a.masked(Field::PLoadLive0, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PLoadLive1, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PLoadLive2, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PLoadLive3, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PLoadLive4, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.savr_dead = [
        a.masked(Field::PSavrDead0, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PSavrDead1, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PSavrDead2, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PSavrDead3, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PSavrDead4, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.savr_live = [
        a.masked(Field::PSavrLive0, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PSavrLive1, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PSavrLive2, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PSavrLive3, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PSavrLive4, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.moisture_dead = [
        a.masked(Field::PMoistureDead0, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PMoistureDead1, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PMoistureDead2, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PMoistureDead3, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PMoistureDead4, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.moisture_live = [
        a.masked(Field::PMoistureLive0, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PMoistureLive1, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PMoistureLive2, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PMoistureLive3, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
        a.masked(Field::PMoistureLive4, i)?
            .with_scalar_array(false)
            .with_numpy_scalar(false),
    ];
    r.hoc_dead = if a.optional(Field::PHocDead, i).is_some() {
        a.masked(Field::PHocDead, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.hoc_live = if a.optional(Field::PHocLive, i).is_some() {
        a.masked(Field::PHocLive, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.depth = if a.optional(Field::PDepth, i).is_some() {
        a.masked(Field::PDepth, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.moe_dead = if a.optional(Field::PMoeDead, i).is_some() {
        a.masked(Field::PMoeDead, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.is_defined = if a.optional(Field::PIsDefined, i).is_some() {
        a.masked(Field::PIsDefined, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.scalar_cell = a.optional(Field::GlobalParticleScalar, i).is_some();
    Ok(r)
}
pub fn intermediates(
    a: &PreparedArguments<'_>,
    i: usize,
) -> Result<MaskedFuelbedIntermediates, String> {
    let mut r = MaskedFuelbedIntermediates::new();
    r.sigma = if a.optional(Field::IbSigma, i).is_some() {
        a.masked(Field::IbSigma, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.bulk_density = if a.optional(Field::IbBulkDensity, i).is_some() {
        a.masked(Field::IbBulkDensity, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.packing_ratio = if a.optional(Field::IbPackingRatio, i).is_some() {
        a.masked(Field::IbPackingRatio, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.relative_packing_ratio = if a.optional(Field::IbRelativePackingRatio, i).is_some() {
        a.masked(Field::IbRelativePackingRatio, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.heat_sink = if a.optional(Field::IbHeatSink, i).is_some() {
        a.masked(Field::IbHeatSink, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.propagating_flux = if a.optional(Field::IbPropagatingFlux, i).is_some() {
        a.masked(Field::IbPropagatingFlux, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_heat_dead = if a.optional(Field::IbWHeatDead, i).is_some() {
        a.masked(Field::IbWHeatDead, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_heat_live = if a.optional(Field::IbWHeatLive, i).is_some() {
        a.masked(Field::IbWHeatLive, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_silica_dead = if a.optional(Field::IbWSilicaDead, i).is_some() {
        a.masked(Field::IbWSilicaDead, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_silica_live = if a.optional(Field::IbWSilicaLive, i).is_some() {
        a.masked(Field::IbWSilicaLive, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_moist_dead = if a.optional(Field::IbWMoistDead, i).is_some() {
        a.masked(Field::IbWMoistDead, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.w_moist_live = if a.optional(Field::IbWMoistLive, i).is_some() {
        a.masked(Field::IbWMoistLive, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.wfl_dead = if a.optional(Field::IbWflDead, i).is_some() {
        a.masked(Field::IbWflDead, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.wfl_live = if a.optional(Field::IbWflLive, i).is_some() {
        a.masked(Field::IbWflLive, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.moe_dead = if a.optional(Field::IbMoeDead, i).is_some() {
        a.masked(Field::IbMoeDead, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.moe_live = if a.optional(Field::IbMoeLive, i).is_some() {
        a.masked(Field::IbMoeLive, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.frac_dead = if a.optional(Field::IbFracDead, i).is_some() {
        a.masked(Field::IbFracDead, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.frac_live = if a.optional(Field::IbFracLive, i).is_some() {
        a.masked(Field::IbFracLive, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.depth = if a.optional(Field::IbDepth, i).is_some() {
        a.masked(Field::IbDepth, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    r.is_defined = if a.optional(Field::IbIsDefined, i).is_some() {
        a.masked(Field::IbIsDefined, i)?
    } else {
        MaskedValue::plain(0.0)
    };
    Ok(r)
}
pub fn fields(r: MaskedFuelbedIntermediates) -> Result<Cell, String> {
    crate::masked::outputs(r.named_values())
}
pub fn evaluate(
    operation: &str,
    a: &PreparedArguments<'_>,
    i: usize,
) -> Result<Option<Cell>, String> {
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
                a.masked(Field::Ri, i)?,
                &intermediates(a, i)?,
                &crate::structures::input(a, i)?,
                &crate::prepared::controls::surface(a)?,
                &input_shape(a, i),
            )
            .map_err(crate::arrays::masked_error)?,
        ),
        _ => return Ok(None),
    }))
}
fn input_shape(a: &PreparedArguments<'_>, i: usize) -> SurfaceInputShape {
    let mut shape = SurfaceInputShape::uniform(false);
    shape.wind_speed = a.optional(Field::GlobalScalarWindSpeed, i).is_some();
    shape.wind_direction = a.optional(Field::GlobalScalarWindDirection, i).is_some();
    shape.slope = a.optional(Field::GlobalScalarSlope, i).is_some();
    shape.aspect = a.optional(Field::GlobalScalarAspect, i).is_some();
    shape.canopy_cover = a.optional(Field::GlobalScalarCanopyCover, i).is_some();
    shape.canopy_height = a.optional(Field::GlobalScalarCanopyHeight, i).is_some();
    shape.crown_ratio = a.optional(Field::GlobalScalarCrownRatio, i).is_some();
    shape.user_waf = a.optional(Field::GlobalScalarUserWaf, i).is_some();
    shape
}
