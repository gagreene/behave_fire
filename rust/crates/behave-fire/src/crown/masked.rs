//! Crown helpers preserving shared NumPy MaskedArray arithmetic semantics.
use super::{CrownHeat, CrownRatios, FinalCrownBehavior, FireClassification, PassiveCrownBehavior};
use crate::engine::{SurfaceCell, SurfaceOptions, WafMethod};
use crate::surface::{
    masked::{self, MaskedError, MaskedFuelbedIntermediates, MaskedValue, SurfaceInputShape},
    FuelbedIntermediates,
};

/// Construct a weak Python scalar for a translated source constant.
fn c(value: f64) -> MaskedValue {
    MaskedValue::scalar(value)
}

/// Evaluate the source flame expression before np.where erases its intermediate mask.
fn flame(intensity: MaskedValue) -> MaskedValue {
    MaskedValue::select(
        intensity.greater(c(0.)),
        c(0.2).mul(intensity.pow(c(2.0 / 3.0))),
        c(0.),
    )
}
/// Classify raw ratio comparisons, including zero-dimensional masked comparison normalization.
pub fn classify_crown_fire_type(
    transition: MaskedValue,
    active: MaskedValue,
) -> FireClassification {
    let fire_type = if transition.less(c(1.)) {
        if active.less(c(1.)) {
            0
        } else {
            2
        }
    } else if active.less(c(1.)) {
        1
    } else {
        3
    };
    FireClassification {
        fire_type,
        is_surface_fire: fire_type == 0 || fire_type == 2,
        is_passive_crown: fire_type == 1,
        is_active_crown: fire_type == 3,
    }
}

/// Select final Crown outputs with each branch's independent raw data and precision.
#[allow(clippy::too_many_arguments)]
pub fn assign_final_crown_fire_behavior(
    surface: bool,
    passive: bool,
    surface_rate: MaskedValue,
    surface_heat: MaskedValue,
    surface_intensity: MaskedValue,
    passive_rate: MaskedValue,
    passive_heat: MaskedValue,
    passive_intensity: MaskedValue,
    passive_flame: MaskedValue,
    crown_rate: MaskedValue,
    crown_heat: MaskedValue,
    crown_intensity: MaskedValue,
    crown_flame: MaskedValue,
) -> FinalCrownBehavior<MaskedValue> {
    FinalCrownBehavior {
        final_spread_rate: MaskedValue::select(
            surface,
            surface_rate,
            MaskedValue::select(passive, passive_rate, crown_rate),
        ),
        final_heat_per_unit_area: MaskedValue::select(
            surface,
            surface_heat,
            MaskedValue::select(passive, passive_heat, crown_heat),
        ),
        final_fireline_intensity: MaskedValue::select(
            surface,
            surface_intensity,
            MaskedValue::select(passive, passive_intensity, crown_intensity),
        ),
        final_flame_length: MaskedValue::select(
            surface,
            flame(surface_intensity),
            MaskedValue::select(passive, passive_flame, crown_flame),
        ),
    }
}

/// Calculate masked critical crown spread.
pub fn calculate_critical_crown_fire_spread_rate(density: MaskedValue) -> MaskedValue {
    let cbd = density.mul(c(16.0185));
    let safe = MaskedValue::select(cbd.greater(c(1e-7)), cbd, c(1.));
    MaskedValue::select(cbd.greater(c(1e-7)), c(3.).div(safe).mul(c(3.28084)), c(0.))
}
/// Calculate masked critical surface intensity.
pub fn calculate_critical_surface_fireline_intensity(
    height: MaskedValue,
    moisture: MaskedValue,
) -> MaskedValue {
    let mf = moisture.maximum(c(30.));
    let cbh = height.mul(c(0.3048)).maximum(c(0.1));
    c(0.010)
        .mul(cbh)
        .mul(c(460.).add(c(25.9).mul(mf)))
        .pow(c(1.5))
        .mul(c(0.2886719))
}
/// Calculate masked transition and active ratios.
pub fn calculate_crown_fire_ratios(
    surface: MaskedValue,
    crown: MaskedValue,
    critical_surface: MaskedValue,
    critical_crown: MaskedValue,
) -> CrownRatios<MaskedValue> {
    let safe_surface =
        MaskedValue::select(critical_surface.greater(c(1e-7)), critical_surface, c(1.));
    let safe_crown = MaskedValue::select(critical_crown.greater(c(1e-7)), critical_crown, c(1.));
    CrownRatios {
        crown_fire_transition_ratio: MaskedValue::select(
            critical_surface.greater(c(1e-7)),
            surface.div(safe_surface),
            c(0.),
        ),
        crown_fire_active_ratio: MaskedValue::select(
            critical_crown.greater(c(1e-7)),
            crown.div(safe_crown),
            c(0.),
        ),
    }
}
/// Calculate masked crown fraction.
pub fn calculate_crown_fraction_burned(
    surface: MaskedValue,
    critical: MaskedValue,
    crowning: MaskedValue,
) -> MaskedValue {
    let denom = crowning.sub(critical);
    MaskedValue::select(
        denom.greater(c(1e-7)),
        surface.sub(critical).div(denom),
        c(0.),
    )
    .maximum(c(0.))
    .minimum(c(1.))
}
/// Calculate masked canopy heat and intensity.
pub fn calculate_crown_heat_and_intensity(
    heat: MaskedValue,
    rate: MaskedValue,
    base: MaskedValue,
    height: MaskedValue,
    density: MaskedValue,
) -> CrownHeat<MaskedValue> {
    let load = density.mul(height.sub(base)).maximum(c(0.));
    let canopy = load.mul(c(8000.));
    let crown = heat.add(canopy);
    let intensity = rate.div(c(60.)).mul(crown);
    CrownHeat {
        crown_fuel_load: load,
        canopy_heat_per_unit_area: canopy,
        crown_fire_heat_per_unit_area: crown,
        crown_fire_line_intensity: intensity,
        crown_flame_length: flame(intensity),
    }
}
/// Calculate masked passive behavior.
pub fn calculate_passive_crown_fire_behavior(
    rate: MaskedValue,
    heat: MaskedValue,
    crown: MaskedValue,
    canopy: MaskedValue,
    fraction: MaskedValue,
) -> PassiveCrownBehavior<MaskedValue> {
    let passive_rate = rate.add(fraction.mul(crown.sub(rate)));
    let passive_heat = heat.add(fraction.mul(canopy));
    let intensity = passive_heat.mul(passive_rate).div(c(60.));
    PassiveCrownBehavior {
        passive_crown_fire_spread_rate: passive_rate,
        passive_crown_fire_heat_per_unit_area: passive_heat,
        passive_crown_fire_line_intensity: intensity,
        passive_crown_fire_flame_length: flame(intensity),
    }
}
/// Calculate masked critical surface rate.
pub fn calculate_surface_fire_critical_spread_rate(
    critical: MaskedValue,
    heat: MaskedValue,
) -> MaskedValue {
    let safe = MaskedValue::select(heat.greater(c(1e-7)), heat, c(1.));
    MaskedValue::select(heat.greater(c(1e-7)), c(60.).mul(critical).div(safe), c(0.))
}
/// Calculate masked crowning surface rate.
pub fn calculate_crowning_surface_fire_spread_rate(
    ri: MaskedValue,
    ib: &FuelbedIntermediates,
    critical: MaskedValue,
    height: MaskedValue,
) -> MaskedValue {
    calculate_crowning_surface_fire_spread_rate_with_shape(ri, ib, critical, height, false)
}
/// Retain the explicit source shape tuple's zero-dimensional grid choice for the crowning helper.
pub fn calculate_crowning_surface_fire_spread_rate_with_shape(
    ri: MaskedValue,
    ib: &FuelbedIntermediates,
    critical: MaskedValue,
    height: MaskedValue,
    scalar_grid: bool,
) -> MaskedValue {
    let mut ib = MaskedFuelbedIntermediates::from_plain(ib);
    macro_rules! rank {($($key:ident),+)=>{$(ib.$key=ib.$key.with_scalar_array(scalar_grid);)+}}
    rank!(
        sigma,
        bulk_density,
        packing_ratio,
        relative_packing_ratio,
        heat_sink,
        propagating_flux,
        w_heat_dead,
        w_heat_live,
        w_silica_dead,
        w_silica_live,
        w_moist_dead,
        w_moist_live,
        wfl_dead,
        wfl_live,
        moe_dead,
        moe_live,
        frac_dead,
        frac_live,
        depth,
        is_defined
    );
    try_calculate_crowning_surface_fire_spread_rate_with_intermediates(
        ri,
        &ib,
        critical,
        height,
        scalar_grid,
    )
    .expect("ordinary fuelbed fields cannot form invalid masked scalar boolean expressions")
}

/// Calculate crowning surface spread with independently masked fuelbed fields and the source grid rank.
///
/// Invalid zero-dimensional masked boolean combinations retain the shared error for adapters to map to TypeError.
pub fn try_calculate_crowning_surface_fire_spread_rate_with_intermediates(
    ri: MaskedValue,
    ib: &MaskedFuelbedIntermediates,
    critical: MaskedValue,
    height: MaskedValue,
    scalar_grid: bool,
) -> Result<MaskedValue, MaskedError> {
    let sink = ib.heat_sink;
    let flux = ib.propagating_flux;
    let ros0 = ri
        .mul(flux)
        .div(MaskedValue::select(sink.greater(c(1e-7)), sink, c(1.)));
    let r10 = critical.div(c(3.34));
    // Keep the reference coefficient's decimal spelling for traceability.
    #[allow(clippy::excessive_precision)]
    let a = MaskedValue::select(
        ros0.greater(c(1e-7)),
        r10.div(MaskedValue::select(ros0.greater(c(1e-7)), ros0, c(1.)))
            .sub(c(1.))
            .maximum(c(0.))
            .div(c(0.0016102128596515481)),
        c(0.),
    );
    let wind =
        MaskedValue::select(a.greater(c(0.)), a.pow(c(1. / 1.4308256324729873)), c(0.)).div(c(0.4));
    let mut input = SurfaceCell::new(10);
    input.wind_speed = wind.value;
    input.canopy_height = height.value;
    let controls = SurfaceOptions {
        waf_method: WafMethod::UserInput,
        ..SurfaceOptions::default()
    };
    input.user_waf = Some(0.4);
    let mut shape = SurfaceInputShape::uniform(scalar_grid);
    shape.wind_speed = wind.is_scalar_array;
    shape.canopy_height = height.is_scalar_array;
    let result = masked::try_run_surface_fire_with_intermediates_and_shape(
        ri, ib, &input, &controls, &shape,
    )?;
    let scalar = scalar_grid
        && ri.is_scalar_array
        && critical.is_scalar_array
        && [
            ib.sigma,
            ib.packing_ratio,
            ib.relative_packing_ratio,
            ib.propagating_flux,
            ib.heat_sink,
            ib.is_defined,
        ]
        .iter()
        .all(|v| v.is_scalar_array);
    Ok(MaskedValue::plain(result.spread_rate).with_scalar_array(scalar))
}
