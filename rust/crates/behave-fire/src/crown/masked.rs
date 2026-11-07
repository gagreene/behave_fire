//! Crown helpers preserving shared NumPy MaskedArray arithmetic semantics.
use super::arithmetic::CrownNumber;
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
fn flame<T: CrownNumber>(intensity: T) -> T {
    T::select(
        intensity.greater(T::scalar(0.)),
        T::scalar(0.2).mul(intensity.pow(T::scalar(2.0 / 3.0))),
        T::scalar(0.),
    )
}
/// Classify raw ratio comparisons, including zero-dimensional masked comparison normalization.
pub fn classify_crown_fire_type<T: CrownNumber>(transition: T, active: T) -> FireClassification {
    let fire_type = if transition.less(T::scalar(1.)) {
        if active.less(T::scalar(1.)) {
            0
        } else {
            2
        }
    } else if active.less(T::scalar(1.)) {
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
pub fn assign_final_crown_fire_behavior<T: CrownNumber>(
    surface: bool,
    passive: bool,
    surface_rate: T,
    surface_heat: T,
    surface_intensity: T,
    passive_rate: T,
    passive_heat: T,
    passive_intensity: T,
    passive_flame: T,
    crown_rate: T,
    crown_heat: T,
    crown_intensity: T,
    crown_flame: T,
) -> FinalCrownBehavior<T> {
    FinalCrownBehavior {
        final_spread_rate: T::select(
            surface,
            surface_rate,
            T::select(passive, passive_rate, crown_rate),
        ),
        final_heat_per_unit_area: T::select(
            surface,
            surface_heat,
            T::select(passive, passive_heat, crown_heat),
        ),
        final_fireline_intensity: T::select(
            surface,
            surface_intensity,
            T::select(passive, passive_intensity, crown_intensity),
        ),
        final_flame_length: T::select(
            surface,
            flame(surface_intensity),
            T::select(passive, passive_flame, crown_flame),
        ),
    }
}

/// Calculate masked critical crown spread.
pub fn calculate_critical_crown_fire_spread_rate<T: CrownNumber>(density: T) -> T {
    let cbd = density.mul(T::scalar(16.0185));
    let safe = T::select(cbd.greater(T::scalar(1e-7)), cbd, T::scalar(1.));
    T::select(
        cbd.greater(T::scalar(1e-7)),
        T::scalar(3.).div(safe).mul(T::scalar(3.28084)),
        T::scalar(0.),
    )
}
/// Calculate masked critical surface intensity.
pub fn calculate_critical_surface_fireline_intensity<T: CrownNumber>(height: T, moisture: T) -> T {
    let mf = moisture.maximum(T::scalar(30.));
    let cbh = height.mul(T::scalar(0.3048)).maximum(T::scalar(0.1));
    T::scalar(0.010)
        .mul(cbh)
        .mul(T::scalar(460.).add(T::scalar(25.9).mul(mf)))
        .pow(T::scalar(1.5))
        .mul(T::scalar(0.2886719))
}
/// Calculate masked transition and active ratios.
pub fn calculate_crown_fire_ratios<T: CrownNumber>(
    surface: T,
    crown: T,
    critical_surface: T,
    critical_crown: T,
) -> CrownRatios<T> {
    let safe_surface = T::select(
        critical_surface.greater(T::scalar(1e-7)),
        critical_surface,
        T::scalar(1.),
    );
    let safe_crown = T::select(
        critical_crown.greater(T::scalar(1e-7)),
        critical_crown,
        T::scalar(1.),
    );
    CrownRatios {
        crown_fire_transition_ratio: T::select(
            critical_surface.greater(T::scalar(1e-7)),
            surface.div(safe_surface),
            T::scalar(0.),
        ),
        crown_fire_active_ratio: T::select(
            critical_crown.greater(T::scalar(1e-7)),
            crown.div(safe_crown),
            T::scalar(0.),
        ),
    }
}
/// Calculate masked crown fraction.
pub fn calculate_crown_fraction_burned<T: CrownNumber>(surface: T, critical: T, crowning: T) -> T {
    let denom = crowning.sub(critical);
    T::select(
        denom.greater(T::scalar(1e-7)),
        surface.sub(critical).div(denom),
        T::scalar(0.),
    )
    .clip_unit()
}
/// Calculate masked canopy heat and intensity.
pub fn calculate_crown_heat_and_intensity<T: CrownNumber>(
    heat: T,
    rate: T,
    base: T,
    height: T,
    density: T,
) -> CrownHeat<T> {
    let load = density.mul(height.sub(base)).maximum(T::scalar(0.));
    let canopy = load.mul(T::scalar(8000.));
    let crown = heat.add(canopy);
    let intensity = rate.div(T::scalar(60.)).mul(crown);
    CrownHeat {
        crown_fuel_load: load,
        canopy_heat_per_unit_area: canopy,
        crown_fire_heat_per_unit_area: crown,
        crown_fire_line_intensity: intensity,
        crown_flame_length: flame(intensity),
    }
}
/// Calculate masked passive behavior.
pub fn calculate_passive_crown_fire_behavior<T: CrownNumber>(
    rate: T,
    heat: T,
    crown: T,
    canopy: T,
    fraction: T,
) -> PassiveCrownBehavior<T> {
    let passive_rate = rate.add(fraction.mul(crown.sub(rate)));
    let passive_heat = heat.add(fraction.mul(canopy));
    let intensity = passive_heat.mul(passive_rate).div(T::scalar(60.));
    PassiveCrownBehavior {
        passive_crown_fire_spread_rate: passive_rate,
        passive_crown_fire_heat_per_unit_area: passive_heat,
        passive_crown_fire_line_intensity: intensity,
        passive_crown_fire_flame_length: flame(intensity),
    }
}
/// Calculate masked critical surface rate.
pub fn calculate_surface_fire_critical_spread_rate<T: CrownNumber>(critical: T, heat: T) -> T {
    let safe = T::select(heat.greater(T::scalar(1e-7)), heat, T::scalar(1.));
    T::select(
        heat.greater(T::scalar(1e-7)),
        T::scalar(60.).mul(critical).div(safe),
        T::scalar(0.),
    )
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
