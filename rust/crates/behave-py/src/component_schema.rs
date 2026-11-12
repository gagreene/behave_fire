//! Fixed component result columns; no input values or scientific prototype cells.
use crate::operation::Operation;
use behave_fire::{crown, engine::FieldKind, surface};

/// One column descriptor, allocated once before output storage or worker setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Column {
    pub(crate) name: String,
    pub(crate) kind: FieldKind,
}

fn column(name: &str, kind: FieldKind) -> Column {
    Column {
        name: name.into(),
        kind,
    }
}
fn physical<const N: usize>(values: [(&str, f64); N]) -> Vec<Column> {
    values
        .into_iter()
        .map(|(name, _)| column(name, FieldKind::Physical))
        .collect()
}
fn names(values: &[&str]) -> Vec<Column> {
    values
        .iter()
        .map(|name| column(name, FieldKind::Physical))
        .collect()
}
fn particles() -> Vec<Column> {
    // Only the fixed record's names and array extents are read. The zero payloads
    // are discarded, and no particle/fuelbed or other scientific function runs.
    let record = surface::Particles::default();
    let mut result = physical(record.named_values());
    result.push(column("is_defined", FieldKind::Flag));
    for (name, values) in record.named_arrays() {
        for index in 0..values.len() {
            result.push(column(&format!("{name}.{index}"), FieldKind::Physical));
        }
    }
    result
}
fn fuelbed() -> Vec<Column> {
    let mut result = physical(surface::FuelbedIntermediates::default().named_values());
    result.push(column("is_defined", FieldKind::Flag));
    result
}
fn prefix(mut fields: Vec<Column>, prefix: &str) -> Vec<Column> {
    for field in &mut fields {
        field.name = format!("{prefix}.{}", field.name);
    }
    fields
}

/// Resolve the complete typed output layout without reading any argument or row.
pub(crate) fn schema(operation: Operation) -> Vec<Column> {
    use Operation::*;
    match operation {
        SurfaceBuildParticleArrays => particles(),
        SurfaceCalculateFuelbedIntermediates => fuelbed(),
        SurfaceCalculateSurfaceRateVector => {
            physical(surface::RateVector::default().named_values())
        }
        SurfaceCalculateSurfaceWindCoefficients => {
            physical(surface::WindCoefficients::default().named_values())
        }
        SurfaceRunSurfaceFire | ModelRunSurface => {
            physical(surface::SurfaceResult::default().named_values())
        }
        CrownCalculateCrownHeatAndIntensity => {
            physical(crown::CrownHeat::<f64>::default().named_values())
        }
        CrownCalculateCrownFireRatios => {
            physical(crown::CrownRatios::<f64>::default().named_values())
        }
        CrownCalculatePassiveCrownFireBehavior => {
            physical(crown::PassiveCrownBehavior::<f64>::default().named_values())
        }
        CrownAssignFinalCrownFireBehavior => {
            physical(crown::FinalCrownBehavior::<f64>::default().named_values())
        }
        CrownClassifyCrownFireType => vec![
            column("fire_type", FieldKind::Code),
            column("is_surface_fire", FieldKind::Flag),
            column("is_passive_crown", FieldKind::Flag),
            column("is_active_crown", FieldKind::Flag),
        ],
        CrownCalculateCrownSurfaceFire => {
            let mut fields = prefix(particles(), "p_crown");
            fields.extend(prefix(fuelbed(), "ib_crown"));
            fields.extend(prefix(
                physical(surface::SurfaceResult::default().named_values()),
                "crown_surface",
            ));
            fields.extend(names(&[
                "ri_crown",
                "crown_fire_spread_rate",
                "crown_length_to_width_ratio",
            ]));
            fields
        }
        CrownRunCrownFire => {
            let mut fields = physical(crown::CrownResult::default().named_values());
            fields.push(column("fire_type", FieldKind::Code));
            fields
        }
        ContainRunContainSimArray => {
            let mut fields = vec![column("status", FieldKind::Code)];
            fields.extend(names(&["contained_time", "final_perimeter", "final_area"]));
            fields
        }
        MortalityCalculateCrownScorchMortality => names(&[
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality",
        ]),
        SafetyCalculateSafetyZone => names(&["separation_distance", "radius", "area"]),
        VaporPressureDeficitCalculatorCalculateVpd => names(&["vpd", "actual_vp", "saturated_vp"]),
        _ => names(&[""]),
    }
}
