//! Typed operation dispatch; numerical work belongs to the standalone core.
use crate::column_output::RowWriter;
use crate::operation::{Family, Operation};
use crate::prepared::PreparedArguments;
use behave_fire::crown;

pub(crate) fn widen<T: Into<f64>>(value: T) -> f64 {
    value.into()
}
pub(crate) fn scalar(out: &mut RowWriter<'_>, value: f64) {
    out.f64("", value);
}
pub(crate) fn named(
    out: &mut RowWriter<'_>,
    values: impl IntoIterator<Item = (&'static str, f64)>,
) {
    for (name, value) in values {
        out.f64(name, value);
    }
}
pub(crate) fn classification(out: &mut RowWriter<'_>, value: crown::FireClassification) {
    out.i32("fire_type", value.fire_type);
    out.boolean("is_surface_fire", value.is_surface_fire);
    out.boolean("is_passive_crown", value.is_passive_crown);
    out.boolean("is_active_crown", value.is_active_crown);
}

#[allow(deprecated)]
pub fn evaluate(
    operation: Operation,
    a: &PreparedArguments<'_>,
    index: usize,
    single: bool,
    out: &mut RowWriter<'_>,
) -> Result<(), String> {
    let ordinary_crown = operation == Operation::CrownRunCrownFire
        && !a.has_masks()
        && !a.has_single()
        && !a.has_integers()
        && !a.has_crown_python_payloads();
    if !ordinary_crown
        && (a.has_masks() || a.has_single() || a.has_integers() || a.has_python_payloads())
    {
        if crate::crown_masked::evaluate(operation, a, index, out)?.is_some() {
            return Ok(());
        }
        if crate::masked_records::evaluate(operation, a, index, out)?.is_some() {
            return Ok(());
        }
        if crate::masked::evaluate(operation, a, index, out)?.is_some() {
            return Ok(());
        }
    }
    let result = match operation.family() {
        Family::Surface => crate::surface::evaluate(operation, a, index, single, out)?,
        Family::Crown => crate::crown_dispatch::evaluate(operation, a, index, single, out)?,
        Family::Safety | Family::Spot | Family::Ignite | Family::Mortality => {
            crate::effects::evaluate(operation, a, index, single, out)?
        }
        Family::VaporPressureDeficitCalculator | Family::FineDeadFuelMoistureTool => {
            crate::tools::evaluate(operation, a, index, single, out)?
        }
        Family::Contain => crate::contain::evaluate(operation, a, index, single, out)?,
        Family::Units => crate::units::evaluate(operation, a, index, single, out)?,
        Family::Model => crate::model::evaluate(operation, a, index, single, out)?,
    };
    result.ok_or_else(|| format!("Native operation is unavailable: {}", operation.name()))
}
