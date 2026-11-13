//! Typed operation dispatch; numerical work belongs to the standalone core.
use crate::component_columns::RowWriter;
use crate::operation::{Family, Operation};
use crate::prepared::PreparedArguments;
use behave_fire::crown;

pub(crate) fn widen<T: Into<f64>>(value: T) -> f64 {
    value.into()
}
pub(crate) fn scalar(out: &mut RowWriter<'_, '_>, value: f64) {
    out.f64("", value);
}
pub(crate) fn named(
    out: &mut RowWriter<'_, '_>,
    values: impl IntoIterator<Item = (&'static str, f64)>,
) {
    for (name, value) in values {
        out.f64(name, value);
    }
}
pub(crate) fn classification(out: &mut RowWriter<'_, '_>, value: crown::FireClassification) {
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
    out: &mut RowWriter<'_, '_>,
) -> Result<(), String> {
    let result = match operation.family() {
        Family::Surface => crate::surface::evaluate(operation, a, index, out)?,
        Family::Crown => crate::crown_dispatch::evaluate(operation, a, index, out)?,
        Family::Safety | Family::Spot | Family::Ignite | Family::Mortality => {
            crate::effects::evaluate(operation, a, index, out)?
        }
        Family::VaporPressureDeficitCalculator | Family::FineDeadFuelMoistureTool => {
            crate::tools::evaluate(operation, a, index, out)?
        }
        Family::Units => {
            return Err("unit program must be prepared before component iteration".into())
        }
        Family::Model => crate::model::evaluate(operation, a, index, out)?,
    };
    result.ok_or_else(|| format!("Native operation is unavailable: {}", operation.name()))
}
