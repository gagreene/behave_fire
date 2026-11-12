//! Real particle dependency closures reuse the staged core's fuelbed equations.
use crate::{
    component_columns::Columns,
    prepared::{Field as Input, PreparedArguments},
};
use behave_fire::{
    engine::{CellState, Field, FieldKind, Stage},
    ParallelError,
};

fn field(name: &str) -> Result<Field, String> {
    Field::from_name(&format!("ib.{name}")).ok_or_else(|| format!("unknown fuelbed field {name}"))
}
fn visit(field: Field, inputs: &mut Vec<(Field, Input)>) -> Result<(), String> {
    if field.producer() == Some(Stage::Fuelbed) {
        for &dependency in field.dependencies() {
            visit(dependency, inputs)?;
        }
    } else if !inputs.iter().any(|(prior, _)| *prior == field) {
        let input = Input::from_name(field.name())
            .ok_or_else(|| format!("unknown particle input {}", field.name()))?;
        inputs.push((field, input));
    }
    Ok(())
}
pub(crate) fn dependencies(names: &[String]) -> Result<Vec<(Field, Input)>, String> {
    let mut result = Vec::new();
    for name in names {
        visit(field(name)?, &mut result)?;
    }
    Ok(result)
}
fn value(field: Field, state: &mut CellState) -> Result<f64, String> {
    if state.is_present(field) {
        return state.physical(field).map_err(|e| e.to_string());
    }
    let result =
        behave_fire::engine::evaluate_fuelbed_field(field, |dependency| value(dependency, state))
            .ok_or_else(|| format!("unbound particle field {}", field.name()))??;
    state
        .set_physical(field, result)
        .map_err(|e| e.to_string())?;
    Ok(result)
}
pub(crate) fn run(
    p: &PreparedArguments<'_>,
    names: &[String],
    output: &mut Columns,
    threads: usize,
) -> Result<(), ParallelError<String>> {
    let fields = names
        .iter()
        .map(|name| field(name))
        .collect::<Result<Vec<_>, _>>()
        .map_err(ParallelError::Work)?;
    let inputs = dependencies(names).map_err(ParallelError::Work)?;
    output.fill(threads, |index, row| {
        let mut state = CellState::new();
        for &(field, input) in &inputs {
            if field.kind() == FieldKind::Flag {
                state
                    .set_flag(field, p.flag(input, index)?)
                    .map_err(|e| e.to_string())?;
            } else {
                state
                    .set_physical(field, p.n(input, index)?)
                    .map_err(|e| e.to_string())?;
            }
        }
        for &field in &fields {
            if field == Field::FuelbedDefined {
                row.boolean(
                    "is_defined",
                    state
                        .flag(Field::ParticleDefined)
                        .map_err(|e| e.to_string())?,
                );
            } else {
                row.f64("", value(field, &mut state)?);
            }
        }
        Ok(())
    })
}
