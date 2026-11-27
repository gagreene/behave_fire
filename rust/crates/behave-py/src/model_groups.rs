//! Capture and validate every dependency group before one detached execution.
use behave_fire::GridShape;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::{BTreeMap, BTreeSet};

struct OwnedGroup {
    fields: Vec<String>,
    shape: Vec<usize>,
    grid: GridShape,
    values: BTreeMap<String, Vec<f64>>,
    controls: crate::model_f64::TypedControls,
}
fn item<'py>(group: &Bound<'py, PyDict>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    group
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("missing group {name}")))
}

#[pyfunction]
#[pyo3(signature=(kind, groups, registry, settings, threads=1, _owned_callback=None))]
pub fn model_groups<'py>(
    py: Python<'py>,
    kind: &str,
    groups: &Bound<'py, PyList>,
    registry: &Bound<'py, PyDict>,
    settings: BTreeMap<String, u8>,
    threads: usize,
    _owned_callback: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let fuels = crate::model_f64::fuels(registry)?;
    let options = crate::model_f64::options(settings)?;
    let mut owned = Vec::new();
    let mut all_outputs = BTreeSet::new();
    for value in groups.iter() {
        let group = value.cast::<PyDict>()?;
        let fields: Vec<String> = item(group, "outputs")?.extract()?;
        for name in &fields {
            if !all_outputs.insert(name.clone()) {
                return Err(PyValueError::new_err(
                    "duplicate output across model groups",
                ));
            }
        }
        let shape: Vec<usize> = item(group, "shape")?.extract()?;
        let grid = crate::model_fields::checked_shape(shape.clone())?;
        let input = item(group, "inputs")?;
        let input = input.cast::<PyDict>()?;
        let mut values = crate::model_f64::physical(input, &grid)?;
        if let Some(descriptors) = group.get_item("conversions")? {
            crate::model_f64::conversions(&mut values, Some(descriptors.cast::<PyDict>()?), &grid)?;
        }
        if group
            .get_item("derive_ratio")?
            .map(|v| v.extract::<bool>())
            .transpose()?
            .unwrap_or(false)
        {
            crate::model_f64::derive_ratio(&mut values)?;
        }
        let controls = item(group, "controls")?;
        let controls = crate::model_f64::controls(controls.cast::<PyDict>()?, &grid)?;
        owned.push(OwnedGroup {
            fields,
            shape,
            grid,
            values,
            controls,
        });
    }
    let mut ready = Vec::new();
    for group in &owned {
        ready.push(crate::model_fields::ReadyGroup::new(
            kind,
            &group.fields,
            &group.values,
            &group.controls,
            &fuels,
            group.grid.clone(),
            options,
        )?);
    }
    if let Some(callback) = _owned_callback {
        callback.call0()?;
    }
    let results = py
        .detach(|| {
            ready
                .iter()
                .map(|group| group.execute(threads))
                .collect::<Result<Vec<_>, String>>()
        })
        .map_err(PyValueError::new_err)?;
    let result = PyDict::new(py);
    for (batch, group) in results.into_iter().zip(&owned) {
        let fields = crate::model_f64::export(py, batch, &group.shape)?;
        for (name, value) in fields.iter() {
            result.set_item(name, value)?;
        }
    }
    Ok(result)
}
