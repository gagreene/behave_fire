//! Dependency-restricted primitive groups; unbound inputs are errors, never filler values.
#[cfg(feature = "allocation-probe")]
use crate::allocation_probe::Observer;
use behave_fire::batch::{CrownBatch, SurfaceBatch};
use behave_fire::engine::{
    CrownPlan, Field, ModelBatch, OutputRequest, PreparedFuels, SurfaceOptions,
};
#[cfg(not(feature = "allocation-probe"))]
use behave_fire::output::NoopObserver as Observer;
use behave_fire::output::{RunObserver, RunPhase};
use behave_fire::{GridShape, PerCell};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;
use std::convert::Infallible;

pub(crate) enum Program<'a> {
    Crown(crate::crown_fields::Program<'a>),
    Surface(crate::surface_fields::Program<'a>),
}
pub(crate) struct ReadyGroup<'a> {
    shape: GridShape,
    request: OutputRequest,
    program: Program<'a>,
    validity: Option<PerCell<'a, bool>>,
}
pub(crate) fn checked_shape(shape: Vec<usize>) -> PyResult<GridShape> {
    let extent = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d.max(1)))
        .and_then(|n| n.checked_mul(8));
    if shape.len() > 64 || extent.map_or(true, |n| n > isize::MAX as usize) {
        return Err(PyValueError::new_err("invalid model field extent"));
    }
    GridShape::new(shape).map_err(|e| PyValueError::new_err(e.to_string()))
}
impl<'a> ReadyGroup<'a> {
    pub(crate) fn new(
        kind: &str,
        names: &[String],
        values: &'a BTreeMap<String, Vec<f64>>,
        controls: &'a crate::model_f64::TypedControls,
        fuels: &'a PreparedFuels,
        shape: GridShape,
        options: SurfaceOptions,
    ) -> PyResult<Self> {
        let allowed = match kind {
            "surface" => OutputRequest::surface(),
            "crown" => OutputRequest::crown(),
            _ => return Err(PyValueError::new_err("unsupported restricted model family")),
        };
        let mut fields = Vec::new();
        for name in names {
            let field = Field::from_name(name)
                .ok_or_else(|| PyValueError::new_err(format!("unknown model field {name}")))?;
            if !allowed.fields().contains(&field) {
                return Err(PyValueError::new_err(format!(
                    "invalid model output {name}"
                )));
            }
            fields.push(field);
        }
        let program = if kind == "crown" {
            Program::Crown(crate::crown_fields::Program::new(
                values,
                &fields,
                fuels,
                CrownPlan::new(options),
            )?)
        } else {
            let ids = controls
                .ids
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("missing fuel_model_grid"))?;
            let ids = if ids.len() == 1 {
                PerCell::Scalar(ids[0])
            } else {
                PerCell::Cells(ids)
            };
            Program::Surface(crate::surface_fields::Program::new(
                values, ids, &fields, fuels, options,
            )?)
        };
        let validity = controls.validity.as_ref().map(|v| {
            if v.len() == 1 {
                PerCell::Scalar(v[0])
            } else {
                PerCell::Cells(v)
            }
        });
        Ok(Self {
            shape,
            request: OutputRequest::new(&fields),
            program,
            validity,
        })
    }
    pub(crate) fn execute(&self, threads: usize) -> Result<ModelBatch, String> {
        let valid = |index| {
            self.validity
                .map_or(true, |v| *v.get(index).expect("validated validity"))
        };
        match &self.program {
            Program::Crown(program) => CrownBatch::run_with_request_observed(
                self.shape.clone(),
                &self.request,
                threads,
                |i| {
                    Ok::<_, Infallible>(if valid(i) {
                        Observer.observe(RunPhase::Kernel, || program.evaluate(i))
                    } else {
                        program.missing()
                    })
                },
                &Observer,
            )
            .map(ModelBatch::Crown)
            .map_err(|e| e.to_string()),
            Program::Surface(program) => SurfaceBatch::run_with_request_observed(
                self.shape.clone(),
                &self.request,
                threads,
                |i| {
                    Ok::<_, Infallible>(if valid(i) {
                        Observer.observe(RunPhase::Kernel, || program.evaluate(i))
                    } else {
                        program.missing()
                    })
                },
                &Observer,
            )
            .map(ModelBatch::Surface)
            .map_err(|e| e.to_string()),
        }
    }
}

#[pyfunction]
#[pyo3(signature=(kind, requested, inputs, control_values, registry, shape, settings, threads=1, _owned_callback=None))]
#[allow(clippy::too_many_arguments)]
pub fn model_fields<'py>(
    py: Python<'py>,
    kind: &str,
    requested: Vec<String>,
    inputs: &Bound<'py, PyDict>,
    control_values: &Bound<'py, PyDict>,
    registry: &Bound<'py, PyDict>,
    shape: Vec<usize>,
    settings: BTreeMap<String, u8>,
    threads: usize,
    _owned_callback: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let grid = checked_shape(shape.clone())?;
    let values = crate::model_f64::physical(inputs, &grid)?;
    let controls = crate::model_f64::controls(control_values, &grid)?;
    let fuels = crate::model_f64::fuels(registry)?;
    let options = crate::model_f64::options(settings)?;
    let ready = ReadyGroup::new(kind, &requested, &values, &controls, &fuels, grid, options)?;
    if let Some(callback) = _owned_callback {
        callback.call0()?;
    }
    let output = py
        .detach(|| ready.execute(threads))
        .map_err(PyValueError::new_err)?;
    crate::model_f64::export(py, output, &shape)
}
