//! Python-owned staged sessions; no Python references survive detached calculation.
use crate::model_f64 as model;
use behave_fire::engine::{Field, FieldKind, GridSession, OutputRequest, Stage, SurfaceSource};
use behave_fire::output::ColumnValues;
use behave_fire::raster::FuelbedColumns;
use behave_fire::PerCell;
use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{Element, IntoPyArray};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;

/// Preserve a structural core error as a Python value error.
fn error(e: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(e.to_string())
}
/// Resolve one public session field name before detached calculation.
fn field(name: &str) -> PyResult<Field> {
    Field::from_name(name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown grid field {name}")))
}
/// Resolve an output or retention request without evaluating science.
fn fields(names: Vec<String>) -> PyResult<Vec<Field>> {
    names.iter().map(|name| field(name)).collect()
}
/// Resolve a public staged-execution name.
fn stage(name: &str) -> PyResult<Stage> {
    match name {
        "particles" => Ok(Stage::Particles),
        "fuelbed" => Ok(Stage::Fuelbed),
        "reaction" => Ok(Stage::Reaction),
        "surface_wind" => Ok(Stage::SurfaceWind),
        "surface_spread" => Ok(Stage::SurfaceSpread),
        "surface_geometry" => Ok(Stage::SurfaceGeometry),
        "crown_spread" => Ok(Stage::CrownSpread),
        "crown_critical" => Ok(Stage::CrownCritical),
        "crown_heat" => Ok(Stage::CrownHeat),
        "crown_ratios" => Ok(Stage::CrownRatios),
        "crown_fraction" => Ok(Stage::CrownFraction),
        "crown_passive" => Ok(Stage::CrownPassive),
        "crown_final" => Ok(Stage::CrownFinal),
        _ => Err(PyValueError::new_err(format!("unknown grid stage {name}"))),
    }
}
/// Copy a typed caller column into native-owned session storage.
fn column(field: Field, value: &Bound<'_, PyAny>) -> PyResult<ColumnValues> {
    Ok(match field.kind() {
        FieldKind::Physical => ColumnValues::Float64(model::array(value, field.name())?),
        FieldKind::Code => ColumnValues::Int32(model::array(value, field.name())?),
        FieldKind::Flag => ColumnValues::Bool(model::array(value, field.name())?),
    })
}
/// Copy a retained column to Python, flattening only unsupported transport ranks.
fn array<'py, T: Element + Copy>(
    py: Python<'py>,
    shape: &[usize],
    values: &[T],
) -> PyResult<Bound<'py, PyAny>> {
    let count = shape.iter().product();
    let values = if values.len() == 1 {
        vec![values[0]; count]
    } else {
        values.to_vec()
    };
    // rust-numpy's ndarray conversion has a 32-axis ceiling. Transfer a flat
    // owned array above that ceiling; the public wrapper restores logical axes.
    let flat_shape = [count];
    let numpy_shape = if shape.len() > 32 {
        &flat_shape[..]
    } else {
        shape
    };
    Ok(ArrayD::from_shape_vec(IxDyn(numpy_shape), values)
        .map_err(error)?
        .into_pyarray(py)
        .into_any())
}

/// The complete independently owned core snapshot, guarded by PyO3's mutable borrow.
#[pyclass(name = "GridSession", module = "behave_fire._rust")]
pub struct NativeGridSession {
    inner: GridSession,
}

#[pymethods]
impl NativeGridSession {
    #[new]
    #[pyo3(signature=(kind, inputs, controls, registry, shape, settings, outputs, keep, threads=1, fuelbed=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kind: &str,
        inputs: &Bound<'_, PyDict>,
        controls: &Bound<'_, PyDict>,
        registry: &Bound<'_, PyDict>,
        shape: Vec<usize>,
        settings: BTreeMap<String, u8>,
        outputs: Vec<String>,
        keep: Vec<String>,
        threads: usize,
        fuelbed: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let grid = crate::model_fields::checked_shape(shape)?;
        let values = model::physical(inputs, &grid)?;
        let controls = model::controls(controls, &grid)?;
        let fuels = model::fuels(registry)?;
        let options = model::options(settings)?;
        let mut bed = BTreeMap::new();
        let mut defined = None;
        let source = if let Some(source) = fuelbed {
            if kind != "surface" {
                return Err(PyValueError::new_err(
                    "only Surface accepts supplied fuelbeds",
                ));
            }
            for (key, value) in source.iter() {
                let name: String = key.extract()?;
                if !matches!(
                    name.as_str(),
                    "sigma"
                        | "bulk_density"
                        | "packing_ratio"
                        | "relative_packing_ratio"
                        | "heat_sink"
                        | "propagating_flux"
                        | "w_heat_dead"
                        | "w_heat_live"
                        | "w_silica_dead"
                        | "w_silica_live"
                        | "w_moist_dead"
                        | "w_moist_live"
                        | "wfl_dead"
                        | "wfl_live"
                        | "moe_dead"
                        | "moe_live"
                        | "frac_dead"
                        | "frac_live"
                        | "depth"
                        | "is_defined"
                ) {
                    return Err(PyValueError::new_err(format!(
                        "unknown supplied fuelbed column {name}"
                    )));
                }
                if name == "is_defined" {
                    defined = Some(model::array::<bool>(&value, &name)?);
                } else {
                    bed.insert(name.clone(), model::array::<f64>(&value, &name)?);
                }
            }
            let v = defined
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("missing fuelbed is_defined"))?;
            let p = |name| model::column(&bed, name);
            let columns = FuelbedColumns {
                sigma: p("sigma")?,
                bulk_density: p("bulk_density")?,
                packing_ratio: p("packing_ratio")?,
                relative_packing_ratio: p("relative_packing_ratio")?,
                heat_sink: p("heat_sink")?,
                propagating_flux: p("propagating_flux")?,
                w_heat_dead: p("w_heat_dead")?,
                w_heat_live: p("w_heat_live")?,
                w_silica_dead: p("w_silica_dead")?,
                w_silica_live: p("w_silica_live")?,
                w_moist_dead: p("w_moist_dead")?,
                w_moist_live: p("w_moist_live")?,
                wfl_dead: p("wfl_dead")?,
                wfl_live: p("wfl_live")?,
                moe_dead: p("moe_dead")?,
                moe_live: p("moe_live")?,
                frac_dead: p("frac_dead")?,
                frac_live: p("frac_live")?,
                depth: p("depth")?,
                is_defined: if v.len() == 1 {
                    PerCell::Scalar(v[0])
                } else {
                    PerCell::Cells(v)
                },
            };
            Some(SurfaceSource::Fuelbeds(columns))
        } else {
            None
        };
        let run = model::configured_run(
            kind,
            grid.clone(),
            &values,
            &controls,
            &fuels,
            options,
            source.as_ref(),
        )?;
        let inner = GridSession::new(
            &run,
            OutputRequest::new(&fields(outputs)?),
            &fields(keep)?,
            threads,
        )
        .map_err(error)?;
        Ok(Self { inner })
    }
    fn __reduce__(&self) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "Grid sessions cannot be pickled; reconstruct from inputs.",
        ))
    }
    fn clear(&mut self, name: &str, persistent: bool) -> PyResult<()> {
        let f = field(name)?;
        if persistent {
            self.inner.clear_persistent(f)
        } else {
            self.inner.clear_injection(f)
        }
        .map_err(error)
    }
    fn get<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        let f = field(name)?;
        let view = self.inner.get(f).map_err(error)?;
        let shape = self.inner.shape().dimensions();
        match f.kind() {
            FieldKind::Physical => array(py, shape, view.physical().map_err(error)?),
            FieldKind::Code => array(py, shape, view.code().map_err(error)?),
            FieldKind::Flag => array(py, shape, view.flag().map_err(error)?),
        }
    }
    fn reset(&mut self) {
        self.inner.reset();
    }
    fn run_fused(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .run_fused_observed(&crate::model_observer::Observer)
        })
        .map_err(error)
    }
    fn run_remaining(&mut self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            self.inner
                .run_remaining_observed(&crate::model_observer::Observer)
        })
        .map_err(error)
    }
    fn run_stage(&mut self, py: Python<'_>, name: &str) -> PyResult<()> {
        let s = stage(name)?;
        py.detach(|| {
            self.inner
                .run_stage_observed(s, &crate::model_observer::Observer)
        })
        .map_err(error)
    }
    fn set(&mut self, name: &str, value: &Bound<'_, PyAny>, persistent: bool) -> PyResult<()> {
        let f = field(name)?;
        let value = column(f, value)?;
        if persistent {
            return self.inner.persistent(f, value).map_err(error);
        }
        match value {
            ColumnValues::Float64(v) => self.inner.set_physical(f, v),
            ColumnValues::Int32(v) => self.inner.set_code(f, v),
            ColumnValues::Bool(v) => self.inner.set_flag(f, v),
        }
        .map_err(error)
    }
    fn set_fuels(&mut self, registry: &Bound<'_, PyDict>) -> PyResult<()> {
        self.inner.set_fuels(model::fuels(registry)?).map_err(error)
    }
    fn set_outputs(&mut self, outputs: Vec<String>, keep: Vec<String>) -> PyResult<()> {
        self.inner
            .set_outputs(OutputRequest::new(&fields(outputs)?), &fields(keep)?)
            .map_err(error)
    }
    fn set_threads(&mut self, threads: usize) {
        self.inner.set_threads(threads);
    }
}

#[pyfunction]
pub fn grid_fields() -> Vec<(&'static str, &'static str)> {
    Field::ALL
        .iter()
        .map(|&f| {
            (
                f.name(),
                match f.kind() {
                    FieldKind::Physical => "physical",
                    FieldKind::Code => "code",
                    FieldKind::Flag => "flag",
                },
            )
        })
        .collect()
}
