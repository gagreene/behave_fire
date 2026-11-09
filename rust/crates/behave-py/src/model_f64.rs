//! Direct normalized model boundary; no legacy numeric tags enter cell loops.
use crate::component_units::Convert;
use behave_fire::engine::{
    run_model_observed, CrownPlan, ModelBatch, ModelRun, OutputRequest, PreparedFuel,
    PreparedFuels, SurfaceOptions, SurfacePlan, SurfaceSource, WafMethod, WindHeight,
    WindOrientation,
};
use behave_fire::output::ColumnValues;
use behave_fire::raster::{
    CrownColumns, CrownRaster, CrownTweakColumns, SurfaceColumns, SurfaceRaster,
    SurfaceTweakColumns,
};
use behave_fire::units::{DensityUnit, FractionUnit, LengthUnit, SlopeUnit, SpeedUnit};
use behave_fire::{GridShape, PerCell};
use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{Element, IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Default)]
struct Timings {
    kernel: AtomicU64,
    encoder: AtomicU64,
}
impl behave_fire::output::RunObserver for Timings {
    fn observe<T>(&self, phase: behave_fire::output::RunPhase, work: impl FnOnce() -> T) -> T {
        let started = Instant::now();
        #[cfg(feature = "allocation-probe")]
        let _scope = crate::allocation_probe::Scope::enter(match phase {
            behave_fire::output::RunPhase::Kernel => crate::allocation_probe::KERNEL,
            behave_fire::output::RunPhase::Encoder => crate::allocation_probe::ENCODER,
        });
        let value = work();
        let elapsed = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let counter = match phase {
            behave_fire::output::RunPhase::Kernel => &self.kernel,
            behave_fire::output::RunPhase::Encoder => &self.encoder,
        };
        counter.fetch_add(elapsed, Ordering::Relaxed);
        value
    }
}

pub(crate) struct TypedControls {
    pub(crate) ids: Option<Vec<i32>>,
    pub(crate) validity: Option<Vec<bool>>,
}

fn convert<T: Convert>(values: &mut Vec<f64>, codes: &[i32]) -> PyResult<()> {
    let units = codes
        .iter()
        .map(|&code| T::try_from(code).map_err(|e| PyValueError::new_err(e.to_string())))
        .collect::<PyResult<Vec<_>>>()?;
    if units.is_empty() || values.is_empty() {
        values.clear();
        return Ok(());
    }
    if values.len() == 1 && units.len() > 1 {
        let value = values[0];
        values
            .try_reserve_exact(units.len() - 1)
            .map_err(|e| pyo3::exceptions::PyMemoryError::new_err(e.to_string()))?;
        values.resize(units.len(), value);
    }
    if units.len() != 1 && units.len() != values.len() {
        return Err(PyValueError::new_err(
            "unit selector length differs from values",
        ));
    }
    for (index, value) in values.iter_mut().enumerate() {
        *value = units[if units.len() == 1 { 0 } else { index }].convert(*value, true);
    }
    Ok(())
}

pub(crate) fn conversions(
    values: &mut BTreeMap<String, Vec<f64>>,
    descriptors: Option<&Bound<'_, PyDict>>,
    shape: &GridShape,
) -> PyResult<()> {
    let Some(descriptors) = descriptors else {
        return Ok(());
    };
    for (name, descriptor) in descriptors.iter() {
        let name: String = name.extract()?;
        let family: String = descriptor.get_item(0)?.extract()?;
        let codes = array::<i32>(
            &descriptor.get_item(1)?,
            &format!("unit selector for {name}"),
        )?;
        if codes.len() != 1 && codes.len() != shape.len() {
            return Err(PyValueError::new_err(
                "wrong normalized unit-selector length",
            ));
        }
        let target = values
            .get_mut(&name)
            .ok_or_else(|| PyValueError::new_err(format!("missing conversion input {name}")))?;
        match family.as_str() {
            "Density" => convert::<DensityUnit>(target, &codes)?,
            "Fraction" => convert::<FractionUnit>(target, &codes)?,
            "Length" => convert::<LengthUnit>(target, &codes)?,
            "Slope" => convert::<SlopeUnit>(target, &codes)?,
            "Speed" => convert::<SpeedUnit>(target, &codes)?,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "unsupported model unit family {family}"
                )))
            }
        }
    }
    Ok(())
}

pub(crate) fn array<T: Element + Copy>(value: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<T>> {
    let ndarray = value.py().import("numpy")?.getattr("ndarray")?;
    if !value.get_type().is(&ndarray) {
        return Err(PyTypeError::new_err(format!(
            "{name} must be a normalized plain ndarray; masks must be handled before this boundary"
        )));
    }
    if !value
        .getattr("flags")?
        .getattr("aligned")?
        .extract::<bool>()?
    {
        return Err(PyTypeError::new_err(format!(
            "{name} must have aligned normalized storage"
        )));
    }
    if !value
        .getattr("dtype")?
        .getattr("isnative")?
        .extract::<bool>()?
    {
        return Err(PyTypeError::new_err(format!(
            "{name} must have normalized native-endian storage"
        )));
    }
    // rust-numpy's borrowed ndarray view supports at most 32 axes. Flatten
    // normalized base arrays while attached; shape restoration uses the checked
    // logical grid and never needs the original Python storage after detach.
    let flattened = if value.getattr("ndim")?.extract::<usize>()? > 32 {
        Some(value.call_method1("ravel", ("C",))?)
    } else {
        None
    };
    let value = flattened.as_ref().unwrap_or(value);
    let view = value
        .extract::<PyReadonlyArrayDyn<'_, T>>()
        .map_err(|_| PyTypeError::new_err(format!("wrong normalized dtype for {name}")))?;
    let array = view.as_array();
    // Normalized C-order columns can use a bulk owned copy. Strided private
    // inputs retain logical C-order traversal rather than copying raw memory.
    Ok(if let Some(values) = array.as_slice() {
        values.to_vec()
    } else {
        array.iter().copied().collect()
    })
}

fn required_array<T: Element + Copy>(values: &Bound<'_, PyDict>, name: &str) -> PyResult<Vec<T>> {
    let value = values
        .get_item(name)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {name}")))?;
    array(&value, name)
}

pub(crate) fn physical(
    values: &Bound<'_, PyDict>,
    shape: &GridShape,
) -> PyResult<BTreeMap<String, Vec<f64>>> {
    let mut result = BTreeMap::new();
    for (name, value) in values.iter() {
        let name: String = name.extract()?;
        let data = array::<f64>(&value, &name)?;
        if data.len() != 1 && data.len() != shape.len() {
            return Err(PyValueError::new_err(format!(
                "{name} has wrong normalized column length"
            )));
        }
        result.insert(name, data);
    }
    Ok(result)
}

pub(crate) fn column<'a>(
    values: &'a BTreeMap<String, Vec<f64>>,
    name: &str,
) -> PyResult<PerCell<'a, f64>> {
    let data = values
        .get(name)
        .ok_or_else(|| PyValueError::new_err(format!("missing {name}")))?;
    Ok(if data.len() == 1 {
        PerCell::Scalar(data[0])
    } else {
        PerCell::Cells(data)
    })
}

pub(crate) fn controls(values: &Bound<'_, PyDict>, shape: &GridShape) -> PyResult<TypedControls> {
    let ids = values
        .get_item("fuel_model_grid")?
        .map(|v| array::<i32>(&v, "fuel_model_grid"))
        .transpose()?;
    let valid = values
        .get_item("validity")?
        .map(|v| array::<bool>(&v, "validity"))
        .transpose()?;
    for (name, length) in [
        ("fuel_model_grid", ids.as_ref().map(Vec::len)),
        ("validity", valid.as_ref().map(Vec::len)),
    ] {
        if let Some(length) = length {
            if length != 1 && length != shape.len() {
                return Err(PyValueError::new_err(format!(
                    "{name} has wrong typed column length"
                )));
            }
        }
    }
    Ok(TypedControls {
        ids,
        validity: valid,
    })
}

pub(crate) fn fuels(values: &Bound<'_, PyDict>) -> PyResult<PreparedFuels> {
    let names = [
        "depth",
        "moe_dead",
        "hoc_dead",
        "hoc_live",
        "dead_1h",
        "dead_10h",
        "dead_100h",
        "live_herb",
        "live_woody",
        "savr_1h",
        "savr_lh",
        "savr_lw",
    ];
    let mut columns = Vec::new();
    for name in names {
        columns.push(required_array::<f64>(values, name)?);
    }
    let dynamic = required_array::<bool>(values, "is_dynamic")?;
    let defined = required_array::<bool>(values, "is_defined")?;
    let count = columns[0].len();
    if count == 0
        || columns.iter().any(|c| c.len() != count)
        || dynamic.len() != count
        || defined.len() != count
    {
        return Err(PyValueError::new_err(
            "inconsistent normalized fuel registry lengths",
        ));
    }
    let mut rows = Vec::new();
    rows.try_reserve_exact(count)
        .map_err(|e| pyo3::exceptions::PyMemoryError::new_err(e.to_string()))?;
    for i in 0..count {
        let mut row = PreparedFuel::default();
        row.fuel_bed_depth = columns[0][i];
        row.moisture_of_extinction_dead = columns[1][i];
        row.heat_of_combustion_dead = columns[2][i];
        row.heat_of_combustion_live = columns[3][i];
        row.dead_1h = columns[4][i];
        row.dead_10h = columns[5][i];
        row.dead_100h = columns[6][i];
        row.live_herb = columns[7][i];
        row.live_woody = columns[8][i];
        row.savr_1h = columns[9][i];
        row.savr_live_herb = columns[10][i];
        row.savr_live_woody = columns[11][i];
        row.is_dynamic = dynamic[i];
        row.is_defined = defined[i];
        rows.push(row);
    }
    PreparedFuels::from_rows(rows).map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) fn options(values: BTreeMap<String, u8>) -> PyResult<SurfaceOptions> {
    let orientation = match values.get("wind_orientation").copied().unwrap_or(0) {
        0 => WindOrientation::Upslope,
        1 => WindOrientation::North,
        _ => return Err(PyValueError::new_err("invalid wind orientation")),
    };
    let height = match values.get("wind_height").copied().unwrap_or(0) {
        0 => WindHeight::TwentyFoot,
        1 => WindHeight::TenMetre,
        _ => return Err(PyValueError::new_err("invalid wind height")),
    };
    let waf = match values.get("waf_method").copied().unwrap_or(0) {
        0 => WafMethod::CrownRatio,
        1 => WafMethod::UserInput,
        _ => return Err(PyValueError::new_err("invalid WAF method")),
    };
    Ok(SurfaceOptions {
        wind_orientation: orientation,
        wind_height: height,
        waf_method: waf,
    })
}

pub(crate) fn export<'py>(
    py: Python<'py>,
    batch: ModelBatch,
    shape: &[usize],
) -> PyResult<Bound<'py, PyDict>> {
    let columns = match batch {
        ModelBatch::Surface(v) => v.into_named_columns(),
        ModelBatch::Crown(v) => v.into_named_columns(),
    };
    export_columns(py, columns, shape)
}

/// Move existing primitive column buffers directly into owned NumPy arrays.
pub(crate) fn export_columns<'py, S: AsRef<str>>(
    py: Python<'py>,
    columns: impl IntoIterator<Item = (S, ColumnValues)>,
    shape: &[usize],
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    for (name, values) in columns {
        if shape.len() > 32 {
            // The conversion side has the same rust-numpy 32-axis limitation.
            // Transfer the owning vector once, then restore logical axes with
            // NumPy's checked reshape while attached to the interpreter.
            let flat = match values {
                ColumnValues::Float64(v) => v.into_pyarray(py).into_any(),
                ColumnValues::Int32(v) => v.into_pyarray(py).into_any(),
                ColumnValues::Bool(v) => v.into_pyarray(py).into_any(),
            };
            result.set_item(
                name.as_ref(),
                flat.call_method1("reshape", (shape.to_vec(),))?,
            )?;
            continue;
        }
        match values {
            ColumnValues::Float64(v) => result.set_item(
                name.as_ref(),
                ArrayD::from_shape_vec(IxDyn(shape), v)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py),
            )?,
            ColumnValues::Int32(v) => result.set_item(
                name.as_ref(),
                ArrayD::from_shape_vec(IxDyn(shape), v)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py),
            )?,
            ColumnValues::Bool(v) => result.set_item(
                name.as_ref(),
                ArrayD::from_shape_vec(IxDyn(shape), v)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
                    .into_pyarray(py),
            )?,
        }
    }
    Ok(result)
}

pub(crate) fn derive_ratio(values: &mut BTreeMap<String, Vec<f64>>) -> PyResult<()> {
    let height = values
        .get("canopy_height")
        .ok_or_else(|| PyValueError::new_err("missing canopy_height"))?;
    let base = values
        .get("canopy_base_height")
        .ok_or_else(|| PyValueError::new_err("missing canopy_base_height"))?;
    let count = if height.is_empty() || base.is_empty() {
        0
    } else {
        height.len().max(base.len())
    };
    let mut ratio = Vec::new();
    ratio
        .try_reserve_exact(count)
        .map_err(|e| pyo3::exceptions::PyMemoryError::new_err(e.to_string()))?;
    for index in 0..count {
        ratio.push(behave_fire::canopy_crown_ratio(
            height[if height.len() == 1 { 0 } else { index }],
            base[if base.len() == 1 { 0 } else { index }],
        ));
    }
    values.insert("crown_ratio".into(), ratio);
    Ok(())
}

#[pyfunction]
#[pyo3(signature=(kind, inputs, control_values, registry, shape, settings, threads=1, conversions=None, derive_ratio=false, profile=false, _owned_callback=None))]
#[allow(clippy::too_many_arguments)]
pub fn model_batch<'py>(
    py: Python<'py>,
    kind: &str,
    inputs: &Bound<'py, PyDict>,
    control_values: &Bound<'py, PyDict>,
    registry: &Bound<'py, PyDict>,
    shape: Vec<usize>,
    settings: BTreeMap<String, u8>,
    threads: usize,
    conversions: Option<&Bound<'py, PyDict>>,
    derive_ratio: bool,
    profile: bool,
    _owned_callback: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let started = Instant::now();
    if shape.len() > 64 {
        return Err(PyValueError::new_err("too many model dimensions"));
    }
    let extent = shape
        .iter()
        .try_fold(1usize, |n, &axis| n.checked_mul(axis.max(1)))
        .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()));
    if extent.map_or(true, |bytes| bytes > isize::MAX as usize) {
        return Err(PyValueError::new_err(
            "model shape exceeds NumPy's byte/stride extent limit",
        ));
    }
    let grid = GridShape::new(shape.clone()).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut values = physical(inputs, &grid)?;
    self::conversions(&mut values, conversions, &grid)?;
    if derive_ratio {
        self::derive_ratio(&mut values)?;
    }
    let controls = controls(control_values, &grid)?;
    let fuels = fuels(registry)?;
    let options = options(settings)?;
    let run = configured_run(kind, grid, &values, &controls, &fuels, options)?;
    let request = match run {
        ModelRun::Surface { .. } => OutputRequest::surface(),
        ModelRun::Crown { .. } => OutputRequest::crown(),
    };
    if let Some(callback) = _owned_callback {
        callback.call0()?;
    }
    let copied = started.elapsed().as_secs_f64();
    let timings = Timings::default();
    let detached = Instant::now();
    let result = py
        .detach(|| {
            if profile {
                run_model_observed(&run, &request, threads, &timings)
            } else {
                #[cfg(feature = "allocation-probe")]
                {
                    run_model_observed(&run, &request, threads, &crate::model_observer::Observer)
                }
                #[cfg(not(feature = "allocation-probe"))]
                {
                    behave_fire::engine::run_model(&run, &request, threads)
                }
            }
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let detached = detached.elapsed().as_secs_f64();
    let transfer = Instant::now();
    let result = export(py, result, &shape)?;
    if profile {
        for (name, value) in [
            ("owned_copy_seconds", copied),
            ("detached_seconds", detached),
            (
                "kernel_seconds",
                timings.kernel.load(Ordering::Relaxed) as f64 / 1e9,
            ),
            (
                "encoder_seconds",
                timings.encoder.load(Ordering::Relaxed) as f64 / 1e9,
            ),
            ("numpy_transfer_seconds", transfer.elapsed().as_secs_f64()),
            ("appended_bytes", 0.),
            ("concatenate_seconds", 0.),
        ] {
            result.set_item(
                format!("__timing.{name}"),
                ArrayD::from_elem(IxDyn(&[]), value).into_pyarray(py),
            )?;
        }
    }
    Ok(result)
}

/// Borrow one configured model from validated, independently owned boundary columns.
pub(crate) fn configured_run<'a>(
    kind: &str,
    grid: GridShape,
    values: &'a BTreeMap<String, Vec<f64>>,
    controls: &'a TypedControls,
    fuels: &'a PreparedFuels,
    options: SurfaceOptions,
) -> PyResult<ModelRun<'a>> {
    let TypedControls { ids, validity } = controls;
    // These operands are not read under their selected configuration. Omit them
    // from the required scientific columns rather than inventing dependencies.
    let aspect = if options.wind_orientation == WindOrientation::North {
        column(values, "aspect")?
    } else {
        values
            .get("aspect")
            .map(|_| column(values, "aspect"))
            .transpose()?
            .unwrap_or(PerCell::Scalar(0.))
    };
    let validity = validity.as_ref().map(|v| {
        if v.len() == 1 {
            PerCell::Scalar(v[0])
        } else {
            PerCell::Cells(v)
        }
    });
    Ok(match kind {
        "surface" => {
            let ids = ids
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("missing fuel_model_grid"))?;
            let fields = SurfaceColumns {
                fuel_model_grid: if ids.len() == 1 {
                    PerCell::Scalar(ids[0])
                } else {
                    PerCell::Cells(ids)
                },
                m1h: column(values, "m1h")?,
                m10h: column(values, "m10h")?,
                m100h: column(values, "m100h")?,
                mlh: column(values, "mlh")?,
                mlw: column(values, "mlw")?,
                wind_speed: column(values, "wind_speed")?,
                wind_direction: column(values, "wind_direction")?,
                slope: column(values, "slope")?,
                aspect,
                canopy_cover: if options.waf_method == WafMethod::UserInput
                    && values.contains_key("user_waf")
                {
                    values
                        .get("canopy_cover")
                        .map(|_| column(values, "canopy_cover"))
                        .transpose()?
                        .unwrap_or(PerCell::Scalar(f64::NAN))
                } else {
                    column(values, "canopy_cover")?
                },
                canopy_height: if options.waf_method == WafMethod::UserInput
                    && values.contains_key("user_waf")
                {
                    values
                        .get("canopy_height")
                        .map(|_| column(values, "canopy_height"))
                        .transpose()?
                        .unwrap_or(PerCell::Scalar(f64::NAN))
                } else {
                    column(values, "canopy_height")?
                },
                crown_ratio: if options.waf_method == WafMethod::UserInput
                    && values.contains_key("user_waf")
                {
                    values
                        .get("crown_ratio")
                        .map(|_| column(values, "crown_ratio"))
                        .transpose()?
                        .unwrap_or(PerCell::Scalar(f64::NAN))
                } else {
                    column(values, "crown_ratio")?
                },
                user_waf: values.get("user_waf").map(|v| {
                    if v.len() == 1 {
                        PerCell::Scalar(v[0])
                    } else {
                        PerCell::Cells(v)
                    }
                }),
                validity,
            };
            ModelRun::Surface {
                raster: SurfaceRaster::new(grid, fields)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                plan: SurfacePlan::new(options),
                source: SurfaceSource::Fuels(fuels),
                tweaks: SurfaceTweakColumns::default(),
            }
        }
        "crown" => {
            let fields = CrownColumns {
                m1h: column(values, "m1h")?,
                m10h: column(values, "m10h")?,
                m100h: column(values, "m100h")?,
                mlh: column(values, "mlh")?,
                mlw: column(values, "mlw")?,
                wind_speed: column(values, "wind_speed")?,
                wind_direction: column(values, "wind_direction")?,
                slope: column(values, "slope")?,
                aspect,
                canopy_base_height: column(values, "canopy_base_height")?,
                canopy_height: column(values, "canopy_height")?,
                canopy_bulk_density: column(values, "canopy_bulk_density")?,
                moisture_foliar: column(values, "moisture_foliar")?,
                surface_ros: column(values, "surface_ros")?,
                surface_fli: column(values, "surface_fli")?,
                surface_hpua: column(values, "surface_hpua")?,
                validity,
            };
            ModelRun::Crown {
                raster: CrownRaster::new(grid, fields)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
                plan: CrownPlan::new(options),
                fuels,
                tweaks: CrownTweakColumns::default(),
            }
        }
        _ => return Err(PyValueError::new_err("unknown f64 model kind")),
    })
}
