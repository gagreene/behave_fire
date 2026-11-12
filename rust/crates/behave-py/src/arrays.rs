//! Owned array boundary. No Python references survive into detached execution.
use crate::column_output::{Buffer, ColumnBundle};
use crate::operation::Operation;
use crate::prepared::{Arguments, PreparedArguments};

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{Element, IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;

fn flatten_base_array<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // Strip ndarray subclasses before selecting ravel, so overridden methods
    // cannot supply a high-rank or malformed result to rust-numpy's view path.
    let base = value
        .py()
        .import("numpy")?
        .getattr("asarray")?
        .call1((value,))?;
    base.call_method1("ravel", ("C",))
}

pub fn masked_error(error: behave_fire::surface::masked::MaskedError) -> String {
    format!("{}: {error:?}", error.python_exception())
}

#[pyfunction]
#[pyo3(signature=(operation, arrays, strings, shape, single, integers=None, threads=None, profile=false, _owned_callback=None))]
// Keep the existing keyword-based Python boundary explicit for PyO3.
#[allow(clippy::too_many_arguments)]
pub fn batch<'py>(
    py: Python<'py>,
    operation: &str,
    arrays: &Bound<'py, PyDict>,
    strings: BTreeMap<String, String>,
    shape: &Bound<'py, PyAny>,
    single: bool,
    integers: Option<&Bound<'py, PyDict>>,
    threads: Option<&Bound<'py, PyAny>>,
    profile: bool,
    _owned_callback: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let started = std::time::Instant::now();
    let requested = if let Some(value) = threads {
        let value = value
            .call_method0("__index__")
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("threads must be an integer"))?;
        if value.lt(0)? {
            return Err(PyValueError::new_err("threads must be nonnegative"));
        }
        if value.gt(usize::MAX)? {
            usize::MAX
        } else {
            value.extract::<usize>()?
        }
    } else {
        1
    };
    let operation = Operation::parse(operation).ok_or_else(|| {
        PyNotImplementedError::new_err(format!("Native operation is unavailable: {operation}"))
    })?;
    let shape: Vec<usize> = shape
        .extract()
        .map_err(|_| PyValueError::new_err("shape dimensions must be nonnegative integers"))?;
    let max_dimensions = if numpy::npyffi::is_numpy_2(py) {
        64
    } else {
        32
    };
    if shape.len() > max_dimensions {
        return Err(PyValueError::new_err(format!(
            "shape exceeds NumPy's maximum of {max_dimensions} dimensions"
        )));
    }
    // Empty dimensions do not remove the stride extents of the other axes.
    // Check their byte bound before allocating or entering detached execution.
    let extent = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d.max(1)))
        .ok_or_else(|| PyValueError::new_err("shape stride extent overflow"))?;
    if extent > (isize::MAX as usize) / std::mem::size_of::<f64>() {
        return Err(PyValueError::new_err(
            "shape exceeds the maximum native stride byte extent",
        ));
    }
    let count = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| PyValueError::new_err("shape size overflow"))?;
    if count > (isize::MAX as usize) / std::mem::size_of::<f64>() {
        return Err(PyValueError::new_err(
            "shape exceeds the maximum native array byte length",
        ));
    }
    // Keep measured thresholds at pipeline boundaries, not in generic map_chunks.
    let threshold = if operation.main_model() {
        behave_fire::batch::PIPELINE_PARALLEL_MIN_CELLS
    } else {
        10_000
    };
    let requested = if count < threshold { 1 } else { requested };
    let mut values = BTreeMap::new();
    for (key, value) in arrays.iter() {
        let key: String = key.extract()?;
        let array: PyReadonlyArrayDyn<'_, f64> = value.extract()?;
        // rust-numpy 0.27's ndarray view conversion has a fixed 32-axis buffer.
        // Flatten only higher-rank inputs, while attached, preserving C order.
        let flattened = if array.ndim() > 32 {
            Some(flatten_base_array(&value)?)
        } else {
            None
        };
        let array = match flattened.as_ref() {
            Some(value) => value.extract::<PyReadonlyArrayDyn<'_, f64>>()?,
            None => array,
        };
        if array.ndim() > 32 {
            return Err(PyValueError::new_err(
                "flattened numeric input exceeds rust-numpy's supported view rank",
            ));
        }
        let view = array.as_array();
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(view.len())
            .map_err(|error| pyo3::exceptions::PyMemoryError::new_err(error.to_string()))?;
        owned.extend(view.iter().copied());
        if owned.len() != count && owned.len() != 1 {
            return Err(PyValueError::new_err(format!("invalid length for {key}")));
        }
        values.insert(key, owned);
    }
    let mut owned_integers = BTreeMap::new();
    if let Some(integers) = integers {
        for (key, value) in integers.iter() {
            let name: String = key.extract()?;
            let (kind, array): (String, PyReadonlyArrayDyn<'_, u64>) = value.extract()?;
            let flattened = if array.ndim() > 32 {
                Some(flatten_base_array(array.as_any())?)
            } else {
                None
            };
            let array = match flattened.as_ref() {
                Some(value) => value.extract::<PyReadonlyArrayDyn<'_, u64>>()?,
                None => array,
            };
            if array.ndim() > 32 {
                return Err(PyValueError::new_err(
                    "flattened integer input exceeds rust-numpy's supported view rank",
                ));
            }
            let view = array.as_array();
            let mut bits = Vec::new();
            bits.try_reserve_exact(view.len())
                .map_err(|error| pyo3::exceptions::PyMemoryError::new_err(error.to_string()))?;
            bits.extend(view.iter().copied());
            if bits.len() != count && bits.len() != 1 {
                return Err(PyValueError::new_err(format!(
                    "invalid integer length for {name}"
                )));
            }
            owned_integers.insert(name, (kind, bits));
        }
    }
    let arguments = Arguments {
        values,
        strings,
        integers: owned_integers,
    };
    let copied = started.elapsed().as_secs_f64();
    if let Some(callback) = _owned_callback {
        callback.call0()?;
    }
    let mut concatenate_seconds = 0.0;
    let mut appended_bytes = 0usize;
    let detached_started = std::time::Instant::now();
    let result = py
        .detach(|| {
            let grid_shape =
                behave_fire::GridShape::new(shape.clone()).map_err(|error| error.to_string())?;
            if count == 0 {
                let prototype_arguments = arguments.empty_prototype();
                let prototype = PreparedArguments::new(&prototype_arguments, &grid_shape)?;
                return crate::parallel::empty(operation, &prototype, single);
            }
            let prepared = PreparedArguments::new(&arguments, &grid_shape)?;
            if let Some(result) = crate::parallel::evaluate(operation, &prepared, count, requested)
            {
                return result.map(|(columns, seconds, bytes)| {
                    concatenate_seconds = seconds;
                    appended_bytes = bytes;
                    columns
                });
            }
            let chunks = behave_fire::map_chunks(count, requested, |range| {
                crate::parallel::chunk(operation, &prepared, range, single)
            })
            .map_err(crate::parallel::error)?;
            appended_bytes = chunks.iter().skip(1).map(ColumnBundle::bytes).sum();
            let concatenate_started = std::time::Instant::now();
            let output = crate::parallel::merge(chunks, count)?;
            concatenate_seconds = concatenate_started.elapsed().as_secs_f64();
            Ok::<_, String>(output)
        })
        .map_err(|error| {
            if error.starts_with("Native operation") {
                PyNotImplementedError::new_err(error)
            } else if error.starts_with("RuntimeError:") {
                pyo3::exceptions::PyRuntimeError::new_err(error)
            } else if error.starts_with("TypeError:") {
                pyo3::exceptions::PyTypeError::new_err(error)
            } else if error.starts_with("AttributeError:") {
                pyo3::exceptions::PyAttributeError::new_err(error)
            } else if error.starts_with("MemoryError:") {
                pyo3::exceptions::PyMemoryError::new_err(error)
            } else if error.starts_with("OverflowError:") {
                pyo3::exceptions::PyOverflowError::new_err(error)
            } else if error.starts_with("ZeroDivisionError:") {
                pyo3::exceptions::PyZeroDivisionError::new_err(error)
            } else {
                PyValueError::new_err(error)
            }
        })?;
    let detached_seconds = detached_started.elapsed().as_secs_f64();
    let transfer_started = std::time::Instant::now();
    let output = transfer_columns(py, result, &shape)?;
    if profile {
        for (name, value) in [
            ("__timing.owned_copy_seconds", copied),
            ("__timing.detached_seconds", detached_seconds),
            (
                "__timing.numpy_transfer_seconds",
                transfer_started.elapsed().as_secs_f64(),
            ),
            ("__timing.concatenate_seconds", concatenate_seconds),
            ("__timing.appended_bytes", appended_bytes as f64),
        ] {
            let array = ArrayD::from_shape_vec(IxDyn(&[]), vec![value])
                .map_err(|error| PyValueError::new_err(error.to_string()))?;
            output.set_item(name, array.into_pyarray(py))?;
        }
    }
    Ok(output)
}

fn set_array<T: Element>(
    output: &Bound<'_, PyDict>,
    name: &str,
    values: Vec<T>,
    shape: &[usize],
) -> PyResult<()> {
    if shape.len() > 32 {
        let array = values.into_pyarray(output.py());
        output.set_item(name, array.call_method1("reshape", (shape.to_vec(),))?)?;
    } else {
        let array = ArrayD::from_shape_vec(IxDyn(shape), values)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        output.set_item(name, array.into_pyarray(output.py()))?;
    }
    Ok(())
}

fn transfer_columns<'py>(
    py: Python<'py>,
    mut bundle: ColumnBundle,
    shape: &[usize],
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    let schemas = PyDict::new(py);
    bundle.columns.sort_by(|a, b| a.name.cmp(&b.name));
    for column in bundle.columns {
        let name = column.name;
        match column.data {
            Buffer::F64(v) => set_array(&output, &name, v, shape)?,
            Buffer::F32(v) => set_array(&output, &name, v, shape)?,
            Buffer::I8(v) => set_array(&output, &name, v, shape)?,
            Buffer::I16(v) => set_array(&output, &name, v, shape)?,
            Buffer::I32(v) => set_array(&output, &name, v, shape)?,
            Buffer::I64(v) => set_array(&output, &name, v, shape)?,
            Buffer::U8(v) => set_array(&output, &name, v, shape)?,
            Buffer::U16(v) => set_array(&output, &name, v, shape)?,
            Buffer::U32(v) => set_array(&output, &name, v, shape)?,
            Buffer::U64(v) => set_array(&output, &name, v, shape)?,
            Buffer::Bool(v) => set_array(&output, &name, v, shape)?,
        }
        if let Some(metadata) = column.metadata {
            let schema = PyDict::new(py);
            schema.set_item("masked", metadata.masked)?;
            schema.set_item("single", metadata.single)?;
            schema.set_item("complex", metadata.complex)?;
            schema.set_item("object", metadata.object)?;
            schema.set_item("object_numpy", metadata.object_numpy)?;
            schema.set_item("integer_kind", metadata.integer_kind)?;
            schema.set_item("integer_type", metadata.integer_type)?;
            schema.set_item("fill_present", metadata.fill_present)?;
            schema.set_item("fill", metadata.fill)?;
            if metadata.masked {
                if let Some(mask) = metadata.mask {
                    set_array(&schema, "mask", mask, shape)?;
                }
            }
            if metadata.complex {
                if let Some(imaginary) = metadata.imaginary {
                    set_array(&schema, "imag", imaginary, shape)?;
                }
            }
            if let Some(dynamic) = metadata.dynamic {
                macro_rules! compact {
                    ($key:literal,$values:expr,$default:expr) => {{
                        let values = $values;
                        let first = values.first().copied().unwrap_or($default);
                        if values.iter().all(|value| *value == first) {
                            schema.set_item($key, first)?;
                        } else {
                            set_array(&schema, $key, values, shape)?;
                        }
                    }};
                }
                if metadata.object {
                    compact!("single", dynamic.single, metadata.single);
                    compact!("integer_kind", dynamic.integer_kind, metadata.integer_kind);
                    compact!("integer_type", dynamic.integer_type, metadata.integer_type);
                    compact!("complex", dynamic.complex, metadata.complex);
                    compact!("object_numpy", dynamic.object_numpy, metadata.object_numpy);
                }
                if metadata.integer_any && (metadata.object || !metadata.complex) {
                    set_array(&schema, "bits", dynamic.bits, shape)?;
                }
            }
            schemas.set_item(name, schema)?;
        }
    }
    output.set_item("__schema", schemas)?;
    Ok(output)
}
