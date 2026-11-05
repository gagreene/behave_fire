//! Owned array boundary. No Python references survive into detached execution.

use numpy::ndarray::{ArrayD, IxDyn};
use numpy::{IntoPyArray, PyReadonlyArrayDyn, PyUntypedArrayMethods};
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

#[derive(Clone)]
pub struct Arguments {
    values: BTreeMap<String, Vec<f64>>,
    strings: BTreeMap<String, String>,
    integers: BTreeMap<String, (String, Vec<u64>)>,
}

impl Arguments {
    fn empty_prototype(&self) -> Self {
        let mut result = self.clone();
        for (name, values) in &mut result.values {
            if values.is_empty() {
                values.push(if name.starts_with("_mask.") { 0.0 } else { 1.0 });
            }
        }
        for (_, bits) in result.integers.values_mut() {
            if bits.is_empty() {
                bits.push(1);
            }
        }
        result
    }
    pub fn integer(&self, name: &str, index: usize) -> Option<(&str, u64)> {
        let (kind, bits) = self.integers.get(name)?;
        bits.get(if bits.len() == 1 { 0 } else { index })
            .map(|value| (kind.as_str(), *value))
    }
    pub fn has_crown_python_payloads(&self) -> bool {
        self.values.keys().any(|key| {
            key.starts_with("_object.")
                || (key.starts_with("_weak.")
                    && !matches!(
                        key.as_str(),
                        "_weak.wind_speed_units"
                            | "_weak.slope_units"
                            | "_weak.aspect"
                            | "_weak.moisture_foliar"
                    ))
        })
    }
    pub fn has_python_payloads(&self) -> bool {
        self.values
            .keys()
            .any(|key| key.starts_with("_object.") || key.starts_with("_weak."))
    }
    pub fn has_masks(&self) -> bool {
        self.values.keys().any(|key| key.starts_with("_masked."))
    }
    pub fn has_single(&self) -> bool {
        self.values.keys().any(|key| key.starts_with("_single."))
    }
    pub fn has_integers(&self) -> bool {
        self.integers.keys().any(|key| {
            let field = key.rsplit('.').next().unwrap_or(key);
            !field.ends_with("units")
                && !field.starts_with('_')
                && !matches!(
                    field,
                    "max_steps"
                        | "charge_type"
                        | "fuel_model_grid"
                        | "equation_number_grid"
                        | "is_crown"
                        | "is_surface_fire"
                        | "is_passive_crown"
                        | "is_active_crown"
                        | "is_defined"
                        | "is_dynamic"
                )
        })
    }
    pub fn masked(
        &self,
        name: &str,
        index: usize,
    ) -> Result<behave_fire::surface::masked::MaskedValue, String> {
        use behave_fire::surface::masked::{
            IntegerKind, IntegerValue, MaskedValue, MathProfile, Precision,
        };
        let profile = match (
            self.optional("_simd_exp", index).unwrap_or(0.0) != 0.0,
            self.optional("_svml_atan2", index).unwrap_or(0.0) != 0.0,
        ) {
            (true, true) => MathProfile::NumpySimdSvmlAtan2,
            (false, true) => MathProfile::LibmSvmlAtan2,
            (true, false) => MathProfile::NumpySimd,
            (false, false) => MathProfile::Libm,
        };
        let mut integer = self
            .integer(name, index)
            .map(|(kind, bits)| {
                IntegerKind::from_name(kind)
                    .map(|kind| IntegerValue::from_bits(kind, bits))
                    .ok_or_else(|| format!("TypeError: unsupported integer dtype {kind}"))
            })
            .transpose()?;
        if let Some(code) = self.optional(&format!("_object_kind.{name}"), index) {
            integer = if (0.0..=8.0).contains(&code) {
                let kinds = [
                    IntegerKind::I8,
                    IntegerKind::I16,
                    IntegerKind::I32,
                    IntegerKind::I64,
                    IntegerKind::U8,
                    IntegerKind::U16,
                    IntegerKind::U32,
                    IntegerKind::U64,
                    IntegerKind::Bool,
                ];
                Some(IntegerValue::from_bits(
                    kinds[code as usize],
                    self.integer(name, index).map_or(0, |(_, bits)| bits),
                ))
            } else {
                None
            };
        }
        if self
            .optional(&format!("_weak_bool.{name}"), index)
            .is_some()
        {
            return Ok(
                MaskedValue::python_bool(integer.is_some_and(|v| v.raw_bits != 0))
                    .with_math_profile(profile),
            );
        }
        if self
            .optional(&format!("_weak_integer.{name}"), index)
            .is_some()
        {
            if let Some(integer) = integer {
                return Ok(if integer.kind.is_signed() {
                    MaskedValue::python_int(integer.as_i128() as i64)
                } else {
                    MaskedValue::python_uint(integer.raw_bits)
                }
                .with_math_profile(profile));
            }
        }
        if self.optional(&format!("_weak.{name}"), index).is_some() {
            return Ok(MaskedValue::scalar(self.n(name, index)?).with_math_profile(profile));
        }
        let value = MaskedValue::new(
            self.n(name, index)?,
            self.optional(&format!("_mask.{name}"), index)
                .unwrap_or(0.0)
                != 0.0,
            self.optional(&format!("_masked.{name}"), index).is_some(),
            if self.optional(&format!("_single.{name}"), index).is_some()
                || self.optional(&format!("_object_kind.{name}"), index) == Some(10.0)
            {
                Precision::F32
            } else {
                Precision::F64
            },
        )
        .with_scalar_array(self.optional(&format!("_scalar.{name}"), index).is_some())
        .with_fill_value(self.optional(&format!("_fill.{name}"), index))
        .with_math_profile(profile)
        .with_numpy_scalar(
            self.optional(&format!("_numpy_scalar.{name}"), index)
                .is_some(),
        );
        let mut value = integer.map_or(value, |integer| value.with_integer(integer));
        if self.optional(&format!("_object.{name}"), index).is_some() {
            value = value.with_object_payload(
                self.optional(&format!("_object_numpy.{name}"), index)
                    .unwrap_or(0.0)
                    != 0.0,
            );
        }
        value = value.with_numpy_integer_type(
            self.optional(&format!("_integer_type.{name}"), index)
                .filter(|number| (0.0..=10.0).contains(number))
                .map(|number| number as u8),
        );
        Ok(value.with_empty_array(self.optional(&format!("_empty.{name}"), index).is_some()))
    }
    pub fn n(&self, name: &str, index: usize) -> Result<f64, String> {
        let values = self
            .values
            .get(name)
            .ok_or_else(|| format!("missing argument {name}"))?;
        values
            .get(if values.len() == 1 { 0 } else { index })
            .copied()
            .ok_or_else(|| format!("invalid length for {name}"))
    }
    pub fn optional(&self, name: &str, index: usize) -> Option<f64> {
        self.n(name, index).ok()
    }
    pub fn s(&self, name: &str) -> Result<&str, String> {
        self.strings
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| format!("missing mode {name}"))
    }
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
    if !crate::dispatch::supported(operation) {
        return Err(PyNotImplementedError::new_err(format!(
            "Native operation is unavailable: {operation}"
        )));
    }
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
    let threshold = if matches!(operation, "model.run_surface" | "crown.run_crown_fire") {
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
    let operation = operation.to_owned();
    let copied = started.elapsed().as_secs_f64();
    if let Some(callback) = _owned_callback {
        callback.call0()?;
    }
    let mut concatenate_seconds = 0.0;
    let mut appended_bytes = 0usize;
    let detached_started = std::time::Instant::now();
    let result = py
        .detach(|| {
            let mut output: BTreeMap<String, Vec<f64>> = BTreeMap::new();
            if count == 0 {
                let prototype = arguments.empty_prototype();
                let fields = if operation == "model.run_surface" {
                    crate::structures::surface_output(crate::parallel::surface(&prototype, 0)?)
                } else {
                    crate::dispatch::evaluate(&operation, &prototype, 0, single)?
                };
                for (name, value) in fields {
                    let retain = name.starts_with("__single.")
                        || name.starts_with("__masked.")
                        || name.starts_with("__integer_dtype.")
                        || name.starts_with("__integer_type.")
                        || name.starts_with("__complex.")
                        || name.starts_with("__object.")
                        || name.starts_with("__object_numpy.")
                        || name.starts_with("__fill");
                    output.insert(name, if retain { vec![value] } else { Vec::new() });
                }
                return Ok::<_, String>(output);
            }
            if let Some(result) =
                crate::parallel::evaluate(&operation, &arguments, &shape, requested)
            {
                return result.map(|(columns, seconds, bytes)| {
                    concatenate_seconds = seconds;
                    appended_bytes = bytes;
                    columns
                });
            }
            let chunks = behave_fire::map_chunks(count, requested, |range| {
                crate::parallel::chunk(&operation, &arguments, range, single)
            })
            .map_err(crate::parallel::error)?;
            appended_bytes = chunks
                .iter()
                .skip(1)
                .flat_map(|chunk| chunk.values())
                .map(|values| values.len() * std::mem::size_of::<f64>())
                .sum();
            let concatenate_started = std::time::Instant::now();
            output = crate::parallel::merge(chunks, count)?;
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
    let output = PyDict::new(py);
    for (name, values) in result {
        let dimensions = if count == 0 && values.len() == 1 {
            &[][..]
        } else {
            shape.as_slice()
        };
        if dimensions.len() > 32 {
            // Transfer owned storage through the supported one-axis conversion,
            // then use NumPy's structural reshape for its valid higher ranks.
            let array = values.into_pyarray(py);
            output.set_item(name, array.call_method1("reshape", (dimensions.to_vec(),))?)?;
            continue;
        }
        let array = ArrayD::from_shape_vec(IxDyn(dimensions), values)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        output.set_item(name, array.into_pyarray(py))?;
    }
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
