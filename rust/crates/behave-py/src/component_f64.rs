//! Owned primitive component execution with static layouts and exact controls.
use crate::prepared::{
    primitive::{Arguments, Prepared},
    Field,
};
use crate::{
    component_columns::Columns,
    component_schema, model_f64,
    operation::{Family, Operation},
};
use behave_fire::engine::SurfaceOptions;
use pyo3::exceptions::{PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;
pub(crate) const OPERATIONS: &[Operation] = Operation::ALL;

/// Return fixed particle dependency plans without evaluating scientific cells.
#[pyfunction]
pub fn component_dependencies() -> PyResult<BTreeMap<String, Vec<String>>> {
    let schema = component_schema::schema(Operation::SurfaceCalculateFuelbedIntermediates);
    let mut result = BTreeMap::new();
    for field in schema {
        let inputs = crate::component_fuelbed::dependencies(std::slice::from_ref(&field.name))
            .map_err(PyValueError::new_err)?;
        result.insert(
            field.name,
            inputs
                .into_iter()
                .map(|(_, field)| field.name().to_owned())
                .collect(),
        );
    }
    Ok(result)
}

/// Capture all operands before detached execution, with no mask or dtype decoder.
#[pyfunction]
#[pyo3(signature=(operation, physical, codes, flags, shape, threads=None, _owned_callback=None, settings=None, outputs=None))]
#[allow(clippy::too_many_arguments)]
pub fn component_batch<'py>(
    py: Python<'py>,
    operation: &str,
    physical: &Bound<'py, PyDict>,
    codes: &Bound<'py, PyDict>,
    flags: &Bound<'py, PyDict>,
    shape: &Bound<'py, PyAny>,
    threads: Option<&Bound<'py, PyAny>>,
    _owned_callback: Option<&Bound<'py, PyAny>>,
    settings: Option<BTreeMap<String, String>>,
    outputs: Option<Vec<String>>,
) -> PyResult<Bound<'py, PyDict>> {
    let operation = Operation::parse(operation)
        .filter(|v| !v.main_model())
        .ok_or_else(|| {
            PyNotImplementedError::new_err("use the configured model boundary for full model calls")
        })?;
    let shape = shape.extract::<Vec<usize>>().map_err(|_| {
        PyValueError::new_err("shape dimensions must be nonnegative native integers")
    })?;
    let grid = crate::model_fields::checked_shape(shape.clone())?;
    let requested = if let Some(value) = threads {
        let count = value
            .call_method0("__index__")
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("threads must be an integer"))?;
        if count.lt(0)? {
            return Err(PyValueError::new_err("threads must be nonnegative"));
        }
        if count.gt(usize::MAX)? {
            usize::MAX
        } else {
            count.extract::<usize>()?
        }
    } else {
        1
    };
    let threads = if grid.len() < 10_000 { 1 } else { requested };
    let mut arguments = Arguments {
        physical: model_f64::physical(physical, &grid)?,
        ..Arguments::default()
    };
    for (name, value) in codes.iter() {
        let name: String = name.extract()?;
        arguments
            .codes
            .insert(name.clone(), model_f64::array::<i64>(&value, &name)?);
    }
    for (name, value) in flags.iter() {
        let name: String = name.extract()?;
        arguments
            .flags
            .insert(name.clone(), model_f64::array::<bool>(&value, &name)?);
    }
    let settings = settings.unwrap_or_default();
    let options = SurfaceOptions {
        wind_orientation: crate::prepared::controls::orientation(
            settings
                .get("wind_orientation_mode")
                .map_or("RelativeToUpslope", String::as_str),
        ),
        wind_height: crate::prepared::controls::height(
            settings
                .get("wind_height_mode")
                .map_or("TwentyFoot", String::as_str),
        ),
        waf_method: crate::prepared::controls::waf(
            settings
                .get("waf_method")
                .map_or("CrownRatio", String::as_str),
        ),
    };
    let prepared = Prepared::new(&arguments, &grid)
        .map_err(PyValueError::new_err)?
        .with_options(options);
    let mut schema = component_schema::schema(operation);
    if let Some(request) = outputs {
        for name in &request {
            if !schema.iter().any(|f| &f.name == name) {
                return Err(PyValueError::new_err(format!(
                    "unknown component output {name}"
                )));
            }
        }
        schema.retain(|field| request.contains(&field.name));
    }
    let names = schema
        .iter()
        .map(|field| field.name.clone())
        .collect::<Vec<_>>();
    if operation.family() != Family::Units {
        let mut required = crate::component_selected::required(operation, &names)
            .unwrap_or_else(|| crate::component_requirements::fields(operation).to_vec());
        if operation == Operation::SurfaceCalculateSurfaceWindAdjustmentFactor
            && options.waf_method == behave_fire::engine::WafMethod::UserInput
            && prepared.has(Field::UserWaf)
        {
            required = vec![Field::UserWaf];
        }
        if options.wind_orientation == behave_fire::engine::WindOrientation::Upslope {
            required.retain(|f| *f != Field::Aspect);
        }
        prepared.require(&required).map_err(PyValueError::new_err)?;
        let needs_weather = operation != Operation::CrownCalculateCrownSurfaceFire
            || names.iter().any(|name| {
                name.starts_with("crown_surface.")
                    || matches!(
                        name.as_str(),
                        "crown_fire_spread_rate" | "crown_length_to_width_ratio"
                    )
            });
        if needs_weather
            && options.wind_orientation == behave_fire::engine::WindOrientation::North
            && matches!(
                operation,
                Operation::SurfaceCalculateDirectionOfMaxSpread
                    | Operation::SurfaceCalculateSurfaceRateVector
                    | Operation::SurfaceRunSurfaceFire
                    | Operation::CrownCalculateCrownSurfaceFire
            )
        {
            prepared
                .require(&[Field::Aspect])
                .map_err(PyValueError::new_err)?;
        }
        if matches!(
            operation,
            Operation::SurfaceRunSurfaceFire
                | Operation::SurfaceCalculateSurfaceWindAdjustmentFactor
        ) && !(options.waf_method == behave_fire::engine::WafMethod::UserInput
            && prepared.has(Field::UserWaf))
        {
            prepared
                .require(&[Field::CanopyCover, Field::CanopyHeight, Field::CrownRatio])
                .map_err(PyValueError::new_err)?;
        }
    }
    let mut result = Columns::new(schema, grid.len()).map_err(PyValueError::new_err)?;
    if let Some(callback) = _owned_callback {
        callback.call0()?;
    }
    if !names.is_empty() {
        py.detach(|| {
            if let Some(value) =
                crate::component_units::run(operation, &arguments, &prepared, &mut result, threads)
            {
                return value;
            }
            if let Some(value) =
                crate::component_selected::run(operation, &prepared, &names, &mut result, threads)
            {
                return value;
            }
            result.fill(threads, |index, row| {
                crate::dispatch::evaluate(operation, &prepared, index, row)
            })
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    }
    model_f64::export_columns(
        py,
        result
            .columns
            .into_iter()
            .map(|(field, values)| (field.name, values)),
        &shape,
    )
}
