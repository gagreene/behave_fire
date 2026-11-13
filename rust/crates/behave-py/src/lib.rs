//! Explicit owned NumPy bindings for the standalone Rust implementation.
#[cfg(feature = "allocation-probe")]
mod allocation_probe;
mod capabilities;
mod component_columns;
mod component_f64;
mod component_fuelbed;
mod component_independent;
mod component_requirements;
mod component_schema;
mod component_selected;
mod component_units;
mod crown;
mod crown_fields;
mod data;
mod dispatch;
mod model_f64;
mod model_fields;
mod model_groups;
mod model_observer;
mod operation;
mod prepared;
mod state;
mod structures;
mod surface_fields;
use pyo3::prelude::*;

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "allocation-probe")]
    {
        module.add_function(wrap_pyfunction!(_allocation_start, module)?)?;
        module.add_function(wrap_pyfunction!(_allocation_stop, module)?)?;
    }
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("__binding_version__", env!("CARGO_PKG_VERSION"))?;
    module.add("__core_version__", behave_fire::VERSION)?;
    module.add("__binding_api_version__", 4_u32)?;
    let contracts = pyo3::types::PyDict::new(module.py());
    for &op in operation::Operation::ALL {
        contracts.set_item(op.name(), op.numeric_contract())?;
    }
    module.add("__numeric_contracts__", contracts)?;
    module.add("__backend_complete__", true)?;
    module.add(
        "__capabilities__",
        pyo3::types::PyTuple::new(module.py(), capabilities::OPERATIONS.iter().copied())?,
    )?;
    module.add_function(wrap_pyfunction!(component_f64::component_batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        component_f64::component_dependencies,
        module
    )?)?;
    module.add(
        "__primitive_operations__",
        pyo3::types::PyTuple::new(
            module.py(),
            component_f64::OPERATIONS.iter().map(|op| op.name()),
        )?,
    )?;
    module.add_function(wrap_pyfunction!(model_f64::model_batch, module)?)?;
    module.add_class::<grid_session::NativeGridSession>()?;
    module.add_function(wrap_pyfunction!(grid_session::grid_fields, module)?)?;
    module.add_function(wrap_pyfunction!(model_fields::model_fields, module)?)?;
    module.add_function(wrap_pyfunction!(model_groups::model_groups, module)?)?;
    module.add_function(wrap_pyfunction!(data::defaults, module)?)?;
    module.add_function(wrap_pyfunction!(data::missing_mortality, module)?)?;
    module.add_function(wrap_pyfunction!(state::state_call, module)?)?;
    Ok(())
}

mod surface;

mod crown_dispatch;

mod effects;

mod grid_session;

mod tools;

mod model;

#[cfg(feature = "allocation-probe")]
#[pyfunction]
fn _allocation_start() {
    allocation_probe::start();
}

#[cfg(feature = "allocation-probe")]
#[pyfunction]
fn _allocation_stop() -> Vec<(usize, usize, usize, usize)> {
    allocation_probe::stop().to_vec()
}
