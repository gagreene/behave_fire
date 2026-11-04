//! Explicit owned NumPy bindings for the standalone Rust implementation.
mod arrays;
mod capabilities;
mod crown;
mod data;
mod dispatch;
mod masked;
mod state;
mod structures;
use pyo3::prelude::*;

#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("__binding_version__", env!("CARGO_PKG_VERSION"))?;
    module.add("__core_version__", behave_fire::VERSION)?;
    module.add("__binding_api_version__", 1_u32)?;
    module.add("__backend_complete__", false)?;
    module.add(
        "__capabilities__",
        pyo3::types::PyTuple::new(module.py(), capabilities::OPERATIONS.iter().copied())?,
    )?;
    module.add_function(wrap_pyfunction!(arrays::batch, module)?)?;
    module.add_function(wrap_pyfunction!(data::defaults, module)?)?;
    module.add_function(wrap_pyfunction!(data::missing_mortality, module)?)?;
    module.add_function(wrap_pyfunction!(state::state_call, module)?)?;
    Ok(())
}

mod masked_records;

mod crown_masked;

mod surface;

mod crown_dispatch;

mod effects;

mod tools;

mod contain;

mod units;

mod model;
