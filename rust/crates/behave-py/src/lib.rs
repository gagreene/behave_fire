//! Explicit owned NumPy bindings for the standalone Rust implementation.
#[cfg(feature = "allocation-probe")]
mod allocation_probe;
mod arrays;
mod capabilities;
mod controls;
mod crown;
mod data;
mod dispatch;
mod masked;
mod math_profiles;
mod state;
mod structures;
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
    module.add("__binding_api_version__", 2_u32)?;
    module.add("__backend_complete__", true)?;
    module.add(
        "__capabilities__",
        pyo3::types::PyTuple::new(module.py(), capabilities::OPERATIONS.iter().copied())?,
    )?;
    module.add_function(wrap_pyfunction!(arrays::batch, module)?)?;
    module.add_function(wrap_pyfunction!(
        math_profiles::atan2_profile_candidates,
        module
    )?)?;
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

mod parallel;

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
