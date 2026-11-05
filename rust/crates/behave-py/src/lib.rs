//! Unpublished Python binding boundary for the standalone core.
//!
//! Maturin packages this library as `behave_fire._rust`. The incomplete marker
//! prevents automatic backend selection before native coverage is complete.

use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// Initialize diagnostic metadata without exposing unimplemented calculations.
#[pymodule]
fn _rust(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("__binding_version__", env!("CARGO_PKG_VERSION"))?;
    module.add("__core_version__", behave_fire::VERSION)?;
    module.add("__binding_api_version__", 1_u32)?;
    module.add("__capabilities__", PyTuple::empty(module.py()))?;
    module.add("__backend_complete__", false)?;
    Ok(())
}
