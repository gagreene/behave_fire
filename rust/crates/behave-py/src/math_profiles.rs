//! Bounded intrinsic configuration probes; never accepts user calculation data.

use pyo3::prelude::*;

/// Return fixed operands and available native atan2 fingerprints for binding calibration.
#[pyfunction]
pub fn atan2_profile_candidates() -> (Vec<u32>, Vec<u32>, Vec<u32>, Option<Vec<u32>>) {
    let (y, x, libm, svml) = behave_fire::surface::masked::atan2_profile_candidates();
    (
        y.to_vec(),
        x.to_vec(),
        libm.to_vec(),
        svml.map(|bits| bits.to_vec()),
    )
}
