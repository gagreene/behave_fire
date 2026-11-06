//! Scalar surface fire stages in reference base units.
mod fuelbed;
mod geometry;
/// NumPy-compatible masked-array scalar stages, including raw masked data and precision.
pub mod masked;
mod particles;
mod reaction;
mod types;
mod wind;
pub use fuelbed::*;
pub use geometry::*;
pub use particles::*;
pub use reaction::*;
pub use types::*;
pub use wind::*;

/// Native single-precision variants for helpers whose Python arithmetic retains float32.
pub mod single {
    pub use super::fuelbed::single::*;
    pub use super::geometry::single::*;
    pub use super::reaction::single::*;
    pub use super::wind::single::*;
}
