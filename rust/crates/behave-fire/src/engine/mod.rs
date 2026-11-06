//! Typed scalar science and reusable numerical inputs, independent of array runtimes.
mod crown;
mod fuel;
mod options;
mod surface;
mod tweaks;

pub use crown::{CrownCell, CrownPlan};
pub use fuel::{PreparedFuel, PreparedFuels};
pub use options::{SurfaceOptions, WafMethod, WindHeight, WindOrientation};
pub use surface::{SurfaceCell, SurfacePlan};
pub use tweaks::{CrownTweaks, SurfaceTweaks};
