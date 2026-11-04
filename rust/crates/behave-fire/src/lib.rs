#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod grid;
mod model;

pub mod crown;
pub mod fuel_models;
pub mod math;
pub mod species;
pub mod surface;
pub mod units;

/// Embedded tables with no build-time or runtime Python dependency.
pub mod generated {
    /// Complete reference and correction tables for fine dead fuel moisture.
    pub mod fine_dead_fuel_moisture;
}

pub use error::BehaveError;
pub use fuel_models::{FuelModel, FuelModels};
pub use grid::{effective_worker_count, GridOutput, GridShape, PerCell};
pub use model::BehaveRun;
pub use species::{SpeciesMasterTable, SpeciesRecord};

/// Version of the standalone Rust core, independent of the Python distribution.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
