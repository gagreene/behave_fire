#![doc = include_str!("../README.md")]
#![deny(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod grid;
mod model;
mod parallel;
// The private SVML adapter audits its CPU guard, scalar ABI, and assembly calls.
#[allow(unsafe_code)]
mod svml;

pub mod batch;
pub mod contain;
pub mod crown;
pub mod engine;
pub mod fine_dead_fuel_moisture;
pub mod fire_size;
pub mod fuel_models;
pub mod ignite;
pub mod math;
pub mod mortality;
pub mod raster;
pub mod safety;
pub mod slope;
pub mod species;
pub mod spot;
pub mod surface;
pub mod units;
pub mod vapor_pressure_deficit;

/// Embedded tables with no build-time or runtime Python dependency.
pub mod generated {
    /// Complete reference and correction tables for fine dead fuel moisture.
    pub mod fine_dead_fuel_moisture;
}

pub use error::BehaveError;
pub use fuel_models::{FuelModel, FuelModels};
pub use grid::{effective_worker_count, GridOutput, GridShape, PerCell};
pub use model::{canopy_crown_ratio, BehaveRun};
pub use parallel::{map_chunks, ParallelError};
pub use species::{SpeciesMasterTable, SpeciesRecord};

/// Version of the standalone Rust core, independent of the Python distribution.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
