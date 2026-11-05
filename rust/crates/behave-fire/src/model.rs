//! Owned registry state shared by the standalone model's calculation methods.

use crate::FuelModels;

/// A model owns the fuel snapshot used by its calculations.
///
/// Later mutations of the registry supplied by a caller do not affect this model.
#[derive(Clone, Debug)]
pub struct BehaveRun {
    fuel_models: FuelModels,
}

impl BehaveRun {
    /// Borrow the immutable registry snapshot used by this model.
    pub fn fuel_models(&self) -> &FuelModels {
        &self.fuel_models
    }

    /// Take ownership of a registry snapshot.
    pub fn new(fuel_models: FuelModels) -> Self {
        Self { fuel_models }
    }
}
