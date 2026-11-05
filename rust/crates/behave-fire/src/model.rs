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
    /// Run crown calculations using the supplied surface result and owned fuels.
    pub fn do_crown_run(
        &self,
        input: &crate::crown::CrownInput,
    ) -> Result<crate::crown::CrownResult, crate::BehaveError> {
        crate::crown::run(input, &self.fuel_models)
    }

    /// Run the surface pipeline using this model's fuel snapshot.
    pub fn do_surface_run(
        &self,
        input: &crate::surface::SurfaceInput,
    ) -> Result<crate::surface::SurfaceResult, crate::BehaveError> {
        crate::surface::run(input, &self.fuel_models)
    }

    /// Borrow the immutable registry snapshot used by this model.
    pub fn fuel_models(&self) -> &FuelModels {
        &self.fuel_models
    }

    /// Take ownership of a registry snapshot.
    pub fn new(fuel_models: FuelModels) -> Self {
        Self { fuel_models }
    }
}
