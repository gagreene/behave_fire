//! Surface raster execution assembles only the current scalar cell on the stack.
use super::{
    tile_shape, validity, FuelbedColumns, SurfaceColumns, SurfaceRasterOutput, SurfaceTweakColumns,
};
use crate::engine::{PreparedFuels, SurfacePlan, SurfaceTweaks};
use crate::surface::SurfaceResult;
use crate::{BehaveError, GridShape, ParallelError};
use std::ops::Range;

#[derive(Clone, Copy)]
enum SurfaceSource<'a, 'c> {
    Fuels(&'a PreparedFuels),
    Fuelbeds(&'a FuelbedColumns<'c>),
}

/// Validated borrowed Surface columns in logical row-major order.
#[derive(Clone, Debug)]
pub struct SurfaceRaster<'a> {
    shape: GridShape,
    inputs: SurfaceColumns<'a>,
}

impl<'a> SurfaceRaster<'a> {
    /// Borrow the validated inputs when creating an independently owned session.
    pub fn columns(&self) -> &SurfaceColumns<'a> {
        &self.inputs
    }

    /// Validate every provided column once, before allocation or worker execution.
    pub fn new(shape: GridShape, inputs: SurfaceColumns<'a>) -> Result<Self, BehaveError> {
        inputs.validate(&shape)?;
        Ok(Self { shape, inputs })
    }

    /// Evaluate one checked cell into caller-managed output storage.
    ///
    /// Returns `None` outside the raster. Invalid pixels return the same default
    /// result as `run`; validity remains available in the caller's input columns.
    pub fn evaluate_cell(
        &self,
        index: usize,
        plan: &SurfacePlan,
        fuels: &PreparedFuels,
        tweaks: SurfaceTweaks,
    ) -> Option<SurfaceResult> {
        if index >= self.shape.len() {
            return None;
        }
        if !self.inputs.valid(index) {
            return Some(SurfaceResult::default());
        }
        let cell = self.inputs.cell(index);
        Some(plan.run_cell(&cell, fuels.for_grid(cell.fuel_model_grid), tweaks))
    }
    /// Borrow one validated spatial cell; invalid or out-of-range indices return None.
    pub fn input_cell(&self, index: usize) -> Option<crate::engine::SurfaceCell> {
        (index < self.shape.len() && self.inputs.valid(index)).then(|| self.inputs.cell(index))
    }

    /// Borrow the checked logical dimensions.
    pub fn shape(&self) -> &GridShape {
        &self.shape
    }

    /// Run prepared fuel records and scalar/per-cell overrides into owned columns.
    pub fn run(
        &self,
        plan: &SurfacePlan,
        fuels: &PreparedFuels,
        tweaks: &SurfaceTweakColumns<'_>,
        threads: usize,
    ) -> Result<SurfaceRasterOutput, ParallelError<BehaveError>> {
        self.run_range(
            0..self.shape.len(),
            self.shape.clone(),
            plan,
            SurfaceSource::Fuels(fuels),
            tweaks,
            threads,
        )
    }

    /// Resume supplied fuelbed columns, ignoring the grid's fuel IDs and moisture stages.
    pub fn run_from_fuelbeds(
        &self,
        plan: &SurfacePlan,
        fuelbeds: &FuelbedColumns<'_>,
        tweaks: &SurfaceTweakColumns<'_>,
        threads: usize,
    ) -> Result<SurfaceRasterOutput, ParallelError<BehaveError>> {
        self.run_range(
            0..self.shape.len(),
            self.shape.clone(),
            plan,
            SurfaceSource::Fuelbeds(fuelbeds),
            tweaks,
            threads,
        )
    }

    /// Evaluate a logical range, returning a one-dimensional tile in the same order.
    /// Input/override slices still describe the full raster; external window readers
    /// can instead construct a raster whose shape is already the window shape.
    pub fn run_tile(
        &self,
        range: Range<usize>,
        plan: &SurfacePlan,
        fuels: &PreparedFuels,
        tweaks: &SurfaceTweakColumns<'_>,
        threads: usize,
    ) -> Result<SurfaceRasterOutput, ParallelError<BehaveError>> {
        let shape = tile_shape(&range, self.shape.len()).map_err(ParallelError::Work)?;
        self.run_range(
            range,
            shape,
            plan,
            SurfaceSource::Fuels(fuels),
            tweaks,
            threads,
        )
    }

    fn run_range(
        &self,
        range: Range<usize>,
        shape: GridShape,
        plan: &SurfacePlan,
        source: SurfaceSource<'_, '_>,
        tweaks: &SurfaceTweakColumns<'_>,
        threads: usize,
    ) -> Result<SurfaceRasterOutput, ParallelError<BehaveError>> {
        tweaks.validate(&self.shape).map_err(ParallelError::Work)?;
        if let SurfaceSource::Fuelbeds(columns) = source {
            columns.validate(&self.shape).map_err(ParallelError::Work)?;
        }
        let start = range.start;
        let mut output = SurfaceRasterOutput::run_with(shape, threads, |offset| {
            let index = start + offset;
            if !self.inputs.valid(index) {
                return Ok::<_, BehaveError>(SurfaceResult::default());
            }
            let overrides = tweaks.cell(index);
            Ok(match source {
                SurfaceSource::Fuelbeds(columns) => {
                    plan.run_from_fuelbed(&self.inputs.cell(index), &columns.cell(index), overrides)
                }
                SurfaceSource::Fuels(fuels) => self
                    .evaluate_cell(index, plan, fuels, overrides)
                    .expect("validated raster index"),
            })
        })?;
        output.validity = validity(self.inputs.validity, range)?;
        Ok(output)
    }
}
