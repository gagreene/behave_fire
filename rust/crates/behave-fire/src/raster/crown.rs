//! Crown raster execution borrows one model-10 fuel and independent Surface columns.
use super::{tile_shape, validity, CrownColumns, CrownRasterOutput, CrownTweakColumns};
use crate::crown::CrownResult;
use crate::engine::{CrownCell, CrownPlan, CrownTweaks, PreparedFuels};
use crate::{BehaveError, GridShape, ParallelError};
use std::ops::Range;

/// Validated borrowed base-unit Crown inputs and supplied Surface values.
#[derive(Clone, Debug)]
pub struct CrownRaster<'a> {
    shape: GridShape,
    inputs: CrownColumns<'a>,
}

impl<'a> CrownRaster<'a> {
    /// Borrow the validated inputs when creating an independently owned session.
    pub fn columns(&self) -> &CrownColumns<'a> {
        &self.inputs
    }

    /// Validate all column lengths before any native calculation or worker launch.
    pub fn new(shape: GridShape, inputs: CrownColumns<'a>) -> Result<Self, BehaveError> {
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
        plan: &CrownPlan,
        fuels: &PreparedFuels,
        tweaks: CrownTweaks,
    ) -> Option<CrownResult> {
        if index >= self.shape.len() {
            return None;
        }
        if !self.inputs.valid(index) {
            return Some(CrownResult::default());
        }
        Some(plan.run_cell(
            &self.inputs.cell(index),
            &self.inputs.surface(index),
            fuels.for_grid(10),
            tweaks,
        ))
    }

    /// Read validated spatial inputs for separately typed supplied Surface values.
    ///
    /// Returns `None` for an out-of-range or explicitly invalid pixel.
    pub fn input_cell(&self, index: usize) -> Option<CrownCell> {
        (index < self.shape.len() && self.inputs.valid(index)).then(|| self.inputs.cell(index))
    }
    /// Read independent supplied Surface quantities at a validated cell.
    pub(crate) fn supplied_surface(
        &self,
        index: usize,
    ) -> Option<crate::crown::SurfaceFireInputs<f64>> {
        (index < self.shape.len() && self.inputs.valid(index)).then(|| self.inputs.surface(index))
    }

    /// Borrow the checked logical dimensions.
    pub fn shape(&self) -> &GridShape {
        &self.shape
    }

    /// Compute a raster using independent Surface fields and Crown override columns.
    pub fn run(
        &self,
        plan: &CrownPlan,
        fuels: &PreparedFuels,
        tweaks: &CrownTweakColumns<'_>,
        threads: usize,
    ) -> Result<CrownRasterOutput, ParallelError<BehaveError>> {
        self.run_range(
            0..self.shape.len(),
            self.shape.clone(),
            plan,
            fuels,
            tweaks,
            threads,
        )
    }

    /// Compute a logical range as one row-major tile; callers retain window offsets.
    pub fn run_tile(
        &self,
        range: Range<usize>,
        plan: &CrownPlan,
        fuels: &PreparedFuels,
        tweaks: &CrownTweakColumns<'_>,
        threads: usize,
    ) -> Result<CrownRasterOutput, ParallelError<BehaveError>> {
        let shape = tile_shape(&range, self.shape.len()).map_err(ParallelError::Work)?;
        self.run_range(range, shape, plan, fuels, tweaks, threads)
    }

    fn run_range(
        &self,
        range: Range<usize>,
        shape: GridShape,
        plan: &CrownPlan,
        fuels: &PreparedFuels,
        tweaks: &CrownTweakColumns<'_>,
        threads: usize,
    ) -> Result<CrownRasterOutput, ParallelError<BehaveError>> {
        tweaks.validate(&self.shape).map_err(ParallelError::Work)?;
        let start = range.start;
        let mut output = CrownRasterOutput::run_with(shape, threads, |offset| {
            Ok::<_, BehaveError>(
                self.evaluate_cell(start + offset, plan, fuels, tweaks.cell(start + offset))
                    .expect("validated raster index"),
            )
        })?;
        output.validity = validity(self.inputs.validity, range)?;
        Ok(output)
    }
}
