//! Crown raster execution borrows one model-10 fuel and independent Surface columns.
use super::{tile_shape, validity, CrownColumns, CrownRasterOutput, CrownTweakColumns};
use crate::crown::CrownResult;
use crate::engine::{CrownPlan, PreparedFuels};
use crate::{BehaveError, GridShape, ParallelError};
use std::ops::Range;

/// Validated borrowed base-unit Crown inputs and supplied Surface values.
#[derive(Clone, Debug)]
pub struct CrownRaster<'a> {
    shape: GridShape,
    inputs: CrownColumns<'a>,
}

impl<'a> CrownRaster<'a> {
    /// Validate all column lengths before any native calculation or worker launch.
    pub fn new(shape: GridShape, inputs: CrownColumns<'a>) -> Result<Self, BehaveError> {
        inputs.validate(&shape)?;
        Ok(Self { shape, inputs })
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
        let fuel = fuels.for_grid(10);
        let mut output = CrownRasterOutput::run_with(shape, threads, |offset| {
            let index = start + offset;
            if !self.inputs.valid(index) {
                return Ok::<_, BehaveError>(CrownResult::default());
            }
            Ok(plan.run_cell(
                &self.inputs.cell(index),
                &self.inputs.surface(index),
                fuel,
                tweaks.cell(index),
            ))
        })?;
        output.validity = validity(self.inputs.validity, range)?;
        Ok(output)
    }
}
