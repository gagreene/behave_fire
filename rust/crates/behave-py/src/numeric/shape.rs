//! Compile genuine dependency-local output groups before any cell loop.
use super::contract::BoundaryError;
use super::snapshot::PhysicalSnapshot;
use behave_fire::GridShape;

pub(crate) struct OutputDependency {
    pub output_slot: usize,
    pub input_slots: Vec<usize>,
}
impl OutputDependency {
    pub(crate) fn new(output_slot: usize, input_slots: &[usize]) -> Self {
        let mut input_slots = input_slots.to_vec();
        input_slots.sort_unstable();
        input_slots.dedup();
        Self {
            output_slot,
            input_slots,
        }
    }
}
pub(crate) struct DependencyGroup {
    pub shape: GridShape,
    pub input_slots: Vec<usize>,
    pub output_slots: Vec<usize>,
}

fn broadcast(left: &[usize], right: &[usize]) -> Result<Vec<usize>, BoundaryError> {
    let rank = left.len().max(right.len());
    let mut shape = vec![1; rank];
    for offset in 0..rank {
        let a = left.len().checked_sub(offset + 1).map_or(1, |i| left[i]);
        let b = right.len().checked_sub(offset + 1).map_or(1, |i| right[i]);
        shape[rank - 1 - offset] = if a == b || b == 1 {
            a
        } else if a == 1 {
            b
        } else {
            return Err(BoundaryError::Broadcast {
                left: left.to_vec(),
                right: right.to_vec(),
            });
        };
    }
    Ok(shape)
}

pub(crate) fn plan_groups(
    inputs: &[GridShape],
    outputs: &[OutputDependency],
) -> Result<Vec<DependencyGroup>, BoundaryError> {
    let mut groups: Vec<DependencyGroup> = Vec::new();
    let mut seen = Vec::new();
    for output in outputs {
        if seen.contains(&output.output_slot) {
            return Err(BoundaryError::DuplicateOutput(output.output_slot));
        }
        seen.push(output.output_slot);
        let mut dimensions = Vec::new();
        for &slot in &output.input_slots {
            let shape = inputs.get(slot).ok_or(BoundaryError::MissingInput(slot))?;
            dimensions = broadcast(&dimensions, shape.dimensions())?;
        }
        let shape = GridShape::new(dimensions)?;
        if let Some(group) = groups
            .iter_mut()
            .find(|group| group.shape == shape && group.input_slots == output.input_slots)
        {
            group.output_slots.push(output.output_slot);
        } else {
            groups.push(DependencyGroup {
                shape,
                input_slots: output.input_slots.clone(),
                output_slots: vec![output.output_slot],
            });
        }
    }
    Ok(groups)
}

impl DependencyGroup {
    pub(crate) fn read_physical<'a>(
        &self,
        inputs: &'a [PhysicalSnapshot],
        slot: usize,
    ) -> Result<&'a [f64], BoundaryError> {
        if !self.input_slots.contains(&slot) {
            return Err(BoundaryError::MissingInput(slot));
        }
        inputs
            .get(slot)
            .map(PhysicalSnapshot::values)
            .ok_or(BoundaryError::MissingInput(slot))
    }
}
