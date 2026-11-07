//! Native snapshots own only normalized physical values or exact typed controls.
use super::contract::{BoundaryError, InputRole};
use behave_fire::GridShape;

pub(crate) struct PhysicalSnapshot {
    shape: GridShape,
    values: Vec<f64>,
}
impl PhysicalSnapshot {
    pub(crate) fn new(shape: GridShape, values: Vec<f64>) -> Result<Self, BoundaryError> {
        shape.validate_len("physical input", values.len())?;
        Ok(Self { shape, values })
    }
    pub(crate) fn shape(&self) -> &GridShape {
        &self.shape
    }
    pub(crate) fn values(&self) -> &[f64] {
        &self.values
    }
    pub(crate) fn into_values(self) -> Vec<f64> {
        self.values
    }
}

pub(crate) enum ControlValues {
    Signed(Vec<i64>),
    Unsigned(Vec<u64>),
    Flags(Vec<bool>),
}
impl ControlValues {
    fn len(&self) -> usize {
        match self {
            Self::Signed(values) => values.len(),
            Self::Unsigned(values) => values.len(),
            Self::Flags(values) => values.len(),
        }
    }
}
pub(crate) struct ControlSnapshot {
    role: InputRole,
    shape: GridShape,
    values: ControlValues,
}
impl ControlSnapshot {
    pub(crate) fn new(
        role: InputRole,
        shape: GridShape,
        values: ControlValues,
    ) -> Result<Self, BoundaryError> {
        let valid = match role {
            InputRole::Identifier
            | InputRole::Index
            | InputRole::Selector
            | InputRole::IterationLimit => matches!(
                values,
                ControlValues::Signed(_) | ControlValues::Unsigned(_)
            ),
            InputRole::Flag => matches!(values, ControlValues::Flags(_)),
            _ => false,
        };
        if !valid {
            return Err(BoundaryError::InvalidRole(role));
        }
        shape.validate_len("typed control", values.len())?;
        Ok(Self {
            role,
            shape,
            values,
        })
    }
    pub(crate) fn role(&self) -> InputRole {
        self.role
    }
    pub(crate) fn shape(&self) -> &GridShape {
        &self.shape
    }
    pub(crate) fn signed(&self) -> Option<&[i64]> {
        match &self.values {
            ControlValues::Signed(values) => Some(values),
            _ => None,
        }
    }
    pub(crate) fn unsigned(&self) -> Option<&[u64]> {
        match &self.values {
            ControlValues::Unsigned(values) => Some(values),
            _ => None,
        }
    }
    pub(crate) fn flags(&self) -> Option<&[bool]> {
        match &self.values {
            ControlValues::Flags(values) => Some(values),
            _ => None,
        }
    }
}
