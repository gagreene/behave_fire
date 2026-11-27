//! Physical data and exact controls have different boundary contracts.
use behave_fire::BehaveError;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum InputRole {
    Physical,
    Identifier,
    Index,
    Selector,
    Flag,
    IterationLimit,
    Mode,
    LookupTable,
}

#[derive(Debug)]
pub(crate) enum BoundaryError {
    Shape(BehaveError),
    InvalidRole(InputRole),
    MissingInput(usize),
    DuplicateOutput(usize),
    Broadcast { left: Vec<usize>, right: Vec<usize> },
}
impl From<BehaveError> for BoundaryError {
    fn from(error: BehaveError) -> Self {
        Self::Shape(error)
    }
}
impl fmt::Display for BoundaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(error) => error.fmt(f),
            Self::InvalidRole(role) => write!(f, "invalid storage for {role:?}"),
            Self::MissingInput(slot) => write!(f, "missing declared input slot {slot}"),
            Self::DuplicateOutput(slot) => write!(f, "output slot {slot} was declared twice"),
            Self::Broadcast { left, right } => {
                write!(f, "cannot broadcast {left:?} with {right:?}")
            }
        }
    }
}
impl std::error::Error for BoundaryError {}
