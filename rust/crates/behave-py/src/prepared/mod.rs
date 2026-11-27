//! Primitive component preparation, with no numeric-provenance decoder.
mod keys;
pub(crate) use keys::Field;
pub(crate) mod primitive;
pub(crate) use primitive::Prepared as PreparedArguments;
pub(crate) mod controls;
pub(crate) mod selectors;
