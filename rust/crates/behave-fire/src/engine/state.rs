//! Reusable numeric cell state; absent slots are not fabricated scientific inputs.
use super::{Field, FieldKind};
use crate::BehaveError;
const WORDS: usize = Field::COUNT.div_ceil(64);

/// A small reusable float64 state with separate exact categories and flags.
#[derive(Clone, Debug)]
pub struct CellState {
    values: [f64; Field::COUNT],
    codes: [i32; 2],
    flags: [bool; 6],
    present: [u64; WORDS],
    available: [u64; WORDS],
}
impl Default for CellState {
    fn default() -> Self {
        Self::new()
    }
}
impl CellState {
    /// Construct no present inputs/results; unused payloads cannot be read.
    pub fn new() -> Self {
        Self {
            values: [f64::NAN; Field::COUNT],
            codes: [-1; 2],
            flags: [false; 6],
            present: [0; WORDS],
            available: [0; WORDS],
        }
    }
    fn mark(&mut self, field: Field, available: bool) {
        let (word, mask) = (field as usize / 64, 1_u64 << (field as usize % 64));
        self.present[word] |= mask;
        if available {
            self.available[word] |= mask;
        } else {
            self.available[word] &= !mask;
        }
    }
    fn check(&self, field: Field, kind: FieldKind) -> Result<(), BehaveError> {
        if field.kind() != kind {
            return Err(BehaveError::InvalidFieldAccess {
                field: field.name().into(),
                reason: "wrong typed field accessor",
            });
        }
        if !self.is_present(field) {
            return Err(BehaveError::InvalidFieldAccess {
                field: field.name().into(),
                reason: "field has not been supplied or computed",
            });
        }
        Ok(())
    }
    /// Whether a field was supplied/computed, including an intentional NaN.
    pub fn is_present(&self, field: Field) -> bool {
        self.present[field as usize / 64] & (1_u64 << (field as usize % 64)) != 0
    }
    /// Whether present data are available, distinct from their numeric payload.
    pub fn is_available(&self, field: Field) -> bool {
        self.available[field as usize / 64] & (1_u64 << (field as usize % 64)) != 0
    }
    /// Set a physical value, retaining NaN as present but unavailable.
    pub fn set_physical(&mut self, field: Field, value: f64) -> Result<(), BehaveError> {
        if field.kind() != FieldKind::Physical {
            return Err(BehaveError::InvalidFieldAccess {
                field: field.name().into(),
                reason: "expected physical field",
            });
        }
        self.values[field as usize] = value;
        self.mark(field, !value.is_nan());
        Ok(())
    }
    /// Read a supplied/computed float64 value; known-unavailable values remain NaN.
    pub fn physical(&self, field: Field) -> Result<f64, BehaveError> {
        self.check(field, FieldKind::Physical)?;
        Ok(self.values[field as usize])
    }
    /// Set an exact identifier/category without a float conversion.
    pub fn set_code(&mut self, field: Field, value: i32) -> Result<(), BehaveError> {
        let slot = field
            .code_slot()
            .ok_or_else(|| BehaveError::InvalidFieldAccess {
                field: field.name().into(),
                reason: "expected category/identifier field",
            })?;
        self.codes[slot] = value;
        self.mark(field, field != Field::FireType || value >= 0);
        Ok(())
    }
    /// Read an exact category/identifier.
    pub fn code(&self, field: Field) -> Result<i32, BehaveError> {
        self.check(field, FieldKind::Code)?;
        Ok(self.codes[field.code_slot().expect("validated code")])
    }
    /// Set a definition/validity flag; false is a known value.
    pub fn set_flag(&mut self, field: Field, value: bool) -> Result<(), BehaveError> {
        let slot = field
            .flag_slot()
            .ok_or_else(|| BehaveError::InvalidFieldAccess {
                field: field.name().into(),
                reason: "expected boolean field",
            })?;
        self.flags[slot] = value;
        self.mark(field, true);
        Ok(())
    }
    /// Read a known boolean value.
    pub fn flag(&self, field: Field) -> Result<bool, BehaveError> {
        self.check(field, FieldKind::Flag)?;
        Ok(self.flags[field.flag_slot().expect("validated flag")])
    }
    /// Reset row presence cheaply without permitting reads of stale payloads.
    pub fn reset_cell(&mut self) {
        self.present.fill(0);
        self.available.fill(0);
    }
}
