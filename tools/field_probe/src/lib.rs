//! Compile the actual boundary data modules without requiring a Python linker.
#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/allocation_probe.rs"]
mod allocation_probe;
#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/fields.rs"]
mod fields;

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/keys.rs"]
mod keys;

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/controls.rs"]
mod controls;

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/units.rs"]
mod units;

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/prepared/unit_selectors.rs"]
mod unit_selectors;

#[allow(dead_code)]
#[path = "../../../rust/crates/behave-py/src/column_output.rs"]
mod column_output;
