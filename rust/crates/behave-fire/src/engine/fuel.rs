//! Prepared numeric fuel snapshots without names, dictionaries, or heap-owned fields.
use crate::{FuelModel, FuelModels};

/// Numerical fuel properties consumed by particle and fire calculations.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PreparedFuel {
    /// Fuel bed depth in feet.
    pub fuel_bed_depth: f64,
    /// Dead moisture of extinction as a fraction.
    pub moisture_of_extinction_dead: f64,
    /// Dead heat of combustion in BTU/lb.
    pub heat_of_combustion_dead: f64,
    /// Live heat of combustion in BTU/lb.
    pub heat_of_combustion_live: f64,
    /// One-hour dead loading in lb/ft².
    pub dead_1h: f64,
    /// Ten-hour dead loading in lb/ft².
    pub dead_10h: f64,
    /// Hundred-hour dead loading in lb/ft².
    pub dead_100h: f64,
    /// Live herbaceous loading in lb/ft².
    pub live_herb: f64,
    /// Live woody loading in lb/ft².
    pub live_woody: f64,
    /// One-hour surface area to volume ratio in inverse feet.
    pub savr_1h: f64,
    /// Live herbaceous surface area to volume ratio in inverse feet.
    pub savr_live_herb: f64,
    /// Live woody surface area to volume ratio in inverse feet.
    pub savr_live_woody: f64,
    /// Whether live herbaceous load transfers as its moisture decreases.
    pub is_dynamic: bool,
    /// Whether final Surface outputs represent a defined fuel.
    pub is_defined: bool,
}

impl PreparedFuel {
    /// Copy physical properties without normalizing undefined rows or cloning names.
    pub fn from_model(model: &FuelModel) -> Self {
        Self {
            fuel_bed_depth: model.fuel_bed_depth,
            moisture_of_extinction_dead: model.moisture_of_extinction_dead,
            heat_of_combustion_dead: model.heat_of_combustion_dead,
            heat_of_combustion_live: model.heat_of_combustion_live,
            dead_1h: model.dead_1h,
            dead_10h: model.dead_10h,
            dead_100h: model.dead_100h,
            live_herb: model.live_herb,
            live_woody: model.live_woody,
            savr_1h: model.savr_1h,
            savr_live_herb: model.savr_live_herb,
            savr_live_woody: model.savr_live_woody,
            is_dynamic: model.is_dynamic,
            is_defined: model.is_defined,
        }
    }
}

/// Immutable dense lookup snapshot prepared once for repeated calculations.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedFuels {
    rows: Box<[PreparedFuel; 261]>,
}

impl PreparedFuels {
    /// Snapshot every clamped grid slot using the registry's existing fallback rules.
    pub fn from_registry(fuels: &FuelModels) -> Self {
        Self {
            rows: Box::new(std::array::from_fn(|id| {
                PreparedFuel::from_model(fuels.for_grid(id as i32))
            })),
        }
    }

    /// Borrow a physical record, clamping the identifier to the valid 0–260 grid.
    pub fn for_grid(&self, id: i32) -> &PreparedFuel {
        &self.rows[id.clamp(0, 260) as usize]
    }
}
