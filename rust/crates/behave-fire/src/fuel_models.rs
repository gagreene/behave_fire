//! Owned fuel registries, typed custom registration, and grid lookup semantics.
use std::collections::BTreeMap;

use crate::units::{
    FractionUnit, HeatOfCombustionUnit, LengthUnit, LoadingUnit, SurfaceAreaToVolumeUnit,
};

/// One fuel record in the reference's internal base units.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FuelModel {
    /// Fuel identifier, truncated to three Unicode characters for custom models.
    pub code: String,
    /// Human-readable name.
    pub name: String,
    /// Fuel bed depth in feet.
    pub fuel_bed_depth: f64,
    /// Fractional dead moisture of extinction.
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
    /// One-hour dead SAVR in ft²/ft³.
    pub savr_1h: f64,
    /// Live herbaceous SAVR in ft²/ft³.
    pub savr_live_herb: f64,
    /// Live woody SAVR in ft²/ft³.
    pub savr_live_woody: f64,
    /// Whether dynamic herbaceous load transfer applies.
    pub is_dynamic: bool,
    /// Whether standard registration protects this slot.
    pub is_reserved: bool,
    /// Whether this record populates grid lookup arrays.
    pub is_defined: bool,
}

/// Custom values and the units in which those values were supplied.
///
/// Registration converts all numeric fields exactly once and overrides reserved/
/// defined flags. Scientific values, including negative values and NaNs, are
/// preserved as in Python. Invalid unit codes fail at enum construction.
#[derive(Debug, Clone, PartialEq)]
pub struct CustomFuelModel {
    /// Custom record values, interpreted using the unit fields below.
    pub model: FuelModel,
    /// Units for depth.
    pub length_units: LengthUnit,
    /// Units for dead moisture of extinction.
    pub moisture_units: FractionUnit,
    /// Units for both heat values.
    pub heat_of_combustion_units: HeatOfCombustionUnit,
    /// Units for all five loading values.
    pub loading_units: LoadingUnit,
    /// Units for all three SAVR values.
    pub savr_units: SurfaceAreaToVolumeUnit,
}

impl Default for CustomFuelModel {
    /// Construct an empty record with base units for every dimension.
    fn default() -> Self {
        Self {
            model: FuelModel::default(),
            length_units: LengthUnit::Feet,
            moisture_units: FractionUnit::Fraction,
            heat_of_combustion_units: HeatOfCombustionUnit::BtusPerPound,
            loading_units: LoadingUnit::PoundsPerSquareFoot,
            savr_units: SurfaceAreaToVolumeUnit::SquareFeetOverCubicFeet,
        }
    }
}

/// Mutable fuel registry; cloning captures an independent owned snapshot.
#[derive(Debug, Clone, PartialEq)]
pub struct FuelModels {
    records: BTreeMap<i32, FuelModel>,
    empty: FuelModel,
}

include!("generated/fuel_models.rs");

impl Default for FuelModels {
    /// Construct every standard and reserved record without external resources.
    fn default() -> Self {
        Self {
            records: standard_fuel_models(),
            empty: FuelModel::default(),
        }
    }
}

impl FuelModels {
    /// Remove a custom record; absent slots succeed and reserved slots fail.
    pub fn clear_custom_fuel_model(&mut self, number: i32) -> bool {
        if self.is_fuel_model_reserved(number) {
            return false;
        }
        self.records.remove(&number);
        true
    }

    /// Test whether a record exists, including undefined reserved records.
    pub fn contains(&self, number: i32) -> bool {
        self.records.contains_key(&number)
    }

    /// Look up a fuel for grid math, matching C++'s supported identifiers 1..=256.
    ///
    /// Out-of-range numbers use the undefined row. Index zero is empty even if a
    /// custom model zero exists. Missing and undefined records yield all-zero
    /// properties; direct `get` preserves the original record and unbounded IDs.
    pub fn for_grid(&self, number: i32) -> &FuelModel {
        if !(1..=256).contains(&number) {
            return &self.empty;
        }
        self.get(number)
            .filter(|record| record.is_defined)
            .unwrap_or(&self.empty)
    }

    /// Restore an exact owned snapshot without registration-time normalization.
    ///
    /// Missing standard records stay missing; flags and base-unit values are
    /// retained as supplied. Grid lookup still applies its documented bounds
    /// and empty-record rules independently of this raw metadata snapshot.
    pub fn from_records(records: BTreeMap<i32, FuelModel>) -> Self {
        Self {
            records,
            empty: FuelModel::default(),
        }
    }

    /// Return the actual record for any registered integer, without clamping.
    pub fn get(&self, number: i32) -> Option<&FuelModel> {
        self.records.get(&number)
    }

    /// Return an empty string when the record does not exist.
    pub fn get_fuel_code(&self, number: i32) -> &str {
        self.get(number).map_or("", |record| &record.code)
    }

    /// Return live herbaceous loading, or zero if the slot is absent.
    pub fn get_fuel_load_live_herbaceous(&self, number: i32, units: Option<LoadingUnit>) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.live_herb, |unit| unit.from_base(record.live_herb))
        })
    }

    /// Return live woody loading, or zero if the slot is absent.
    pub fn get_fuel_load_live_woody(&self, number: i32, units: Option<LoadingUnit>) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.live_woody, |unit| unit.from_base(record.live_woody))
        })
    }

    /// Return hundred-hour dead loading, or zero if the slot is absent.
    pub fn get_fuel_load_hundred_hour(&self, number: i32, units: Option<LoadingUnit>) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.dead_100h, |unit| unit.from_base(record.dead_100h))
        })
    }

    /// Return one-hour dead loading, or zero if the slot is absent.
    pub fn get_fuel_load_one_hour(&self, number: i32, units: Option<LoadingUnit>) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.dead_1h, |unit| unit.from_base(record.dead_1h))
        })
    }

    /// Return ten-hour dead loading, or zero if the slot is absent.
    pub fn get_fuel_load_ten_hour(&self, number: i32, units: Option<LoadingUnit>) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.dead_10h, |unit| unit.from_base(record.dead_10h))
        })
    }

    /// Alias for direct, unclamped lookup.
    pub fn get_fuel_model(&self, number: i32) -> Option<&FuelModel> {
        self.get(number)
    }

    /// Return an empty string when the record does not exist.
    pub fn get_fuel_name(&self, number: i32) -> &str {
        self.get(number).map_or("", |record| &record.name)
    }

    /// Return depth in the requested units, or zero when absent.
    pub fn get_fuelbed_depth(&self, number: i32, units: Option<LengthUnit>) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.fuel_bed_depth, |unit| {
                unit.from_base(record.fuel_bed_depth)
            })
        })
    }

    /// Return dead heat in the requested units, or zero when absent.
    pub fn get_heat_of_combustion_dead(
        &self,
        number: i32,
        units: Option<HeatOfCombustionUnit>,
    ) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.heat_of_combustion_dead, |unit| {
                unit.from_base(record.heat_of_combustion_dead)
            })
        })
    }

    /// Return live heat in the requested units, or zero when absent.
    pub fn get_heat_of_combustion_live(
        &self,
        number: i32,
        units: Option<HeatOfCombustionUnit>,
    ) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.heat_of_combustion_live, |unit| {
                unit.from_base(record.heat_of_combustion_live)
            })
        })
    }

    /// Return whether dynamic load transfer applies; absent slots return false.
    pub fn get_is_dynamic(&self, number: i32) -> bool {
        self.get(number).is_some_and(|record| record.is_dynamic)
    }

    /// Return extinction moisture in the requested units, or zero when absent.
    pub fn get_moisture_of_extinction_dead(&self, number: i32, units: Option<FractionUnit>) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.moisture_of_extinction_dead, |unit| {
                unit.from_base(record.moisture_of_extinction_dead)
            })
        })
    }

    /// Borrow an actual registered record mutably, including reserved records.
    /// This explicit low-level access mirrors direct Python dictionary mutation.
    pub fn get_mut(&mut self, number: i32) -> Option<&mut FuelModel> {
        self.records.get_mut(&number)
    }

    /// Return live herbaceous SAVR, or zero when absent.
    pub fn get_savr_live_herbaceous(
        &self,
        number: i32,
        units: Option<SurfaceAreaToVolumeUnit>,
    ) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.savr_live_herb, |unit| {
                unit.from_base(record.savr_live_herb)
            })
        })
    }

    /// Return live woody SAVR, or zero when absent.
    pub fn get_savr_live_woody(&self, number: i32, units: Option<SurfaceAreaToVolumeUnit>) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.savr_live_woody, |unit| {
                unit.from_base(record.savr_live_woody)
            })
        })
    }

    /// Return one-hour dead SAVR, or zero when absent.
    pub fn get_savr_one_hour(&self, number: i32, units: Option<SurfaceAreaToVolumeUnit>) -> f64 {
        self.get(number).map_or(0.0, |record| {
            units.map_or(record.savr_1h, |unit| unit.from_base(record.savr_1h))
        })
    }

    /// Test zero fuel loading using Python truth semantics, including NaNs.
    pub fn is_all_fuel_load_zero(&self, number: i32) -> bool {
        self.get(number).map_or(true, |record| {
            [
                record.dead_1h,
                record.dead_10h,
                record.dead_100h,
                record.live_herb,
                record.live_woody,
            ]
            .iter()
            .all(|value| *value == 0.0)
        })
    }

    /// Return whether the slot contributes to grid lookup arrays.
    pub fn is_fuel_model_defined(&self, number: i32) -> bool {
        (1..=256).contains(&number) && self.get(number).is_some_and(|record| record.is_defined)
    }

    /// Return whether custom registration and clearing protect this slot.
    pub fn is_fuel_model_reserved(&self, number: i32) -> bool {
        self.get(number).is_some_and(|record| record.is_reserved)
    }

    /// Borrow every record, including custom IDs outside the grid range.
    pub fn records(&self) -> &BTreeMap<i32, FuelModel> {
        &self.records
    }

    /// Register custom data after converting every physical field once.
    /// Return false without conversion or mutation for reserved or out-of-range slots.
    pub fn set_custom_fuel_model(&mut self, number: i32, input: CustomFuelModel) -> bool {
        if !(1..=256).contains(&number) || self.is_fuel_model_reserved(number) {
            return false;
        }
        let mut model = input.model;
        model.code = model.code.chars().take(3).collect();
        model.fuel_bed_depth = input.length_units.to_base(model.fuel_bed_depth);
        model.moisture_of_extinction_dead = input
            .moisture_units
            .to_base(model.moisture_of_extinction_dead);
        model.heat_of_combustion_dead = input
            .heat_of_combustion_units
            .to_base(model.heat_of_combustion_dead);
        model.heat_of_combustion_live = input
            .heat_of_combustion_units
            .to_base(model.heat_of_combustion_live);
        model.dead_1h = input.loading_units.to_base(model.dead_1h);
        model.dead_10h = input.loading_units.to_base(model.dead_10h);
        model.dead_100h = input.loading_units.to_base(model.dead_100h);
        model.live_herb = input.loading_units.to_base(model.live_herb);
        model.live_woody = input.loading_units.to_base(model.live_woody);
        model.savr_1h = input.savr_units.to_base(model.savr_1h);
        model.savr_live_herb = input.savr_units.to_base(model.savr_live_herb);
        model.savr_live_woody = input.savr_units.to_base(model.savr_live_woody);
        model.is_reserved = false;
        model.is_defined = true;
        self.records.insert(number, model);
        true
    }

    /// Clone the registry into an independent immutable-by-convention snapshot.
    pub fn snapshot(&self) -> Self {
        self.clone()
    }
}

impl std::ops::Index<i32> for FuelModels {
    type Output = FuelModel;

    /// Index an actual record, panicking for an absent slot like a map index.
    fn index(&self, number: i32) -> &Self::Output {
        &self.records[&number]
    }
}
