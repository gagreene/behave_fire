//! Complete explicit-reference species registry and case-insensitive operations.

/// An integer does not identify a declared species enum variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidSpeciesEnum {
    /// Enum name.
    pub kind: &'static str,
    /// Rejected integer code.
    pub code: i32,
}

impl std::fmt::Display for InvalidSpeciesEnum {
    /// Describe the invalid enum discriminant.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid {} code {}", self.kind, self.code)
    }
}

impl std::error::Error for InvalidSpeciesEnum {}

macro_rules! species_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident = $value:expr),+ $(,)? }) => {
        $(#[$meta])*
        #[repr(i32)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name {
            $(
                #[doc = concat!("Reference variant `", stringify!($variant), "`.")]
                $variant = $value
            ),+
        }

        impl $name {
            /// Every declared variant in Python definition order.
            pub const ALL: &'static [Self] = &[$( Self::$variant ),+];

            /// Return the exact Python discriminant.
            pub const fn code(self) -> i32 { self as i32 }
        }

        impl TryFrom<i32> for $name {
            type Error = InvalidSpeciesEnum;

            /// Validate an integer discriminant.
            fn try_from(code: i32) -> Result<Self, Self::Error> {
                match code {
                    $( $value => Ok(Self::$variant), )+
                    _ => Err(InvalidSpeciesEnum { kind: stringify!($name), code }),
                }
            }
        }
    };
}

species_enum! {
    /// Geographic Area Coordination Centers, preserving the reference codes.
    Gacc { NotSet = -1, Alaska = 1, California = 2, EasternArea = 3, GreatBasin = 4,
        NorthernRockies = 5, Northwest = 6, RockyMountain = 7, SouthernArea = 8, Southwest = 9 }
}

species_enum! {
    /// Mortality equation type; unsupported science remains unsupported.
    EquationType { NotSet = -1, CrownScorch = 0, BoleChar = 1, CrownDamage = 2 }
}

species_enum! {
    /// The explicit reference declares only the unset crown-damage code.
    CrownDamageEquationCode { NotSet = -1 }
}

/// Owned species record with the complete explicit Python field set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeciesRecord {
    /// Species code.
    pub species_code: String,
    /// Scientific name.
    pub scientific_name: String,
    /// Common name.
    pub common_name: String,
    /// Mortality equation selector.
    pub mortality_equation_number: i32,
    /// Bark equation selector.
    pub bark_equation_number: i32,
    /// Crown coefficient selector.
    pub crown_coefficient_code: i32,
    /// Alaska availability.
    pub alaska: bool,
    /// California availability.
    pub california: bool,
    /// Eastern Area availability.
    pub eastern_area: bool,
    /// Great Basin availability.
    pub great_basin: bool,
    /// Northern Rockies availability.
    pub northern_rockies: bool,
    /// Northwest availability.
    pub northwest: bool,
    /// Rocky Mountain availability.
    pub rocky_mountain: bool,
    /// Southern Area availability.
    pub southern_area: bool,
    /// Southwest availability.
    pub southwest: bool,
    /// Mortality equation family.
    pub equation_type: EquationType,
    /// Crown damage equation selector.
    pub crown_damage_equation_code: CrownDamageEquationCode,
}

impl Default for SpeciesRecord {
    /// Construct the same empty sentinel record as the Python constructor.
    fn default() -> Self {
        Self {
            species_code: String::new(),
            scientific_name: String::new(),
            common_name: String::new(),
            mortality_equation_number: -1,
            bark_equation_number: -1,
            crown_coefficient_code: 0,
            alaska: false,
            california: false,
            eastern_area: false,
            great_basin: false,
            northern_rockies: false,
            northwest: false,
            rocky_mountain: false,
            southern_area: false,
            southwest: false,
            equation_type: EquationType::NotSet,
            crown_damage_equation_code: CrownDamageEquationCode::NotSet,
        }
    }
}

impl SpeciesRecord {
    /// Test whether this record is available in a region; NotSet returns false.
    pub fn get_gacc_availability(&self, region: Gacc) -> bool {
        match region {
            Gacc::NotSet => false,
            Gacc::Alaska => self.alaska,
            Gacc::California => self.california,
            Gacc::EasternArea => self.eastern_area,
            Gacc::GreatBasin => self.great_basin,
            Gacc::NorthernRockies => self.northern_rockies,
            Gacc::Northwest => self.northwest,
            Gacc::RockyMountain => self.rocky_mountain,
            Gacc::SouthernArea => self.southern_area,
            Gacc::Southwest => self.southwest,
        }
    }
}

/// Mutable, insertion-ordered species registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeciesMasterTable {
    /// Actual records, preserving direct mutation and custom duplicate behavior.
    pub records: Vec<SpeciesRecord>,
}

include!("generated/species.rs");

impl Default for SpeciesMasterTable {
    /// Load all 24 reference records without Python or external data.
    fn default() -> Self {
        Self {
            records: standard_species(),
        }
    }
}

impl SpeciesMasterTable {
    /// Append a custom record; duplicate species codes are allowed.
    pub fn add_species_record(&mut self, record: SpeciesRecord) {
        self.records.push(record);
    }

    /// Test for the first case-insensitive match before an empty-code sentinel.
    pub fn contains_species_code(&self, code: &str) -> bool {
        self.get_species_record(code).is_some()
    }

    /// Borrow the actual ordered records, including entries after a sentinel.
    pub fn get_all_records(&self) -> &[SpeciesRecord] {
        &self.records
    }

    /// Return the total count, including custom records and empty sentinels.
    pub fn get_number_of_records(&self) -> usize {
        self.records.len()
    }

    /// Return the first case-insensitive match before an empty-code sentinel.
    pub fn get_species_record(&self, code: &str) -> Option<&SpeciesRecord> {
        let index = self.get_species_table_index_from_species_code(code);
        self.get_species_record_at_index(index)
    }

    /// Return an ordered record by index; negative and out-of-range indices fail.
    pub fn get_species_record_at_index(&self, index: i32) -> Option<&SpeciesRecord> {
        usize::try_from(index)
            .ok()
            .and_then(|index| self.records.get(index))
    }

    /// Return the first case-insensitive code and equation match before a sentinel.
    pub fn get_species_record_by_code_and_type(
        &self,
        code: &str,
        equation: EquationType,
    ) -> Option<&SpeciesRecord> {
        let index =
            self.get_species_table_index_from_species_code_and_equation_type(code, equation);
        self.get_species_record_at_index(index)
    }

    /// Filter all actual records by region, preserving order and duplicates.
    pub fn get_species_records_for_gacc_region(&self, region: Gacc) -> Vec<&SpeciesRecord> {
        self.records
            .iter()
            .filter(|record| record.get_gacc_availability(region))
            .collect()
    }

    /// Filter all actual records by region and equation type.
    pub fn get_species_records_for_gacc_region_and_equation_type(
        &self,
        region: Gacc,
        equation: EquationType,
    ) -> Vec<&SpeciesRecord> {
        self.records
            .iter()
            .filter(|record| {
                record.get_gacc_availability(region) && record.equation_type == equation
            })
            .collect()
    }

    /// Return the first matching index, or -1; an empty-code record ends search.
    pub fn get_species_table_index_from_species_code(&self, code: &str) -> i32 {
        let code = code.to_uppercase();
        self.records
            .iter()
            .take_while(|record| !record.species_code.is_empty())
            .position(|record| record.species_code.to_uppercase() == code)
            .map_or(-1, |index| index as i32)
    }

    /// Return the first matching code/equation index, or -1, stopping at a sentinel.
    pub fn get_species_table_index_from_species_code_and_equation_type(
        &self,
        code: &str,
        equation: EquationType,
    ) -> i32 {
        let code = code.to_uppercase();
        self.records
            .iter()
            .take_while(|record| !record.species_code.is_empty())
            .position(|record| {
                record.species_code.to_uppercase() == code && record.equation_type == equation
            })
            .map_or(-1, |index| index as i32)
    }

    /// Replace current records with the complete default reference table.
    pub fn initialize_master_table(&mut self) {
        self.records = standard_species();
    }

    /// Insert a fully typed record; equivalent to Python's fieldwise insertion.
    pub fn insert_record(&mut self, record: SpeciesRecord) {
        self.add_species_record(record);
    }

    /// Remove the first case-insensitive match; duplicate records remain.
    pub fn remove_species_record(&mut self, code: &str) -> bool {
        let index = self.get_species_table_index_from_species_code(code);
        if index < 0 {
            return false;
        }
        self.records.remove(index as usize);
        true
    }
}
