//! Typed scalar conversions preserving explicit Python f64 operation order.
//! Base units and integer codes follow `behave_units.py` at reference 7933789.

/// An integer does not identify a declared unit variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidUnit {
    /// Unit enum name.
    pub kind: &'static str,
    /// Rejected integer code.
    pub code: i32,
}

impl std::fmt::Display for InvalidUnit {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "invalid {} unit code {}", self.kind, self.code)
    }
}

impl std::error::Error for InvalidUnit {}

/// Units for area, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AreaUnit {
    /// Python unit code 0.
    SquareFeet = 0,
    /// Python unit code 1.
    Acres = 1,
    /// Python unit code 2.
    Hectares = 2,
    /// Python unit code 3.
    SquareMeters = 3,
    /// Python unit code 4.
    SquareMiles = 4,
    /// Python unit code 5.
    SquareKilometers = 5,
}

impl AreaUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[
        Self::SquareFeet,
        Self::Acres,
        Self::Hectares,
        Self::SquareMeters,
        Self::SquareMiles,
        Self::SquareKilometers,
    ];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::SquareFeet => value,
            Self::Acres => value * 2.295684e-05,
            Self::Hectares => value * 9.2903036e-06,
            Self::SquareMeters => value * 0.0929030353835,
            Self::SquareMiles => value * 3.5870064279e-08,
            Self::SquareKilometers => value * 9.290304e-08,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::SquareFeet => value,
            Self::Acres => value * 43560.00216057611,
            Self::Hectares => value * 107639.10416709723,
            Self::SquareMeters => value * 10.76391041671,
            Self::SquareMiles => value * 27878400.0,
            Self::SquareKilometers => value * 10763910.416709721,
        }
    }
}

impl TryFrom<i32> for AreaUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::SquareFeet),
            1 => Ok(Self::Acres),
            2 => Ok(Self::Hectares),
            3 => Ok(Self::SquareMeters),
            4 => Ok(Self::SquareMiles),
            5 => Ok(Self::SquareKilometers),
            _ => Err(InvalidUnit {
                kind: "AreaUnit",
                code,
            }),
        }
    }
}

/// Units for basalarea, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BasalAreaUnit {
    /// Python unit code 0.
    SquareFeetPerAcre = 0,
    /// Python unit code 1.
    SquareMetersPerHectare = 1,
}

impl BasalAreaUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[Self::SquareFeetPerAcre, Self::SquareMetersPerHectare];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::SquareFeetPerAcre => value,
            Self::SquareMetersPerHectare => value / 4.35889,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::SquareFeetPerAcre => value,
            Self::SquareMetersPerHectare => value * 4.35889,
        }
    }
}

impl TryFrom<i32> for BasalAreaUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::SquareFeetPerAcre),
            1 => Ok(Self::SquareMetersPerHectare),
            _ => Err(InvalidUnit {
                kind: "BasalAreaUnit",
                code,
            }),
        }
    }
}

/// Units for length, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LengthUnit {
    /// Python unit code 0.
    Feet = 0,
    /// Python unit code 1.
    Inches = 1,
    /// Python unit code 2.
    Millimeters = 2,
    /// Python unit code 3.
    Centimeters = 3,
    /// Python unit code 4.
    Meters = 4,
    /// Python unit code 5.
    Chains = 5,
    /// Python unit code 6.
    Miles = 6,
    /// Python unit code 7.
    Kilometers = 7,
}

impl LengthUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[
        Self::Feet,
        Self::Inches,
        Self::Millimeters,
        Self::Centimeters,
        Self::Meters,
        Self::Chains,
        Self::Miles,
        Self::Kilometers,
    ];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::Feet => value,
            Self::Inches => value * 12.0,
            Self::Millimeters => value * 304.8000000012192,
            Self::Centimeters => value * 30.48,
            Self::Meters => value * 0.3048,
            Self::Chains => value * 0.0151515151515,
            Self::Miles => value * 0.0001893939393939394,
            Self::Kilometers => value * 0.0003048,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::Feet => value,
            Self::Inches => value * 0.08333333333333,
            Self::Millimeters => value * 0.003280839895,
            Self::Centimeters => value * 0.03280839895,
            Self::Meters => value * 3.2808398950131,
            Self::Chains => value * 66.0,
            Self::Miles => value * 5280.0,
            Self::Kilometers => value * 3280.8398950131,
        }
    }
}

impl TryFrom<i32> for LengthUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::Feet),
            1 => Ok(Self::Inches),
            2 => Ok(Self::Millimeters),
            3 => Ok(Self::Centimeters),
            4 => Ok(Self::Meters),
            5 => Ok(Self::Chains),
            6 => Ok(Self::Miles),
            7 => Ok(Self::Kilometers),
            _ => Err(InvalidUnit {
                kind: "LengthUnit",
                code,
            }),
        }
    }
}

/// Units for loading, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LoadingUnit {
    /// Python unit code 0.
    PoundsPerSquareFoot = 0,
    /// Python unit code 1.
    TonsPerAcre = 1,
    /// Python unit code 2.
    TonnesPerHectare = 2,
    /// Python unit code 3.
    KilogramsPerSquareMeter = 3,
}

impl LoadingUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[
        Self::PoundsPerSquareFoot,
        Self::TonsPerAcre,
        Self::TonnesPerHectare,
        Self::KilogramsPerSquareMeter,
    ];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::PoundsPerSquareFoot => value,
            Self::TonsPerAcre => value * 43.53804725927289,
            Self::TonnesPerHectare => value * 9.806650000749439,
            Self::KilogramsPerSquareMeter => value * 4.882394331746218,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::PoundsPerSquareFoot => value,
            Self::TonsPerAcre => value * 0.02296841643,
            Self::TonnesPerHectare => value * 0.10197162129,
            Self::KilogramsPerSquareMeter => value * 0.20481754075,
        }
    }
}

impl TryFrom<i32> for LoadingUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::PoundsPerSquareFoot),
            1 => Ok(Self::TonsPerAcre),
            2 => Ok(Self::TonnesPerHectare),
            3 => Ok(Self::KilogramsPerSquareMeter),
            _ => Err(InvalidUnit {
                kind: "LoadingUnit",
                code,
            }),
        }
    }
}

/// Units for pressure, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PressureUnit {
    /// Python unit code 0.
    Pascal = 0,
    /// Python unit code 1.
    HectoPascal = 1,
    /// Python unit code 2.
    KiloPascal = 2,
    /// Python unit code 3.
    MegaPascal = 3,
    /// Python unit code 4.
    GigaPascal = 4,
    /// Python unit code 5.
    Bar = 5,
    /// Python unit code 6.
    Atmosphere = 6,
    /// Python unit code 7.
    TechnicalAtmosphere = 7,
    /// Python unit code 8.
    PoundPerSquareInch = 8,
}

impl PressureUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[
        Self::Pascal,
        Self::HectoPascal,
        Self::KiloPascal,
        Self::MegaPascal,
        Self::GigaPascal,
        Self::Bar,
        Self::Atmosphere,
        Self::TechnicalAtmosphere,
        Self::PoundPerSquareInch,
    ];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::Pascal => value,
            Self::HectoPascal => value * 0.01,
            Self::KiloPascal => value * 0.001,
            Self::MegaPascal => value * 1e-06,
            Self::GigaPascal => value * 1e-09,
            Self::Bar => value * 1e-05,
            Self::Atmosphere => value * 9.869232667160129e-06,
            Self::TechnicalAtmosphere => value * 1.0197162129779282e-05,
            Self::PoundPerSquareInch => value * 0.00014503774389728312,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::Pascal => value,
            Self::HectoPascal => value * 100.0,
            Self::KiloPascal => value * 1000.0,
            Self::MegaPascal => value * 1000000.0,
            Self::GigaPascal => value * 1000000000.0,
            Self::Bar => value * 100000.0,
            Self::Atmosphere => value * 101325.0,
            Self::TechnicalAtmosphere => value * 98066.5,
            Self::PoundPerSquareInch => value * 6894.757,
        }
    }
}

impl TryFrom<i32> for PressureUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::Pascal),
            1 => Ok(Self::HectoPascal),
            2 => Ok(Self::KiloPascal),
            3 => Ok(Self::MegaPascal),
            4 => Ok(Self::GigaPascal),
            5 => Ok(Self::Bar),
            6 => Ok(Self::Atmosphere),
            7 => Ok(Self::TechnicalAtmosphere),
            8 => Ok(Self::PoundPerSquareInch),
            _ => Err(InvalidUnit {
                kind: "PressureUnit",
                code,
            }),
        }
    }
}

/// Units for surfaceareatovolume, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SurfaceAreaToVolumeUnit {
    /// Python unit code 0.
    SquareFeetOverCubicFeet = 0,
    /// Python unit code 1.
    SquareMetersOverCubicMeters = 1,
    /// Python unit code 2.
    SquareInchesOverCubicInches = 2,
    /// Python unit code 3.
    SquareCentimetersOverCubicCentimeters = 3,
}

impl SurfaceAreaToVolumeUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[
        Self::SquareFeetOverCubicFeet,
        Self::SquareMetersOverCubicMeters,
        Self::SquareInchesOverCubicInches,
        Self::SquareCentimetersOverCubicCentimeters,
    ];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::SquareFeetOverCubicFeet => value,
            Self::SquareMetersOverCubicMeters => value * 3.280839895013123,
            Self::SquareInchesOverCubicInches => value * 0.08333333333333333,
            Self::SquareCentimetersOverCubicCentimeters => value * 0.03280839895013123,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::SquareFeetOverCubicFeet => value,
            Self::SquareMetersOverCubicMeters => value * 0.3048,
            Self::SquareInchesOverCubicInches => value * 12.0,
            Self::SquareCentimetersOverCubicCentimeters => value * 30.48,
        }
    }
}

impl TryFrom<i32> for SurfaceAreaToVolumeUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::SquareFeetOverCubicFeet),
            1 => Ok(Self::SquareMetersOverCubicMeters),
            2 => Ok(Self::SquareInchesOverCubicInches),
            3 => Ok(Self::SquareCentimetersOverCubicCentimeters),
            _ => Err(InvalidUnit {
                kind: "SurfaceAreaToVolumeUnit",
                code,
            }),
        }
    }
}

/// Units for speed, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpeedUnit {
    /// Python unit code 0.
    FeetPerMinute = 0,
    /// Python unit code 1.
    ChainsPerHour = 1,
    /// Python unit code 2.
    MetersPerSecond = 2,
    /// Python unit code 3.
    MetersPerMinute = 3,
    /// Python unit code 4.
    MetersPerHour = 4,
    /// Python unit code 5.
    MilesPerHour = 5,
    /// Python unit code 6.
    KilometersPerHour = 6,
}

impl SpeedUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[
        Self::FeetPerMinute,
        Self::ChainsPerHour,
        Self::MetersPerSecond,
        Self::MetersPerMinute,
        Self::MetersPerHour,
        Self::MilesPerHour,
        Self::KilometersPerHour,
    ];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::FeetPerMinute => value,
            Self::ChainsPerHour => value * 0.9090909090909091,
            Self::MetersPerSecond => value * 0.00508,
            Self::MetersPerMinute => value * 0.3048,
            Self::MetersPerHour => value * 18.288,
            Self::MilesPerHour => value * 0.01136363636,
            Self::KilometersPerHour => value * 0.018288,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::FeetPerMinute => value,
            Self::ChainsPerHour => value * 1.1,
            Self::MetersPerSecond => value * 196.8503937,
            Self::MetersPerMinute => value * 3.28084,
            Self::MetersPerHour => value * 0.0547,
            Self::MilesPerHour => value * 88.0,
            Self::KilometersPerHour => value * 54.680665,
        }
    }
}

impl TryFrom<i32> for SpeedUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::FeetPerMinute),
            1 => Ok(Self::ChainsPerHour),
            2 => Ok(Self::MetersPerSecond),
            3 => Ok(Self::MetersPerMinute),
            4 => Ok(Self::MetersPerHour),
            5 => Ok(Self::MilesPerHour),
            6 => Ok(Self::KilometersPerHour),
            _ => Err(InvalidUnit {
                kind: "SpeedUnit",
                code,
            }),
        }
    }
}

/// Units for fraction, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FractionUnit {
    /// Python unit code 0.
    Fraction = 0,
    /// Python unit code 1.
    Percent = 1,
}

impl FractionUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[Self::Fraction, Self::Percent];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::Fraction => value,
            Self::Percent => value * 100.0,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::Fraction => value,
            Self::Percent => value / 100.0,
        }
    }
}

impl TryFrom<i32> for FractionUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::Fraction),
            1 => Ok(Self::Percent),
            _ => Err(InvalidUnit {
                kind: "FractionUnit",
                code,
            }),
        }
    }
}

/// Units for slope, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SlopeUnit {
    /// Python unit code 0.
    Degrees = 0,
    /// Python unit code 1.
    Percent = 1,
}

impl SlopeUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[Self::Degrees, Self::Percent];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::Degrees => value,
            Self::Percent => value.to_radians().tan() * 100.0,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::Degrees => value,
            Self::Percent => (value / 100.0).atan().to_degrees(),
        }
    }
}

impl TryFrom<i32> for SlopeUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::Degrees),
            1 => Ok(Self::Percent),
            _ => Err(InvalidUnit {
                kind: "SlopeUnit",
                code,
            }),
        }
    }
}

/// Units for density, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DensityUnit {
    /// Python unit code 0.
    PoundsPerCubicFoot = 0,
    /// Python unit code 1.
    KilogramsPerCubicMeter = 1,
}

impl DensityUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[Self::PoundsPerCubicFoot, Self::KilogramsPerCubicMeter];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::PoundsPerCubicFoot => value,
            Self::KilogramsPerCubicMeter => value * 16.01846326520259,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::PoundsPerCubicFoot => value,
            Self::KilogramsPerCubicMeter => value * 0.062427961,
        }
    }
}

impl TryFrom<i32> for DensityUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::PoundsPerCubicFoot),
            1 => Ok(Self::KilogramsPerCubicMeter),
            _ => Err(InvalidUnit {
                kind: "DensityUnit",
                code,
            }),
        }
    }
}

/// Units for heatofcombustion, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeatOfCombustionUnit {
    /// Python unit code 0.
    BtusPerPound = 0,
    /// Python unit code 1.
    KilojoulesPerKilogram = 1,
}

impl HeatOfCombustionUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[Self::BtusPerPound, Self::KilojoulesPerKilogram];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::BtusPerPound => value,
            Self::KilojoulesPerKilogram => value * 2.32779,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::BtusPerPound => value,
            Self::KilojoulesPerKilogram => value * 0.429592,
        }
    }
}

impl TryFrom<i32> for HeatOfCombustionUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::BtusPerPound),
            1 => Ok(Self::KilojoulesPerKilogram),
            _ => Err(InvalidUnit {
                kind: "HeatOfCombustionUnit",
                code,
            }),
        }
    }
}

/// Units for heatsink, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeatSinkUnit {
    /// Python unit code 0.
    BtusPerCubicFoot = 0,
    /// Python unit code 1.
    KilojoulesPerCubicMeter = 1,
}

impl HeatSinkUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[Self::BtusPerCubicFoot, Self::KilojoulesPerCubicMeter];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::BtusPerCubicFoot => value,
            Self::KilojoulesPerCubicMeter => value / 0.026839,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::BtusPerCubicFoot => value,
            Self::KilojoulesPerCubicMeter => value * 0.026839,
        }
    }
}

impl TryFrom<i32> for HeatSinkUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::BtusPerCubicFoot),
            1 => Ok(Self::KilojoulesPerCubicMeter),
            _ => Err(InvalidUnit {
                kind: "HeatSinkUnit",
                code,
            }),
        }
    }
}

/// Units for heatperunitarea, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeatPerUnitAreaUnit {
    /// Python unit code 0.
    BtusPerSquareFoot = 0,
    /// Python unit code 1.
    KilojoulesPerSquareMeter = 1,
    /// Python unit code 2.
    KilowattSecondsPerSquareMeter = 2,
}

impl HeatPerUnitAreaUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[
        Self::BtusPerSquareFoot,
        Self::KilojoulesPerSquareMeter,
        Self::KilowattSecondsPerSquareMeter,
    ];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::BtusPerSquareFoot => value,
            Self::KilojoulesPerSquareMeter => value * 11.363636363636365,
            Self::KilowattSecondsPerSquareMeter => value * 11.363636363636365,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::BtusPerSquareFoot => value,
            Self::KilojoulesPerSquareMeter => value * 0.088,
            Self::KilowattSecondsPerSquareMeter => value * 0.088,
        }
    }
}

impl TryFrom<i32> for HeatPerUnitAreaUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::BtusPerSquareFoot),
            1 => Ok(Self::KilojoulesPerSquareMeter),
            2 => Ok(Self::KilowattSecondsPerSquareMeter),
            _ => Err(InvalidUnit {
                kind: "HeatPerUnitAreaUnit",
                code,
            }),
        }
    }
}

/// Units for heatsourceandreactionintensity, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeatSourceAndReactionIntensityUnit {
    /// Python unit code 0.
    BtusPerSquareFootPerMinute = 0,
    /// Python unit code 1.
    BtusPerSquareFootPerSecond = 1,
    /// Python unit code 2.
    KilojoulesPerSquareMeterPerSecond = 2,
    /// Python unit code 3.
    KilojoulesPerSquareMeterPerMinute = 3,
    /// Python unit code 4.
    KilowattsPerSquareMeter = 4,
}

impl HeatSourceAndReactionIntensityUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[
        Self::BtusPerSquareFootPerMinute,
        Self::BtusPerSquareFootPerSecond,
        Self::KilojoulesPerSquareMeterPerSecond,
        Self::KilojoulesPerSquareMeterPerMinute,
        Self::KilowattsPerSquareMeter,
    ];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::BtusPerSquareFootPerMinute => value,
            Self::BtusPerSquareFootPerSecond => value * 0.016666666666666666,
            Self::KilojoulesPerSquareMeterPerSecond => value * 0.1893939393939394,
            Self::KilojoulesPerSquareMeterPerMinute => value * 11.363636363636365,
            Self::KilowattsPerSquareMeter => value * 1.8939393939393938,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::BtusPerSquareFootPerMinute => value,
            Self::BtusPerSquareFootPerSecond => value * 60.0,
            Self::KilojoulesPerSquareMeterPerSecond => value * 5.28,
            Self::KilojoulesPerSquareMeterPerMinute => value * 0.088,
            Self::KilowattsPerSquareMeter => value * 0.528,
        }
    }
}

impl TryFrom<i32> for HeatSourceAndReactionIntensityUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::BtusPerSquareFootPerMinute),
            1 => Ok(Self::BtusPerSquareFootPerSecond),
            2 => Ok(Self::KilojoulesPerSquareMeterPerSecond),
            3 => Ok(Self::KilojoulesPerSquareMeterPerMinute),
            4 => Ok(Self::KilowattsPerSquareMeter),
            _ => Err(InvalidUnit {
                kind: "HeatSourceAndReactionIntensityUnit",
                code,
            }),
        }
    }
}

/// Units for firelineintensity, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FirelineIntensityUnit {
    /// Python unit code 0.
    BtusPerFootPerSecond = 0,
    /// Python unit code 1.
    BtusPerFootPerMinute = 1,
    /// Python unit code 2.
    KilojoulesPerMeterPerSecond = 2,
    /// Python unit code 3.
    KilojoulesPerMeterPerMinute = 3,
    /// Python unit code 4.
    KilowattsPerMeter = 4,
}

impl FirelineIntensityUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[
        Self::BtusPerFootPerSecond,
        Self::BtusPerFootPerMinute,
        Self::KilojoulesPerMeterPerSecond,
        Self::KilojoulesPerMeterPerMinute,
        Self::KilowattsPerMeter,
    ];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::BtusPerFootPerSecond => value,
            Self::BtusPerFootPerMinute => value * 60.0,
            Self::KilojoulesPerMeterPerSecond => value * 3.464140419,
            Self::KilojoulesPerMeterPerMinute => value * 207.848,
            Self::KilowattsPerMeter => value * 3.464140419,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::BtusPerFootPerSecond => value,
            Self::BtusPerFootPerMinute => value * 0.01666666666666667,
            Self::KilojoulesPerMeterPerSecond => value * 0.2886719,
            Self::KilojoulesPerMeterPerMinute => value * 0.00481120819,
            Self::KilowattsPerMeter => value * 0.2886719,
        }
    }
}

impl TryFrom<i32> for FirelineIntensityUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::BtusPerFootPerSecond),
            1 => Ok(Self::BtusPerFootPerMinute),
            2 => Ok(Self::KilojoulesPerMeterPerSecond),
            3 => Ok(Self::KilojoulesPerMeterPerMinute),
            4 => Ok(Self::KilowattsPerMeter),
            _ => Err(InvalidUnit {
                kind: "FirelineIntensityUnit",
                code,
            }),
        }
    }
}

/// Units for temperature, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TemperatureUnit {
    /// Python unit code 0.
    Fahrenheit = 0,
    /// Python unit code 1.
    Celsius = 1,
    /// Python unit code 2.
    Kelvin = 2,
}

impl TemperatureUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[Self::Fahrenheit, Self::Celsius, Self::Kelvin];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::Fahrenheit => value,
            Self::Celsius => (value - 32.0) * (5.0 / 9.0),
            Self::Kelvin => (value - 32.0) * (5.0 / 9.0) + 273.15,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::Fahrenheit => value,
            Self::Celsius => value * (9.0 / 5.0) + 32.0,
            Self::Kelvin => (value - 273.15) * (9.0 / 5.0) + 32.0,
        }
    }
}

impl TryFrom<i32> for TemperatureUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::Fahrenheit),
            1 => Ok(Self::Celsius),
            2 => Ok(Self::Kelvin),
            _ => Err(InvalidUnit {
                kind: "TemperatureUnit",
                code,
            }),
        }
    }
}

/// Units for time, with the Python integer discriminants.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    /// Python unit code 0.
    Minutes = 0,
    /// Python unit code 1.
    Seconds = 1,
    /// Python unit code 2.
    Hours = 2,
    /// Python unit code 3.
    Days = 3,
    /// Python unit code 4.
    Years = 4,
}

impl TimeUnit {
    /// Every supported unit, ordered by its integer code.
    pub const ALL: &'static [Self] = &[
        Self::Minutes,
        Self::Seconds,
        Self::Hours,
        Self::Days,
        Self::Years,
    ];

    /// Return the exact Python unit code.
    pub const fn code(self) -> i32 {
        self as i32
    }

    /// Convert from the base unit, preserving Python's independent factors.
    pub fn from_base(self, value: f64) -> f64 {
        match self {
            Self::Minutes => value,
            Self::Seconds => value * 60.0,
            Self::Hours => value * 0.016666666666666666,
            Self::Days => value * 0.0006944444444444445,
            Self::Years => value * 1.901285268841737e-06,
        }
    }

    /// Convert to the base unit; temperature and slope apply affine/trigonometric rules.
    pub fn to_base(self, value: f64) -> f64 {
        match self {
            Self::Minutes => value,
            Self::Seconds => value * 0.016666666666666666,
            Self::Hours => value * 60.0,
            Self::Days => value * 1440.0,
            Self::Years => value * 525960.0,
        }
    }
}

impl TryFrom<i32> for TimeUnit {
    type Error = InvalidUnit;

    /// Validate a Python integer discriminant.
    fn try_from(code: i32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::Minutes),
            1 => Ok(Self::Seconds),
            2 => Ok(Self::Hours),
            3 => Ok(Self::Days),
            4 => Ok(Self::Years),
            _ => Err(InvalidUnit {
                kind: "TimeUnit",
                code,
            }),
        }
    }
}
