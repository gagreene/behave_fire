//! Enum-valued numerical configuration; string interpretation belongs to adapters.

/// Reference for wind direction and the returned spread bearing.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WindOrientation {
    /// Clockwise from the upslope direction.
    #[default]
    Upslope,
    /// Clockwise from north, corrected by aspect.
    North,
}

/// Height of the supplied open wind measurement.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WindHeight {
    /// Standard twenty-foot wind.
    #[default]
    TwentyFoot,
    /// Ten-metre wind, converted to twenty-foot wind by dividing by 1.15.
    TenMetre,
}

/// Method for converting open wind to midflame wind.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WafMethod {
    /// Derive adjustment from the fuel bed and canopy.
    #[default]
    CrownRatio,
    /// Use a supplied factor, falling back to canopy adjustment if absent.
    UserInput,
}

/// Immutable control choices shared by a scalar or raster calculation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SurfaceOptions {
    /// Coordinate reference for wind and spread bearings.
    pub wind_orientation: WindOrientation,
    /// Measurement height of the open wind speed.
    pub wind_height: WindHeight,
    /// Wind adjustment calculation method.
    pub waf_method: WafMethod,
}
