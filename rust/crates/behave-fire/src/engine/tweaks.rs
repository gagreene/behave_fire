//! Explicit stage overrides applied before calculating their dependent outputs.

/// Optional Surface stage outputs in component base units.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfaceTweaks {
    /// Reaction intensity in BTU/ft²/min, before slope cap and heat calculations.
    pub reaction_intensity: Option<f64>,
    /// No-wind/no-slope rate in ft/min, before vector and effective-wind calculations.
    pub no_wind_no_slope_spread_rate: Option<f64>,
    /// Midflame wind in ft/min, before calculating wind amplification.
    pub midflame_wind_speed: Option<f64>,
    /// Final dimensionless wind amplification; overrides midflame-derived amplification.
    pub wind_phi: Option<f64>,
    /// Final dimensionless slope amplification, including an explicit override of the cap.
    pub slope_phi: Option<f64>,
    /// Forward rate in ft/min, before shape, intensity, and flame calculations.
    pub spread_rate: Option<f64>,
}

/// Optional Crown stage values; supplied Surface fields remain separate inputs.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CrownTweaks {
    /// Overrides for the internal model-10 Surface calculation.
    pub fm10: SurfaceTweaks,
    /// Critical Surface fireline intensity in BTU/ft/s, before classification.
    pub critical_surface_intensity: Option<f64>,
    /// Critical active-Crown spread rate in ft/min, before classification/back-solving.
    pub critical_crown_rate: Option<f64>,
    /// Fraction burned, before passive and final assignments.
    pub crown_fraction_burned: Option<f64>,
}
