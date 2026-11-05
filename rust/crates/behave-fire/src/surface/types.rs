/// Named scalar equivalents of the Python Particles mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Particles {
    /// Dead fuel loading by particle, in lb/ft².
    pub load_dead: [f64; 5],
    /// Live fuel loading by particle, in lb/ft².
    pub load_live: [f64; 5],
    /// Dead particle SAVR in ft²/ft³.
    pub savr_dead: [f64; 5],
    /// Live particle SAVR in ft²/ft³.
    pub savr_live: [f64; 5],
    /// Dead moisture fractions by particle.
    pub moisture_dead: [f64; 5],
    /// Live moisture fractions by particle.
    pub moisture_live: [f64; 5],
    /// Dead heat of combustion in BTU/lb.
    pub hoc_dead: f64,
    /// Live heat of combustion in BTU/lb.
    pub hoc_live: f64,
    /// Fuelbed depth in feet.
    pub depth: f64,
    /// Dead moisture of extinction as a fraction.
    pub moe_dead: f64,
    /// Whether the fuel lookup slot is defined; false masks final outputs.
    pub is_defined: bool,
}
impl Particles {
    /// Construct a zero-valued record for explicit stage inputs.
    pub fn new() -> Self {
        Self::default()
    }
    /// Return numeric scalar fields under their exact Python keys.
    pub fn named_values(&self) -> [(&'static str, f64); 4] {
        [
            ("hoc_dead", self.hoc_dead),
            ("hoc_live", self.hoc_live),
            ("depth", self.depth),
            ("moe_dead", self.moe_dead),
        ]
    }
    /// Return the five particles per life state in Python order.
    pub fn named_arrays(&self) -> [(&'static str, [f64; 5]); 6] {
        [
            ("load_dead", self.load_dead),
            ("load_live", self.load_live),
            ("savr_dead", self.savr_dead),
            ("savr_live", self.savr_live),
            ("moisture_dead", self.moisture_dead),
            ("moisture_live", self.moisture_live),
        ]
    }
}
/// Named scalar equivalents of the Python FuelbedIntermediates mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct FuelbedIntermediates {
    /// Characteristic SAVR in ft²/ft³.
    pub sigma: f64,
    /// Bulk fuelbed density in lb/ft³.
    pub bulk_density: f64,
    /// Packing ratio, dimensionless.
    pub packing_ratio: f64,
    /// Relative packing ratio, dimensionless.
    pub relative_packing_ratio: f64,
    /// Heat sink in BTU/ft³.
    pub heat_sink: f64,
    /// Propagating flux ratio, dimensionless.
    pub propagating_flux: f64,
    /// Weighted dead heat of combustion in BTU/lb.
    pub w_heat_dead: f64,
    /// Weighted live heat of combustion in BTU/lb.
    pub w_heat_live: f64,
    /// Weighted effective dead silica fraction.
    pub w_silica_dead: f64,
    /// Weighted effective live silica fraction.
    pub w_silica_live: f64,
    /// Surface-area-weighted dead moisture fraction.
    pub w_moist_dead: f64,
    /// Surface-area-weighted live moisture fraction.
    pub w_moist_live: f64,
    /// Size-class-weighted dead net loading in lb/ft².
    pub wfl_dead: f64,
    /// Size-class-weighted live net loading in lb/ft².
    pub wfl_live: f64,
    /// Dead moisture of extinction as a fraction.
    pub moe_dead: f64,
    /// Live moisture of extinction as a fraction.
    pub moe_live: f64,
    /// Fraction of total surface area in dead particles.
    pub frac_dead: f64,
    /// Fraction of total surface area in live particles.
    pub frac_live: f64,
    /// Fuelbed depth in feet.
    pub depth: f64,
    /// Whether the fuel lookup slot is defined; false masks final outputs.
    pub is_defined: bool,
}
impl FuelbedIntermediates {
    /// Construct a zero-valued record for explicit stage inputs.
    pub fn new() -> Self {
        Self::default()
    }
    /// Return numeric scalar fields under their exact Python keys.
    pub fn named_values(&self) -> [(&'static str, f64); 19] {
        [
            ("sigma", self.sigma),
            ("bulk_density", self.bulk_density),
            ("packing_ratio", self.packing_ratio),
            ("relative_packing_ratio", self.relative_packing_ratio),
            ("heat_sink", self.heat_sink),
            ("propagating_flux", self.propagating_flux),
            ("w_heat_dead", self.w_heat_dead),
            ("w_heat_live", self.w_heat_live),
            ("w_silica_dead", self.w_silica_dead),
            ("w_silica_live", self.w_silica_live),
            ("w_moist_dead", self.w_moist_dead),
            ("w_moist_live", self.w_moist_live),
            ("wfl_dead", self.wfl_dead),
            ("wfl_live", self.wfl_live),
            ("moe_dead", self.moe_dead),
            ("moe_live", self.moe_live),
            ("frac_dead", self.frac_dead),
            ("frac_live", self.frac_live),
            ("depth", self.depth),
        ]
    }
}
/// Named scalar equivalents of the Python SurfaceResult mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct SurfaceResult {
    /// Forward spread rate in ft/min.
    pub spread_rate: f64,
    /// Backing spread rate in ft/min.
    pub backing_spread_rate: f64,
    /// Flanking spread rate in ft/min.
    pub flanking_spread_rate: f64,
    /// Flame length in feet.
    pub flame_length: f64,
    /// Fireline intensity in BTU/ft/s.
    pub fireline_intensity: f64,
    /// Heat per unit area in BTU/ft².
    pub heat_per_unit_area: f64,
    /// Effective wind speed in mph.
    pub effective_wind_speed: f64,
    /// Ellipse length-to-width ratio, dimensionless; undefined fuel yields 1.
    pub fire_length_to_width_ratio: f64,
    /// Ellipse eccentricity, dimensionless.
    pub eccentricity: f64,
    /// Direction of maximum spread in degrees.
    pub direction_of_max_spread: f64,
    /// Residence time in minutes.
    pub residence_time: f64,
    /// Reaction intensity in BTU/ft²/min.
    pub reaction_intensity: f64,
    /// Midflame wind speed in ft/min.
    pub midflame_wind_speed: f64,
    /// No-wind, no-slope spread rate in ft/min.
    pub no_wind_no_slope_spread_rate: f64,
}
impl SurfaceResult {
    /// Construct a zero-valued record for explicit stage inputs.
    pub fn new() -> Self {
        Self::default()
    }
    /// Return numeric scalar fields under their exact Python keys.
    pub fn named_values(&self) -> [(&'static str, f64); 14] {
        [
            ("spread_rate", self.spread_rate),
            ("backing_spread_rate", self.backing_spread_rate),
            ("flanking_spread_rate", self.flanking_spread_rate),
            ("flame_length", self.flame_length),
            ("fireline_intensity", self.fireline_intensity),
            ("heat_per_unit_area", self.heat_per_unit_area),
            ("effective_wind_speed", self.effective_wind_speed),
            (
                "fire_length_to_width_ratio",
                self.fire_length_to_width_ratio,
            ),
            ("eccentricity", self.eccentricity),
            ("direction_of_max_spread", self.direction_of_max_spread),
            ("residence_time", self.residence_time),
            ("reaction_intensity", self.reaction_intensity),
            ("midflame_wind_speed", self.midflame_wind_speed),
            (
                "no_wind_no_slope_spread_rate",
                self.no_wind_no_slope_spread_rate,
            ),
        ]
    }
}
/// Named scalar equivalents of the Python WindCoefficients mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct WindCoefficients {
    /// Sigma in ft²/ft³, replaced with 1 when not greater than 1e-7.
    pub safe_sigma: f64,
    /// Albini/Rothermel wind coefficient C, dimensionless.
    pub wind_c: f64,
    /// Albini/Rothermel wind exponent B, dimensionless.
    pub wind_b: f64,
    /// Albini/Rothermel relative-packing exponent E, dimensionless.
    pub wind_e: f64,
}
impl WindCoefficients {
    /// Construct a zero-valued record for explicit stage inputs.
    pub fn new() -> Self {
        Self::default()
    }
    /// Return numeric scalar fields under their exact Python keys.
    pub fn named_values(&self) -> [(&'static str, f64); 4] {
        [
            ("safe_sigma", self.safe_sigma),
            ("wind_c", self.wind_c),
            ("wind_b", self.wind_b),
            ("wind_e", self.wind_e),
        ]
    }
}
/// Named scalar equivalents of the Python RateVector mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct RateVector {
    /// Spread vector component along upslope in ft/min.
    pub x_component: f64,
    /// Spread vector component across upslope in ft/min.
    pub y_component: f64,
    /// Resultant spread vector magnitude in ft/min.
    pub rate_vector: f64,
}
impl RateVector {
    /// Construct a zero-valued record for explicit stage inputs.
    pub fn new() -> Self {
        Self::default()
    }
    /// Return numeric scalar fields under their exact Python keys.
    pub fn named_values(&self) -> [(&'static str, f64); 3] {
        [
            ("x_component", self.x_component),
            ("y_component", self.y_component),
            ("rate_vector", self.rate_vector),
        ]
    }
}
