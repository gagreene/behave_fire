//! Compile-time physical and categorical output identities.
macro_rules! fields {
    ($($field:ident => $name:literal),+ $(,)?) => {
        /// A named output or injectable scientific field.
        #[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
        #[non_exhaustive]
        pub enum Field {$(#[doc=$name] $field),+}
        impl Field {
            /// Every currently defined field in stable order.
            pub const ALL:&'static[Self]=&[$(Self::$field),+];
            /// Resolve a public field name before execution.
            pub fn from_name(name:&str)->Option<Self> {match name {$($name=>Some(Self::$field),)+_=>None}}
            /// The public field name.
            pub const fn name(self)->&'static str {match self {$(Self::$field=>$name),+}}
        }
    }
}
fields! {
    SpreadRate => "spread_rate",
    BackingSpreadRate => "backing_spread_rate",
    FlankingSpreadRate => "flanking_spread_rate",
    FlameLength => "flame_length",
    FirelineIntensity => "fireline_intensity",
    HeatPerUnitArea => "heat_per_unit_area",
    EffectiveWindSpeed => "effective_wind_speed",
    FireLengthToWidthRatio => "fire_length_to_width_ratio",
    Eccentricity => "eccentricity",
    DirectionOfMaxSpread => "direction_of_max_spread",
    ResidenceTime => "residence_time",
    ReactionIntensity => "reaction_intensity",
    MidflameWindSpeed => "midflame_wind_speed",
    NoWindNoSlopeSpreadRate => "no_wind_no_slope_spread_rate",
    CrownFireSpreadRate => "crown_fire_spread_rate",
    CrownFlameLength => "crown_flame_length",
    CrownFireLineIntensity => "crown_fire_line_intensity",
    CrownFireHeatPerUnitArea => "crown_fire_heat_per_unit_area",
    CanopyHeatPerUnitArea => "canopy_heat_per_unit_area",
    CrownCriticalSurfaceFireLineIntensity => "crown_critical_surface_fire_line_intensity",
    CrownCriticalFireSpreadRate => "crown_critical_fire_spread_rate",
    CrownFireTransitionRatio => "crown_fire_transition_ratio",
    CrownFireActiveRatio => "crown_fire_active_ratio",
    CrownLengthToWidthRatio => "crown_length_to_width_ratio",
    SurfaceFireCriticalSpreadRate => "surface_fire_critical_spread_rate",
    CrowningSurfaceFireSpreadRate => "crowning_surface_fire_spread_rate",
    CrownFractionBurned => "crown_fraction_burned",
    PassiveCrownFireSpreadRate => "passive_crown_fire_spread_rate",
    PassiveCrownFireHeatPerUnitArea => "passive_crown_fire_heat_per_unit_area",
    PassiveCrownFireLineIntensity => "passive_crown_fire_line_intensity",
    PassiveCrownFireFlameLength => "passive_crown_fire_flame_length",
    FinalSpreadRate => "final_spread_rate",
    FinalHeatPerUnitArea => "final_heat_per_unit_area",
    FinalFirelineIntensity => "final_fireline_intensity",
    FinalFlameLength => "final_flame_length",
    FireType => "fire_type",
}
/// Default Surface result fields, in public dictionary order.
pub const SURFACE_FIELDS: &[Field] = &[
    Field::SpreadRate,
    Field::BackingSpreadRate,
    Field::FlankingSpreadRate,
    Field::FlameLength,
    Field::FirelineIntensity,
    Field::HeatPerUnitArea,
    Field::EffectiveWindSpeed,
    Field::FireLengthToWidthRatio,
    Field::Eccentricity,
    Field::DirectionOfMaxSpread,
    Field::ResidenceTime,
    Field::ReactionIntensity,
    Field::MidflameWindSpeed,
    Field::NoWindNoSlopeSpreadRate,
];
/// Default Crown result fields, including typed classification.
pub const CROWN_FIELDS: &[Field] = &[
    Field::CrownFireSpreadRate,
    Field::CrownFlameLength,
    Field::CrownFireLineIntensity,
    Field::CrownFireHeatPerUnitArea,
    Field::CanopyHeatPerUnitArea,
    Field::CrownCriticalSurfaceFireLineIntensity,
    Field::CrownCriticalFireSpreadRate,
    Field::CrownFireTransitionRatio,
    Field::CrownFireActiveRatio,
    Field::CrownLengthToWidthRatio,
    Field::SurfaceFireCriticalSpreadRate,
    Field::CrowningSurfaceFireSpreadRate,
    Field::CrownFractionBurned,
    Field::PassiveCrownFireSpreadRate,
    Field::PassiveCrownFireHeatPerUnitArea,
    Field::PassiveCrownFireLineIntensity,
    Field::PassiveCrownFireFlameLength,
    Field::FinalSpreadRate,
    Field::FinalHeatPerUnitArea,
    Field::FinalFirelineIntensity,
    Field::FinalFlameLength,
    Field::FireType,
];

/// A deduplicated, ordered set of requested outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputRequest {
    fields: Vec<Field>,
}
impl OutputRequest {
    /// Request named fields, retaining the first occurrence of each.
    pub fn new(fields: &[Field]) -> Self {
        let mut unique = Vec::with_capacity(fields.len());
        for &field in fields {
            if !unique.contains(&field) {
                unique.push(field);
            }
        }
        Self { fields: unique }
    }
    /// Request every default Surface result.
    pub fn surface() -> Self {
        Self::new(SURFACE_FIELDS)
    }
    /// Request every default Crown result.
    pub fn crown() -> Self {
        Self::new(CROWN_FIELDS)
    }
    /// Whether a field was selected.
    pub fn contains(&self, field: Field) -> bool {
        self.fields.contains(&field)
    }
    /// Requested fields in caller order.
    pub fn fields(&self) -> &[Field] {
        &self.fields
    }
}
