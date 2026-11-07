//! Compile-time physical and categorical output identities.
macro_rules! fields {
    ($($field:ident => $name:literal),+ $(,)?) => {
        /// A named output or injectable scientific field.
        #[derive(Clone,Copy,Debug,PartialEq,Eq,Hash)]
        #[non_exhaustive]
        #[repr(usize)]
        pub enum Field {$(#[doc=$name] $field),+}
        impl Field {
            /// Every currently defined field in stable order.
            pub const ALL:&'static[Self]=&[$(Self::$field),+];
            /// Number of stable field slots in reusable cell state.
            pub const COUNT:usize=Self::ALL.len();
            /// Resolve a public field name before execution.
            pub fn from_name(name:&str)->Option<Self> {match name {$($name=>Some(Self::$field),)+_=>None}}
            /// The public field name.
            pub const fn name(self)->&'static str {match self {$(Self::$field=>$name),+}}
        }
    }
}
fields! {
    FuelModelGrid => "fuel_model_grid",
    M1h => "m1h",
    M10h => "m10h",
    M100h => "m100h",
    Mlh => "mlh",
    Mlw => "mlw",
    WindSpeed => "wind_speed",
    WindDirection => "wind_direction",
    Slope => "slope",
    Aspect => "aspect",
    CanopyCover => "canopy_cover",
    CanopyHeight => "canopy_height",
    CrownRatio => "crown_ratio",
    UserWaf => "user_waf",
    CanopyBaseHeight => "canopy_base_height",
    CanopyBulkDensity => "canopy_bulk_density",
    MoistureFoliar => "moisture_foliar",
    SurfaceRos => "surface_ros",
    SurfaceFli => "surface_fli",
    SurfaceHpua => "surface_hpua",
    Validity => "validity",
    ParticleHocDead => "p.hoc_dead",
    ParticleHocLive => "p.hoc_live",
    ParticleDepth => "p.depth",
    ParticleMoeDead => "p.moe_dead",
    ParticleDefined => "p.is_defined",
    FuelbedDefined => "ib.is_defined",
    WindB => "wind_b",
    WindC => "wind_c",
    WindE => "wind_e",
    Wind20 => "wind20",
    Waf => "waf",
    WindPhi => "wind_phi",
    SlopePhi => "slope_phi",
    VectorX => "vector.x_component",
    VectorY => "vector.y_component",
    RateVector => "vector.rate_vector",
    CrownFuelLoad => "crown_fuel_load",
    IsSurfaceFire => "is_surface_fire",
    IsPassiveCrown => "is_passive_crown",
    IsActiveCrown => "is_active_crown",
    ParticleLoadDead0 => "p.load_dead.0",
    ParticleLoadDead1 => "p.load_dead.1",
    ParticleLoadDead2 => "p.load_dead.2",
    ParticleLoadDead3 => "p.load_dead.3",
    ParticleLoadDead4 => "p.load_dead.4",
    ParticleLoadLive0 => "p.load_live.0",
    ParticleLoadLive1 => "p.load_live.1",
    ParticleLoadLive2 => "p.load_live.2",
    ParticleLoadLive3 => "p.load_live.3",
    ParticleLoadLive4 => "p.load_live.4",
    ParticleSavrDead0 => "p.savr_dead.0",
    ParticleSavrDead1 => "p.savr_dead.1",
    ParticleSavrDead2 => "p.savr_dead.2",
    ParticleSavrDead3 => "p.savr_dead.3",
    ParticleSavrDead4 => "p.savr_dead.4",
    ParticleSavrLive0 => "p.savr_live.0",
    ParticleSavrLive1 => "p.savr_live.1",
    ParticleSavrLive2 => "p.savr_live.2",
    ParticleSavrLive3 => "p.savr_live.3",
    ParticleSavrLive4 => "p.savr_live.4",
    ParticleMoistureDead0 => "p.moisture_dead.0",
    ParticleMoistureDead1 => "p.moisture_dead.1",
    ParticleMoistureDead2 => "p.moisture_dead.2",
    ParticleMoistureDead3 => "p.moisture_dead.3",
    ParticleMoistureDead4 => "p.moisture_dead.4",
    ParticleMoistureLive0 => "p.moisture_live.0",
    ParticleMoistureLive1 => "p.moisture_live.1",
    ParticleMoistureLive2 => "p.moisture_live.2",
    ParticleMoistureLive3 => "p.moisture_live.3",
    ParticleMoistureLive4 => "p.moisture_live.4",
    FuelbedSigma => "ib.sigma",
    FuelbedBulkDensity => "ib.bulk_density",
    FuelbedPackingRatio => "ib.packing_ratio",
    FuelbedRelativePackingRatio => "ib.relative_packing_ratio",
    FuelbedHeatSink => "ib.heat_sink",
    FuelbedPropagatingFlux => "ib.propagating_flux",
    FuelbedWHeatDead => "ib.w_heat_dead",
    FuelbedWHeatLive => "ib.w_heat_live",
    FuelbedWSilicaDead => "ib.w_silica_dead",
    FuelbedWSilicaLive => "ib.w_silica_live",
    FuelbedWMoistDead => "ib.w_moist_dead",
    FuelbedWMoistLive => "ib.w_moist_live",
    FuelbedWflDead => "ib.wfl_dead",
    FuelbedWflLive => "ib.wfl_live",
    FuelbedMoeDead => "ib.moe_dead",
    FuelbedMoeLive => "ib.moe_live",
    FuelbedFracDead => "ib.frac_dead",
    FuelbedFracLive => "ib.frac_live",
    FuelbedDepth => "ib.depth",
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

/// Physical data and typed control/flag slots have different storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    /// Float64 scientific quantity.
    Physical,
    /// Int32 identifier or category.
    Code,
    /// Boolean definition/validity/classification field.
    Flag,
}
impl Field {
    /// The field's fixed storage role, unrelated to NumPy numeric dtypes.
    pub const fn kind(self) -> FieldKind {
        match self {
            Self::FireType | Self::FuelModelGrid => FieldKind::Code,
            Self::ParticleDefined
            | Self::FuelbedDefined
            | Self::Validity
            | Self::IsSurfaceFire
            | Self::IsPassiveCrown
            | Self::IsActiveCrown => FieldKind::Flag,
            _ => FieldKind::Physical,
        }
    }
    pub(crate) const fn code_slot(self) -> Option<usize> {
        match self {
            Self::FireType => Some(0),
            Self::FuelModelGrid => Some(1),
            _ => None,
        }
    }
    pub(crate) const fn flag_slot(self) -> Option<usize> {
        match self {
            Self::ParticleDefined => Some(0),
            Self::FuelbedDefined => Some(1),
            Self::Validity => Some(2),
            Self::IsSurfaceFire => Some(3),
            Self::IsPassiveCrown => Some(4),
            Self::IsActiveCrown => Some(5),
            _ => None,
        }
    }
}
