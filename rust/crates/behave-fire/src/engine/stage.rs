//! Stages group shared equations; field dependencies support partial injections.
use super::Field;
/// Selected full model, independent of numeric input storage dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pipeline {
    /// Surface fuel and ellipse chain.
    Surface,
    /// Model-10 spread and independent supplied-Surface Crown chain.
    Crown,
}
/// A public scientific stage group; automatic completion operates at field level.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Particles scientific stage.
    Particles,
    /// Fuelbed scientific stage.
    Fuelbed,
    /// Reaction scientific stage.
    Reaction,
    /// SurfaceWind scientific stage.
    SurfaceWind,
    /// SurfaceSpread scientific stage.
    SurfaceSpread,
    /// SurfaceGeometry scientific stage.
    SurfaceGeometry,
    /// CrownSpread scientific stage.
    CrownSpread,
    /// CrownCritical scientific stage.
    CrownCritical,
    /// CrownHeat scientific stage.
    CrownHeat,
    /// CrownRatios scientific stage.
    CrownRatios,
    /// CrownFraction scientific stage.
    CrownFraction,
    /// CrownPassive scientific stage.
    CrownPassive,
    /// CrownFinal scientific stage.
    CrownFinal,
}
impl Stage {
    /// All supported stage groups in dependency-compatible default order.
    pub const ALL: &'static [Self] = &[
        Self::Particles,
        Self::Fuelbed,
        Self::Reaction,
        Self::SurfaceWind,
        Self::SurfaceSpread,
        Self::SurfaceGeometry,
        Self::CrownSpread,
        Self::CrownCritical,
        Self::CrownHeat,
        Self::CrownRatios,
        Self::CrownFraction,
        Self::CrownPassive,
        Self::CrownFinal,
    ];
    /// Every field this explicit stage recomputes, including typed flags/codes.
    pub const fn writes(self) -> &'static [Field] {
        match self {
            Self::Particles => &[
                Field::ParticleHocDead,
                Field::ParticleHocLive,
                Field::ParticleDepth,
                Field::ParticleMoeDead,
                Field::ParticleDefined,
                Field::ParticleLoadDead0,
                Field::ParticleLoadDead1,
                Field::ParticleLoadDead2,
                Field::ParticleLoadDead3,
                Field::ParticleLoadDead4,
                Field::ParticleLoadLive0,
                Field::ParticleLoadLive1,
                Field::ParticleLoadLive2,
                Field::ParticleLoadLive3,
                Field::ParticleLoadLive4,
                Field::ParticleSavrDead0,
                Field::ParticleSavrDead1,
                Field::ParticleSavrDead2,
                Field::ParticleSavrDead3,
                Field::ParticleSavrDead4,
                Field::ParticleSavrLive0,
                Field::ParticleSavrLive1,
                Field::ParticleSavrLive2,
                Field::ParticleSavrLive3,
                Field::ParticleSavrLive4,
                Field::ParticleMoistureDead0,
                Field::ParticleMoistureDead1,
                Field::ParticleMoistureDead2,
                Field::ParticleMoistureDead3,
                Field::ParticleMoistureDead4,
                Field::ParticleMoistureLive0,
                Field::ParticleMoistureLive1,
                Field::ParticleMoistureLive2,
                Field::ParticleMoistureLive3,
                Field::ParticleMoistureLive4,
            ],
            Self::Fuelbed => &[
                Field::FuelbedDefined,
                Field::FuelbedSigma,
                Field::FuelbedBulkDensity,
                Field::FuelbedPackingRatio,
                Field::FuelbedRelativePackingRatio,
                Field::FuelbedHeatSink,
                Field::FuelbedPropagatingFlux,
                Field::FuelbedWHeatDead,
                Field::FuelbedWHeatLive,
                Field::FuelbedWSilicaDead,
                Field::FuelbedWSilicaLive,
                Field::FuelbedWMoistDead,
                Field::FuelbedWMoistLive,
                Field::FuelbedWflDead,
                Field::FuelbedWflLive,
                Field::FuelbedMoeDead,
                Field::FuelbedMoeLive,
                Field::FuelbedFracDead,
                Field::FuelbedFracLive,
                Field::FuelbedDepth,
            ],
            Self::Reaction => &[Field::ReactionIntensity],
            Self::SurfaceWind => &[
                Field::WindB,
                Field::WindC,
                Field::WindE,
                Field::Wind20,
                Field::Waf,
                Field::MidflameWindSpeed,
                Field::WindPhi,
                Field::SlopePhi,
            ],
            Self::SurfaceSpread => &[
                Field::NoWindNoSlopeSpreadRate,
                Field::VectorX,
                Field::VectorY,
                Field::RateVector,
                Field::SpreadRate,
                Field::EffectiveWindSpeed,
                Field::DirectionOfMaxSpread,
            ],
            Self::SurfaceGeometry => &[
                Field::FireLengthToWidthRatio,
                Field::Eccentricity,
                Field::BackingSpreadRate,
                Field::FlankingSpreadRate,
                Field::ResidenceTime,
                Field::HeatPerUnitArea,
                Field::FirelineIntensity,
                Field::FlameLength,
            ],
            Self::CrownSpread => &[Field::CrownFireSpreadRate, Field::CrownLengthToWidthRatio],
            Self::CrownCritical => &[
                Field::CrownCriticalSurfaceFireLineIntensity,
                Field::CrownCriticalFireSpreadRate,
            ],
            Self::CrownHeat => &[
                Field::CrownFuelLoad,
                Field::CanopyHeatPerUnitArea,
                Field::CrownFireHeatPerUnitArea,
                Field::CrownFireLineIntensity,
                Field::CrownFlameLength,
            ],
            Self::CrownRatios => &[
                Field::CrownFireTransitionRatio,
                Field::CrownFireActiveRatio,
                Field::FireType,
                Field::IsSurfaceFire,
                Field::IsPassiveCrown,
                Field::IsActiveCrown,
            ],
            Self::CrownFraction => &[
                Field::SurfaceFireCriticalSpreadRate,
                Field::CrowningSurfaceFireSpreadRate,
                Field::CrownFractionBurned,
            ],
            Self::CrownPassive => &[
                Field::PassiveCrownFireSpreadRate,
                Field::PassiveCrownFireHeatPerUnitArea,
                Field::PassiveCrownFireLineIntensity,
                Field::PassiveCrownFireFlameLength,
            ],
            Self::CrownFinal => &[
                Field::FinalSpreadRate,
                Field::FinalHeatPerUnitArea,
                Field::FinalFirelineIntensity,
                Field::FinalFlameLength,
            ],
        }
    }
}
impl Field {
    /// The one producing stage; caller/algorithm inputs have no producer.
    pub const fn producer(self) -> Option<Stage> {
        match self {
            Self::ParticleHocDead
            | Self::ParticleHocLive
            | Self::ParticleDepth
            | Self::ParticleMoeDead
            | Self::ParticleDefined
            | Self::ParticleLoadDead0
            | Self::ParticleLoadDead1
            | Self::ParticleLoadDead2
            | Self::ParticleLoadDead3
            | Self::ParticleLoadDead4
            | Self::ParticleLoadLive0
            | Self::ParticleLoadLive1
            | Self::ParticleLoadLive2
            | Self::ParticleLoadLive3
            | Self::ParticleLoadLive4
            | Self::ParticleSavrDead0
            | Self::ParticleSavrDead1
            | Self::ParticleSavrDead2
            | Self::ParticleSavrDead3
            | Self::ParticleSavrDead4
            | Self::ParticleSavrLive0
            | Self::ParticleSavrLive1
            | Self::ParticleSavrLive2
            | Self::ParticleSavrLive3
            | Self::ParticleSavrLive4
            | Self::ParticleMoistureDead0
            | Self::ParticleMoistureDead1
            | Self::ParticleMoistureDead2
            | Self::ParticleMoistureDead3
            | Self::ParticleMoistureDead4
            | Self::ParticleMoistureLive0
            | Self::ParticleMoistureLive1
            | Self::ParticleMoistureLive2
            | Self::ParticleMoistureLive3
            | Self::ParticleMoistureLive4 => Some(Stage::Particles),
            Self::FuelbedDefined
            | Self::FuelbedSigma
            | Self::FuelbedBulkDensity
            | Self::FuelbedPackingRatio
            | Self::FuelbedRelativePackingRatio
            | Self::FuelbedHeatSink
            | Self::FuelbedPropagatingFlux
            | Self::FuelbedWHeatDead
            | Self::FuelbedWHeatLive
            | Self::FuelbedWSilicaDead
            | Self::FuelbedWSilicaLive
            | Self::FuelbedWMoistDead
            | Self::FuelbedWMoistLive
            | Self::FuelbedWflDead
            | Self::FuelbedWflLive
            | Self::FuelbedMoeDead
            | Self::FuelbedMoeLive
            | Self::FuelbedFracDead
            | Self::FuelbedFracLive
            | Self::FuelbedDepth => Some(Stage::Fuelbed),
            Self::ReactionIntensity => Some(Stage::Reaction),
            Self::WindB
            | Self::WindC
            | Self::WindE
            | Self::Wind20
            | Self::Waf
            | Self::MidflameWindSpeed
            | Self::WindPhi
            | Self::SlopePhi => Some(Stage::SurfaceWind),
            Self::NoWindNoSlopeSpreadRate
            | Self::VectorX
            | Self::VectorY
            | Self::RateVector
            | Self::SpreadRate
            | Self::EffectiveWindSpeed
            | Self::DirectionOfMaxSpread => Some(Stage::SurfaceSpread),
            Self::FireLengthToWidthRatio
            | Self::Eccentricity
            | Self::BackingSpreadRate
            | Self::FlankingSpreadRate
            | Self::ResidenceTime
            | Self::HeatPerUnitArea
            | Self::FirelineIntensity
            | Self::FlameLength => Some(Stage::SurfaceGeometry),
            Self::CrownFireSpreadRate | Self::CrownLengthToWidthRatio => Some(Stage::CrownSpread),
            Self::CrownCriticalSurfaceFireLineIntensity | Self::CrownCriticalFireSpreadRate => {
                Some(Stage::CrownCritical)
            }
            Self::CrownFuelLoad
            | Self::CanopyHeatPerUnitArea
            | Self::CrownFireHeatPerUnitArea
            | Self::CrownFireLineIntensity
            | Self::CrownFlameLength => Some(Stage::CrownHeat),
            Self::CrownFireTransitionRatio
            | Self::CrownFireActiveRatio
            | Self::FireType
            | Self::IsSurfaceFire
            | Self::IsPassiveCrown
            | Self::IsActiveCrown => Some(Stage::CrownRatios),
            Self::SurfaceFireCriticalSpreadRate
            | Self::CrowningSurfaceFireSpreadRate
            | Self::CrownFractionBurned => Some(Stage::CrownFraction),
            Self::PassiveCrownFireSpreadRate
            | Self::PassiveCrownFireHeatPerUnitArea
            | Self::PassiveCrownFireLineIntensity
            | Self::PassiveCrownFireFlameLength => Some(Stage::CrownPassive),
            Self::FinalSpreadRate
            | Self::FinalHeatPerUnitArea
            | Self::FinalFirelineIntensity
            | Self::FinalFlameLength => Some(Stage::CrownFinal),
            _ => None,
        }
    }
    /// Possible direct reads; control/selection rules choose actual required inputs.
    ///
    /// This graph supports common-grid staged invalidation/liveness. Python output
    /// shape groups retain their operation-specific exact shape/read contracts.
    /// Optional user WAF and unused upslope aspect are resolved by configuration,
    /// not eagerly fetched as mandatory inputs by an automatic stage executor.
    pub const fn dependencies(self) -> &'static [Self] {
        match self {
            Self::BackingSpreadRate => &[Self::SpreadRate, Self::Eccentricity],
            Self::CanopyHeatPerUnitArea => &[Self::CrownFuelLoad],
            Self::CrownCriticalFireSpreadRate => &[Self::CanopyBulkDensity],
            Self::CrownCriticalSurfaceFireLineIntensity => {
                &[Self::CanopyBaseHeight, Self::MoistureFoliar]
            }
            Self::CrownFireActiveRatio => {
                &[Self::CrownFireSpreadRate, Self::CrownCriticalFireSpreadRate]
            }
            Self::CrownFireHeatPerUnitArea => &[Self::SurfaceHpua, Self::CanopyHeatPerUnitArea],
            Self::CrownFireLineIntensity => {
                &[Self::CrownFireSpreadRate, Self::CrownFireHeatPerUnitArea]
            }
            Self::CrownFireSpreadRate => &[Self::SpreadRate],
            Self::CrownFireTransitionRatio => &[
                Self::SurfaceFli,
                Self::CrownCriticalSurfaceFireLineIntensity,
            ],
            Self::CrownFlameLength => &[Self::CrownFireLineIntensity],
            Self::CrownFractionBurned => &[
                Self::SurfaceRos,
                Self::SurfaceFireCriticalSpreadRate,
                Self::CrowningSurfaceFireSpreadRate,
            ],
            Self::CrownFuelLoad => &[
                Self::CanopyHeight,
                Self::CanopyBaseHeight,
                Self::CanopyBulkDensity,
            ],
            Self::CrowningSurfaceFireSpreadRate => &[
                Self::ReactionIntensity,
                Self::FuelbedSigma,
                Self::FuelbedPackingRatio,
                Self::FuelbedRelativePackingRatio,
                Self::FuelbedPropagatingFlux,
                Self::FuelbedHeatSink,
                Self::FuelbedDefined,
                Self::CrownCriticalFireSpreadRate,
            ],
            Self::CrownLengthToWidthRatio => &[Self::EffectiveWindSpeed],
            Self::DirectionOfMaxSpread => {
                &[Self::VectorX, Self::VectorY, Self::RateVector, Self::Aspect]
            }
            Self::Eccentricity => &[Self::FireLengthToWidthRatio],
            Self::EffectiveWindSpeed => &[
                Self::SpreadRate,
                Self::NoWindNoSlopeSpreadRate,
                Self::WindB,
                Self::WindC,
                Self::WindE,
                Self::FuelbedRelativePackingRatio,
            ],
            Self::FinalFirelineIntensity => &[
                Self::FireType,
                Self::SurfaceFli,
                Self::PassiveCrownFireLineIntensity,
                Self::CrownFireLineIntensity,
            ],
            Self::FinalFlameLength => &[
                Self::FireType,
                Self::SurfaceFli,
                Self::PassiveCrownFireFlameLength,
                Self::CrownFlameLength,
            ],
            Self::FinalHeatPerUnitArea => &[
                Self::FireType,
                Self::SurfaceHpua,
                Self::PassiveCrownFireHeatPerUnitArea,
                Self::CrownFireHeatPerUnitArea,
            ],
            Self::FinalSpreadRate => &[
                Self::FireType,
                Self::SurfaceRos,
                Self::PassiveCrownFireSpreadRate,
                Self::CrownFireSpreadRate,
            ],
            Self::FireLengthToWidthRatio => &[Self::EffectiveWindSpeed],
            Self::FirelineIntensity => &[Self::SpreadRate, Self::HeatPerUnitArea],
            Self::FireType => &[Self::CrownFireTransitionRatio, Self::CrownFireActiveRatio],
            Self::FlameLength => &[Self::FirelineIntensity],
            Self::FlankingSpreadRate => &[
                Self::SpreadRate,
                Self::BackingSpreadRate,
                Self::FireLengthToWidthRatio,
            ],
            Self::FuelbedBulkDensity => &[
                Self::ParticleLoadDead0,
                Self::ParticleLoadDead1,
                Self::ParticleLoadDead2,
                Self::ParticleLoadDead3,
                Self::ParticleLoadDead4,
                Self::ParticleLoadLive0,
                Self::ParticleLoadLive1,
                Self::ParticleLoadLive2,
                Self::ParticleLoadLive3,
                Self::ParticleLoadLive4,
                Self::ParticleDepth,
            ],
            Self::FuelbedDefined => &[Self::ParticleDefined],
            Self::FuelbedDepth => &[Self::ParticleDepth],
            Self::FuelbedFracDead => &[
                Self::ParticleLoadDead0,
                Self::ParticleLoadDead1,
                Self::ParticleLoadDead2,
                Self::ParticleLoadDead3,
                Self::ParticleLoadDead4,
                Self::ParticleSavrDead0,
                Self::ParticleSavrDead1,
                Self::ParticleSavrDead2,
                Self::ParticleSavrDead3,
                Self::ParticleSavrDead4,
                Self::ParticleLoadLive0,
                Self::ParticleLoadLive1,
                Self::ParticleLoadLive2,
                Self::ParticleLoadLive3,
                Self::ParticleLoadLive4,
                Self::ParticleSavrLive0,
                Self::ParticleSavrLive1,
                Self::ParticleSavrLive2,
                Self::ParticleSavrLive3,
                Self::ParticleSavrLive4,
            ],
            Self::FuelbedFracLive => &[Self::FuelbedFracDead],
            Self::FuelbedHeatSink => &[
                Self::FuelbedFracDead,
                Self::FuelbedFracLive,
                Self::ParticleLoadDead0,
                Self::ParticleLoadDead1,
                Self::ParticleLoadDead2,
                Self::ParticleLoadDead3,
                Self::ParticleLoadDead4,
                Self::ParticleSavrDead0,
                Self::ParticleSavrDead1,
                Self::ParticleSavrDead2,
                Self::ParticleSavrDead3,
                Self::ParticleSavrDead4,
                Self::ParticleLoadLive0,
                Self::ParticleLoadLive1,
                Self::ParticleLoadLive2,
                Self::ParticleLoadLive3,
                Self::ParticleLoadLive4,
                Self::ParticleSavrLive0,
                Self::ParticleSavrLive1,
                Self::ParticleSavrLive2,
                Self::ParticleSavrLive3,
                Self::ParticleSavrLive4,
                Self::ParticleMoistureDead0,
                Self::ParticleMoistureDead1,
                Self::ParticleMoistureDead2,
                Self::ParticleMoistureDead3,
                Self::ParticleMoistureDead4,
                Self::ParticleMoistureLive0,
                Self::ParticleMoistureLive1,
                Self::ParticleMoistureLive2,
                Self::ParticleMoistureLive3,
                Self::ParticleMoistureLive4,
                Self::FuelbedBulkDensity,
            ],
            Self::FuelbedMoeDead => &[Self::ParticleMoeDead],
            Self::FuelbedMoeLive => &[
                Self::ParticleLoadDead0,
                Self::ParticleLoadDead1,
                Self::ParticleLoadDead2,
                Self::ParticleLoadDead3,
                Self::ParticleLoadDead4,
                Self::ParticleSavrDead0,
                Self::ParticleSavrDead1,
                Self::ParticleSavrDead2,
                Self::ParticleSavrDead3,
                Self::ParticleSavrDead4,
                Self::ParticleLoadLive0,
                Self::ParticleLoadLive1,
                Self::ParticleLoadLive2,
                Self::ParticleLoadLive3,
                Self::ParticleLoadLive4,
                Self::ParticleSavrLive0,
                Self::ParticleSavrLive1,
                Self::ParticleSavrLive2,
                Self::ParticleSavrLive3,
                Self::ParticleSavrLive4,
                Self::ParticleMoistureDead0,
                Self::ParticleMoistureDead1,
                Self::ParticleMoistureDead2,
                Self::ParticleMoistureDead3,
                Self::ParticleMoistureDead4,
                Self::FuelbedMoeDead,
            ],
            Self::FuelbedPackingRatio => &[
                Self::ParticleLoadDead0,
                Self::ParticleLoadDead1,
                Self::ParticleLoadDead2,
                Self::ParticleLoadDead3,
                Self::ParticleLoadDead4,
                Self::ParticleLoadLive0,
                Self::ParticleLoadLive1,
                Self::ParticleLoadLive2,
                Self::ParticleLoadLive3,
                Self::ParticleLoadLive4,
                Self::ParticleDepth,
            ],
            Self::FuelbedPropagatingFlux => &[Self::FuelbedSigma, Self::FuelbedPackingRatio],
            Self::FuelbedRelativePackingRatio => &[Self::FuelbedPackingRatio, Self::FuelbedSigma],
            Self::FuelbedSigma => &[
                Self::FuelbedFracDead,
                Self::FuelbedFracLive,
                Self::ParticleLoadDead0,
                Self::ParticleLoadDead1,
                Self::ParticleLoadDead2,
                Self::ParticleLoadDead3,
                Self::ParticleLoadDead4,
                Self::ParticleSavrDead0,
                Self::ParticleSavrDead1,
                Self::ParticleSavrDead2,
                Self::ParticleSavrDead3,
                Self::ParticleSavrDead4,
                Self::ParticleLoadLive0,
                Self::ParticleLoadLive1,
                Self::ParticleLoadLive2,
                Self::ParticleLoadLive3,
                Self::ParticleLoadLive4,
                Self::ParticleSavrLive0,
                Self::ParticleSavrLive1,
                Self::ParticleSavrLive2,
                Self::ParticleSavrLive3,
                Self::ParticleSavrLive4,
            ],
            Self::FuelbedWflDead => &[
                Self::ParticleLoadDead0,
                Self::ParticleLoadDead1,
                Self::ParticleLoadDead2,
                Self::ParticleLoadDead3,
                Self::ParticleLoadDead4,
                Self::ParticleSavrDead0,
                Self::ParticleSavrDead1,
                Self::ParticleSavrDead2,
                Self::ParticleSavrDead3,
                Self::ParticleSavrDead4,
            ],
            Self::FuelbedWflLive => &[
                Self::ParticleLoadLive0,
                Self::ParticleLoadLive1,
                Self::ParticleLoadLive2,
                Self::ParticleLoadLive3,
                Self::ParticleLoadLive4,
                Self::ParticleSavrLive0,
                Self::ParticleSavrLive1,
                Self::ParticleSavrLive2,
                Self::ParticleSavrLive3,
                Self::ParticleSavrLive4,
            ],
            Self::FuelbedWHeatDead => &[
                Self::ParticleLoadDead0,
                Self::ParticleLoadDead1,
                Self::ParticleLoadDead2,
                Self::ParticleLoadDead3,
                Self::ParticleLoadDead4,
                Self::ParticleSavrDead0,
                Self::ParticleSavrDead1,
                Self::ParticleSavrDead2,
                Self::ParticleSavrDead3,
                Self::ParticleSavrDead4,
                Self::ParticleHocDead,
            ],
            Self::FuelbedWHeatLive => &[
                Self::ParticleLoadLive0,
                Self::ParticleLoadLive1,
                Self::ParticleLoadLive2,
                Self::ParticleLoadLive3,
                Self::ParticleLoadLive4,
                Self::ParticleSavrLive0,
                Self::ParticleSavrLive1,
                Self::ParticleSavrLive2,
                Self::ParticleSavrLive3,
                Self::ParticleSavrLive4,
                Self::ParticleHocLive,
            ],
            Self::FuelbedWMoistDead => &[
                Self::ParticleLoadDead0,
                Self::ParticleLoadDead1,
                Self::ParticleLoadDead2,
                Self::ParticleLoadDead3,
                Self::ParticleLoadDead4,
                Self::ParticleSavrDead0,
                Self::ParticleSavrDead1,
                Self::ParticleSavrDead2,
                Self::ParticleSavrDead3,
                Self::ParticleSavrDead4,
                Self::ParticleMoistureDead0,
                Self::ParticleMoistureDead1,
                Self::ParticleMoistureDead2,
                Self::ParticleMoistureDead3,
                Self::ParticleMoistureDead4,
            ],
            Self::FuelbedWMoistLive => &[
                Self::ParticleLoadLive0,
                Self::ParticleLoadLive1,
                Self::ParticleLoadLive2,
                Self::ParticleLoadLive3,
                Self::ParticleLoadLive4,
                Self::ParticleSavrLive0,
                Self::ParticleSavrLive1,
                Self::ParticleSavrLive2,
                Self::ParticleSavrLive3,
                Self::ParticleSavrLive4,
                Self::ParticleMoistureLive0,
                Self::ParticleMoistureLive1,
                Self::ParticleMoistureLive2,
                Self::ParticleMoistureLive3,
                Self::ParticleMoistureLive4,
            ],
            Self::FuelbedWSilicaDead => &[
                Self::ParticleLoadDead0,
                Self::ParticleLoadDead1,
                Self::ParticleLoadDead2,
                Self::ParticleLoadDead3,
                Self::ParticleLoadDead4,
                Self::ParticleSavrDead0,
                Self::ParticleSavrDead1,
                Self::ParticleSavrDead2,
                Self::ParticleSavrDead3,
                Self::ParticleSavrDead4,
            ],
            Self::FuelbedWSilicaLive => &[
                Self::ParticleLoadLive0,
                Self::ParticleLoadLive1,
                Self::ParticleLoadLive2,
                Self::ParticleLoadLive3,
                Self::ParticleLoadLive4,
                Self::ParticleSavrLive0,
                Self::ParticleSavrLive1,
                Self::ParticleSavrLive2,
                Self::ParticleSavrLive3,
                Self::ParticleSavrLive4,
            ],
            Self::HeatPerUnitArea => &[Self::ReactionIntensity, Self::ResidenceTime],
            Self::IsActiveCrown => &[Self::FireType],
            Self::IsPassiveCrown => &[Self::FireType],
            Self::IsSurfaceFire => &[Self::FireType],
            Self::MidflameWindSpeed => &[Self::Waf, Self::Wind20],
            Self::NoWindNoSlopeSpreadRate => &[
                Self::ReactionIntensity,
                Self::FuelbedPropagatingFlux,
                Self::FuelbedHeatSink,
            ],
            Self::ParticleDefined => &[Self::FuelModelGrid],
            Self::ParticleDepth => &[Self::FuelModelGrid],
            Self::ParticleHocDead => &[Self::FuelModelGrid],
            Self::ParticleHocLive => &[Self::FuelModelGrid],
            Self::ParticleLoadDead0 => &[Self::FuelModelGrid],
            Self::ParticleLoadDead1 => &[Self::FuelModelGrid],
            Self::ParticleLoadDead2 => &[Self::FuelModelGrid],
            Self::ParticleLoadDead3 => &[Self::FuelModelGrid, Self::Mlh],
            Self::ParticleLoadDead4 => &[Self::FuelModelGrid],
            Self::ParticleLoadLive0 => &[Self::FuelModelGrid, Self::Mlh],
            Self::ParticleLoadLive1 => &[Self::FuelModelGrid],
            Self::ParticleLoadLive2 => &[Self::FuelModelGrid],
            Self::ParticleLoadLive3 => &[Self::FuelModelGrid],
            Self::ParticleLoadLive4 => &[Self::FuelModelGrid],
            Self::ParticleMoeDead => &[Self::FuelModelGrid],
            Self::ParticleMoistureDead0 => &[Self::M1h],
            Self::ParticleMoistureDead1 => &[Self::M10h],
            Self::ParticleMoistureDead2 => &[Self::M100h],
            Self::ParticleMoistureDead3 => &[Self::M1h],
            Self::ParticleMoistureDead4 => &[],
            Self::ParticleMoistureLive0 => &[Self::Mlh],
            Self::ParticleMoistureLive1 => &[Self::Mlw],
            Self::ParticleMoistureLive2 => &[],
            Self::ParticleMoistureLive3 => &[],
            Self::ParticleMoistureLive4 => &[],
            Self::ParticleSavrDead0 => &[Self::FuelModelGrid],
            Self::ParticleSavrDead1 => &[Self::FuelModelGrid],
            Self::ParticleSavrDead2 => &[Self::FuelModelGrid],
            Self::ParticleSavrDead3 => &[Self::FuelModelGrid],
            Self::ParticleSavrDead4 => &[Self::FuelModelGrid],
            Self::ParticleSavrLive0 => &[Self::FuelModelGrid],
            Self::ParticleSavrLive1 => &[Self::FuelModelGrid],
            Self::ParticleSavrLive2 => &[Self::FuelModelGrid],
            Self::ParticleSavrLive3 => &[Self::FuelModelGrid],
            Self::ParticleSavrLive4 => &[Self::FuelModelGrid],
            Self::PassiveCrownFireFlameLength => &[Self::PassiveCrownFireLineIntensity],
            Self::PassiveCrownFireHeatPerUnitArea => &[
                Self::SurfaceHpua,
                Self::CrownFractionBurned,
                Self::CanopyHeatPerUnitArea,
            ],
            Self::PassiveCrownFireLineIntensity => &[
                Self::PassiveCrownFireSpreadRate,
                Self::PassiveCrownFireHeatPerUnitArea,
            ],
            Self::PassiveCrownFireSpreadRate => &[
                Self::SurfaceRos,
                Self::CrownFractionBurned,
                Self::CrownFireSpreadRate,
            ],
            Self::RateVector => &[Self::VectorX, Self::VectorY],
            Self::ReactionIntensity => &[
                Self::FuelbedSigma,
                Self::FuelbedRelativePackingRatio,
                Self::FuelbedWMoistDead,
                Self::FuelbedWMoistLive,
                Self::FuelbedMoeDead,
                Self::FuelbedMoeLive,
                Self::FuelbedWSilicaDead,
                Self::FuelbedWSilicaLive,
                Self::FuelbedWflDead,
                Self::FuelbedWflLive,
                Self::FuelbedWHeatDead,
                Self::FuelbedWHeatLive,
            ],
            Self::ResidenceTime => &[Self::FuelbedSigma],
            Self::SlopePhi => &[
                Self::Slope,
                Self::FuelbedPackingRatio,
                Self::ReactionIntensity,
            ],
            Self::SpreadRate => &[Self::NoWindNoSlopeSpreadRate, Self::RateVector],
            Self::SurfaceFireCriticalSpreadRate => &[
                Self::CrownCriticalSurfaceFireLineIntensity,
                Self::SurfaceHpua,
            ],
            Self::VectorX => &[
                Self::NoWindNoSlopeSpreadRate,
                Self::WindPhi,
                Self::SlopePhi,
                Self::WindDirection,
                Self::Aspect,
            ],
            Self::VectorY => &[
                Self::NoWindNoSlopeSpreadRate,
                Self::WindPhi,
                Self::SlopePhi,
                Self::WindDirection,
                Self::Aspect,
            ],
            Self::Waf => &[
                Self::CanopyCover,
                Self::CanopyHeight,
                Self::CrownRatio,
                Self::UserWaf,
                Self::FuelbedDepth,
            ],
            Self::Wind20 => &[Self::WindSpeed],
            Self::WindB => &[Self::FuelbedSigma],
            Self::WindC => &[Self::FuelbedSigma],
            Self::WindE => &[Self::FuelbedSigma],
            Self::WindPhi => &[
                Self::FuelbedSigma,
                Self::MidflameWindSpeed,
                Self::WindB,
                Self::WindC,
                Self::WindE,
                Self::FuelbedRelativePackingRatio,
            ],
            _ => &[],
        }
    }
}
