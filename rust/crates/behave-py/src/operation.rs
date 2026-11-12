//! Decode operation identity once, before any cell loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Family {
    Contain,
    Crown,
    FineDeadFuelMoistureTool,
    Ignite,
    Model,
    Mortality,
    Safety,
    Spot,
    Surface,
    Units,
    VaporPressureDeficitCalculator,
}
macro_rules! operations {
    ($($variant:ident: $family:ident => $name:literal),+ $(,)?) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum Operation { $($variant),+ }
        impl Operation {
            pub(crate) const ALL: &'static [Self] = &[$(Self::$variant),+];
            pub(crate) fn parse(name: &str) -> Option<Self> { match name { $($name => Some(Self::$variant),)+ _ => None } }
            pub(crate) const fn name(self) -> &'static str { match self { $(Self::$variant => $name),+ } }
            pub(crate) const fn family(self) -> Family { match self { $(Self::$variant => Family::$family),+ } }
            pub(crate) const fn main_model(self) -> bool { matches!(self, Self::ModelRunSurface | Self::CrownRunCrownFire) }
        }
    };
}
impl Operation {
    pub(crate) fn numeric_contract(self) -> &'static str {
        "f64"
    }
}
operations! {
    ContainRunContainSimArray: Contain => "contain.run_contain_sim_array",
    CrownAssignFinalCrownFireBehavior: Crown => "crown.assign_final_crown_fire_behavior",
    CrownCalculateCriticalCrownFireSpreadRate: Crown => "crown.calculate_critical_crown_fire_spread_rate",
    CrownCalculateCriticalSurfaceFirelineIntensity: Crown => "crown.calculate_critical_surface_fireline_intensity",
    CrownCalculateCrownFireRatios: Crown => "crown.calculate_crown_fire_ratios",
    CrownCalculateCrownFractionBurned: Crown => "crown.calculate_crown_fraction_burned",
    CrownCalculateCrownHeatAndIntensity: Crown => "crown.calculate_crown_heat_and_intensity",
    CrownCalculateCrownSurfaceFire: Crown => "crown.calculate_crown_surface_fire",
    CrownCalculateCrowningSurfaceFireSpreadRate: Crown => "crown.calculate_crowning_surface_fire_spread_rate",
    CrownCalculatePassiveCrownFireBehavior: Crown => "crown.calculate_passive_crown_fire_behavior",
    CrownCalculateSurfaceFireCriticalSpreadRate: Crown => "crown.calculate_surface_fire_critical_spread_rate",
    CrownClassifyCrownFireType: Crown => "crown.classify_crown_fire_type",
    CrownRunCrownFire: Crown => "crown.run_crown_fire",
    FineDeadFuelMoistureToolCalculateFineDeadFuelMoisture: FineDeadFuelMoistureTool => "fine_dead_fuel_moisture_tool.calculate_fine_dead_fuel_moisture",
    IgniteCalculateFirebrandIgnitionProbability: Ignite => "ignite.calculate_firebrand_ignition_probability",
    IgniteCalculateFuelTemperature: Ignite => "ignite.calculate_fuel_temperature",
    IgniteCalculateLightningIgnitionProbability: Ignite => "ignite.calculate_lightning_ignition_probability",
    ModelCanopyCrownRatio: Model => "model.canopy_crown_ratio",
    ModelRunSurface: Model => "model.run_surface",
    MortalityCalculateCrownScorchMortality: Mortality => "mortality.calculate_crown_scorch_mortality",
    MortalityCalculateScorchHeight: Mortality => "mortality.calculate_scorch_height",
    SafetyCalculateSafetyZone: Safety => "safety.calculate_safety_zone",
    SpotCalculateSpottingFromBurningPile: Spot => "spot.calculate_spotting_from_burning_pile",
    SpotCalculateSpottingFromSurfaceFire: Spot => "spot.calculate_spotting_from_surface_fire",
    SpotCalculateSpottingFromTorchingTrees: Spot => "spot.calculate_spotting_from_torching_trees",
    SurfaceBuildParticleArrays: Surface => "surface.build_particle_arrays",
    SurfaceCalculateBackingSpreadRate: Surface => "surface.calculate_backing_spread_rate",
    SurfaceCalculateDirectionOfMaxSpread: Surface => "surface.calculate_direction_of_max_spread",
    SurfaceCalculateEccentricity: Surface => "surface.calculate_eccentricity",
    SurfaceCalculateEffectiveWindSpeed: Surface => "surface.calculate_effective_wind_speed",
    SurfaceCalculateFireArea: Surface => "surface.calculate_fire_area",
    SurfaceCalculateFireLength: Surface => "surface.calculate_fire_length",
    SurfaceCalculateFireLengthToWidthRatio: Surface => "surface.calculate_fire_length_to_width_ratio",
    SurfaceCalculateFirePerimeter: Surface => "surface.calculate_fire_perimeter",
    SurfaceCalculateFireWidth: Surface => "surface.calculate_fire_width",
    SurfaceCalculateFirelineIntensity: Surface => "surface.calculate_fireline_intensity",
    SurfaceCalculateFlameLength: Surface => "surface.calculate_flame_length",
    SurfaceCalculateFlankingSpreadRate: Surface => "surface.calculate_flanking_spread_rate",
    SurfaceCalculateForwardSpreadRate: Surface => "surface.calculate_forward_spread_rate",
    SurfaceCalculateFuelbedIntermediates: Surface => "surface.calculate_fuelbed_intermediates",
    SurfaceCalculateHeatPerUnitArea: Surface => "surface.calculate_heat_per_unit_area",
    SurfaceCalculateMidflameWindSpeed: Surface => "surface.calculate_midflame_wind_speed",
    SurfaceCalculateNoWindNoSlopeSpreadRate: Surface => "surface.calculate_no_wind_no_slope_spread_rate",
    SurfaceCalculateReactionIntensity: Surface => "surface.calculate_reaction_intensity",
    SurfaceCalculateReactionIntensityOutput: Surface => "surface.calculate_reaction_intensity_output",
    SurfaceCalculateResidenceTime: Surface => "surface.calculate_residence_time",
    SurfaceCalculateSlopePhi: Surface => "surface.calculate_slope_phi",
    SurfaceCalculateSurfaceRateVector: Surface => "surface.calculate_surface_rate_vector",
    SurfaceCalculateSurfaceWindAdjustmentFactor: Surface => "surface.calculate_surface_wind_adjustment_factor",
    SurfaceCalculateSurfaceWindCoefficients: Surface => "surface.calculate_surface_wind_coefficients",
    SurfaceCalculateTwentyFootWindSpeed: Surface => "surface.calculate_twenty_foot_wind_speed",
    SurfaceCalculateWindAdjustmentFactor: Surface => "surface.calculate_wind_adjustment_factor",
    SurfaceCalculateWindPhi: Surface => "surface.calculate_wind_phi",
    SurfaceRunSurfaceFire: Surface => "surface.run_surface_fire",
    UnitsAreaFromBase: Units => "units.Area.from_base",
    UnitsAreaToBase: Units => "units.Area.to_base",
    UnitsBasalAreaFromBase: Units => "units.BasalArea.from_base",
    UnitsBasalAreaToBase: Units => "units.BasalArea.to_base",
    UnitsDensityFromBase: Units => "units.Density.from_base",
    UnitsDensityToBase: Units => "units.Density.to_base",
    UnitsFirelineIntensityFromBase: Units => "units.FirelineIntensity.from_base",
    UnitsFirelineIntensityToBase: Units => "units.FirelineIntensity.to_base",
    UnitsFractionFromBase: Units => "units.Fraction.from_base",
    UnitsFractionToBase: Units => "units.Fraction.to_base",
    UnitsHeatOfCombustionFromBase: Units => "units.HeatOfCombustion.from_base",
    UnitsHeatOfCombustionToBase: Units => "units.HeatOfCombustion.to_base",
    UnitsHeatPerUnitAreaFromBase: Units => "units.HeatPerUnitArea.from_base",
    UnitsHeatPerUnitAreaToBase: Units => "units.HeatPerUnitArea.to_base",
    UnitsHeatSinkFromBase: Units => "units.HeatSink.from_base",
    UnitsHeatSinkToBase: Units => "units.HeatSink.to_base",
    UnitsHeatSourceAndReactionIntensityFromBase: Units => "units.HeatSourceAndReactionIntensity.from_base",
    UnitsHeatSourceAndReactionIntensityToBase: Units => "units.HeatSourceAndReactionIntensity.to_base",
    UnitsLengthFromBase: Units => "units.Length.from_base",
    UnitsLengthToBase: Units => "units.Length.to_base",
    UnitsLoadingFromBase: Units => "units.Loading.from_base",
    UnitsLoadingToBase: Units => "units.Loading.to_base",
    UnitsPressureFromBase: Units => "units.Pressure.from_base",
    UnitsPressureToBase: Units => "units.Pressure.to_base",
    UnitsSlopeFromBase: Units => "units.Slope.from_base",
    UnitsSlopeToBase: Units => "units.Slope.to_base",
    UnitsSpeedFromBase: Units => "units.Speed.from_base",
    UnitsSpeedToBase: Units => "units.Speed.to_base",
    UnitsSurfaceAreaToVolumeFromBase: Units => "units.SurfaceAreaToVolume.from_base",
    UnitsSurfaceAreaToVolumeToBase: Units => "units.SurfaceAreaToVolume.to_base",
    UnitsTemperatureFromBase: Units => "units.Temperature.from_base",
    UnitsTemperatureToBase: Units => "units.Temperature.to_base",
    UnitsTimeFromBase: Units => "units.Time.from_base",
    UnitsTimeToBase: Units => "units.Time.to_base",
    VaporPressureDeficitCalculatorCalculateVpd: VaporPressureDeficitCalculator => "vapor_pressure_deficit_calculator.calculate_vpd",
}
