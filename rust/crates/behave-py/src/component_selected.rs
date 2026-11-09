//! Independent ratio and final-branch outputs consume only their actual operands.
use crate::{
    component_columns::Columns,
    operation::Operation,
    prepared::{Field, PreparedArguments},
};
use behave_fire::{crown, ParallelError};

pub(crate) fn required(operation: Operation, names: &[String]) -> Option<Vec<Field>> {
    use Field::*;
    if operation == Operation::SurfaceCalculateFuelbedIntermediates {
        return Some(
            crate::component_fuelbed::dependencies(names)
                .expect("validated fuelbed schema")
                .into_iter()
                .map(|(_, input)| input)
                .collect(),
        );
    }
    if operation == Operation::CrownCalculateCrownSurfaceFire {
        let weather = names.iter().any(|name| {
            name.starts_with("crown_surface.")
                || matches!(
                    name.as_str(),
                    "crown_fire_spread_rate" | "crown_length_to_width_ratio"
                )
        });
        return Some(
            crate::component_requirements::fields(operation)
                .iter()
                .copied()
                .filter(|field| {
                    weather
                        || !matches!(
                            field,
                            WindSpeed | WindDirection | Slope | WindSpeedUnits | SlopeUnits
                        )
                })
                .collect(),
        );
    }
    let mut result = Vec::new();
    for name in names {
        let fields: &[Field] = match (operation, name.as_str()) {
            (Operation::CrownCalculateCrownFireRatios, "crown_fire_transition_ratio") => {
                &[SurfaceFli, CriticalSurfaceFli]
            }
            (Operation::CrownCalculateCrownFireRatios, "crown_fire_active_ratio") => {
                &[CrownRos, CriticalCrownRos]
            }
            (
                Operation::CrownCalculateCrownHeatAndIntensity,
                "crown_fuel_load" | "canopy_heat_per_unit_area",
            ) => &[CanopyBaseHeight, CanopyHeight, CanopyBulkDensity],
            (Operation::CrownCalculateCrownHeatAndIntensity, "crown_fire_heat_per_unit_area") => &[
                SurfaceHpua,
                CanopyBaseHeight,
                CanopyHeight,
                CanopyBulkDensity,
            ],
            (
                Operation::CrownCalculateCrownHeatAndIntensity,
                "crown_fire_line_intensity" | "crown_flame_length",
            ) => &[
                SurfaceHpua,
                CrownRos,
                CanopyBaseHeight,
                CanopyHeight,
                CanopyBulkDensity,
            ],
            (
                Operation::CrownCalculatePassiveCrownFireBehavior,
                "passive_crown_fire_spread_rate",
            ) => &[SurfaceRos, CrownRos, Cfb],
            (
                Operation::CrownCalculatePassiveCrownFireBehavior,
                "passive_crown_fire_heat_per_unit_area",
            ) => &[SurfaceHpua, CanopyHpua, Cfb],
            (
                Operation::CrownCalculatePassiveCrownFireBehavior,
                "passive_crown_fire_line_intensity" | "passive_crown_fire_flame_length",
            ) => &[SurfaceRos, SurfaceHpua, CrownRos, CanopyHpua, Cfb],
            (Operation::VaporPressureDeficitCalculatorCalculateVpd, "saturated_vp") => {
                &[Temperature, TempUnits, OutputUnits]
            }
            (Operation::VaporPressureDeficitCalculatorCalculateVpd, "actual_vp" | "vpd") => &[
                Temperature,
                TempUnits,
                RelativeHumidity,
                RhUnits,
                OutputUnits,
            ],
            (
                Operation::MortalityCalculateCrownScorchMortality,
                "crown_length_scorch" | "crown_volume_scorch",
            ) => &[ScorchHeightFt, TreeHeightFt, CrownRatio],
            (Operation::MortalityCalculateCrownScorchMortality, "probability_mortality") => &[
                ScorchHeightFt,
                TreeHeightFt,
                CrownRatio,
                DbhInches,
                Coefficients0,
                Coefficients1,
                Coefficients2,
            ],
            (Operation::SafetyCalculateSafetyZone, "separation_distance") => &[FlameHeightFt],
            (Operation::SafetyCalculateSafetyZone, "radius" | "area") => &[
                FlameHeightFt,
                NumberOfPersonnel,
                AreaPerPersonSqft,
                NumberOfEquipment,
                AreaPerEquipmentSqft,
            ],
            (Operation::CrownAssignFinalCrownFireBehavior, "final_spread_rate") => &[
                IsSurfaceFire,
                IsPassiveCrown,
                SurfaceRos,
                PassiveRos,
                CrownRos,
            ],
            (Operation::CrownAssignFinalCrownFireBehavior, "final_heat_per_unit_area") => &[
                IsSurfaceFire,
                IsPassiveCrown,
                SurfaceHpua,
                PassiveHpua,
                CrownHpua,
            ],
            (Operation::CrownAssignFinalCrownFireBehavior, "final_fireline_intensity") => &[
                IsSurfaceFire,
                IsPassiveCrown,
                SurfaceFli,
                PassiveFli,
                CrownFli,
            ],
            (Operation::CrownAssignFinalCrownFireBehavior, "final_flame_length") => &[
                IsSurfaceFire,
                IsPassiveCrown,
                SurfaceFli,
                PassiveFl,
                CrownFl,
            ],
            _ => return None,
        };
        for &field in fields {
            if !result.contains(&field) {
                result.push(field);
            }
        }
    }
    Some(result)
}

pub(crate) fn run(
    operation: Operation,
    p: &PreparedArguments<'_>,
    names: &[String],
    output: &mut Columns,
    threads: usize,
) -> Option<Result<(), ParallelError<String>>> {
    if operation == Operation::SurfaceCalculateFuelbedIntermediates {
        return Some(crate::component_fuelbed::run(p, names, output, threads));
    }
    if let Some(result) = crate::component_independent::run(operation, p, names, output, threads) {
        return Some(result);
    }
    let select = |name| names.iter().any(|v| v == name);
    if operation == Operation::CrownCalculateCrownFireRatios {
        let transition = select("crown_fire_transition_ratio");
        let active = select("crown_fire_active_ratio");
        return Some(output.fill(threads, |i, row| {
            if transition {
                row.f64(
                    "crown_fire_transition_ratio",
                    crown::available::ratio(
                        p.n(Field::SurfaceFli, i)?,
                        p.n(Field::CriticalSurfaceFli, i)?,
                    ),
                );
            }
            if active {
                row.f64(
                    "crown_fire_active_ratio",
                    crown::available::ratio(
                        p.n(Field::CrownRos, i)?,
                        p.n(Field::CriticalCrownRos, i)?,
                    ),
                );
            }
            Ok(())
        }));
    }
    if operation != Operation::CrownAssignFinalCrownFireBehavior {
        return None;
    }
    let spread = select("final_spread_rate");
    let heat = select("final_heat_per_unit_area");
    let intensity = select("final_fireline_intensity");
    let flame = select("final_flame_length");
    Some(output.fill(threads, |i, row| {
        let surface = p.flag(Field::IsSurfaceFire, i)?;
        let passive = p.flag(Field::IsPassiveCrown, i)?;
        let branch = |s, t, c| {
            p.n(
                if surface {
                    s
                } else if passive {
                    t
                } else {
                    c
                },
                i,
            )
        };
        if spread {
            row.f64(
                "final_spread_rate",
                branch(Field::SurfaceRos, Field::PassiveRos, Field::CrownRos)?,
            );
        }
        if heat {
            row.f64(
                "final_heat_per_unit_area",
                branch(Field::SurfaceHpua, Field::PassiveHpua, Field::CrownHpua)?,
            );
        }
        if intensity {
            row.f64(
                "final_fireline_intensity",
                branch(Field::SurfaceFli, Field::PassiveFli, Field::CrownFli)?,
            );
        }
        if flame {
            row.f64(
                "final_flame_length",
                if surface {
                    crown::available::flame_length(p.n(Field::SurfaceFli, i)?)
                } else {
                    p.n(
                        if passive {
                            Field::PassiveFl
                        } else {
                            Field::CrownFl
                        },
                        i,
                    )?
                },
            );
        }
        Ok(())
    }))
}
