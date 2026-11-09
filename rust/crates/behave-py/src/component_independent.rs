//! Dependency-restricted component stages share scalar science, never dummy inputs.
use crate::{
    component_columns::Columns,
    operation::Operation,
    prepared::{Field, PreparedArguments},
};
use behave_fire::{crown, mortality, safety, vapor_pressure_deficit as vpd, ParallelError};

pub(crate) fn run(
    operation: Operation,
    p: &PreparedArguments<'_>,
    names: &[String],
    output: &mut Columns,
    threads: usize,
) -> Option<Result<(), ParallelError<String>>> {
    use Operation::*;
    if operation == CrownCalculateCrownSurfaceFire {
        return Some(crown_surface(p, names, output, threads));
    }
    let fields: &[&str] = match operation {
        CrownCalculateCrownHeatAndIntensity => &[
            "crown_fuel_load",
            "canopy_heat_per_unit_area",
            "crown_fire_heat_per_unit_area",
            "crown_fire_line_intensity",
            "crown_flame_length",
        ],
        CrownCalculatePassiveCrownFireBehavior => &[
            "passive_crown_fire_spread_rate",
            "passive_crown_fire_heat_per_unit_area",
            "passive_crown_fire_line_intensity",
            "passive_crown_fire_flame_length",
        ],
        MortalityCalculateCrownScorchMortality => &[
            "crown_length_scorch",
            "crown_volume_scorch",
            "probability_mortality",
        ],
        VaporPressureDeficitCalculatorCalculateVpd => &["vpd", "actual_vp", "saturated_vp"],
        SafetyCalculateSafetyZone => &["separation_distance", "radius", "area"],
        _ => return None,
    };
    let mask = fields.iter().enumerate().fold(0u8, |m, (i, name)| {
        if names.iter().any(|n| n == name) {
            m | (1 << i)
        } else {
            m
        }
    });
    Some(match operation {
        CrownCalculateCrownHeatAndIntensity => output.fill(threads, |i, row| {
            let n = |field| p.n(field, i);
            let load = crown::calculate_canopy_fuel_load(
                n(Field::CanopyBaseHeight)?,
                n(Field::CanopyHeight)?,
                n(Field::CanopyBulkDensity)?,
            );
            let canopy = crown::calculate_canopy_heat_per_unit_area(load);
            let heat = if mask & 28 != 0 {
                n(Field::SurfaceHpua)? + canopy
            } else {
                f64::NAN
            };
            let intensity = if mask & 24 != 0 {
                (n(Field::CrownRos)? / 60.) * heat
            } else {
                f64::NAN
            };
            if mask & 1 != 0 {
                row.f64("crown_fuel_load", load);
            }
            if mask & 2 != 0 {
                row.f64("canopy_heat_per_unit_area", canopy);
            }
            if mask & 4 != 0 {
                row.f64("crown_fire_heat_per_unit_area", heat);
            }
            if mask & 8 != 0 {
                row.f64("crown_fire_line_intensity", intensity);
            }
            if mask & 16 != 0 {
                row.f64(
                    "crown_flame_length",
                    crown::available::flame_length(intensity),
                );
            }
            Ok(())
        }),
        CrownCalculatePassiveCrownFireBehavior => output.fill(threads, |i, row| {
            let n = |field| p.n(field, i);
            let fraction = n(Field::Cfb)?;
            let rate = if mask & 13 != 0 {
                crown::calculate_passive_spread_rate(
                    n(Field::SurfaceRos)?,
                    n(Field::CrownRos)?,
                    fraction,
                )
            } else {
                f64::NAN
            };
            let heat = if mask & 14 != 0 {
                crown::calculate_passive_heat_per_unit_area(
                    n(Field::SurfaceHpua)?,
                    n(Field::CanopyHpua)?,
                    fraction,
                )
            } else {
                f64::NAN
            };
            let intensity = if mask & 12 != 0 {
                crown::calculate_passive_intensity(heat, rate)
            } else {
                f64::NAN
            };
            if mask & 1 != 0 {
                row.f64("passive_crown_fire_spread_rate", rate);
            }
            if mask & 2 != 0 {
                row.f64("passive_crown_fire_heat_per_unit_area", heat);
            }
            if mask & 4 != 0 {
                row.f64("passive_crown_fire_line_intensity", intensity);
            }
            if mask & 8 != 0 {
                row.f64(
                    "passive_crown_fire_flame_length",
                    crown::available::flame_length(intensity),
                );
            }
            Ok(())
        }),
        MortalityCalculateCrownScorchMortality => output.fill(threads, |i, row| {
            let n = |field| p.n(field, i);
            let (cls, cvs) = mortality::calculate_crown_scorch_fractions(
                n(Field::ScorchHeightFt)?,
                n(Field::TreeHeightFt)?,
                n(Field::CrownRatio)?,
            );
            if mask & 1 != 0 {
                row.f64("crown_length_scorch", cls);
            }
            if mask & 2 != 0 {
                row.f64("crown_volume_scorch", cvs);
            }
            if mask & 4 != 0 {
                let value = mortality::calculate_crown_scorch_mortality_with_coefficients(
                    n(Field::ScorchHeightFt)?,
                    n(Field::TreeHeightFt)?,
                    n(Field::CrownRatio)?,
                    n(Field::DbhInches)?,
                    [
                        n(Field::Coefficients0)?,
                        n(Field::Coefficients1)?,
                        n(Field::Coefficients2)?,
                    ],
                );
                row.f64("probability_mortality", value.probability_mortality);
            }
            Ok(())
        }),
        VaporPressureDeficitCalculatorCalculateVpd => output.fill(threads, |i, row| {
            let sat = vpd::saturated_vapor_pressure_hpa(
                p.n(Field::Temperature, i)?,
                p.temperature_units(i)?,
            );
            let unit = p.pressure_units(i)?;
            let actual = if mask & 3 != 0 {
                p.humidity_units(i)?
                    .to_base(p.n(Field::RelativeHumidity, i)?)
                    * sat
            } else {
                f64::NAN
            };
            if mask & 1 != 0 {
                row.f64("vpd", unit.from_base((sat - actual) * 100.));
            }
            if mask & 2 != 0 {
                row.f64("actual_vp", unit.from_base(actual * 100.));
            }
            if mask & 4 != 0 {
                row.f64("saturated_vp", unit.from_base(sat * 100.));
            }
            Ok(())
        }),
        SafetyCalculateSafetyZone => output.fill(threads, |i, row| {
            let n = |field| p.n(field, i);
            let flame = n(Field::FlameHeightFt)?;
            if mask & 6 != 0 {
                let value = safety::calculate_safety_zone(
                    flame,
                    n(Field::NumberOfPersonnel)?,
                    n(Field::AreaPerPersonSqft)?,
                    n(Field::NumberOfEquipment)?,
                    n(Field::AreaPerEquipmentSqft)?,
                );
                if mask & 1 != 0 {
                    row.f64("separation_distance", value.separation_distance);
                }
                if mask & 2 != 0 {
                    row.f64("radius", value.radius);
                }
                if mask & 4 != 0 {
                    row.f64("area", value.area);
                }
            } else {
                row.f64(
                    "separation_distance",
                    safety::calculate_separation_distance(flame),
                );
            }
            Ok(())
        }),
        _ => unreachable!(),
    })
}

fn crown_surface(
    p: &PreparedArguments<'_>,
    names: &[String],
    output: &mut Columns,
    threads: usize,
) -> Result<(), ParallelError<String>> {
    let particles = names.iter().any(|n| n.starts_with("p_crown."));
    let fuelbed = names.iter().any(|n| n.starts_with("ib_crown."));
    let surface = names.iter().any(|n| n.starts_with("crown_surface."));
    let reaction = names.iter().any(|n| n == "ri_crown");
    let spread = names.iter().any(|n| n == "crown_fire_spread_rate");
    let length = names.iter().any(|n| n == "crown_length_to_width_ratio");
    let weather = surface || spread || length;
    output.fill(threads, |i, row| {
        let fuel = crate::structures::fuel(p, i)?;
        let full = if weather {
            Some(crown::calculate_crown_surface_fire(
                &crate::crown::input(p, i)?,
                &fuel,
                &p.crown_options(),
                behave_fire::engine::SurfaceTweaks::default(),
            ))
        } else {
            None
        };
        let (particle, bed, ri) = if let Some(full) = &full {
            (full.p_crown, full.ib_crown, full.ri_crown)
        } else {
            let particle = behave_fire::surface::build_particle_arrays_from_prepared(
                &fuel,
                p.n(Field::M1h, i)?,
                p.n(Field::M10h, i)?,
                p.n(Field::M100h, i)?,
                p.n(Field::Mlh, i)?,
                p.n(Field::Mlw, i)?,
            );
            let bed = fuel.prepare_fuelbed_available(&particle);
            let ri = fuel.reaction_intensity_available(&bed);
            (particle, bed, ri)
        };
        if particles {
            row.with_prefix("p_crown", |row| {
                crate::structures::particles_output(row, particle)
            });
        }
        if fuelbed {
            row.with_prefix("ib_crown", |row| {
                crate::structures::fuelbed_output(row, bed)
            });
        }
        if let Some(full) = full {
            if surface {
                row.with_prefix("crown_surface", |row| {
                    crate::structures::surface_output(row, full.crown_surface)
                });
            }
            if reaction {
                row.f64("ri_crown", ri);
            }
            if spread {
                row.f64("crown_fire_spread_rate", full.crown_fire_spread_rate);
            }
            if length {
                row.f64(
                    "crown_length_to_width_ratio",
                    full.crown_length_to_width_ratio,
                );
            }
        } else if reaction {
            row.f64("ri_crown", ri);
        }
        Ok(())
    })
}
