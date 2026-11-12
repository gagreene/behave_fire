//! Shared field-level fuelbed science for sessions and component dependency groups.
use super::Field;

fn five<E>(read: &mut impl FnMut(Field) -> Result<f64, E>, start: Field) -> Result<[f64; 5], E> {
    let mut result = [0.; 5];
    for (index, value) in result.iter_mut().enumerate() {
        *value = read(Field::ALL[start as usize + index])?;
    }
    Ok(result)
}
fn area<E>(
    read: &mut impl FnMut(Field) -> Result<f64, E>,
    dead: bool,
) -> Result<(f64, [f64; 5]), E> {
    let load = five(
        read,
        if dead {
            Field::ParticleLoadDead0
        } else {
            Field::ParticleLoadLive0
        },
    )?;
    let savr = five(
        read,
        if dead {
            Field::ParticleSavrDead0
        } else {
            Field::ParticleSavrLive0
        },
    )?;
    let (weights, total) = crate::surface::particle_area_weights::<true>(&load, &savr);
    Ok((total, weights))
}

/// Evaluate one physical fuelbed field from actual input or injected dependency values.
pub fn evaluate<E>(
    field: Field,
    mut read: impl FnMut(Field) -> Result<f64, E>,
) -> Option<Result<f64, E>> {
    if field.producer() != Some(super::Stage::Fuelbed) || field.kind() != super::FieldKind::Physical
    {
        return None;
    }
    Some((|| {
        Ok(match field {
            Field::FuelbedDepth => read(Field::ParticleDepth)?,
            Field::FuelbedMoeDead => read(Field::ParticleMoeDead)?,
            Field::FuelbedBulkDensity => crate::surface::bulk_from_loads::<true>(
                &five(&mut read, Field::ParticleLoadDead0)?,
                &five(&mut read, Field::ParticleLoadLive0)?,
                read(Field::ParticleDepth)?,
            ),
            Field::FuelbedPackingRatio => crate::surface::packing_from_loads::<true>(
                &five(&mut read, Field::ParticleLoadDead0)?,
                &five(&mut read, Field::ParticleLoadLive0)?,
                read(Field::ParticleDepth)?,
            ),
            Field::FuelbedWHeatDead => crate::surface::weighted_constant(
                &area(&mut read, true)?.1,
                read(Field::ParticleHocDead)?,
            ),
            Field::FuelbedWHeatLive => crate::surface::weighted_constant(
                &area(&mut read, false)?.1,
                read(Field::ParticleHocLive)?,
            ),
            Field::FuelbedWSilicaDead => {
                crate::surface::weighted_constant(&area(&mut read, true)?.1, 0.01)
            }
            Field::FuelbedWSilicaLive => crate::surface::weighted_values(
                &area(&mut read, false)?.1,
                &[0.01, 0.01, 0., 0., 0.],
            ),
            Field::FuelbedWMoistDead => crate::surface::weighted_values(
                &area(&mut read, true)?.1,
                &five(&mut read, Field::ParticleMoistureDead0)?,
            ),
            Field::FuelbedWMoistLive => crate::surface::weighted_values(
                &area(&mut read, false)?.1,
                &five(&mut read, Field::ParticleMoistureLive0)?,
            ),
            Field::FuelbedWflDead | Field::FuelbedWflLive => {
                let dead = field == Field::FuelbedWflDead;
                let (load, savr) = if dead {
                    (Field::ParticleLoadDead0, Field::ParticleSavrDead0)
                } else {
                    (Field::ParticleLoadLive0, Field::ParticleSavrLive0)
                };
                let net = five(&mut read, load)?.map(|v| v * (1. - 0.0555));
                crate::surface::size_sorted_wfl(
                    &five(&mut read, savr)?,
                    &area(&mut read, dead)?.1,
                    &net,
                )
            }
            Field::FuelbedMoeLive => {
                let sd = five(&mut read, Field::ParticleSavrDead0)?;
                let sl = five(&mut read, Field::ParticleSavrLive0)?;
                let terms = crate::surface::SavrTerms::new(&sd, &sl);
                let ld = five(&mut read, Field::ParticleLoadDead0)?;
                let ll = five(&mut read, Field::ParticleLoadLive0)?;
                let md = five(&mut read, Field::ParticleMoistureDead0)?;
                let fine_d = (0..5).map(|n| ld[n] * terms.exp_dead[n]).sum();
                let fine_dm = (0..5).map(|n| ld[n] * terms.exp_dead[n] * md[n]).sum();
                let fine_l = (0..5).map(|n| ll[n] * terms.exp_live500[n]).sum();
                crate::surface::live_extinction_from_fines::<true>(
                    fine_d,
                    fine_dm,
                    fine_l,
                    read(Field::FuelbedMoeDead)?,
                )
            }
            Field::FuelbedHeatSink => {
                let terms = crate::surface::SavrTerms::new(
                    &five(&mut read, Field::ParticleSavrDead0)?,
                    &five(&mut read, Field::ParticleSavrLive0)?,
                );
                let dead = crate::surface::heat_sink_side(
                    read(Field::FuelbedFracDead)?,
                    &area(&mut read, true)?.1,
                    &five(&mut read, Field::ParticleMoistureDead0)?,
                    &terms.exp_dead,
                );
                let live = crate::surface::heat_sink_side(
                    read(Field::FuelbedFracLive)?,
                    &area(&mut read, false)?.1,
                    &five(&mut read, Field::ParticleMoistureLive0)?,
                    &terms.exp_live138,
                );
                (dead + live) * read(Field::FuelbedBulkDensity)?
            }
            Field::FuelbedFracDead => {
                let (dead, _) = area(&mut read, true)?;
                let (live, _) = area(&mut read, false)?;
                crate::surface::dead_area_fraction::<true>(dead, live)
            }
            Field::FuelbedFracLive => 1. - read(Field::FuelbedFracDead)?,
            Field::FuelbedSigma => {
                let (_, wd) = area(&mut read, true)?;
                let (_, wl) = area(&mut read, false)?;
                crate::surface::mixture_sigma(
                    read(Field::FuelbedFracDead)?,
                    read(Field::FuelbedFracLive)?,
                    &wd,
                    &five(&mut read, Field::ParticleSavrDead0)?,
                    &wl,
                    &five(&mut read, Field::ParticleSavrLive0)?,
                )
            }
            Field::FuelbedRelativePackingRatio => {
                crate::surface::relative_packing_from_sigma::<true>(
                    read(Field::FuelbedSigma)?,
                    read(Field::FuelbedPackingRatio)?,
                )
            }
            Field::FuelbedPropagatingFlux => crate::surface::propagating_flux_from_sigma::<true>(
                read(Field::FuelbedSigma)?,
                read(Field::FuelbedPackingRatio)?,
            ),
            _ => unreachable!("validated physical fuelbed field"),
        })
    })())
}
