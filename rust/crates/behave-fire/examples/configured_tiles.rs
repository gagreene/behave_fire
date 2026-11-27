//! Read raster windows in the caller; compute f64 cells with per-cell injections.
use behave_fire::crown::SurfaceFireInputs;
use behave_fire::engine::{
    run_model, CrownCell, CrownPlan, CrownTweaks, ModelBatch, ModelRun, OutputRequest,
    PreparedFuels, SurfaceTweaks,
};
use behave_fire::output::ColumnValues;
use behave_fire::raster::{CrownColumns, CrownRaster, CrownTweakColumns, SurfaceTweakColumns};
use behave_fire::{FuelModels, GridShape, PerCell};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let plan = CrownPlan::default();
    // A raster reader owns these row-major columns. The core needs no raster library.
    let wind = (0..4099).map(|i| (i % 501) as f64).collect::<Vec<_>>();
    let reaction = (0..wind.len())
        .map(|i| 900. + (i % 7) as f64)
        .collect::<Vec<_>>();
    let fraction = (0..wind.len())
        .map(|i| (i % 11) as f64 / 10.)
        .collect::<Vec<_>>();
    for threads in [1, 4] {
        for start in (0..wind.len()).step_by(1031) {
            let end = (start + 1031).min(wind.len());
            let raster = CrownRaster::new(
                GridShape::new(vec![end - start])?,
                CrownColumns {
                    wind_speed: PerCell::Cells(&wind[start..end]),
                    canopy_base_height: PerCell::Scalar(4.),
                    canopy_height: PerCell::Scalar(30.),
                    canopy_bulk_density: PerCell::Scalar(0.01),
                    surface_ros: PerCell::Scalar(2.),
                    surface_fli: PerCell::Scalar(200.),
                    surface_hpua: PerCell::Scalar(1000.),
                    ..CrownColumns::default()
                },
            )?;
            let run = ModelRun::Crown {
                raster,
                plan,
                fuels: &fuels,
                tweaks: CrownTweakColumns {
                    fm10: SurfaceTweakColumns {
                        reaction_intensity: Some(PerCell::Cells(&reaction[start..end])),
                        ..SurfaceTweakColumns::default()
                    },
                    crown_fraction_burned: Some(PerCell::Cells(&fraction[start..end])),
                    ..CrownTweakColumns::default()
                },
            };
            let ModelBatch::Crown(output) = run_model(&run, &OutputRequest::crown(), threads)?
            else {
                unreachable!("Crown configuration returns Crown columns")
            };
            // Validate every stored field against explicit scalar composition outside timing.
            for (name, column) in output.into_named_columns() {
                for index in start..end {
                    let mut supplied = SurfaceFireInputs::new();
                    supplied.surface_ros = 2.;
                    supplied.surface_fli = 200.;
                    supplied.surface_hpua = 1000.;
                    let expected = plan.run_cell(
                        &CrownCell {
                            wind_speed: wind[index],
                            canopy_base_height: 4.,
                            canopy_height: 30.,
                            canopy_bulk_density: 0.01,
                            ..CrownCell::default()
                        },
                        &supplied,
                        fuels.for_grid(10),
                        CrownTweaks {
                            fm10: SurfaceTweaks {
                                reaction_intensity: Some(reaction[index]),
                                ..SurfaceTweaks::default()
                            },
                            crown_fraction_burned: Some(fraction[index]),
                            ..CrownTweaks::default()
                        },
                    );
                    match &column {
                        ColumnValues::Float64(values) => {
                            let wanted = expected
                                .named_values()
                                .into_iter()
                                .find(|(key, _)| *key == name)
                                .expect("known model output")
                                .1;
                            let actual = values[index - start];
                            assert!(
                                actual.to_bits() == wanted.to_bits()
                                    || (actual.is_nan() && wanted.is_nan()),
                                "{name}/{index}"
                            );
                        }
                        ColumnValues::Int32(values) => {
                            assert_eq!(values[index - start], expected.fire_type)
                        }
                        ColumnValues::Bool(_) => {
                            unreachable!("model output is physical or categorical")
                        }
                    }
                }
            }
        }
    }
    println!("Configured raster tiles and all outputs match scalar composition at 1/4 workers");
    Ok(())
}
