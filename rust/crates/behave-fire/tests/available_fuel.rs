//! Missing fuel prerequisites never silently become finite reaction contributions.
use behave_fire::engine::{
    run_model, CrownPlan, ModelBatch, ModelRun, OutputRequest, PreparedFuel, PreparedFuels,
    SurfaceCell, SurfaceOptions, SurfacePlan, SurfaceSource, SurfaceTweaks, WafMethod,
};
use behave_fire::raster::{
    CrownColumns, CrownRaster, CrownTweakColumns, FuelbedColumns, SurfaceColumns, SurfaceRaster,
    SurfaceTweakColumns,
};
use behave_fire::surface::FuelbedIntermediates;
use behave_fire::{FuelModels, GridShape, PerCell};

fn supplied(
    bed: FuelbedIntermediates,
    tweaks: SurfaceTweakColumns<'_>,
) -> behave_fire::batch::SurfaceBatch {
    let run = ModelRun::Surface {
        raster: SurfaceRaster::new(
            GridShape::new(vec![1]).unwrap(),
            SurfaceColumns {
                wind_speed: PerCell::Scalar(88.),
                user_waf: Some(PerCell::Scalar(0.4)),
                ..SurfaceColumns::default()
            },
        )
        .unwrap(),
        plan: SurfacePlan::new(SurfaceOptions {
            waf_method: WafMethod::UserInput,
            ..SurfaceOptions::default()
        }),
        source: SurfaceSource::Fuelbeds(FuelbedColumns::uniform(bed)),
        tweaks,
    };
    let ModelBatch::Surface(output) = run_model(&run, &OutputRequest::surface(), 1).unwrap() else {
        unreachable!()
    };
    output
}

#[test]
fn missing_prepared_fuel_properties_propagate_in_surface_and_crown() {
    let registry = FuelModels::default();
    for property in ["depth", "extinction"] {
        let rows = (0..=260)
            .map(|id| {
                let mut row = PreparedFuel::from_model(registry.for_grid(id));
                if [10, 101, 124].contains(&id) {
                    if property == "depth" {
                        row.fuel_bed_depth = f64::NAN;
                    } else {
                        row.moisture_of_extinction_dead = f64::NAN;
                    }
                }
                row
            })
            .collect();
        let fuels = PreparedFuels::from_rows(rows).unwrap();
        for id in [10, 101, 124] {
            let raster = SurfaceRaster::new(
                GridShape::new(vec![1]).unwrap(),
                SurfaceColumns {
                    fuel_model_grid: PerCell::Scalar(id),
                    wind_speed: PerCell::Scalar(88.),
                    user_waf: Some(PerCell::Scalar(0.4)),
                    ..SurfaceColumns::default()
                },
            )
            .unwrap();
            let plan = SurfacePlan::new(SurfaceOptions {
                waf_method: WafMethod::UserInput,
                ..SurfaceOptions::default()
            });
            let run = ModelRun::Surface {
                raster,
                plan,
                source: SurfaceSource::Fuels(&fuels),
                tweaks: SurfaceTweakColumns::default(),
            };
            let ModelBatch::Surface(output) =
                run_model(&run, &OutputRequest::surface(), 1).unwrap()
            else {
                unreachable!()
            };
            assert!(output.reaction_intensity[0].is_nan(), "{id}/{property}");
            assert!(output.heat_per_unit_area[0].is_nan(), "{id}/{property}");
            assert!(output.spread_rate[0].is_nan(), "{id}/{property}");
            assert!(
                output.residence_time[0].is_finite(),
                "independent sigma {id}/{property}"
            );
            assert_eq!(output.midflame_wind_speed, [35.2]);
            // Public scalar plans share the configured model's missing-data policy.
            assert!(plan
                .run_cell(
                    &SurfaceCell::new(id),
                    fuels.for_grid(id),
                    SurfaceTweaks::default()
                )
                .reaction_intensity
                .is_nan());
        }
        let run = ModelRun::Crown {
            raster: CrownRaster::new(
                GridShape::new(vec![1]).unwrap(),
                CrownColumns {
                    wind_speed: PerCell::Scalar(88.),
                    canopy_base_height: PerCell::Scalar(4.),
                    canopy_height: PerCell::Scalar(30.),
                    canopy_bulk_density: PerCell::Scalar(0.01),
                    surface_ros: PerCell::Scalar(2.),
                    surface_fli: PerCell::Scalar(200.),
                    surface_hpua: PerCell::Scalar(1000.),
                    ..CrownColumns::default()
                },
            )
            .unwrap(),
            plan: CrownPlan::default(),
            fuels: &fuels,
            tweaks: CrownTweakColumns::default(),
        };
        let ModelBatch::Crown(output) = run_model(&run, &OutputRequest::crown(), 1).unwrap() else {
            unreachable!()
        };
        assert!(
            output.crown_fire_spread_rate[0].is_nan(),
            "Crown/{property}"
        );
        assert!(output.canopy_heat_per_unit_area[0].is_finite());
        assert!(output.crown_critical_surface_fire_line_intensity[0].is_finite());
    }
}

#[test]
fn missing_supplied_reaction_prerequisites_preserve_independent_outputs() {
    let fuels = PreparedFuels::from_registry(&FuelModels::default());
    let original =
        SurfacePlan::default().prepare_fuelbed(&SurfaceCell::new(124), fuels.for_grid(124));
    for property in [
        "relative_packing_ratio",
        "moe_dead",
        "moe_live",
        "w_silica_dead",
        "w_silica_live",
    ] {
        let mut bed = original;
        match property {
            "relative_packing_ratio" => bed.relative_packing_ratio = f64::NAN,
            "moe_dead" => bed.moe_dead = f64::NAN,
            "moe_live" => bed.moe_live = f64::NAN,
            "w_silica_dead" => bed.w_silica_dead = f64::NAN,
            _ => bed.w_silica_live = f64::NAN,
        }
        let output = supplied(bed, SurfaceTweakColumns::default());
        assert!(output.reaction_intensity[0].is_nan(), "{property}");
        assert!(output.heat_per_unit_area[0].is_nan(), "{property}");
        assert!(output.spread_rate[0].is_nan(), "{property}");
        assert_eq!(
            output.residence_time[0].to_bits(),
            (384. / original.sigma).to_bits()
        );
        assert_eq!(output.midflame_wind_speed, [35.2]);
        let overridden = supplied(
            bed,
            SurfaceTweakColumns {
                reaction_intensity: Some(PerCell::Scalar(333.)),
                ..SurfaceTweakColumns::default()
            },
        );
        assert_eq!(overridden.reaction_intensity, [333.]);
        assert!(overridden.heat_per_unit_area[0].is_finite());
        // A known false physical threshold bypasses unused missing reaction inputs.
        bed.sigma = 0.;
        let zero = supplied(bed, SurfaceTweakColumns::default());
        assert_eq!(zero.reaction_intensity, [0.]);
        assert_eq!(zero.heat_per_unit_area, [0.]);
    }
}
